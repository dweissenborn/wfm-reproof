"""Independent re-proof of arXiv:2609.18182 (WFM: Wiki Foundation Model for Complex Agentic Reasoning).

Measures everything the paper lets an outsider check, on the paper's own public benchmarks
(HippoRAG 2 subsets: 1,000 questions each from HotpotQA, 2WikiMultihopQA and MuSiQue) with its
stated embedder (all-MiniLM-L6-v2). No training, no API calls, CPU only.

Lab 1  re-measures the paper's "Native RAG" baseline and tests the published cells for
       whole-number offsets from the measurement.
Lab 2  holds embedder and queries fixed and varies only the stored representation: flat passages,
       triples, a triple-only entity graph (the paper's "Ordinary graph" ablation), zero-LLM title
       links, NER entity links and the full Wiki Graph. Scores Recall@k with paired bootstrap CIs,
       storage and build cost, and repeats MuSiQue with three OpenIE LLMs.
Lab 3  audits the paper's recall tables from its TeX source.
Lab 4  asks what fits in a model's context window: how many tokens of retrieved context are read
       before the gold answer appears, per representation.
Lab 5  reverse-engineers the structure the link walk uses (overlay.py): how evidence passages are
       linked (same home, direct reference, shared identifier), whether dense retrieval gets a
       foothold, and how many missed passages sit one hop from a seed. Compares the PageRank walk
       with a one-hop expansion that runs as a single filtered vector query.

Run: uv run reproof.py   (fetches pinned inputs to data/, writes results/)
"""

import hashlib
import http.client
import io
import json
import math
import platform
import re
import tarfile
import time
import urllib.request
from datetime import date
from importlib.metadata import version
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import overlay

ROOT = Path(__file__).parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
PAGE_TEMPLATE = ROOT / "page.html"
PAGE_OUT = RESULTS / "wfm-reproof.html"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DIM = 384
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")  # the paper's table order
KS = (2, 5, 10, 20)
METHODS = ("flat", "triples", "triple_graph", "id_links", "wiki_links", "wiki_graph")
LLM_CALLS_PER_PASSAGE = {"flat": 0, "triples": 2, "triple_graph": 2, "id_links": 0, "wiki_links": 1, "wiki_graph": 2}
CONTEXT_METHODS = ("facts", "flat", "id_links", "wiki_links", "wiki_graph")
# Store-friendly stand-ins for the PageRank walk over an overlay (Lab 5).
LAB5_VARIANTS = {"expand": overlay.expand_once, "walk_once": overlay.walk_once}
BUDGETS = (64, 128, 256, 512, 1024, 2048, 4096)
CONTEXT_PASSAGES = 60  # ranked passages kept per query: enough to fill the largest budget
CONTEXT_FACTS = 600
# HippoRAG 2 graph-search defaults: fact linking, passage node weight, PPR damping.
TOP_FACTS = 5
TOP_PHRASES = 5
TOP_SEED_PASSAGES = 5  # link-only graphs have no facts, so they seed from as many passages
PASSAGE_NODE_WEIGHT = 0.05
DAMPING = 0.5
PPR_ITERS = 30  # residual shrinks by DAMPING per step: 0.5**30 < 1e-9
BATCH = 100
BOOTSTRAPS = 2000
TITLE_MIN_CHARS = 4  # shorter titles ("It", "Ra") would link half the corpus
TITLE_MAX_WORDS = 8
# Instruction + one-shot demo chars per passage across HippoRAG's two OpenIE calls (NER, then
# triples), measured from its src/openie_extraction_instructions.py.
OPENIE_PROMPT_CHARS = 2400
CHARS_PER_TOKEN = 4
ROUNDING = 0.005  # half a unit in the paper's last reported digit

HIPPORAG = "https://raw.githubusercontent.com/OSU-NLP-Group/HippoRAG"
MAIN = f"{HIPPORAG}/1438aba3fc44ff10573e5a5e1e7cc3c7f9794aff"
LEGACY = f"{HIPPORAG}/b144c46df14cabe5f5822d8caded4bec5f709461"
PAPER_SOURCE = "https://arxiv.org/src/2609.18182v1"
PAPER_EXP_TEX_SHA1 = "4ae1c48ce21d5d2b45b10c87369b54f11c65e4e2"
DOWNLOADS = {  # path under data/ -> (pinned URL, SHA-1)
    "v2/2wikimultihopqa.json": (f"{MAIN}/reproduce/dataset/2wikimultihopqa.json", "8cc40ddc6b4fd3d023d0fb47a0cb95381bfefae8"),
    "v2/2wikimultihopqa_corpus.json": (f"{MAIN}/reproduce/dataset/2wikimultihopqa_corpus.json", "beeaf13ecd5ccfa7f493dce997c1bd7b7f3471b8"),
    "v2/hotpotqa.json": (f"{MAIN}/reproduce/dataset/hotpotqa.json", "1f198572a8f1014c7c0cda519a3bcaeb76830e2a"),
    "v2/hotpotqa_corpus.json": (f"{MAIN}/reproduce/dataset/hotpotqa_corpus.json", "8566c8b193d2626ecf0f751817bb32f1d839142b"),
    "v2/musique.json": (f"{MAIN}/reproduce/dataset/musique.json", "32878a39b4a99c6e276be1562cd30f0e87485e05"),
    "v2/musique_corpus.json": (f"{MAIN}/reproduce/dataset/musique_corpus.json", "22bd0dc18f4cf1595821887e0157b13b8f6e432d"),
    "v1/2wikimultihopqa.json": (f"{LEGACY}/data/2wikimultihopqa.json", "8cc40ddc6b4fd3d023d0fb47a0cb95381bfefae8"),
    "v1/2wikimultihopqa_corpus.json": (f"{LEGACY}/data/2wikimultihopqa_corpus.json", "beeaf13ecd5ccfa7f493dce997c1bd7b7f3471b8"),
    "v1/hotpotqa.json": (f"{LEGACY}/data/hotpotqa.json", "8e5a4b41fdeb1a56e4eaa261a4dfa8e8c630609d"),
    "v1/hotpotqa_corpus.json": (f"{LEGACY}/data/hotpotqa_corpus.json", "a188647f19b075752677509427c8ae1b263fc28e"),
    "v1/musique.json": (f"{LEGACY}/data/musique.json", "32878a39b4a99c6e276be1562cd30f0e87485e05"),
    "v1/musique_corpus.json": (f"{LEGACY}/data/musique_corpus.json", "22bd0dc18f4cf1595821887e0157b13b8f6e432d"),
    "openie/2wikimultihopqa.json": (
        f"{LEGACY}/output/openie_2wikimultihopqa_results_ner_gpt-3.5-turbo-1106_6119.json",
        "2df85aa3edcbbb824895ff5fd15fdc2795f899a5",
    ),
    "openie/hotpotqa.json": (
        f"{LEGACY}/output/openie_hotpotqa_results_ner_gpt-3.5-turbo-1106_9221.json",
        "bf37f44271b295f6bad8ada2b2f15ca4ba04cff7",
    ),
    "openie/musique.json": (
        f"{LEGACY}/output/openie_musique_results_ner_gpt-3.5-turbo-1106_11656.json",
        "e750d1bee8170eb6b28c2ede2b84f4e4ce6c47c9",
    ),
    "openie/musique-gpt-4o-mini.json": (
        f"{MAIN}/outputs/musique/openie_results_ner_gpt-4o-mini.json",
        "ed047703134f28b048de0e32f8c3719a9cdc63ef",
    ),
    "openie/musique-llama-3.3-70b.json": (
        f"{MAIN}/outputs/musique/openie_results_ner_meta-llama_Llama-3.3-70B-Instruct.json",
        "8901148bb407150cb69e036838db0a4ccefc6fff",
    ),
}

_model = None


def fetch(url, attempts=3):
    """Bytes at url, read in chunks (one large read() stops at 1 MiB against arXiv) and retried on drops."""
    request = urllib.request.Request(url, headers={"User-Agent": "wfm-reproof"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return b"".join(iter(lambda: response.read(1 << 16), b""))
        except (OSError, http.client.HTTPException):
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)


def ensure_data():
    """Download every pinned input once and refuse to run on anything that fails its SHA-1."""
    for rel, (url, sha1) in DOWNLOADS.items():
        path = DATA / rel
        if not path.exists():
            print(f"fetch {rel}", flush=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(fetch(url))
        digest = hashlib.sha1(path.read_bytes()).hexdigest()
        if digest != sha1:
            raise SystemExit(f"{rel}: SHA-1 {digest} does not match pinned {sha1}; delete the file to re-fetch")


def paper_tex():
    """The paper's experiments section (exp.tex) from its pinned arXiv v1 source."""
    path = DATA / "paper" / "exp.tex"
    if not path.exists():
        print("fetch paper source", flush=True)
        with tarfile.open(fileobj=io.BytesIO(fetch(PAPER_SOURCE)), mode="r:gz") as tar:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(tar.extractfile("exp.tex").read())
    raw = path.read_bytes()
    if hashlib.sha1(raw).hexdigest() != PAPER_EXP_TEX_SHA1:
        raise SystemExit("data/paper/exp.tex does not match the pinned arXiv v1 source")
    return raw.decode("utf-8")


def load(version, name):
    """Return (queries, docs); docs use HippoRAG's "title\\ntext" passage format."""
    queries = json.loads((DATA / version / f"{name}.json").read_text(encoding="utf-8"))
    corpus = json.loads((DATA / version / f"{name}_corpus.json").read_text(encoding="utf-8"))
    if isinstance(corpus, dict):  # v1 hotpotqa stores {title: [sentences]}
        corpus = [{"title": t, "text": "".join(s)} for t, s in corpus.items()]
    return queries, [f"{d['title']}\n{d['text']}" for d in corpus]


def gold_indices(queries, docs, name):
    """Corpus rows of each query's supporting passages (HippoRAG 2 get_gold_docs rules)."""
    row = {d: i for i, d in enumerate(docs)}
    golds = []
    for sample in queries:
        if "supporting_facts" in sample:
            titles = {t for t, _ in sample["supporting_facts"]}
            sep = "" if name == "hotpotqa" else " "
            gold = {f"{t}\n{sep.join(s)}" for t, s in sample["context"] if t in titles}
        else:
            gold = {f"{p['title']}\n{p['paragraph_text']}" for p in sample["paragraphs"] if p["is_supporting"]}
        missing = gold - row.keys()
        assert gold and not missing, f"{name}: gold passage missing from corpus: {sorted(missing)[:1]}"
        golds.append({row[g] for g in gold})
    return golds


def embed(texts):
    """L2-normalised MiniLM vectors and encode seconds, cached by model revision and content."""
    digest = hashlib.sha1("\x1e".join([MODEL_REVISION, *texts]).encode("utf-8")).hexdigest()[:16]
    path, seconds = CACHE / f"{digest}.npy", CACHE / f"{digest}.seconds"
    if not path.exists():
        global _model
        if _model is None:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(MODEL, revision=MODEL_REVISION, device="cpu")
        print(f"embed {len(texts)} texts", flush=True)
        start = time.perf_counter()
        vectors = _model.encode(texts, batch_size=128, normalize_embeddings=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors.astype(np.float32))
        seconds.write_text(f"{time.perf_counter() - start:.1f}")
    return np.load(path), float(seconds.read_text())


def top_k(scores, k):
    """Indices of each row's k best scores, best first; ties go to the lower index so runs repeat exactly."""
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def recall_matrix(ranked, golds):
    """Per-query Recall@k in percent; one row per query, one column per k in KS."""
    return np.array([[100 * len(g.intersection(r[:k].tolist())) / len(g) for k in KS] for r, g in zip(ranked, golds)])


def whole_number_gaps(measured, published):
    """Measured minus published per k, and whether every gap is a whole number within the paper's rounding."""
    gaps = [m - p for m, p in zip(measured, published)]
    return gaps, all(abs(g - round(g)) <= ROUNDING + 1e-9 for g in gaps)


def lab1(published):
    """Native RAG (flat MiniLM retrieval) on the HippoRAG 2 subsets, storage variants, and gaps to the paper."""
    results = {}
    for i, name in enumerate(DATASETS):
        queries, docs = load("v2", name)
        golds = gold_indices(queries, docs, name)
        q, _ = embed([s["question"] for s in queries])
        p, p_seconds = embed(docs)
        text_only, _ = embed([d.split("\n", 1)[1] for d in docs])
        int8 = np.round(p * (127 / np.abs(p).max())).astype(np.int8)
        variants = {
            "title+text fp32": (q @ p.T, DIM * 4),
            "text-only fp32": (q @ text_only.T, DIM * 4),
            "title+text int8": (q @ int8.T.astype(np.float32), DIM),
            "title+text 1-bit": (q @ np.sign(p).T, DIM // 8),
        }
        recalls = {v: recall_matrix(top_k(s, max(KS)), golds).mean(axis=0).tolist() for v, (s, _) in variants.items()}
        paper = published[4 * i : 4 * i + 4]
        gaps = {v: dict(zip(("gaps", "whole_numbers"), whole_number_gaps(r, paper))) for v, r in recalls.items()}
        results[name] = {
            "passages": len(docs),
            "embed_seconds": p_seconds,
            "paper_native_rag": paper,
            "variants": {v: {"recall": recalls[v], "bytes_per_vector": variants[v][1], **gaps[v]} for v in variants},
        }
    return results


def norm(phrase):
    """HippoRAG phrase normalisation: lowercase, alphanumerics and single spaces only."""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", phrase.lower()).split())


def title_links(docs):
    """Zero-LLM wiki links: a passage's page title is its home; every other title its text names is a mention.

    Titles are matched case-kept, so "Home" the film does not match "home" the noun.
    """
    homes = []
    for doc in docs:
        key = " ".join(overlay.words(re.sub(r"\s*\([^)]*\)\s*$", "", doc.split("\n", 1)[0])))
        homes.append(key if len(key) >= TITLE_MIN_CHARS else None)
    finder = overlay.register_finder([h for h in homes if h], TITLE_MAX_WORDS)
    return overlay.link([d.split("\n", 1)[1] for d in docs], homes, [finder])


def bipartite(node_passage):
    """Undirected node-passage graph with the nodes first and the passages after them."""
    return sp.bmat([[None, node_passage], [node_passage.T, None]]).tocsr()


def build(openie_file, docs):
    """Facts, entities, graphs, storage and build-cost accounting for one corpus and one OpenIE run."""
    oie = json.loads((DATA / "openie" / openie_file).read_text(encoding="utf-8"))["docs"]
    assert len(oie) == len(docs)
    facts, entities = {}, {}
    fact_passage, triple_mention, ner_mention = set(), set(), set()
    for i, doc in enumerate(oie):
        for t in doc["extracted_triples"]:
            if len(t) != 3 or not all(isinstance(x, str) for x in t) or not (norm(t[0]) and norm(t[2])):
                continue
            fact_passage.add((facts.setdefault(tuple(t), len(facts)), i))
            for x in (t[0], t[2]):
                triple_mention.add((entities.setdefault(norm(x), len(entities)), i))
        for x in doc["extracted_entities"]:
            if isinstance(x, str) and norm(x):
                ner_mention.add((entities.setdefault(norm(x), len(entities)), i))

    n_fact, n_ent, n_pass = len(facts), len(entities), len(docs)
    fact_s = np.array([entities[norm(t[0])] for t in facts])
    fact_o = np.array([entities[norm(t[2])] for t in facts])
    pairs = np.array(sorted(fact_passage))
    ent_pass_triple = overlay.incidence(triple_mention, n_ent, n_pass)
    ent_pass_ner = overlay.incidence(ner_mention, n_ent, n_pass)
    ent_pass_wiki = overlay.incidence(triple_mention | ner_mention, n_ent, n_pass)
    titles, title_home, title_mention = title_links(docs)
    title_home = overlay.incidence(title_home, len(titles), n_pass)
    title_mention = overlay.incidence(title_mention, len(titles), n_pass)
    title_pass = title_home + title_mention
    name_bytes = np.array([len(e.encode("utf-8")) for e in entities])
    s, o = fact_s[pairs[:, 0]], fact_o[pairs[:, 0]]
    keep = s != o
    ee = sp.csr_matrix((np.ones(keep.sum(), np.float32), (s[keep], o[keep])), shape=(n_ent, n_ent))
    ee = (ee + ee.T).tocsr()

    sizes = {
        "passage text": sum(len(d.encode("utf-8")) for d in docs),
        "passage vectors": n_pass * DIM * 4,
        "fact text": sum(len("\t".join(t).encode("utf-8")) for t in facts),
        "fact vectors": n_fact * DIM * 4,
        "fact provenance + endpoints": len(pairs) * 8 + n_fact * 8,
        "entity names": int(name_bytes.sum()),
        "entity-entity edges": ee.nnz // 2 * 12,
        "entity-passage links": ent_pass_wiki.nnz * 8,
        "NER entity names": int(name_bytes[np.unique(ent_pass_ner.nonzero()[0])].sum()),
        "NER entity-passage links": ent_pass_ner.nnz * 8,
        "title index": sum(len(t.encode("utf-8")) for t in titles),
        "title links": title_pass.nnz * 8,
    }
    uses = {
        "flat": ["passage text", "passage vectors"],
        "triples": ["fact text", "fact vectors", "fact provenance + endpoints"],
        "triple_graph": ["fact text", "fact vectors", "fact provenance + endpoints", "entity names", "entity-entity edges"],
        "id_links": ["passage text", "passage vectors", "title index", "title links"],
        "wiki_links": ["passage text", "passage vectors", "NER entity names", "NER entity-passage links"],
        "wiki_graph": ["passage text", "passage vectors", "fact text", "fact vectors", "fact provenance + endpoints",
                       "entity names", "entity-entity edges", "entity-passage links"],
    }
    entity_chars = sum(len(json.dumps(d["extracted_entities"])) for d in oie)
    triple_chars = sum(len(json.dumps(d["extracted_triples"])) for d in oie)
    passage_chars = sum(len(d) for d in docs)
    return {
        "fact_texts": [" ".join(t) for t in facts],
        "fact_s": fact_s,
        "fact_o": fact_o,
        "pairs": pairs,
        "passages_per_entity": np.asarray(ent_pass_triple.sum(axis=1)).ravel(),
        "ent_pass_triple": ent_pass_triple,
        "graphs": {
            "triple_graph": ee,
            "id_links": bipartite(title_pass),
            "wiki_links": bipartite(ent_pass_ner),
            "wiki_graph": sp.bmat([[ee, ent_pass_wiki], [ent_pass_wiki.T, None]]).tocsr(),
        },
        "overlays": {"id_links": (title_home, title_mention), "wiki_links": (None, ent_pass_ner)},
        "stats": {
            "passages": n_pass,
            "unique_facts": n_fact,
            "entities": n_ent,
            "entity_entity_edges": ee.nnz // 2,
            "entity_passage_links": ent_pass_wiki.nnz,
            "titles": len(titles),
            "title_links": title_pass.nnz,
        },
        "storage_bytes": {m: sum(sizes[c] for c in parts) for m, parts in uses.items()},
        "storage_bytes_int8_vectors": {
            m: sum(sizes[c] // 4 if "vectors" in c else sizes[c] for c in parts) for m, parts in uses.items()
        },
        "storage_components": sizes,
        "llm_calls_to_build": {m: LLM_CALLS_PER_PASSAGE[m] * n_pass for m in METHODS},
        "openie_tokens_estimate": {
            "input": (2 * passage_chars + entity_chars + n_pass * OPENIE_PROMPT_CHARS) / CHARS_PER_TOKEN,
            "output": (entity_chars + triple_chars) / CHARS_PER_TOKEN,
        },
    }


def fact_max_per_passage(fact_scores, pairs, n_passages):
    """Passage score = best score among the facts extracted from it; fact-less passages rank last."""
    by_passage = pairs[np.argsort(pairs[:, 1], kind="stable")]
    starts = np.flatnonzero(np.r_[True, np.diff(by_passage[:, 1]) > 0])
    out = np.full((len(fact_scores), n_passages), -np.inf, np.float32)
    out[:, by_passage[starts, 1]] = np.maximum.reduceat(fact_scores[:, by_passage[:, 0]], starts, axis=1)
    return out


def entity_seeds(fact_scores, fact_s, fact_o, passages_per_entity):
    """HippoRAG 2 phrase weights: top facts' min-max scores / entity passage count, top phrases kept."""
    lo, hi = fact_scores.min(axis=1, keepdims=True), fact_scores.max(axis=1, keepdims=True)
    scores = (fact_scores - lo) / (hi - lo)
    top = np.argpartition(-scores, TOP_FACTS - 1, axis=1)[:, :TOP_FACTS]
    top_scores = np.take_along_axis(scores, top, axis=1)
    rows = np.repeat(np.arange(len(scores)), TOP_FACTS)
    weight = np.zeros((len(scores), len(passages_per_entity)), np.float32)
    count = np.zeros_like(weight)
    for ends in (fact_s[top], fact_o[top]):
        np.add.at(weight, (rows, ends.ravel()), (top_scores / passages_per_entity[ends]).ravel())
        np.add.at(count, (rows, ends.ravel()), 1)
    weight = np.divide(weight, count, out=weight, where=count > 0)
    keep = np.argpartition(-weight, TOP_PHRASES - 1, axis=1)[:, :TOP_PHRASES]
    seeds = np.zeros_like(weight)
    np.put_along_axis(seeds, keep, np.take_along_axis(weight, keep, axis=1), axis=1)
    return seeds


def transition(adj):
    """Column-stochastic transition matrix and dangling-node mask of an undirected graph."""
    deg = np.asarray(adj.sum(axis=0), dtype=np.float64).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    return (adj @ sp.diags(inv)).astype(np.float32).tocsr(), deg == 0


def ppr(trans, dangling, seeds):
    """Personalised PageRank, one column per query; dangling mass restarts at the seeds."""
    reset = seeds / seeds.sum(axis=0, keepdims=True)
    rank = reset.copy()
    for _ in range(PPR_ITERS):
        rank = (1 - DAMPING) * reset + DAMPING * (trans @ rank + reset * rank[dangling].sum(axis=0))
    return rank


def paired_ci(a, b, rng):
    """Mean of a-b with a 95% paired-bootstrap interval over queries."""
    diffs = (a - b)[rng.integers(0, len(a), (BOOTSTRAPS, len(a)))].mean(axis=1)
    return [float((a - b).mean()), *np.percentile(diffs, [2.5, 97.5]).tolist()]


def answer_norm(text):
    """SQuAD-style answer normalisation: lowercase, punctuation to spaces, no articles, single spaces."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answers_for(queries):
    """Normalised gold answers per query, and which queries test retrieval (not yes/no, not named in the question)."""
    answers, eligible = [], []
    for sample in queries:
        aliases = {answer_norm(a) for a in [sample["answer"], *sample.get("answer_aliases", [])]} - {""}
        question = f" {answer_norm(sample['question'])} "
        answers.append(sorted(aliases))
        eligible.append(bool(aliases) and not aliases & {"yes", "no"} and not any(f" {a} " in question for a in aliases))
    return answers, np.array(eligible)


def token_counts(texts):
    """Approximate LLM tokens per context unit, plus one for the separator between units."""
    return [math.ceil(len(t) / CHARS_PER_TOKEN) + 1 for t in texts]


def tokens_to_answer(ranked, unit_text, unit_tokens, answers):
    """Context tokens read, in rank order, until each query's answer first appears; inf if it never does."""
    need = np.full(len(ranked), np.inf)
    for i, (row, aliases) in enumerate(zip(ranked, answers)):
        used = 0
        for u in row:
            text = f" {unit_text[u]} "
            ends = [text.find(f" {a} ") + len(a) + 1 for a in aliases if f" {a} " in text]
            if ends:
                need[i] = used + math.ceil(min(ends) / CHARS_PER_TOKEN)
                break
            used += unit_tokens[u]
            if used >= BUDGETS[-1]:
                break
    return need


def evaluate(name, openie_file, rng):
    """Recall, paired CIs, storage, build cost and context coverage of every representation for one corpus."""
    queries, docs = load("v1", name)
    golds = gold_indices(queries, docs, name)
    g = build(openie_file, docs)
    q, _ = embed([s["question"] for s in queries])
    p, p_seconds = embed(docs)
    f, f_seconds = embed(g["fact_texts"])
    walks = {m: transition(a) for m, a in g["graphs"].items()}
    link_nodes = {m: g["graphs"][m].shape[0] - len(docs) for m in ("id_links", "wiki_links", "wiki_graph")}
    node_units = {m: mention if home is None else home + mention for m, (home, mention) in g["overlays"].items()}
    ranked = {m: [] for m in METHODS}
    variants = {v: {m: [] for m in node_units} for v in LAB5_VARIANTS}
    fact_ranked = []
    for start in range(0, len(q), BATCH):
        qb = q[start : start + BATCH]
        passage_scores, fact_scores = qb @ p.T, qb @ f.T
        seeds = entity_seeds(fact_scores, g["fact_s"], g["fact_o"], g["passages_per_entity"])
        ranked["flat"].append(top_k(passage_scores, CONTEXT_PASSAGES))
        ranked["triples"].append(top_k(fact_max_per_passage(fact_scores, g["pairs"], len(docs)), CONTEXT_PASSAGES))
        rank = ppr(*walks["triple_graph"], seeds.T)
        ranked["triple_graph"].append(top_k((g["ent_pass_triple"].T @ rank).T, CONTEXT_PASSAGES))
        lo = passage_scores.min(axis=1, keepdims=True)
        normalised = (passage_scores - lo) / np.ptp(passage_scores, axis=1, keepdims=True)
        best = np.argpartition(-normalised, TOP_SEED_PASSAGES - 1, axis=1)[:, :TOP_SEED_PASSAGES]
        for m in ("id_links", "wiki_links"):
            link_seeds = np.zeros((len(qb), link_nodes[m] + len(docs)), np.float32)
            np.put_along_axis(link_seeds, best + link_nodes[m], np.take_along_axis(normalised, best, axis=1), axis=1)
            ranked[m].append(top_k(ppr(*walks[m], link_seeds.T)[link_nodes[m] :].T, CONTEXT_PASSAGES))
        rank = ppr(*walks["wiki_graph"], np.hstack([seeds, normalised * PASSAGE_NODE_WEIGHT]).T)
        ranked["wiki_graph"].append(top_k(rank[link_nodes["wiki_graph"] :].T, CONTEXT_PASSAGES))
        fact_ranked.append(top_k(fact_scores, CONTEXT_FACTS))
        for m, node_unit in node_units.items():
            for v, rank_fn in LAB5_VARIANTS.items():
                variants[v][m].append(top_k(rank_fn(passage_scores, node_unit, TOP_SEED_PASSAGES), CONTEXT_PASSAGES))
    ranked = {m: np.vstack(r) for m, r in ranked.items()}
    variants = {v: {m: np.vstack(r) for m, r in by_overlay.items()} for v, by_overlay in variants.items()}
    recalls = {m: recall_matrix(r, golds) for m, r in ranked.items()}
    contrasts = [
        ("wiki_graph", "flat"),
        ("wiki_graph", "triple_graph"),
        ("wiki_graph", "wiki_links"),
        ("wiki_links", "flat"),
        ("id_links", "flat"),
        ("wiki_links", "id_links"),
        ("triples", "flat"),
        ("triple_graph", "flat"),
    ]

    answers, eligible = answers_for(queries)
    units = {
        "passages": ([answer_norm(d) for d in docs], token_counts(docs)),
        "facts": ([answer_norm(t) for t in g["fact_texts"]], token_counts(g["fact_texts"])),
    }
    def coverage(rows, kind):
        """Answer-in-context share per budget and the median tokens read, over eligible questions."""
        need = tokens_to_answer(rows, *units[kind], answers)[eligible]
        median = float(np.median(need))
        return {"coverage": [100 * float(np.mean(need <= b)) for b in BUDGETS], "median_tokens": median if math.isfinite(median) else None}

    context = {m: coverage(*((np.vstack(fact_ranked), "facts") if m == "facts" else (ranked[m], "passages"))) for m in CONTEXT_METHODS}
    structure_rng = np.random.default_rng(0)  # its own draws, so Lab 2's intervals stay put
    structure = {}
    for m, (home, mention) in g["overlays"].items():
        walked = {"walk": ranked[m], **{v: variants[v][m] for v in LAB5_VARIANTS}}
        structure[m] = {
            **overlay.profile(home, mention, golds, ranked["flat"], walked, TOP_SEED_PASSAGES, max(KS)),
            "variants": {},
        }
        for v in LAB5_VARIANTS:
            rec = recall_matrix(variants[v][m], golds)
            structure[m]["variants"][v] = {
                "recall": rec.mean(axis=0).tolist(),
                "minus_walk_ci95": {f"R@{KS[c]}": paired_ci(rec[:, c], recalls[m][:, c], structure_rng) for c in (1, 3)},
                "context": coverage(variants[v][m], "passages"),
            }
    return {
        "recall": {m: r.mean(axis=0).tolist() for m, r in recalls.items()},
        "paired_diff_ci95": {
            f"{a} - {b}": {f"R@{KS[c]}": paired_ci(recalls[a][:, c], recalls[b][:, c], rng) for c in (1, 3)}
            for a, b in contrasts
        },
        "context": {"eligible_questions": int(eligible.sum()), "methods": context},
        "structure": structure,
        "embed_seconds": {"passages": p_seconds, "facts": f_seconds},
        **{k: g[k] for k in ("stats", "storage_bytes", "storage_bytes_int8_vectors", "storage_components", "llm_calls_to_build", "openie_tokens_estimate")},
    }


def lab2():
    """Every representation on every dataset, plus MuSiQue under two stronger OpenIE LLMs."""
    rng = np.random.default_rng(0)
    results = {name: evaluate(name, f"{name}.json", rng) for name in DATASETS}
    results["musique extractors"] = {
        "gpt-3.5-turbo": results["musique"],
        **{llm: evaluate("musique", f"musique-{llm}.json", rng) for llm in ("gpt-4o-mini", "llama-3.3-70b")},
    }
    return results


def audit(tex):
    """Re-read the paper's recall tables from its TeX source and flag rows where Recall@k falls as k grows."""
    tables, violations = {}, []
    for label, per in (("tab:multihop-recall", 4), ("tab:memory-recall", 3)):
        body = tex.split(f"\\label{{{label}}}", 1)[1].split("\\end{tabular}", 1)[0]
        rows = {}
        for line in body.splitlines():
            cells = [c.strip() for c in line.strip().removesuffix("\\\\").split("&")]
            nums = [re.sub(r"\\textbf\{(.*)\}", r"\1", c) for c in cells[1:]]
            if nums and all(re.fullmatch(r"\d+(\.\d+)?", n) for n in nums):
                rows[re.sub(r"\\textsuperscript\{(\w+)\}", r"\1", cells[0])] = [float(n) for n in nums]
        tables[label] = rows
        for method, vals in rows.items():
            for block in range(len(vals) // per):
                seg = vals[block * per : (block + 1) * per]
                if any(x > y for x, y in zip(seg, seg[1:])):
                    violations.append({"table": label, "method": method, "block": block, "values": seg})
    admission = re.search(r"[^.]*planning values[^.]*\.", tex).group(0).strip()
    return {"tables": tables, "non_monotonic_recall": violations, "admission": admission}


def selftest():
    """Smallest checks that fail if ranking, recall, PPR, linking or answer matching breaks."""
    assert top_k(np.array([[0.1, 0.9, 0.5, 0.9]]), 3).tolist() == [[1, 3, 2]]
    assert recall_matrix(np.array([[3, 1] + [9] * 18]), [{1, 2}])[0].tolist() == [50.0] * 4
    path = sp.csr_matrix(np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], np.float32))
    rank = ppr(*transition(path), np.array([[1.0], [0], [0], [0]], np.float32)).ravel()
    assert rank[0] > rank[1] > rank[2] > rank[3] and abs(rank.sum() - 1) < 1e-5
    pairs = np.array([[0, 1], [1, 1], [2, 0]])
    assert fact_max_per_passage(np.array([[0.25, 0.75, 0.5]]), pairs, 3).tolist() == [[0.5, 0.75, -np.inf]]
    assert whole_number_gaps([51.275, 61.8], [31.28, 31.8]) == ([51.275 - 31.28, 61.8 - 31.8], True)
    assert not whole_number_gaps([58.0], [44.24])[1]
    assert title_links(["Alpha Beta\nWe met Gamma Delta and a home.", "Gamma Delta (band)\nNo links."]) == (
        {"Alpha Beta": 0, "Gamma Delta": 1},
        {(0, 0), (1, 1)},
        {(1, 0)},
    )
    overlay.selftest()
    assert answer_norm("The U.S. Army's 1st") == "u s army s 1st"
    assert tokens_to_answer(np.array([[0, 1]]), ["alpha beta", "gamma delta"], [3, 3], [["delta"]]).tolist() == [6.0]


def mb(n):
    """Bytes as megabytes with one decimal."""
    return f"{n / 1e6:.1f} MB"


def report(r):
    """Markdown summary of all four labs next to the paper's own numbers."""
    names = {"hotpotqa": "HotpotQA", "2wikimultihopqa": "2Wiki", "musique": "MuSiQue"}
    f1 = lambda xs: " / ".join(f"{x:.1f}" for x in xs)  # noqa: E731
    lines = ["# WFM re-proof results", "", f"Run {r['meta']['date']} on {r['meta']['platform']} in {r['meta']['seconds'] / 60:.0f} min.", ""]
    lines += ["## Lab 1 - Native RAG re-measured (all-MiniLM-L6-v2, HippoRAG 2 subsets)", ""]
    lines += ["| Dataset | Passages | Measured R@2/5/10/20 | Paper R@2/5/10/20 | Gap per k | Whole-number gaps |", "|---|---|---|---|---|---|"]
    for n in DATASETS:
        d = r["lab1"][n]
        v = d["variants"]["title+text fp32"]
        gaps = " / ".join(f"{g:+.3f}" for g in v["gaps"])
        lines.append(f"| {names[n]} | {d['passages']} | {f1(v['recall'])} | {f1(d['paper_native_rag'])} | {gaps} | {'yes' if v['whole_numbers'] else 'no'} |")
    lines += ["", "Storage variants of the flat index (R@5 / R@20, bytes per vector):", ""]
    lines += ["| Dataset | " + " | ".join(r["lab1"][DATASETS[0]]["variants"]) + " |", "|---" * 5 + "|"]
    for n in DATASETS:
        vs = r["lab1"][n]["variants"].values()
        lines.append(f"| {names[n]} | " + " | ".join(f"{v['recall'][1]:.1f} / {v['recall'][3]:.1f} ({v['bytes_per_vector']} B)" for v in vs) + " |")

    lines += ["", "## Lab 2 - Representations (v1 corpora, GPT-3.5 OpenIE)", ""]
    lines += ["| Dataset | Method | R@2 | R@5 | R@10 | R@20 | Storage fp32 | vs flat | Storage int8 | vs flat | LLM calls |", "|---" * 11 + "|"]
    for n in DATASETS:
        d = r["lab2"][n]
        for m in METHODS:
            sizes = [d["storage_bytes"][m], d["storage_bytes_int8_vectors"][m]]
            flats = [d["storage_bytes"]["flat"], d["storage_bytes_int8_vectors"]["flat"]]
            storage = " | ".join(f"{mb(s)} | {s / f:.1f}x" for s, f in zip(sizes, flats))
            recall = " | ".join(f"{x:.1f}" for x in d["recall"][m])
            lines.append(f"| {names[n]} | {m} | {recall} | {storage} | {d['llm_calls_to_build'][m]} |")
    lines += ["", "Average R@20 across the three datasets (the paper's ablation metric; WFM claims "
              f"{np.mean([r['audit']['tables']['tab:multihop-recall']['WFM'][4 * i + 3] for i in range(3)]):.2f}):", ""]
    for m in METHODS:
        lines.append(f"- {m}: {np.mean([r['lab2'][n]['recall'][m][3] for n in DATASETS]):.2f}")
    lines += ["", "Paired bootstrap 95% CI of recall differences (points):", ""]
    for n in DATASETS:
        for contrast, cis in r["lab2"][n]["paired_diff_ci95"].items():
            lines.append(f"- {names[n]} {contrast}: " + "; ".join(f"{k} {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]" for k, v in cis.items()))
    lines += ["", "Extractor robustness on MuSiQue (R@5 / R@20):", ""]
    lines += ["| OpenIE LLM | Unique facts | " + " | ".join(METHODS) + " |", "|---" * (len(METHODS) + 2) + "|"]
    for llm, d in r["lab2"]["musique extractors"].items():
        cells = " | ".join(f"{d['recall'][m][1]:.1f} / {d['recall'][m][3]:.1f}" for m in METHODS)
        lines.append(f"| {llm} | {d['stats']['unique_facts']} | {cells} |")
    lines += ["", "Build cost:", ""]
    for n in DATASETS:
        d = r["lab2"][n]
        t = d["openie_tokens_estimate"]
        lines.append(
            f"- {names[n]}: {d['stats']}; embed passages {d['embed_seconds']['passages']}s vs facts {d['embed_seconds']['facts']}s (CPU); "
            f"OpenIE ~{t['input'] / 1e6:.1f}M input + {t['output'] / 1e6:.1f}M output tokens"
        )

    lines += ["", "## Lab 4 - What fits in the model's context", ""]
    lines += ["Share of questions whose gold answer appears within the first N context tokens (answers not yes/no and not named in the question).", ""]
    lines += ["| Dataset | Context | " + " | ".join(f"{b} tok" for b in BUDGETS) + " | Median tokens |", "|---" * (len(BUDGETS) + 3) + "|"]
    for n in DATASETS:
        c = r["lab2"][n]["context"]
        for m in CONTEXT_METHODS:
            cov = " | ".join(f"{x:.1f}" for x in c["methods"][m]["coverage"])
            median = c["methods"][m]["median_tokens"]
            lines.append(f"| {names[n]} ({c['eligible_questions']} q) | {m} | {cov} | {'>' + str(BUDGETS[-1]) if median is None else f'{median:.0f}'} |")

    lines += ["", "## Lab 5 - The structure the link walk uses", ""]
    lines += ["Gold pairs: evidence passages of one question, by what links them. Missed: gold passages outside the dense top 20.", ""]
    head = ["Dataset", "Overlay", "Mentions/unit", "Units sharing an id", "Links on top 1% ids", "Gold pairs: same home / direct ref / shared id / none",
            "Dense foothold (gold in top 5)", "Missed one hop from a seed", "Missed recovered: walk / expand / walk once",
            "R@20 walk / expand / walk once", "Expand - walk R@20 (95% CI)", "Walk once - walk R@20 (95% CI)"]
    lines += ["| " + " | ".join(head) + " |", "|---" * len(head) + "|"]
    for n in DATASETS:
        for m, s in r["lab2"][n]["structure"].items():
            gp, v = s["gold_pairs"], s["variants"]
            cis = [v[x]["minus_walk_ci95"]["R@20"] for x in LAB5_VARIANTS]
            recovered = " / ".join(f"{s['missed_recovered'][x]:.1f}%" for x in ("walk", *LAB5_VARIANTS))
            recall = " / ".join(f"{x:.1f}" for x in (r["lab2"][n]["recall"][m][3], *(v[x]["recall"][3] for x in LAB5_VARIANTS)))
            lines.append(
                f"| {names[n]} | {m} | {s['mentions_per_unit']:.2f} | {s['units_sharing_an_identifier']:.1f}% | {s['links_on_top_1pct_identifiers']:.1f}% | "
                f"{gp['same home']:.1f} / {gp['direct reference']:.1f} / {gp['shared identifier']:.1f} / {gp['not linked']:.1f} | "
                f"{s['dense_foothold']:.1f}% | {s['missed_one_hop_from_a_seed']:.1f}% of {s['missed_gold_units']} | {recovered} | {recall} | "
                + " | ".join(f"{c[0]:+.2f} [{c[1]:+.2f}, {c[2]:+.2f}]" for c in cis) + " |"
            )

    lines += ["", "## Lab 3 - Audit of the paper's tables", "", f"> {r['audit']['admission']}", ""]
    for v in r["audit"]["non_monotonic_recall"]:
        lines.append(f"- Recall falls as k grows: {v['method']} in {v['table']} block {v['block']}: {v['values']}")
    return "\n".join(lines) + "\n"


def write_page(results):
    """Fill the page template with the results so the published page always matches this run.

    Writes a page fragment for embedding in a host page, and a standalone copy that opens directly in a browser.
    """
    page = PAGE_TEMPLATE.read_text(encoding="utf-8")
    assert "/*RESULTS*/null" in page
    filled = page.replace("/*RESULTS*/null", json.dumps(results, allow_nan=False))
    PAGE_OUT.write_text(filled, encoding="utf-8")
    head = '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    PAGE_OUT.with_suffix(".standalone.html").write_text(head + filled.replace("</style>", "</style>\n</head>\n<body>", 1) + "\n</body>\n</html>\n", encoding="utf-8")


def main():
    start = time.perf_counter()
    selftest()
    ensure_data()
    audit_results = audit(paper_tex())
    results = {
        "audit": audit_results,
        "lab1": lab1(audit_results["tables"]["tab:multihop-recall"]["Native RAG"]),
        "lab2": lab2(),
        "settings": {"ks": KS, "budgets": BUDGETS, "methods": METHODS, "context_methods": CONTEXT_METHODS},
    }
    results["meta"] = {
        "date": date.today().isoformat(),
        "platform": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "versions": {pkg: version(pkg) for pkg in ("numpy", "scipy", "torch", "sentence-transformers")},
        "model": f"{MODEL}@{MODEL_REVISION[:7]}",
        "seconds": round(time.perf_counter() - start),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    (RESULTS / "report.md").write_text(report(results), encoding="utf-8")
    if PAGE_TEMPLATE.exists():
        write_page(results)
    print(report(results))


if __name__ == "__main__":
    main()
