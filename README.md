# wfm-reproof

An independent re-proof of **arXiv:2609.18182, "WFM: Wiki Foundation Model for Complex Agentic Reasoning"** (Dong et al., Tencent Youtu Lab, v1, 16 Sep 2026).

The paper argues that agents should store knowledge as an *LLM Wiki* (dense passages joined by entity links) rather than as flat chunks or knowledge-graph triples. It then reports large retrieval gains for WFM, its trained graph model. WFM has no released code, so this package measures everything an outsider can check, on the paper's own public test sets, with the embedding model the paper names:

- the paper's flat-retrieval baseline ("Native RAG");
- the representation claim (flat passages vs triples vs a wiki graph), with storage and build cost;
- the paper's recall tables, read from its TeX source;
- how much of a model's context window each representation needs before the answer appears;
- which structure in the data the link walk relies on, generalized in `overlay.py` for any corpus whose records name each other by identifier.

No training, no API calls, CPU only. One script, pinned inputs, deterministic output.

## In short

**The LLM Wiki idea holds up; the paper's numbers don't.** Linking passages the way a wiki does beats both flat chunks and knowledge-graph triples on the paper's own test sets. The margins the paper reports, however, rest on a flat baseline that is 14–30 points too low.

**Best bang for the buck:**
- **Links:** link each passage to the identifiers its text names, by exact string match with no LLM.
- **Ranking:** take one degree-weighted step from the dense top five.

That reaches 82.0 average Recall@20 against 72.8 for flat retrieval, at 1.02× storage and zero LLM calls. It needs no graph database and no training. A training-free full wiki graph reaches 85.1, against the 86.2 the paper claims for its trained model, at about 8× the storage.

## Findings

Measured on the HippoRAG 2 test sets with all-MiniLM-L6-v2. Full tables are in [`results/report.md`](results/report.md); [`results/wfm-reproof.standalone.html`](results/wfm-reproof.standalone.html) is an offline findings page with charts.

### 1. The flat baseline is too low, and on two datasets it is the measurement minus whole numbers

Recall@2 / 5 / 10 / 20, `title\ntext` passages:

| Dataset | Measured | Paper ("Native RAG") | Measured minus paper |
|---|---|---|---|
| HotpotQA | 58.000 / 74.000 / 80.550 / 85.250 | 44.24 / 55.35 / 62.68 / 68.10 | +13.760 / +18.650 / +17.870 / +17.150 |
| 2Wiki | 51.275 / 61.800 / 66.225 / 69.225 | 31.28 / 31.80 / 46.22 / 49.22 | +19.995 / +30.000 / +20.005 / +20.005 |
| MuSiQue | 37.275 / 49.275 / 57.100 / 64.442 | 17.28 / 19.28 / 37.10 / 44.44 | +19.995 / +29.995 / +20.000 / +20.002 |

- **Whole-number offsets.** On 2Wiki and MuSiQue, all eight published cells equal the measurement minus exactly 20, 30, 20 and 20, within the paper's ±0.005 rounding.
  - Independent errors would each land that close to a whole number about 1% of the time. All eight together would happen about once in 10^16.
  - The same index without titles shows no such pattern.
- **The paper's own disclaimer.** Its Implementation Details section says: *"The newly completed cells and diagnostic sweeps are constructed planning values and require validation with measured runs before external use."*
- **An impossible row.** The HippoRAG-IRCoT row reports HotpotQA Recall@10 (76.10) below Recall@5 (76.90), which no single ranking can produce.

### 2. Wiki-style representations win, by less than claimed

Average Recall@20 over the three datasets, with storage relative to flat passages at fp32:

| Representation | Avg R@20 | Storage vs flat | LLM calls per passage |
|---|---|---|---|
| flat passages | 72.84 | 1.0× | 0 |
| triples | 72.61 | 6.7–8.1× | 2 |
| triple graph (the paper's "Ordinary graph") | 74.20 | 6.8–8.3× | 2 |
| title links (no LLM) | 82.02 | 1.02× | 0 |
| NER entity links | 83.26 | 1.08–1.09× | 1 |
| wiki graph | 85.06 | 7.9–9.3× | 2 |
| *WFM, as claimed (not reproducible)* | *86.20* | | |

- **Gains are real.** The wiki graph beats flat retrieval at Recall@20 by +8.65, +19.85 and +8.15 points on HotpotQA, 2Wiki and MuSiQue, with 95% intervals that exclude zero.
- **But smaller than claimed.** The paper implies +25.1, +40.9 and +30.8 (WFM against its Native RAG).
- **Passage nodes matter more than the paper says.** Adding them to the triple graph gains 10.86 points of average Recall@20, where the paper's ablation reports 7.48. This holds with GPT-3.5, GPT-4o-mini and Llama-3.3-70B triples.
- **No training needed to get close.** A training-free wiki graph (HippoRAG 2's published method) reaches 85.06, against WFM's claimed 86.20.
- **Triples alone don't help.** They never beat the passages they came from, except on the entity-centric 2Wiki questions.

### 3. Storage: links are nearly free, facts cost about 8×

- **Why triples are big.** One vector per fact, at 8–11 facts per passage, is what makes triple stores 7–9× larger than flat stores. With int8 vectors everywhere the ratio is still 4.5–6.5×.
- **Quantization is cheap.** int8 vectors cut flat storage 4× and move recall by at most 0.15 points. 1-bit vectors cut it 32× for 2–4 points.
- **Title links depend on the questions.** Found by exact string match with no LLM, they add +6.8 and +21.7 Recall@20 points on HotpotQA and 2Wiki at 1.02× storage. On MuSiQue, whose questions don't follow page links, they change nothing (−1.0, within noise). There, LLM entity links (+4.1) or the full graph (+8.2) are needed.

### 4. Model context: facts are compact but lose the answer

Median context tokens read before the gold answer appears (questions with yes/no answers, or answers named in the question, excluded):

| Context | HotpotQA (830 q) | 2Wiki (521 q) | MuSiQue (977 q) |
|---|---|---|---|
| facts | 1,007 | >4,096 | >4,096 |
| flat passages | 243 | 3,261 | 1,273 |
| title links | 302 | 682 | 805 |
| NER links | 256 | 684 | 706 |
| wiki graph | 265 | 390 | 896 |

- **Facts lose the answer.** Each costs a tenth of a passage, yet facts carry the answer less often at nearly every budget from 64 to 4,096 tokens.
- **Links pay off past the top five.** The link-only methods keep the dense top five, so their gains start once the budget runs past those five passages (roughly 750 tokens) and peak around 2,048.
- **Where the fact layer earns its storage:** re-ranking the first slots. At 256 tokens on 2Wiki, the wiki graph has the answer for 33% of questions and flat passages for 12%.

### 5. Structure: what the link walk actually uses

Measured by Lab 5 through `overlay.py`:

| | HotpotQA | 2Wiki | MuSiQue |
|---|---|---|---|
| Dense foothold (an evidence passage in the dense top 5) | 95.0% | 97.7% | 90.6% |
| Evidence pairs linked: title links / NER links | 68.2% / 76.6% | 44.1% / 55.5% | 35.4% / 64.1% |
| Missed evidence one hop from a seed: title / NER | 73.6% / 86.5% | 81.6% / 88.9% | 48.6% / 72.2% |
| Missed evidence recovered by the PageRank walk: title / NER | 71.0% / 69.6% | 81.1% / 75.2% | 21.5% / 29.5% |
| Recall@20 vs the walk, reached units ranked by similarity: title / NER | −2.2 / −5.3 | −2.3 / −17.4 | −0.4 / −1.8 |
| Recall@20 vs the walk, one step weighted by identifier degree: title / NER | +0.5 / +0.2 | +2.0 / +0.1 | +1.1 / −0.1 |

- **Footholds are rarely the problem.** Dense search finds part of the evidence for 91–98% of questions.
- **Linkage decides whether links help.** Under both overlays, the three datasets rank the same way on "missed evidence one hop from a seed" as on Recall@20 gain.
- **Ranking decides how much they help.**
  - Ranking the reached units by similarity to the question loses up to 17 points. The evidence dense search missed is, by definition, unlike the question.
  - One walk step that splits weight by identifier degree matches the full PageRank walk. No graph engine is needed, only identifier degrees.
- **Keep the seeds in dense order.**
  - The step leaves the dense top five in place, so Recall@2 stays at the dense level (57.2 / 51.3 / 37.3). The PageRank walk reorders them and drops to 48.4 / 46.8 / 33.0 with title links.
  - The step's median tokens-to-answer matches or beats the walk's in all six cases.

## Best bang for the buck

In order of return on effort, measured on these benchmarks:

1. **Prefix each chunk with its home's name** (title, document number). This is free and worth 0.3–3.6 points of Recall@5.
2. **Store vectors as int8.** Storage drops 4×, and recall moves at most 0.15 points.
3. **Link chunks to the identifiers their text names, by exact match against the register of homes.**
   - Costs: no LLM, and 1.02× storage.
   - Average Recall@20 rises from 72.8 to 82.0.
   - This pays off when questions follow names, as on HotpotQA and 2Wiki.
4. **Rank with one degree-weighted step** (`overlay.walk_once`), not similarity expansion and not PageRank.
   - It lands between −0.13 and +1.95 points of PageRank's Recall@20 across the six dataset and overlay pairs.
   - It keeps the dense top five intact.
   - It runs as three filtered queries on any store that can filter on an array field.
5. **Only then, LLM entity links** (1 LLM call per passage), for corpora whose questions don't follow names. They give +4.1 on MuSiQue and +1.2 on average over title links.
6. **The full wiki graph last.** It adds +1.8 more on average, at about 8× the storage and 2 LLM calls per passage.

**Skip:**
- triples as the store;
- facts as model context;
- ranking the linked neighbors by similarity to the question.

### The code

`overlay.py` is the benchmark-free core. With your own units, home keys and dense scores:

```python
import numpy as np
import overlay

texts = [...]   # one entry per retrievable unit (chunk)
homes = [...]   # each unit's home key, e.g. "doc:DOC 1042", or None
finders = [overlay.register_finder([h for h in homes if h]),   # homes named in the text
           overlay.pattern_finder("rfc", r"\bRFC \d{3,4}\b")]   # typed ids, even without a home in the corpus
ids, home_pairs, mention_pairs = overlay.link(texts, homes, finders)
node_unit = overlay.incidence(home_pairs | mention_pairs, len(ids), len(texts))

scores = query_vectors @ unit_vectors.T   # dense cosine scores, queries x units
ranked = np.argsort(-overlay.walk_once(scores, node_unit, seeds=5), axis=1, kind="stable")
```

**How keys match:**
- **Case is kept**, so "Home" the film does not match "home" the noun.
- **Separators are ignored**, so `DOC-1042` and `DOC 1042` match.
- **A lowercase kind prefix** (`doc:`, `rfc:`) is matched on the words after it, so a mention resolves to the same identifier as the home it names.

### On a document store, without a graph database

Links are not a separate store. They are key fields on each unit, and units that share a key are neighbors:

```text
unit   unitId, homeKey, linkKeys[] (homeKey first, then mentions), text, vector, acl, citation
key    kind-prefixed, words joined by single spaces, case kept: "doc:DOC 1042", "rfc:RFC 9110", "ticket:TKT 4821"
```

The step as three queries, in generic SQL over a store with array fields and vector search. The service does the scoring:

```sql
-- 1. seeds: the dense top 5, returning their keys
SELECT TOP 5 c.id, c.linkKeys FROM c
WHERE ARRAY_CONTAINS(c.acl, @principal)
ORDER BY VECTOR_DISTANCE(c.embedding, @queryVector)

-- 2. degrees of the seeds' keys (or keep a count per key at ingest)
SELECT k AS key, COUNT(1) AS degree FROM c JOIN k IN c.linkKeys
WHERE ARRAY_CONTAINS(c.acl, @principal) AND ARRAY_CONTAINS(@seedKeys, k)
GROUP BY k

-- 3. units sharing a key, skipping keys above a degree cap
SELECT c.id, c.linkKeys, c.citation FROM c
WHERE ARRAY_CONTAINS(c.acl, @principal)
  AND EXISTS(SELECT VALUE k FROM k IN c.linkKeys WHERE ARRAY_CONTAINS(@keys, k))
```

**Scoring:** score each stepped unit as the sum, over seeds and shared keys, of `seed weight / seed key count / key degree`. A key shared by three units passes most of a seed's weight; a key shared by a thousand passes almost none.

**Packing the context:** seeds first in dense order, then stepped units by score, then the rest of the dense list.

**Access control:** every query applies the caller's access, so a hidden unit can't be a seed, count toward a degree, or be returned.

## When the structure applies

Nothing in the method is specific to Wikipedia. It needs six properties in the data, and Lab 5 measures each one:

1. **Self-describing units.** A unit embeds its home's name with its text.
2. **One home per unit.** Every unit belongs to exactly one identifiable thing: a page, a document number, a ticket. Units with the same home are linked.
3. **Mentions of other identifiers.** Units name other things by identifier.
   - Exact matches against the register of homes give *direct references*.
   - Typed patterns give *shared identifiers*, even when the corpus holds no home for them.
   - Structured fields (parent ids, linked items) are mentions too, with no text extraction needed.
4. **A dense foothold.** Expansion starts from the dense top five. If those miss all of the evidence, links cannot help.
5. **Evidence one hop away.** What links recover sits one identifier away from a dense hit.
6. **Specific identifiers.** A link is worth the inverse of its identifier's degree. Hubs connect everything and so say nothing; weight them down or leave them out. Examples: an audit field, a lookup table, a country everyone mentions, a scope identifier that every record carries.

**Where it does not help:**
- **Questions that don't follow names.** On MuSiQue, title links change nothing.
- **Corpora with no dense foothold.**
- **Identifiers that are mostly hubs.** When one identifier touches most units, a single step reaches nearly everything and ranking falls back to its degree weights. Drop such identifiers before linking.

**Check a corpus before building its overlay.** Run `overlay.profile` on a sample:
- **Without labels:** mentions per unit, and the share of units that share an identifier with another unit. Together these give the structural potential.
- **With 30–50 labeled questions** (question → evidence units):
  - the dense foothold rate;
  - how evidence pairs are linked;
  - the share of missed evidence one hop from a seed.

  These predicted the gain on the benchmarks. A corpus that looks like MuSiQue under title links will not benefit.
- **Hubs:** the share of links held by the top 1% of identifiers, and the largest degree.

## Corpus

Three public multi-hop QA benchmarks in their HippoRAG 2 subsets: 1,000 questions each from HotpotQA (9,811 passages), 2WikiMultihopQA (6,119) and MuSiQue (11,656). The embedder is all-MiniLM-L6-v2, and the extracted facts are HippoRAG's published OpenIE output.

[`CORPUS.md`](CORPUS.md) has the details:
- per-dataset sizes and how each was built;
- fact, entity and link counts;
- storage per representation;
- every pinned file with its source commit, size and SHA-1;
- links to each source's license.

## Run it

Requires [uv](https://docs.astral.sh/uv/) and about 1 GB of disk (PyTorch CPU, data, embedding cache).

```bash
uv run reproof.py          # all five labs: self-checks, downloads, embeddings, results
uv run python overlay.py   # the overlay's own self-test
```

A cold run takes about 20 minutes on a laptop CPU, and 5–10 minutes once `cache/` holds the embeddings. It writes:

| File | Contents |
|---|---|
| `results/results.json` | Every measured number, the audit, versions and run time |
| `results/report.md` | The same results as Markdown tables |
| `results/wfm-reproof.html` | Findings page fragment (template `page.html` filled with this run) |
| `results/wfm-reproof.standalone.html` | The same page as a self-contained file that opens offline |

The script runs its self-checks first, then refuses to continue if any input differs from its pinned SHA-1.

## What each lab does

**Lab 1: baseline.**
- **Setup:** dense retrieval with `sentence-transformers/all-MiniLM-L6-v2` (revision `1110a24`) over the HippoRAG 2 subsets. Passages are embedded as `title\ntext`, and gold passages are matched with HippoRAG 2's own `get_gold_docs` rules.
- **Check:** Recall@2/5/10/20 is compared with the paper's "Native RAG" row, and each gap is checked for a whole number within the paper's ±0.005 rounding.
- **Variants:** int8 and 1-bit versions of the same index, and the index without titles.

**Lab 2: representations.** Same questions and embedder; only the stored representation changes.

| Representation | What is stored | LLM calls per passage to build |
|---|---|---|
| `flat` | passage text, one vector per passage | 0 |
| `triples` | LLM-extracted facts, one vector per fact; a passage ranks by its best fact | 2 |
| `triple_graph` | facts plus the entity graph they form (the paper's "Ordinary graph" ablation) | 2 |
| `id_links` | passages plus links to the page titles their text names, found by exact string match | 0 |
| `wiki_links` | passages plus links to LLM-extracted named entities | 1 |
| `wiki_graph` | passages, entities, facts and every link between them (the paper's Wiki Graph schema) | 2 |

- **Search:** the graph representations use personalized PageRank at HippoRAG 2's default settings (damping 0.5, top 5 facts, top 5 phrases, passage node weight 0.05). The link-only graphs seed from the top 5 dense passages.
- **No tuning:** nothing is trained or tuned on the test questions.
- **Extractors:** MuSiQue is repeated with GPT-4o-mini and Llama-3.3-70B extractions.
- **Intervals:** differences carry 95% paired-bootstrap intervals over the 1,000 questions (2,000 resamples, fixed seed).

**Lab 3: audit.** Parses the recall tables straight out of the paper's `exp.tex`. It flags any row where recall falls as k grows, which is impossible for one ranked list, and quotes the paper's own statement about constructed values.

**Lab 4: model context.** Recall@k counts passages, but a model's context window counts tokens.
- For each question, the lab reads a method's ranked output in order (facts for the fact store, passages for the others).
- It counts approximate tokens (4 characters each, plus one per unit) until the normalized gold answer first appears.
- It reports the share of questions with the answer inside budgets from 64 to 4,096 tokens.

**Lab 5: structure.** Runs the benchmarks through `overlay.py` and traces every question's evidence through two overlays: title links (no LLM) and NER links. It reports:
- the dense foothold;
- how evidence pairs are linked (same home, direct reference, shared identifier);
- hubness;
- the share of missed evidence one hop from a seed;
- how much of it three rankers recover: the PageRank walk, a one-hop expansion ranked by similarity, and one walk step weighted by identifier degree.

## Not tested
- **WFM itself.** No code or weights were released. That covers its query-conditioned attention model, the attention-variance regularizer, and the claim that it prevents attention collapse.
- **The 10.5× distributed-training speedup,** which needs the authors' code and a multi-GPU cluster.
- **LLM-judged answer accuracy,** and the long-term memory benchmarks (PersonaMem, RHELM).

## Caveats
- **Link-shaped benchmarks.** Many HotpotQA and 2Wiki questions were built by following Wikipedia hyperlinks and Wikidata relations, which favors link-based retrieval. MuSiQue is the less link-shaped test.
- **Answer-in-context is a proxy** (Lab 4). It shows the answer string reached the model, not that the model would use it.
- **Approximate tokens.** Token counts use 4 characters per token for every representation.
- **What storage counts.** The bytes each representation needs to rank and return evidence: text, fp32 vectors (with an int8 variant), index entries and graph edges. It excludes any database or ANN index overhead.

## License

**Code:** MIT; see [`LICENSE`](LICENSE).

**Data:** the package downloads, but does not redistribute, third-party data. Each source keeps its own terms; see [`CORPUS.md`](CORPUS.md#licenses).
