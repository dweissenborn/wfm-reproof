# Corpus

Everything this package measures comes from public benchmark data. `reproof.py` downloads every file from a pinned commit and refuses to run if any file differs from its SHA-1. Nothing here is redistributed.

## The benchmarks

These are the HippoRAG 2 evaluation subsets of three multi-hop question-answering datasets. Each has 1,000 questions, the passage corpus they are answered from, and gold supporting passages. These are the test sets the WFM paper reports on.

| Dataset | Questions | Passages (current set) | Passages (v1 set, OpenIE coverage) | How its questions were built |
|---|---|---|---|---|
| HotpotQA | 1,000 | 9,811 | 9,221 | Crowd-written over pairs of Wikipedia pages: bridge questions follow a hyperlink, comparison questions compare two entities |
| 2WikiMultihopQA | 1,000 | 6,119 | 6,119 | Templates over Wikipedia pages joined by Wikidata relations |
| MuSiQue | 1,000 | 11,656 | 11,656 | Composed from single-hop questions; built to resist shortcuts |

- **Unit format.** Each passage is one retrievable unit, embedded as `title\ntext`. The title is the passage's home: the wiki page it came from.
- **Gold passages.** They are matched with HippoRAG 2's own `get_gold_docs` rules. HotpotQA and 2Wiki name gold passages through `supporting_facts`; MuSiQue flags them with `is_supporting`.
- **Two versions.** 2Wiki and MuSiQue are byte-identical between the two HippoRAG versions. HotpotQA differs (9,811 vs 9,221 passages), so Lab 1 uses the current HotpotQA set and Labs 2, 4 and 5 use the older set that the published OpenIE output covers.
- **How link-shaped the questions are.** This matters for the results. Many HotpotQA and 2Wiki questions follow links that are visible in the text, so link-based retrieval starts with an advantage. MuSiQue questions compose facts that are often not linked by name, and title links do nothing there.

## Extracted facts and entities

The triples and named entities are the OpenIE output HippoRAG published for these exact corpora. Nothing was re-extracted.

| Dataset | Unique facts | Entities | Entity–entity edges | Entity–passage links | Distinct titles | Title incidences, home + mention (no LLM) |
|---|---|---|---|---|---|---|
| HotpotQA (v1) | 98,674 | 96,914 | 94,922 | 145,650 | 8,978 | 19,234 |
| 2Wiki | 50,674 | 52,552 | 48,799 | 77,642 | 5,987 | 9,507 |
| MuSiQue | 107,451 | 105,785 | 103,315 | 161,672 | 9,738 | 27,510 |

- **Extractors.** The main runs use GPT-3.5-turbo-1106 extractions. MuSiQue is repeated with GPT-4o-mini and Llama-3.3-70B extractions to rule out a weak extractor.
- **Title links** are found without any model. A passage's title is its home, and every other title its text names, matched exactly with case kept, is a mention. Titles shorter than 4 characters are skipped, because a title like "It" would link half the corpus.

## Storage per representation

Measured bytes needed to rank and return evidence (text, fp32 vectors, index entries and graph edges), in MB:

| Dataset | flat | triples | triple graph | title links | NER links | wiki graph |
|---|---|---|---|---|---|---|
| HotpotQA | 19.4 | 157.7 | 160.6 | 19.7 | 21.1 | 181.2 |
| 2Wiki | 12.1 | 81.0 | 82.5 | 12.3 | 13.2 | 95.2 |
| MuSiQue | 23.8 | 171.7 | 174.9 | 24.2 | 25.6 | 199.9 |

## Embedder

`sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, the embedding model the paper names. It produces 384 dimensions, L2-normalized, on CPU. Embeddings are cached under `cache/`, keyed by model revision and content: about 870 MB for every text the labs embed.

## Pinned files

Two commits of [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) are used: `1438aba` (current) and `b144c46` (legacy). All files are fetched from `raw.githubusercontent.com`. They total about 178 MB under `data/`.

| Local path under `data/` | Source (commit:path) | Size | SHA-1 |
|---|---|---|---|
| `v2/2wikimultihopqa.json` | `1438aba:reproduce/dataset/2wikimultihopqa.json` | 6.5 MB | `8cc40ddc6b4fd3d023d0fb47a0cb95381bfefae8` |
| `v2/2wikimultihopqa_corpus.json` | `1438aba:reproduce/dataset/2wikimultihopqa_corpus.json` | 3.1 MB | `beeaf13ecd5ccfa7f493dce997c1bd7b7f3471b8` |
| `v2/hotpotqa.json` | `1438aba:reproduce/dataset/hotpotqa.json` | 8.2 MB | `1f198572a8f1014c7c0cda519a3bcaeb76830e2a` |
| `v2/hotpotqa_corpus.json` | `1438aba:reproduce/dataset/hotpotqa_corpus.json` | 6.4 MB | `8566c8b193d2626ecf0f751817bb32f1d839142b` |
| `v2/musique.json` | `1438aba:reproduce/dataset/musique.json` | 12.5 MB | `32878a39b4a99c6e276be1562cd30f0e87485e05` |
| `v2/musique_corpus.json` | `1438aba:reproduce/dataset/musique_corpus.json` | 6.2 MB | `22bd0dc18f4cf1595821887e0157b13b8f6e432d` |
| `v1/2wikimultihopqa.json` | `b144c46:data/2wikimultihopqa.json` | 6.5 MB | `8cc40ddc6b4fd3d023d0fb47a0cb95381bfefae8` |
| `v1/2wikimultihopqa_corpus.json` | `b144c46:data/2wikimultihopqa_corpus.json` | 3.1 MB | `beeaf13ecd5ccfa7f493dce997c1bd7b7f3471b8` |
| `v1/hotpotqa.json` | `b144c46:data/hotpotqa.json` | 8.1 MB | `8e5a4b41fdeb1a56e4eaa261a4dfa8e8c630609d` |
| `v1/hotpotqa_corpus.json` | `b144c46:data/hotpotqa_corpus.json` | 5.9 MB | `a188647f19b075752677509427c8ae1b263fc28e` |
| `v1/musique.json` | `b144c46:data/musique.json` | 12.5 MB | `32878a39b4a99c6e276be1562cd30f0e87485e05` |
| `v1/musique_corpus.json` | `b144c46:data/musique_corpus.json` | 6.2 MB | `22bd0dc18f4cf1595821887e0157b13b8f6e432d` |
| `openie/2wikimultihopqa.json` | `b144c46:output/openie_2wikimultihopqa_results_ner_gpt-3.5-turbo-1106_6119.json` | 10.1 MB | `2df85aa3edcbbb824895ff5fd15fdc2795f899a5` |
| `openie/hotpotqa.json` | `b144c46:output/openie_hotpotqa_results_ner_gpt-3.5-turbo-1106_9221.json` | 22.1 MB | `bf37f44271b295f6bad8ada2b2f15ca4ba04cff7` |
| `openie/musique.json` | `b144c46:output/openie_musique_results_ner_gpt-3.5-turbo-1106_11656.json` | 21.2 MB | `e750d1bee8170eb6b28c2ede2b84f4e4ce6c47c9` |
| `openie/musique-gpt-4o-mini.json` | `1438aba:outputs/musique/openie_results_ner_gpt-4o-mini.json` | 22.7 MB | `ed047703134f28b048de0e32f8c3719a9cdc63ef` |
| `openie/musique-llama-3.3-70b.json` | `1438aba:outputs/musique/openie_results_ner_meta-llama_Llama-3.3-70B-Instruct.json` | 24.2 MB | `8901148bb407150cb69e036838db0a4ccefc6fff` |
| `paper/exp.tex` | `exp.tex` from `https://arxiv.org/src/2609.18182v1` | <0.1 MB | `4ae1c48ce21d5d2b45b10c87369b54f11c65e4e2` |

## Licenses

This package downloads but does not redistribute the data. Each source carries its own terms; check them before reusing the data:
- [HotpotQA](https://hotpotqa.github.io/)
- [2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop)
- [MuSiQue](https://github.com/StonyBrookNLP/musique)
- [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG), the subsets and the OpenIE output
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- the [arXiv source of 2609.18182](https://arxiv.org/abs/2609.18182)
