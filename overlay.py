"""Identifier overlay for a dense corpus: the structure behind the wiki-link gains in reproof.py.

A corpus of retrievable units (chunks) keeps its text and vectors. The overlay adds two kinds of
links from each unit to identifiers:

- home: the one thing the unit belongs to (a wiki page title, a document number, a ticket id);
- mention: other things its text names, found by exact match against a register of known
  identifiers (every home in the corpus) or by a typed pattern (standard numbers, ticket ids).

Units that share an identifier are one hop apart. Retrieval keeps the dense top hits in dense order and
takes one step through their identifiers, splitting each seed's weight by identifier degree
(walk_once), so it needs only identifier degrees and a filter on an array field, not a graph engine.
Ranking the reached units by similarity to the query instead (expand_once) is kept for comparison: it
loses recall, because the evidence dense search missed is unlike the query by definition. Nothing here
is specific to the benchmarks; reproof.py measures it on them.
"""

import re

import numpy as np
import scipy.sparse as sp

SEED_TIER = 8.0  # cosine scores lie in [-1, 1], so tiers 8 and 4 apart never interleave
REACHED_TIER = 4.0
KIND_PREFIX = re.compile(r"([a-z][a-z0-9_-]*):(.*)\Z", re.S)  # "doc:DOC 1042"; "Star Wars: Episode IV" is not a kind


def words(text):
    """Case-kept alphanumeric runs: "DOC-1042" and "DOC 1042" become the same words."""
    return re.sub(r"[^A-Za-z0-9]", " ", text).split()


def register_finder(keys, max_words=8):
    """Find known identifiers (every home key in the corpus) in text as exact, case-kept word runs.

    Returns the keys as given, so a mention resolves to the same identifier as the home it names. A key may
    carry a lowercase kind prefix ("doc:DOC 1042"); text is matched on the words after the prefix, which is
    how documents cite each other. Keys from different kinds whose words coincide are all returned.
    """
    index = {}
    for key in keys:
        prefixed = KIND_PREFIX.match(key)
        index.setdefault(" ".join(words(prefixed.group(2) if prefixed else key)), set()).add(key)
    lengths = sorted({n for n in (len(run.split()) for run in index) if 0 < n <= max_words})

    def find(text):
        w = words(text)
        runs = {" ".join(w[j : j + n]) for n in lengths for j in range(len(w) - n + 1)}
        return set().union(*(index[run] for run in runs if run in index))

    return find


def pattern_finder(kind, regex):
    """Find typed identifiers by pattern, whether or not the corpus holds their home (e.g. standard numbers)."""
    compiled = re.compile(regex)
    return lambda text: {f"{kind}:{' '.join(words(m.group(0)))}" for m in compiled.finditer(text)}


def link(texts, homes, finders):
    """Identifier ids plus (identifier, unit) home and mention pairs; homes are registered first, in unit order."""
    ids = {}
    for home in homes:
        if home:
            ids.setdefault(home, len(ids))
    home_pairs, mention_pairs = set(), set()
    for unit, (text, home) in enumerate(zip(texts, homes)):
        own = ids.get(home) if home else None
        if own is not None:
            home_pairs.add((own, unit))
        for find in finders:
            for key in find(text):
                node = ids.setdefault(key, len(ids))
                if node != own:
                    mention_pairs.add((node, unit))
    return ids, home_pairs, mention_pairs


def incidence(pairs, n_ids, n_units):
    """Binary identifier-by-unit sparse matrix from (identifier, unit) pairs."""
    pairs = np.array(sorted(pairs), dtype=np.int64).reshape(-1, 2)
    return sp.csr_matrix((np.ones(len(pairs), np.float32), (pairs[:, 0], pairs[:, 1])), shape=(n_ids, n_units))


def expand_once(scores, node_unit, seeds=5):
    """Rank the dense top seeds first, then units sharing an identifier with a seed, then the rest; dense order within each tier."""
    order = np.argsort(-scores, axis=1, kind="stable")[:, :seeds]
    is_seed = np.zeros(scores.shape, bool)
    np.put_along_axis(is_seed, order, True, axis=1)
    reached = (node_unit.T @ (node_unit @ is_seed.T.astype(np.float32))).T > 0
    return scores + REACHED_TIER * reached + SEED_TIER * is_seed


def walk_once(scores, node_unit, seeds=5):
    """Rank the dense top seeds first, then units by one step of the walk, then the rest in dense order.

    One step goes seed -> identifier -> unit and splits the weight by degree at each hop, so a rare
    identifier passes most of a seed's weight to its few units while a hub spreads it thin. It needs
    only identifier degrees, which a document store can count, not a graph engine.
    """
    order = np.argsort(-scores, axis=1, kind="stable")[:, :seeds]
    lo = scores.min(axis=1, keepdims=True)
    normalised = (scores - lo) / np.ptp(scores, axis=1, keepdims=True)
    weight = np.zeros(scores.shape, np.float32)
    np.put_along_axis(weight, order, np.take_along_axis(normalised, order, axis=1), axis=1)
    is_seed = np.zeros(scores.shape, bool)
    np.put_along_axis(is_seed, order, True, axis=1)
    unit_degree = np.maximum(np.asarray(node_unit.sum(axis=0)).ravel(), 1)
    id_degree = np.maximum(np.asarray(node_unit.sum(axis=1)).ravel(), 1)
    id_mass = (node_unit @ (weight / unit_degree).T) / id_degree[:, None]
    mass = (node_unit.T @ id_mass).T
    mass /= np.maximum(mass.max(axis=1, keepdims=True), 1e-12)
    # Seeds keep their dense order; reached units rank by walk mass, dense score breaking ties.
    return np.where(is_seed, SEED_TIER + scores, REACHED_TIER * (mass > 0) + mass + 1e-3 * scores)


def entries(compressed, i):
    """Indices stored in row i of a CSR matrix or column i of a CSC matrix, as a set of ints."""
    return set(compressed.indices[compressed.indptr[i] : compressed.indptr[i + 1]].tolist())


def profile(home, mention, golds=None, dense_ranked=None, walked=None, seeds=5, k=20):
    """How much structure an overlay has and, given labelled questions, how much of it the evidence chains use.

    home and mention are identifier-by-unit matrices (home may be None). golds lists each question's
    evidence units; dense_ranked is the dense ranking per question; walked maps a method name to its ranking.
    """
    both = sp.csr_matrix(mention if home is None else home + mention, copy=True)
    both.data[:] = 1
    degree = np.asarray(both.sum(axis=1)).ravel()
    shared_units = np.asarray(both[degree >= 2].sum(axis=0)).ravel() > 0
    top = np.sort(degree)[::-1][: max(1, len(degree) // 100)]
    out = {
        "identifiers": int((degree > 0).sum()),
        "mentions_per_unit": mention.nnz / both.shape[1],
        "units_sharing_an_identifier": 100 * float(shared_units.mean()),
        "links_on_top_1pct_identifiers": 100 * float(top.sum() / max(1, both.nnz)),
        "max_identifier_degree": int(degree.max()),
    }
    if golds is None:
        return out

    unit_ids, id_units = both.tocsc(), both
    home_ids = home.tocsc() if home is not None else None
    mention_ids = mention.tocsc()
    pair_kinds = {"same home": 0, "direct reference": 0, "shared identifier": 0, "not linked": 0}
    foothold = missed = walkable = 0
    recovered = dict.fromkeys(walked or {}, 0)
    for q, gold in enumerate(golds):
        gold = sorted(gold)
        for i, a in enumerate(gold):
            for b in gold[i + 1 :]:
                if home_ids is not None and entries(home_ids, a) & entries(home_ids, b):
                    pair_kinds["same home"] += 1
                elif home_ids is not None and (
                    entries(mention_ids, a) & entries(home_ids, b) or entries(mention_ids, b) & entries(home_ids, a)
                ):
                    pair_kinds["direct reference"] += 1
                elif entries(unit_ids, a) & entries(unit_ids, b):
                    pair_kinds["shared identifier"] += 1
                else:
                    pair_kinds["not linked"] += 1
        if dense_ranked is None:
            continue
        seed_units = dense_ranked[q, :seeds].tolist()
        foothold += bool(set(seed_units) & set(gold))
        seed_ids = set().union(*(entries(unit_ids, u) for u in seed_units))
        reach = set().union(*(entries(id_units, n) for n in seed_ids))
        miss = set(gold) - set(dense_ranked[q, :k].tolist())
        missed += len(miss)
        walkable += len(miss & reach)
        for name, ranked in (walked or {}).items():
            recovered[name] += len(miss & set(ranked[q, :k].tolist()))
    pairs = sum(pair_kinds.values())
    out["gold_pairs"] = {kind: 100 * n / max(1, pairs) for kind, n in pair_kinds.items()}
    if dense_ranked is not None:
        out["dense_foothold"] = 100 * foothold / len(golds)
        out["missed_gold_units"] = missed
        out["missed_one_hop_from_a_seed"] = 100 * walkable / max(1, missed)
        out["missed_recovered"] = {name: 100 * n / max(1, missed) for name, n in recovered.items()}
    return out


def selftest():
    """Smallest checks that fail if matching, linking, expansion or profiling breaks."""
    find = register_finder(["Alpha Beta", "Gamma Delta"])
    assert find("We met Gamma-Delta and a home.") == {"Gamma Delta"}
    # Kind-prefixed homes match on the words after the prefix and come back as given; a colon in a title is not a kind.
    prefixed = register_finder(["doc:DOC 1042", "ticket:DOC 1042", "Star Wars: Episode IV"])
    assert prefixed("See DOC-1042 and Star Wars Episode IV.") == {"doc:DOC 1042", "ticket:DOC 1042", "Star Wars: Episode IV"}
    assert prefixed("see doc 1042") == set()  # case is kept
    spec = pattern_finder("rfc", r"\bRFC \d{3,4}\b")
    assert spec("Per RFC 9110 and RFC 9111.") == {"rfc:RFC 9110", "rfc:RFC 9111"}
    ids, home, mention = link(["cites Gamma Delta", "no links", "also Gamma Delta"], ["Alpha Beta", "Gamma Delta", None], [find])
    assert ids == {"Alpha Beta": 0, "Gamma Delta": 1} and home == {(0, 0), (1, 1)} and mention == {(1, 0), (1, 2)}
    m = incidence(mention, 2, 3)
    ranked = expand_once(np.array([[0.9, 0.1, 0.2]]), m + incidence(home, 2, 3), seeds=1)
    assert np.argsort(-ranked[0]).tolist() == [0, 2, 1]
    # A specific identifier (only units 0 and 1) beats a hub (units 0, 2, 3, 4) even when unit 2 is closer to the query.
    hub = incidence({(0, 0), (0, 1), (1, 0), (1, 2), (1, 3), (1, 4)}, 2, 5)
    stepped = walk_once(np.array([[0.9, 0.1, 0.5, 0.2, 0.2]]), hub, seeds=1)
    assert np.argsort(-stepped[0], kind="stable").tolist()[:2] == [0, 1]
    # Seeds stay in dense order even when the walk sends more weight back to the lower one.
    both_seeds = walk_once(np.array([[0.9, 0.8, 0.1, 0.1, 0.1]]), hub, seeds=2)
    assert np.argsort(-both_seeds[0], kind="stable").tolist()[:2] == [0, 1]
    p = profile(incidence(home, 2, 3), m, golds=[{0, 1}], dense_ranked=np.array([[0, 2, 1]]), seeds=1, k=1)
    assert p["gold_pairs"]["direct reference"] == 100 and p["dense_foothold"] == 100
    assert p["missed_gold_units"] == 1 and p["missed_one_hop_from_a_seed"] == 100
    assert abs(profile(None, m)["units_sharing_an_identifier"] - 200 / 3) < 1e-9


if __name__ == "__main__":
    selftest()
    print("overlay selftest ok")
