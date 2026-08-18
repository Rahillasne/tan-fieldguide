#!/usr/bin/env python3
"""Corpus-scale retrieval eval across all descriptors + fusion.

Rankings: v0 mean-spectrum (SAM), v1 SpectralEarth CLS, v1b token-mean,
v2 bag-of-spectral-words, and RRF fusion (v0 + v2 + v1b).

Metrics:
  purity@10   — fraction of top-10 same-pseudo-class neighbors, with the
                query's whole SITE excluded (semantic generalization;
                every chip is a query, n=3558)
  site-reid@1 — top-1 cross-scene neighbor is the query's own site
                (identity robustness; no site exclusion)
  plume p@10  — cross-SITE plume retrieval (13 queries; small-n caveat)
"""

import json
from pathlib import Path

import numpy as np

INDEX_DIR = Path(__file__).parent / "data" / "index"
K = 10
RRF_K = 60


def normalized(a):
    return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-9, None)


def load():
    meta = json.loads((INDEX_DIR / "meta.json").read_text())
    aux = json.loads((INDEX_DIR / "aux.json").read_text())
    chips = [c for s in meta["scenes"] for c in s["chips"]]
    assert aux["chip_ids"] == [c["id"] for c in chips]
    scene_of = np.array([c["scene"] for c in chips])
    site_of = np.array([aux["site_of_scene"][s] for s in scene_of])
    cls = np.array(aux["chip_class"])
    plume = np.array([c["plume"] for c in chips])
    embs = {
        "v0 mean-spectrum": np.load(INDEX_DIR / "emb.npy"),
        "v1 SpecEarth-CLS": np.load(INDEX_DIR / "emb_v1.npy"),
        "v1b SpecEarth-tok": np.load(INDEX_DIR / "emb_v1b.npy"),
        "v2 spectral-words": np.load(INDEX_DIR / "emb_v2.npy"),
    }
    return chips, scene_of, site_of, cls, plume, embs


def orders_from(emb):
    sim = normalized(emb.astype(np.float64)) @ normalized(emb.astype(np.float64)).T
    np.fill_diagonal(sim, -np.inf)
    return np.argsort(-sim, axis=1)


def rrf(order_list):
    n = order_list[0].shape[0]
    score = np.zeros((n, n))
    for order in order_list:
        ranks = np.empty_like(order)
        rows = np.arange(n)[:, None]
        ranks[rows, order] = np.arange(n)[None, :]
        score += 1.0 / (RRF_K + ranks)
    np.fill_diagonal(score, -np.inf)
    return np.argsort(-score, axis=1)


def evaluate(name, order, scene_of, site_of, cls, plume):
    n = order.shape[0]
    pur, reid = [], 0
    for q in range(n):
        row = order[q]
        cross_site = row[site_of[row] != site_of[q]][:K]
        pur.append((cls[cross_site] == cls[q]).mean())
        nn = row[scene_of[row] != scene_of[q]][0]
        reid += site_of[nn] == site_of[q]
    precs = []
    for q in np.where(plume)[0]:
        row = order[q]
        cross_site = row[site_of[row] != site_of[q]][:K]
        precs.append(plume[cross_site].mean())
    print(f"  {name:<20} purity@10={np.mean(pur):.3f}  "
          f"site-reid@1={reid/n:.3f}  plume-p@10={np.mean(precs):.3f}")


def main():
    chips, scene_of, site_of, cls, plume, embs = load()
    n_cls = {c: int((cls == c).sum()) for c in sorted(set(cls))}
    print(f"{len(chips)} chips | classes {n_cls} | random purity = "
          f"{sum((v/len(chips))**2 for v in n_cls.values()):.3f}")
    orders = {name: orders_from(e) for name, e in embs.items()}
    for name, order in orders.items():
        evaluate(name, order, scene_of, site_of, cls, plume)
    fused = rrf([orders["v0 mean-spectrum"], orders["v2 spectral-words"],
                 orders["v1b SpecEarth-tok"]])
    evaluate("FUSION v0+v2+v1b", fused, scene_of, site_of, cls, plume)


if __name__ == "__main__":
    main()
