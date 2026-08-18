#!/usr/bin/env python3
"""Composition descriptors, land-cover pseudo-classes, and site groups.

From the per-chip pixel samples (sample_pixels.py):

1. Bag-of-spectral-words: fit a spherical k-means codebook (K=64) on
   L2-normalized pixel spectra, describe each chip as the Hellinger-
   transformed histogram of its pixels' codewords -> data/index/emb_v2.npy.
   Composition-aware: a half-water/half-desert chip matches other mixed
   shoreline chips, not the meaningless average of the two.

2. Physics pseudo-classes per chip (water / snow-ice / vegetation /
   bright / dark / barren) from band indices on the sampled pixels —
   corpus-scale ground truth for the retrieval eval, and a viewer badge.

3. Same-site scene groups (scene centers within ~0.35 deg): retrieval
   should not "discover" the query's own site wearing a different date.

Outputs: emb_v2.npy, codebook.npz, aux.json {chip_class, chip_class_frac,
site_of_scene}.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).parent
INDEX_DIR = ROOT / "data" / "index"
SAMPLES_DIR = INDEX_DIR / "pixel_samples"
K = 64
CLASSES = ["water", "snow_ice", "vegetation", "bright", "dark", "barren"]


def band(grid_nm, nm):
    return int(np.argmin(np.abs(np.asarray(grid_nm) - nm)))


def classify_pixels(spec, grid_nm):
    """spec (n, D) reflectance -> class index per pixel."""
    g = lambda nm: spec[:, band(grid_nm, nm)]
    r560, r640, r860, r1610 = g(560), g(640), g(860), g(1610)
    lo, hi = band(grid_nm, 450), band(grid_nm, 1300)
    bright = spec[:, lo:hi].mean(axis=1)
    ndvi = (r860 - r640) / np.clip(r860 + r640, 1e-6, None)
    ndwi = (r560 - r860) / np.clip(r560 + r860, 1e-6, None)
    ndsi = (r560 - r1610) / np.clip(r560 + r1610, 1e-6, None)
    cls = np.full(len(spec), CLASSES.index("barren"))
    cls[bright > 0.28] = CLASSES.index("bright")
    cls[ndvi > 0.35] = CLASSES.index("vegetation")
    cls[(ndsi > 0.4) & (r560 > 0.25)] = CLASSES.index("snow_ice")
    cls[(ndwi > 0.1) & (r860 < 0.08)] = CLASSES.index("water")
    cls[(bright < 0.07) & (cls == CLASSES.index("barren"))] = CLASSES.index("dark")
    return cls


def site_groups(scenes):
    """scene id -> canonical site id (min scene id of its spatial group)."""
    lat = np.array([s["lat"] for s in scenes])
    lon = np.array([s["lon"] for s in scenes])
    sid = [s["scene"] for s in scenes]
    site_of = {s: s for s in sid}
    def find(a):
        while site_of[a] != a:
            site_of[a] = site_of[site_of[a]]
            a = site_of[a]
        return a
    for i in range(len(scenes)):
        for j in range(i + 1, len(scenes)):
            if (abs(lat[i] - lat[j]) < 0.35
                    and abs((lon[i] - lon[j]) * np.cos(np.radians(lat[i]))) < 0.35):
                ra, rb = find(sid[i]), find(sid[j])
                if ra != rb:
                    site_of[max(ra, rb)] = min(ra, rb)
    return {s: find(s) for s in sid}


def main():
    meta = json.loads((INDEX_DIR / "meta.json").read_text())
    grid_nm = np.array(meta["grid_nm"])
    all_samples, order_ids = [], []
    for s in meta["scenes"]:
        z = np.load(SAMPLES_DIR / f"{s['scene']}.npz")
        want = [c["id"] for c in s["chips"]]
        assert list(z["ids"]) == want, f"order mismatch {s['scene']}"
        all_samples.append(z["samples"])
        order_ids += want
    samples = np.concatenate(all_samples)          # (n_chips, P, D)
    n_chips, P, D = samples.shape
    print(f"{n_chips} chips x {P} pixels x {D} bands")

    px = samples.reshape(-1, D)
    unit = px / np.clip(np.linalg.norm(px, axis=1, keepdims=True), 1e-9, None)
    rng = np.random.default_rng(0)
    fit_idx = rng.choice(len(unit), min(400_000, len(unit)), replace=False)
    km = MiniBatchKMeans(n_clusters=K, random_state=0, batch_size=4096,
                         n_init=3, max_iter=200).fit(unit[fit_idx])
    words = km.predict(unit).reshape(n_chips, P)
    hist = np.zeros((n_chips, K), np.float32)
    for i in range(n_chips):
        hist[i] = np.bincount(words[i], minlength=K)
    hist /= hist.sum(axis=1, keepdims=True)
    emb_v2 = np.sqrt(hist)                          # Hellinger: cosine-ready
    np.save(INDEX_DIR / "emb_v2.npy", emb_v2)
    np.savez(INDEX_DIR / "codebook.npz", centers=km.cluster_centers_.astype(np.float32),
             grid_nm=grid_nm)

    cls_px = classify_pixels(px, grid_nm).reshape(n_chips, P)
    frac = np.stack([(cls_px == k).mean(axis=1) for k in range(len(CLASSES))], axis=1)
    chip_class = [CLASSES[int(i)] for i in frac.argmax(axis=1)]
    counts = {c: chip_class.count(c) for c in CLASSES}
    print("chip classes:", counts)

    aux = {
        "classes": CLASSES,
        "chip_ids": order_ids,
        "chip_class": chip_class,
        "chip_class_frac": np.round(frac, 3).tolist(),
        "site_of_scene": site_groups(meta["scenes"]),
    }
    (INDEX_DIR / "aux.json").write_text(json.dumps(aux))
    n_sites = len(set(aux["site_of_scene"].values()))
    print(f"emb_v2 {emb_v2.shape}; {n_sites} sites / {len(meta['scenes'])} scenes")


if __name__ == "__main__":
    main()
