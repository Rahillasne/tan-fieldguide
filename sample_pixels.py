#!/usr/bin/env python3
"""Sample per-chip pixel spectra for composition descriptors and class labels.

One pass over the scene HDF5s: for each chip in data/index/meta.json, sample
SAMPLES_PER_CHIP valid pixels, resample their spectra to the index's
canonical grid, and store per-scene npz under data/index/pixel_samples/.
Idempotent per scene. Everything downstream (bag-of-spectral-words codebook,
chip histograms, land-cover pseudo-classes) derives from these samples
without touching the scenes again.
"""

import json
from pathlib import Path

import h5py
import numpy as np

from core import CHIP, SR_PATH, good_band_mask, interp_matrix

ROOT = Path(__file__).parent
INDEX_DIR = ROOT / "data" / "index"
OUT_DIR = INDEX_DIR / "pixel_samples"
SCENES_DIR = ROOT / "data" / "scenes"
SAMPLES_PER_CHIP = 256


def sample_scene(sid, chips, grid_nm, rng):
    out_npz = OUT_DIR / f"{sid}.npz"
    if out_npz.exists():
        return f"{sid}: already sampled"
    with h5py.File(SCENES_DIR / f"{sid}_ortho_sr_hdf5.h5", "r") as f:
        g = f[SR_PATH]
        sr_ds = g["surface_reflectance"]
        wl = sr_ds.attrs["wavelengths"].astype(float)
        good = good_band_mask(wl, sr_ds.attrs["good_wavelengths"])
        cube = sr_ds[:]
        bad_px = (g["nodata_pixels"][:] > 0) | (g["beta_cloud_mask"][:] > 0) | (g["beta_cirrus_mask"][:] > 0)

    W = interp_matrix(wl[good], grid_nm)  # (D, n_good)
    out = np.empty((len(chips), SAMPLES_PER_CHIP, len(grid_nm)), np.float32)
    for ci, ch in enumerate(chips):
        r0, c0 = ch["r"], ch["c"]
        block = cube[good][:, r0:r0 + CHIP, c0:c0 + CHIP]
        valid = ~bad_px[r0:r0 + CHIP, c0:c0 + CHIP]
        flat = block.reshape(block.shape[0], -1)[:, valid.ravel()]  # (n_good, n_valid)
        n_valid = flat.shape[1]
        pick = rng.choice(n_valid, SAMPLES_PER_CHIP, replace=n_valid < SAMPLES_PER_CHIP)
        out[ci] = (W @ flat[:, pick]).T
    np.savez_compressed(out_npz, samples=out, ids=np.array([c["id"] for c in chips]))
    return f"{sid}: sampled {len(chips)} chips"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads((INDEX_DIR / "meta.json").read_text())
    grid_nm = np.array(meta["grid_nm"])
    rng = np.random.default_rng(0)
    for s in meta["scenes"]:
        try:
            print(sample_scene(s["scene"], s["chips"], grid_nm, rng), flush=True)
        except Exception as e:
            print(f"{s['scene']}: FAILED {e!r}", flush=True)


if __name__ == "__main__":
    main()
