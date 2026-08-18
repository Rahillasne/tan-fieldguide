#!/usr/bin/env python3
"""v1 arm: frozen SpectralEarth SpecViT-S embeddings for every indexed chip.

Reads the chips defined in data/index/meta.json, re-extracts each 128x128
cube from the scene HDF5, resamples 426 Tanager bands onto the 202-band
EnMAP grid (linear, from good bands only), applies SpectralEarth's per-band
normalization (reflectance x10000, minus mu, over sigma), and runs the
frozen encoder. Per-scene npz outputs under data/index/v1/; merge aligns
with meta.json chip order into data/index/emb_v1.npy.

Run with .venv-ml (python3.12 + torch). Weights/stats in v1_assets/
(no license published upstream — local evaluation only, do not redistribute).
"""

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from core import ARTIFACT_WINDOWS_NM, CHIP, SR_PATH, good_band_mask, interp_matrix  # noqa: F401

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "v1_assets"))
from spec_vit import SpecViTSmall  # noqa: E402

INDEX_DIR = ROOT / "data" / "index"
V1_DIR = INDEX_DIR / "v1b"  # v1b: stores CLS + patch-token-mean pooling
SCENES_DIR = ROOT / "data" / "scenes"
BATCH = 16


def load_model():
    model = SpecViTSmall(num_classes=0).eval()
    msg = model.load_state_dict(
        torch.load(ROOT / "v1_assets" / "spec_vit_small_mae.pth", map_location="cpu"),
        strict=False)
    assert not msg.missing_keys and not msg.unexpected_keys
    torch.set_num_threads(max(1, (torch.get_num_threads() or 8) - 2))
    return model


def embed_scene(model, sid, chips, enmap_nm, mu, sigma):
    out_npz = V1_DIR / f"{sid}.npz"
    if out_npz.exists():
        return f"{sid}: already embedded"
    h5_path = SCENES_DIR / f"{sid}_ortho_sr_hdf5.h5"
    with h5py.File(h5_path, "r") as f:
        g = f[SR_PATH]
        sr_ds = g["surface_reflectance"]
        wl = sr_ds.attrs["wavelengths"].astype(float)
        good = good_band_mask(wl, sr_ds.attrs["good_wavelengths"])
        cube = sr_ds[:]
        bad_px = (g["nodata_pixels"][:] > 0) | (g["beta_cloud_mask"][:] > 0) | (g["beta_cirrus_mask"][:] > 0)

    W = interp_matrix(wl[good], enmap_nm)  # (202, n_good)
    ids, batches, buf = [], [], []
    for ch in chips:
        r0, c0 = ch["r"], ch["c"]
        block = cube[good][:, r0:r0 + CHIP, c0:c0 + CHIP].copy()  # (n_good,128,128)
        bad = bad_px[r0:r0 + CHIP, c0:c0 + CHIP]
        if bad.any():
            fill = block[:, ~bad].mean(axis=1) if (~bad).any() else np.zeros(block.shape[0], np.float32)
            block[:, bad] = fill[:, None]
        cube202 = (W @ block.reshape(block.shape[0], -1)).reshape(202, CHIP, CHIP)
        x = (np.clip(cube202, 0, None) * 10000.0 - mu[:, None, None]) / sigma[:, None, None]
        buf.append(x.astype(np.float32))
        ids.append(ch["id"])
        if len(buf) == BATCH:
            batches.append(np.stack(buf)); buf = []
    if buf:
        batches.append(np.stack(buf))

    cls_out, tok_out = [], []
    with torch.no_grad():
        for b in batches:
            x = torch.from_numpy(b)
            cls_out.append(model(x).numpy())
            toks = model.get_intermediate_layers(x, n=1, norm=True)[0]
            tok_out.append(toks.mean(dim=1).numpy())
    if cls_out:
        emb_cls, emb_tok = np.concatenate(cls_out), np.concatenate(tok_out)
    else:
        emb_cls = emb_tok = np.zeros((0, 384), np.float32)
    np.savez(out_npz, ids=np.array(ids),
             emb=emb_cls.astype(np.float32), emb_tok=emb_tok.astype(np.float32))
    return f"{sid}: {len(ids)} chips embedded"


def merge(meta):
    rows, rows_tok, ok = [], [], True
    for s in meta["scenes"]:
        z = np.load(V1_DIR / f"{s['scene']}.npz", allow_pickle=False)
        ids = list(z["ids"])
        want = [c["id"] for c in s["chips"]]
        if ids != want:
            print(f"order mismatch in {s['scene']}", file=sys.stderr); ok = False; continue
        rows.append(z["emb"])
        rows_tok.append(z["emb_tok"])
    if ok:
        emb = np.concatenate(rows)
        np.save(INDEX_DIR / "emb_v1.npy", emb)
        np.save(INDEX_DIR / "emb_v1b.npy", np.concatenate(rows_tok))
        print(f"merged v1 (cls+tok): {emb.shape}")


def main():
    V1_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads((INDEX_DIR / "meta.json").read_text())
    enmap_nm = np.loadtxt(ROOT / "v1_assets" / "enmap_bands_remapped.csv",
                          delimiter=",", skiprows=1, usecols=1)
    mu = np.load(ROOT / "v1_assets" / "mu.npy")
    sigma = np.load(ROOT / "v1_assets" / "sigma.npy")
    model = load_model()
    t0 = time.time()
    for s in meta["scenes"]:
        try:
            print(embed_scene(model, s["scene"], s["chips"], enmap_nm, mu, sigma), flush=True)
        except Exception as e:
            print(f"{s['scene']}: FAILED {e!r}", flush=True)
    print(f"total {time.time()-t0:.0f}s")
    merge(meta)


if __name__ == "__main__":
    main()
