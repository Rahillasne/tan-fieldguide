#!/usr/bin/env python3
"""Fieldguide index builder: open Tanager SR scenes -> chip descriptors + thumbnails.

Reads every ortho_sr_hdf5 scene in data/scenes/, cuts 128x128 chips, computes a
masked mean-spectrum descriptor on a canonical wavelength grid, renders an RGB
thumbnail per chip, and writes per-scene results to data/index/scenes/.
Run merge() (default after indexing) to assemble data/index/{meta.json,emb.npy}.

Idempotent: scenes with an existing .npz under data/index/scenes/ are skipped.
"""

import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from core import ARTIFACT_WINDOWS_NM, CHIP, SR_PATH, good_band_mask, utm_to_wgs84

ROOT = Path(__file__).parent
SCENES_DIR = ROOT / "data" / "scenes"
CH4_DIR = ROOT / "data" / "ch4"
INDEX_DIR = ROOT / "data" / "index"
THUMBS_DIR = INDEX_DIR / "thumbs"

MAX_BAD_FRAC = 0.30  # reject chips with more nodata+cloud+cirrus than this

# Canonical wavelength grid the descriptor lives on. Points with no good band
# within GAP_NM are dropped (this removes the deep water-vapor windows and any
# masked artifact window without listing them twice).
GRID = np.arange(400.0, 2451.0, 10.0)
GAP_NM = 12.0

RGB_NM = (641.0, 551.0, 471.0)


def parse_grid_corners(struct_metadata):
    ul = re.search(r"UpperLeftPointMtrs=\(([\d.\-]+),([\d.\-]+)\)", struct_metadata)
    lr = re.search(r"LowerRightMtrs=\(([\d.\-]+),([\d.\-]+)\)", struct_metadata)
    return (float(ul.group(1)), float(ul.group(2))), (float(lr.group(1)), float(lr.group(2)))


def load_plume_features():
    """scene_id -> list of lon/lat bounding boxes of open CH4 plume features."""
    plumes = {}
    if not CH4_DIR.exists():
        return plumes
    for p in sorted(CH4_DIR.glob("*.geojson")):
        sid = p.name.split("_ql_ch4")[0]
        try:
            gj = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        boxes = []
        for feat in gj.get("features", []):
            geom = feat.get("geometry") or {}
            coords = np.array(list(_iter_coords(geom)))
            if len(coords):
                boxes.append([coords[:, 0].min(), coords[:, 1].min(),
                              coords[:, 0].max(), coords[:, 1].max()])
        if boxes:
            plumes[sid] = boxes
    return plumes


def _iter_coords(geom):
    t, c = geom.get("type"), geom.get("coordinates", [])
    if t == "Point":
        yield c
    elif t in ("MultiPoint", "LineString"):
        yield from c
    elif t in ("MultiLineString", "Polygon"):
        for ring in c:
            yield from ring
    elif t == "MultiPolygon":
        for poly in c:
            for ring in poly:
                yield from ring


def index_scene(h5_path, plumes):
    sid = h5_path.name.replace("_ortho_sr_hdf5.h5", "")
    out_npz = INDEX_DIR / "scenes" / f"{sid}.npz"
    if out_npz.exists():
        return f"{sid}: already indexed"

    with h5py.File(h5_path, "r") as f:
        g = f[SR_PATH]
        sr_ds = g["surface_reflectance"]
        wl = sr_ds.attrs["wavelengths"].astype(float)
        good = good_band_mask(wl, sr_ds.attrs["good_wavelengths"])

        cube = sr_ds[:]  # (426, y, x) float32, ~1.5 GB
        bad_px = (g["nodata_pixels"][:] > 0) | (g["beta_cloud_mask"][:] > 0) | (g["beta_cirrus_mask"][:] > 0)
        epsg = int(f["HDFEOS/GRIDS/HYP"].attrs["epsg_code"])
        (ulx, uly), _ = parse_grid_corners(
            f["HDFEOS INFORMATION/StructMetadata.0"][()].decode())

    zone, northern = epsg % 100, (epsg // 100) % 10 == 6  # 326xx N / 327xx S
    ny, nx = cube.shape[1], cube.shape[2]

    # canonical grid: keep only points near an actual good band, and none
    # inside an artifact window (so masked windows are visible gaps, not
    # silently bridged by interpolation)
    keep = np.array([np.min(np.abs(wl[good] - g_nm)) <= GAP_NM for g_nm in GRID])
    for lo, hi in ARTIFACT_WINDOWS_NM:
        keep &= ~((GRID >= lo) & (GRID <= hi))
    grid_kept = GRID[keep]

    # per-scene RGB stretch from a spatial subsample of valid pixels
    rgb_idx = [int(np.argmin(np.abs(wl - nm))) for nm in RGB_NM]
    rgb_full = cube[rgb_idx].transpose(1, 2, 0)
    valid_sub = ~bad_px[::4, ::4]
    lo_hi = []
    for c in range(3):
        vals = rgb_full[::4, ::4, c][valid_sub]
        vals = vals[vals > -9000]
        lo_hi.append(np.percentile(vals, [2, 98]) if len(vals) else np.array([0.0, 1.0]))

    chips, embs = [], []
    for r0 in range(0, ny - CHIP + 1, CHIP):
        for c0 in range(0, nx - CHIP + 1, CHIP):
            bad = bad_px[r0:r0 + CHIP, c0:c0 + CHIP]
            bad_frac = float(bad.mean())
            if bad_frac > MAX_BAD_FRAC:
                continue
            block = cube[:, r0:r0 + CHIP, c0:c0 + CHIP]
            spec = block[:, ~bad].mean(axis=1)
            spec_good = spec[good]
            if not np.all(np.isfinite(spec_good)) or spec_good.max() <= 0:
                continue
            desc = np.interp(grid_kept, wl[good], spec_good)

            # chip center in lon/lat (pixel centers; 30 m grid)
            cx = ulx + (c0 + CHIP / 2) * 30.0
            cy = uly - (r0 + CHIP / 2) * 30.0
            lat, lon = utm_to_wgs84(cx, cy, zone, northern)

            # thumbnail
            rgb = rgb_full[r0:r0 + CHIP, c0:c0 + CHIP].copy()
            for c in range(3):
                lo, hi = lo_hi[c]
                rgb[:, :, c] = np.clip((rgb[:, :, c] - lo) / max(hi - lo, 1e-6), 0, 1)
            rgb = (np.power(rgb, 0.7) * 255).astype(np.uint8)
            rgb[bad] = [16, 16, 16]
            chip_id = f"{sid}_{r0}_{c0}"
            Image.fromarray(rgb).save(THUMBS_DIR / f"{chip_id}.jpg", quality=70)

            # plume features are points; pad their bbox by ~2 km so a chip
            # containing a detection gets flagged
            pad_deg = 0.02
            near_plume = any(
                b[0] - pad_deg <= float(lon) <= b[2] + pad_deg
                and b[1] - pad_deg <= float(lat) <= b[3] + pad_deg
                for b in plumes.get(sid, []))
            chips.append({
                "id": chip_id, "scene": sid, "r": r0, "c": c0,
                "lat": round(float(lat), 4), "lon": round(float(lon), 4),
                "bad_frac": round(bad_frac, 3), "plume": near_plume,
            })
            embs.append(desc.astype(np.float32))

    if not chips:
        np.savez(out_npz, emb=np.zeros((0, keep.sum()), np.float32))
        (INDEX_DIR / "scenes" / f"{sid}.json").write_text(json.dumps({"scene": sid, "chips": []}))
        return f"{sid}: 0 usable chips"

    center_lat, center_lon = utm_to_wgs84(ulx + nx * 15.0, uly - ny * 15.0, zone, northern)
    scene_meta = {
        "scene": sid,
        "date": f"{sid[0:4]}-{sid[4:6]}-{sid[6:8]}",
        "time": f"{sid[9:11]}:{sid[11:13]}Z",
        "lat": round(float(center_lat), 3), "lon": round(float(center_lon), 3),
        "has_plume_data": sid in plumes,
        "chips": chips,
    }
    np.savez(out_npz, emb=np.stack(embs), grid=grid_kept)
    (INDEX_DIR / "scenes" / f"{sid}.json").write_text(json.dumps(scene_meta))
    return f"{sid}: {len(chips)} chips"


def merge():
    scenes, embs, grid = [], [], None
    for j in sorted((INDEX_DIR / "scenes").glob("*.json")):
        meta = json.loads(j.read_text())
        if not meta.get("chips"):
            continue
        z = np.load(j.with_suffix(".npz"))
        if grid is None:
            grid = z["grid"]
        elif z["emb"].shape[1] != len(grid):
            print(f"skip {meta['scene']}: grid mismatch", file=sys.stderr)
            continue
        scenes.append(meta)
        embs.append(z["emb"])
    emb = np.concatenate(embs) if embs else np.zeros((0, 0), np.float32)
    np.save(INDEX_DIR / "emb.npy", emb)
    (INDEX_DIR / "meta.json").write_text(json.dumps(
        {"grid_nm": [float(x) for x in (grid if grid is not None else [])],
         "scenes": scenes}))
    n = sum(len(s["chips"]) for s in scenes)
    print(f"merged: {len(scenes)} scenes, {n} chips, emb {emb.shape}")


def main():
    (INDEX_DIR / "scenes").mkdir(parents=True, exist_ok=True)
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)
    plumes = load_plume_features()
    for h5_path in sorted(SCENES_DIR.glob("*_ortho_sr_hdf5.h5")):
        try:
            print(index_scene(h5_path, plumes), flush=True)
        except Exception as e:
            print(f"{h5_path.name}: FAILED {e!r}", flush=True)
    merge()


if __name__ == "__main__":
    main()
