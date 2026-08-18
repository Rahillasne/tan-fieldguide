import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from core import interp_matrix, utm_to_wgs84, good_band_mask, ARTIFACT_WINDOWS_NM  # noqa: E402
from build_v2 import classify_pixels, site_groups, CLASSES  # noqa: E402


def test_interp_matrix_matches_np_interp():
    rng = np.random.default_rng(1)
    src = np.sort(rng.uniform(400, 2500, 80))
    dst = np.linspace(380, 2600, 50)  # includes out-of-range ends
    vals = rng.normal(size=80)
    W = interp_matrix(src, dst)
    np.testing.assert_allclose(W @ vals, np.interp(dst, src, vals), rtol=1e-5, atol=1e-6)


def test_utm_central_meridian_and_equator():
    # easting 500000 lies on the central meridian; zone 11 N -> -117 deg
    lat, lon = utm_to_wgs84(500000.0, 0.0, 11, True)
    assert abs(lon - (-117.0)) < 1e-6
    assert abs(lat) < 1e-6


def test_utm_known_point():
    # Mount Whitney-ish: 11N 384409E 4048768N -> 36.578N, -118.292W (approx)
    lat, lon = utm_to_wgs84(384409.0, 4048768.0, 11, True)
    assert abs(lat - 36.578) < 0.01
    assert abs(lon + 118.292) < 0.01


def test_good_band_mask_removes_artifact_windows():
    wl = np.arange(380.0, 2500.0, 5.0)
    good = np.ones(len(wl), dtype=np.uint8)
    m = good_band_mask(wl, good)
    for lo, hi in ARTIFACT_WINDOWS_NM:
        assert not m[(wl >= lo) & (wl <= hi)].any()
    assert m.sum() > 0.8 * len(wl)


def _flat_spectrum(grid, level):
    return np.full((1, len(grid)), level, dtype=np.float32)


def test_classify_pixels_basic():
    grid = np.arange(400.0, 2451.0, 10.0)
    # water: green > NIR, dark NIR
    water = _flat_spectrum(grid, 0.02)
    water[0, np.argmin(np.abs(grid - 560))] = 0.06
    # vegetation: NIR >> red
    veg = _flat_spectrum(grid, 0.05)
    veg[0, np.argmin(np.abs(grid - 860))] = 0.45
    # bright playa
    bright = _flat_spectrum(grid, 0.45)
    spec = np.concatenate([water, veg, bright])
    got = [CLASSES[i] for i in classify_pixels(spec, grid)]
    assert got == ["water", "vegetation", "bright"]


def test_site_groups_transitive():
    scenes = [
        {"scene": "a", "lat": 10.0, "lon": 20.0},
        {"scene": "b", "lat": 10.2, "lon": 20.2},   # near a
        {"scene": "c", "lat": 10.4, "lon": 20.4},   # near b, not a
        {"scene": "d", "lat": -30.0, "lon": 100.0},  # alone
    ]
    site = site_groups(scenes)
    assert site["a"] == site["b"] == site["c"]
    assert site["d"] == "d"
    assert len(set(site.values())) == 2
