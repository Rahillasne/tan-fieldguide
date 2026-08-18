"""Shared numerics and constants for the Fieldguide pipeline (numpy-only)."""

import numpy as np

CHIP = 128
SR_PATH = "HDFEOS/GRIDS/HYP/Data Fields"

# Instrument artifact windows masked on top of the product's good_wavelengths
# flag (which only covers the two deep water-vapor windows): order-sorting-
# filter seam regions, the O2-A telluric residual, and a known calibration
# artifact near 936 nm.
ARTIFACT_WINDOWS_NM = [(651.0, 671.0), (755.0, 770.0), (929.0, 944.0), (1277.0, 1307.0)]


def good_band_mask(wl, good_flags):
    """Product good_wavelengths flag AND the artifact-window mask."""
    good = good_flags.astype(bool).copy()
    for lo, hi in ARTIFACT_WINDOWS_NM:
        good &= ~((wl >= lo) & (wl <= hi))
    return good


def interp_matrix(src_nm, dst_nm):
    """W (len(dst), len(src)) such that W @ spectra == np.interp per point."""
    W = np.zeros((len(dst_nm), len(src_nm)), dtype=np.float32)
    idx = np.searchsorted(src_nm, dst_nm)
    for i, (j, nm) in enumerate(zip(idx, dst_nm)):
        if j <= 0:
            W[i, 0] = 1.0
        elif j >= len(src_nm):
            W[i, -1] = 1.0
        else:
            t = (nm - src_nm[j - 1]) / (src_nm[j] - src_nm[j - 1])
            W[i, j - 1] = 1.0 - t
            W[i, j] = t
    return W


def utm_to_wgs84(easting, northing, zone, northern):
    """Inverse transverse Mercator (Snyder series), accurate to ~1 m."""
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    e1 = (1 - np.sqrt(1 - e2)) / (1 + np.sqrt(1 - e2))
    k0 = 0.9996
    x = np.asarray(easting, dtype=float) - 500000.0
    y = np.asarray(northing, dtype=float)
    if not northern:
        y = y - 10000000.0
    m = y / k0
    mu = m / (a * (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256))
    phi1 = (mu + (3 * e1 / 2 - 27 * e1**3 / 32) * np.sin(2 * mu)
            + (21 * e1**2 / 16 - 55 * e1**4 / 32) * np.sin(4 * mu)
            + (151 * e1**3 / 96) * np.sin(6 * mu)
            + (1097 * e1**4 / 512) * np.sin(8 * mu))
    ep2 = e2 / (1 - e2)
    c1 = ep2 * np.cos(phi1) ** 2
    t1 = np.tan(phi1) ** 2
    n1 = a / np.sqrt(1 - e2 * np.sin(phi1) ** 2)
    r1 = a * (1 - e2) / (1 - e2 * np.sin(phi1) ** 2) ** 1.5
    d = x / (n1 * k0)
    lat = phi1 - (n1 * np.tan(phi1) / r1) * (
        d**2 / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * ep2) * d**4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * ep2 - 3 * c1**2) * d**6 / 720)
    lon = (d - (1 + 2 * t1 + c1) * d**3 / 6
           + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * ep2 + 24 * t1**2) * d**5 / 120) / np.cos(phi1)
    lon0 = np.radians((zone - 1) * 6 - 180 + 3)
    return np.degrees(lat), np.degrees(lon) + np.degrees(lon0)
