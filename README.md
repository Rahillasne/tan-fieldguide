# Tanager Fieldguide

**Search a hyperspectral archive by what is actually in the pixels.**

Planet's Tanager-1 satellite measures 426 spectral bands from 377 to 2500 nm.
An RGB browse image shows about 15% of that range, and the archive itself can
only be searched by date, location, and cloud cover — none of which say
anything about *what the ground is made of*. Fieldguide closes that gap: click
any chip in any scene and it returns the spectrally most similar chips across
the entire open archive, with both full spectra overlaid.

### ▶ [Live demo](https://planetrahil.github.io/tan-fieldguide/)

149 scenes · 3,558 chips · runs entirely in your browser, nothing to install,
no server and no vector database.

---

## How it works

Every scene is cut into 128×128-pixel chips (3.8 km at 30 m GSD), rejecting
chips that are more than 30% nodata/cloud/cirrus per the product's own masks.
Each chip carries three independent descriptors:

| Descriptor | What it is | Why it's there |
|---|---|---|
| **mean spectrum** | cloud-masked mean reflectance on a canonical 10 nm grid, compared by spectral angle | interpretable by construction — two chips match because their spectra match |
| **spectral words** | 64-word bag-of-spectral-words histogram (spherical k-means codebook over pixel spectra), Hellinger-transformed | composition-aware: a shoreline chip matches other mixed water/land chips instead of the meaningless average of the two |
| **SpectralEarth** | frozen EnMAP-pretrained ViT, patch-token pooled; Tanager's 426 bands resampled onto its 202-band grid | the learned arm — and the yardstick check on whether pretraining transfers to a new sensor |

The default ranking is **reciprocal-rank fusion of all three**, chosen by
measurement rather than taste (below). You can switch between them live.

**Spectral hygiene.** On top of the product's `good_wavelengths` flag (which
covers the two deep water-vapor windows), known instrument artifact windows are
masked out: order-sorting-filter seams at 651–671 and 1277–1307 nm, the O₂-A
telluric residual near 762 nm, and a calibration artifact near 936 nm. Masked
regions appear as *visible gaps* in the plotted spectra — never bridged by
interpolation. Without this the index happily clusters on calibration state
instead of ground content.

**Near-duplicate suppression.** The open archive revisits the same places:
149 scenes cover only 75 distinct sites. Retrieval therefore excludes the
query's whole *site* (scene centers within ~35 km are grouped), and caps
results at 2 chips per scene, so a result list shows genuine spectral analogues
rather than the same terrain on a different date.

## Retrieval evaluation

Every one of the 3,558 chips is used as a query, with its own site excluded.
Labels are physics pseudo-classes (water / snow-ice / vegetation / bright /
dark / barren) derived from band indices on 256 sampled pixels per chip.

| Ranking | class purity@10 |
|---|---|
| mean spectrum | 0.812 |
| spectral words | 0.825 |
| SpectralEarth (frozen, token-pooled) | 0.845 |
| **fusion — the default** | **0.870** |

Random baseline: 0.366.

No single family dominates every probe, which is the point: the archive needs
a queryable index precisely so that methods can be compared on real retrieval
tasks instead of asserted.

## Data

Everything here is built from Planet's **CC-BY 4.0 Tanager-1 open release**
(`tanager1-release2-core-imagery`): ortho surface-reflectance HDF-EOS5 cubes
plus the `ql_ch4_json` methane detections, which are used as qualitative
anchors (amber badges mark chips within ~2 km of a detection) and never as
training data. 151 scenes were processed; 2 yielded no usable chips (fully
clouded).

Imagery © Planet Labs PBC, CC-BY 4.0. Methane detections via the Carbon Mapper
program.

## Reproduce

```sh
python3 -m venv .venv && .venv/bin/pip install numpy h5py pillow scikit-learn
mkdir -p data/scenes data/ch4

gsutil ls gs://open-cogs/planet-stac/tanager1-release2-core-imagery/ortho_sr_hdf5/ > list.txt
xargs -P 4 -n 1 -J % gsutil -q cp -n % data/scenes/ < list.txt   # ~154 GB for all 151
gsutil -m cp "gs://open-cogs/planet-stac/tanager1-release2-core-imagery/ql_ch4_json/*.geojson" data/ch4/

.venv/bin/python build_index.py     # chips, mean-spectrum descriptors, thumbnails
.venv/bin/python sample_pixels.py   # 256 pixel spectra per chip
.venv/bin/python build_v2.py        # codebook, spectral-words, classes, site groups
.venv/bin/python embed_v1.py        # optional: SpectralEarth arm (see note)
.venv/bin/python eval_v2.py         # the table above
.venv/bin/python make_viewer.py     # -> build/fieldguide.html and docs/
```

Every stage is idempotent per scene, so an interrupted run resumes where it
stopped.

> **Note on gsutil:** `gsutil -m cp -I` (reading a file list from stdin)
> silently drops most of its input on macOS. Use the `xargs` form above.

> **Note on SpectralEarth weights:** they are not vendored here — upstream
> publishes no license. Fetch them from the
> [SpectralEarth repo](https://github.com/AABNassim/spectral_earth) and place
> the checkpoint, band grid, and normalization statistics under `v1_assets/`.
> The other two descriptors have no such dependency, and the viewer degrades
> gracefully when the embeddings are absent.

## Files

| File | Role |
|---|---|
| `core.py` | shared numerics: band masks, interpolation matrix, inverse UTM |
| `build_index.py` | scenes → chips, mean-spectrum descriptors, thumbnails, geolocation |
| `sample_pixels.py` | per-chip pixel-spectra samples (feeds everything below) |
| `build_v2.py` | spectral-words codebook + histograms, land-cover classes, site groups |
| `embed_v1.py` | frozen SpectralEarth embeddings |
| `eval_v2.py` | the retrieval evaluation |
| `make_viewer.py`, `viewer_template.html` | static viewer: single self-contained file, or the split-asset site in `docs/` |
| `tests/` | unit tests, run in CI |

## Limitations

- Chip scale is fixed at 128×128 (3.8 km). Objects of interest are often much
  smaller; the mean-spectrum and spectral-words arms are chip-size agnostic and
  could be rebuilt finer, and the ViT already computes ~120 m patch tokens that
  could be indexed directly for sub-chip localization.
- Land-cover labels are physics heuristics, not ground truth.
- With site exclusion enabled, the 13 plume-adjacent chips in the open release
  give no measurable cross-site retrieval signal for any method — too few
  positives to say anything. The badges stay qualitative.
- Search is exact brute force. That is a feature at this scale (milliseconds
  over a few thousand chips) but would need an ANN index beyond ~10⁵ chips.

## License

Code: MIT (see `LICENSE`). Imagery-derived assets under `docs/` (thumbnails,
spectra, metadata): CC-BY 4.0, attribution **Planet Labs PBC**.
