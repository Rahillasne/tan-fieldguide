# Fieldguide — 2026 Tanager Open Data Competition entry

**Live demo:** https://planetrahil.github.io/tan-fieldguide/
**Code:** https://github.com/Planetrahil/tan-fieldguide

## One line

The first similarity-search index over Tanager's open hyperspectral archive:
click any 3.8 km chip and retrieve its spectral lookalikes across 149 scenes,
entirely in the browser.

## Why it matters

Tanager's archive is searchable by date, location, and cloud cover — none of
which describe what is in the pixels, even though every pixel carries 426
bands. Fieldguide makes the spectral dimension itself queryable. Every
"queryable Earth" embedding product shipped to date (AlphaEarth, Clay,
Major TOM, Earth Index, TESSERA) operates on 10 m-class multispectral or SAR
imagery; none ingests imaging spectroscopy. Similarity search where the
distance metric lives in 426 bands is a capability that only a hyperspectral
archive can offer.

## What was built

- A chip index over all 151 open ortho surface-reflectance scenes (149 usable,
  3,558 chips), with the product's own cloud/nodata masks and a hard mask on
  known instrument artifact windows so the index clusters on ground content
  rather than calibration state.
- Three descriptor families — an interpretable mean spectrum, a composition-
  aware bag-of-spectral-words histogram, and a frozen EnMAP-pretrained
  SpectralEarth ViT — switchable live, with reciprocal-rank fusion as the
  measured-best default.
- Near-duplicate suppression: the 149 scenes cover only 75 distinct sites, so
  retrieval excludes the query's own site and caps results per scene.
- A static viewer: a few MB of index, brute-force exact search client-side,
  no server and no vector database.

## Results

All 3,558 chips as queries, query site excluded, physics pseudo-class labels,
random baseline 0.366:

| Ranking | class purity@10 |
|---|---|
| mean spectrum | 0.812 |
| spectral words | 0.825 |
| SpectralEarth (frozen) | 0.845 |
| **fusion (default)** | **0.870** |

The interpretable baseline is competitive with a pretrained foundation model,
and fusing them beats both — a useful, honest result for anyone considering
learned embeddings on a new sensor.

## Data and licensing

Built exclusively on the CC-BY 4.0 Tanager-1 open release
(`tanager1-release2-core-imagery`), imagery © Planet Labs PBC. The open
`ql_ch4_json` methane detections (Carbon Mapper program) are used as
qualitative anchors only, never as training data. Code is MIT.
