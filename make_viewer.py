#!/usr/bin/env python3
"""Bundle the Fieldguide index into viewer outputs.

Three outputs from one template (viewer_template.html):
  build/fieldguide.html          self-contained, doctype, open locally
  build/fieldguide_artifact.html self-contained, no doctype (artifact publish)
  site/                          split assets for static hosting (GitHub Pages):
                                 index.html + meta.json + spec.bin + thumbs/
"""

import base64
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
INDEX_DIR = ROOT / "data" / "index"
BUILD = ROOT / "build"
SITE = ROOT / "site"


def render(template, meta_expr, spec_expr, thumb_fn, spec2_expr, spec3_expr):
    html = template.replace("__DATA_META__", meta_expr)
    html = html.replace("__DATA_SPEC_EXPR__", spec_expr)
    html = html.replace("__DATA_SPEC2_EXPR__", spec2_expr)
    html = html.replace("__DATA_SPEC3_EXPR__", spec3_expr)
    html = html.replace("__THUMB_FN__", thumb_fn)
    assert "__DATA_" not in html and "__THUMB_" not in html
    return html


def inline_f32(arr):
    b64 = base64.b64encode(arr.astype(np.float32).tobytes()).decode()
    return (
        '(() => { const bin = atob("' + b64 + '");'
        " const buf = new Uint8Array(bin.length);"
        " for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);"
        " return new Float32Array(buf.buffer); })()"
    )



# Example queries for the viewer's empty state: the most prototypical chip of
# each class (highest class fraction, mostly-clear), so a first-time visitor
# can see a good result in one click.
EXAMPLE_CLASSES = [
    ("vegetation", "Vegetation"),
    ("water", "Water"),
    ("bright", "Salt flat"),
    ("snow_ice", "Snow & ice"),
    ("barren", "Desert & bare rock"),
]


def pick_examples(aux, chips):
    classes = aux["classes"]
    frac = np.array(aux["chip_class_frac"], dtype=float)
    bad = np.array([c["bad_frac"] for c in chips])
    plume = np.array([c["plume"] for c in chips])
    used_scenes, out = set(), []
    for key, label in EXAMPLE_CLASSES:
        if key not in classes:
            continue
        col = frac[:, classes.index(key)].copy()
        col[bad > 0.02] = -1
        for i in np.argsort(-col):
            if col[i] <= 0:
                break
            if chips[i]["scene"] in used_scenes:
                continue
            used_scenes.add(chips[i]["scene"])
            out.append({"i": int(i), "label": label})
            break
    plume_idx = np.where(plume & (bad <= 0.1))[0]
    if len(plume_idx):
        best = plume_idx[np.argmin(bad[plume_idx])]
        out.append({"i": int(best), "label": "Methane site"})
    return out


def load_optional_emb(path, n_chips, label):
    if not path.exists():
        return None
    cand = np.load(path).astype(np.float32)
    if cand.shape[0] != n_chips:
        print(f"NOTE: {label} rows {cand.shape[0]} != chips {n_chips} — shipping without it")
        return None
    return cand


def main():
    meta = json.loads((INDEX_DIR / "meta.json").read_text())
    emb = np.load(INDEX_DIR / "emb.npy").astype(np.float32)
    chips = [c for s in meta["scenes"] for c in s["chips"]]
    assert emb.shape[0] == len(chips), f"emb rows {emb.shape[0]} != chips {len(chips)}"

    # merge aux (classes + site groups) into the shipped metadata, if built
    aux_path = INDEX_DIR / "aux.json"
    if aux_path.exists():
        aux = json.loads(aux_path.read_text())
        if aux["chip_ids"] == [c["id"] for c in chips]:
            cls_of = dict(zip(aux["chip_ids"], aux["chip_class"]))
            sites = sorted(set(aux["site_of_scene"].values()))
            site_idx = {s: i for i, s in enumerate(sites)}
            for s in meta["scenes"]:
                s["site"] = site_idx[aux["site_of_scene"][s["scene"]]]
                for c in s["chips"]:
                    c["cls"] = cls_of[c["id"]].replace("_", " ")
            meta["examples"] = pick_examples(aux, chips)
        else:
            print("NOTE: aux.json out of date — rerun build_v2.py; shipping without classes")

    template = (ROOT / "viewer_template.html").read_text()
    meta_json = json.dumps(meta, separators=(",", ":"))

    # --- self-contained variants ---
    thumbs = {}
    for ch in chips:
        jpg = (INDEX_DIR / "thumbs" / f"{ch['id']}.jpg").read_bytes()
        thumbs[ch["id"]] = "data:image/jpeg;base64," + base64.b64encode(jpg).decode()
    emb_v1 = load_optional_emb(INDEX_DIR / "emb_v1b.npy", len(chips), "emb_v1b")
    if emb_v1 is None:
        emb_v1 = load_optional_emb(INDEX_DIR / "emb_v1.npy", len(chips), "emb_v1")
    emb_v2 = load_optional_emb(INDEX_DIR / "emb_v2.npy", len(chips), "emb_v2")

    thumb_fn_inline = "(m => id => m[id])(" + json.dumps(thumbs, separators=(",", ":")) + ")"
    spec2_inline = inline_f32(emb_v1) if emb_v1 is not None else "null"
    spec3_inline = inline_f32(emb_v2) if emb_v2 is not None else "null"
    html = render(template, meta_json, inline_f32(emb), thumb_fn_inline,
                  spec2_inline, spec3_inline)

    BUILD.mkdir(exist_ok=True)
    (BUILD / "fieldguide.html").write_text("<!doctype html>\n" + html)
    (BUILD / "fieldguide_artifact.html").write_text(html)

    # --- split-asset site (module script allows top-level await) ---
    SITE.mkdir(exist_ok=True)
    (SITE / "meta.json").write_text(meta_json)
    (SITE / "spec.bin").write_bytes(emb.tobytes())
    (SITE / ".nojekyll").write_text("")
    # loadF32 returns null on a 404 or a truncated body, so a missing file
    # degrades to "that ranking is unavailable" instead of a blank page
    def fetch_f32(name, arr):
        return f"await loadF32('{name}', {arr.shape[1]})"

    spec2_site = spec3_site = "null"
    if emb_v1 is not None:
        (SITE / "spec2.bin").write_bytes(emb_v1.tobytes())
        spec2_site = fetch_f32("spec2.bin", emb_v1)
    if emb_v2 is not None:
        (SITE / "spec3.bin").write_bytes(emb_v2.tobytes())
        spec3_site = fetch_f32("spec3.bin", emb_v2)
    site_html = render(
        template,
        "await (await fetch('meta.json')).json()",
        fetch_f32("spec.bin", emb),
        "id => 'thumbs/' + id + '.jpg'",
        spec2_site,
        spec3_site,
    )
    (SITE / "index.html").write_text("<!doctype html>\n" + site_html)
    if (SITE / "thumbs").exists():
        shutil.rmtree(SITE / "thumbs")
    shutil.copytree(INDEX_DIR / "thumbs", SITE / "thumbs")

    size = (BUILD / "fieldguide.html").stat().st_size / 1e6
    site_mb = sum(p.stat().st_size for p in SITE.rglob("*") if p.is_file()) / 1e6
    print(f"build/fieldguide.html {size:.1f} MB | site/ {site_mb:.1f} MB | "
          f"{len(meta['scenes'])} scenes, {len(chips)} chips, {emb.shape[1]}-pt spectra")


if __name__ == "__main__":
    main()
