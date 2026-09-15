import csv

import numpy as np

from knotvision.export import read_indexed_tiff, write_indexed_tiff
from knotvision.job import read_json


def test_tiff_roundtrip_anisotropic_resolution(tmp_path):
    idx = np.random.RandomState(0).randint(0, 5, (30, 40)).astype(np.int32)
    pal = np.array([[1, 2, 3], [10, 20, 30], [100, 110, 120], [200, 210, 220], [255, 0, 0]], np.uint8)
    p = tmp_path / "d.tiff"
    write_indexed_tiff(p, idx, pal, 420, 380)
    back, rgb = read_indexed_tiff(p)
    assert (back == idx).all() and (rgb == pal).all()
    import tifffile
    with tifffile.TiffFile(str(p)) as tf:
        page = tf.pages[0]
        xr, yr = page.tags["XResolution"].value, page.tags["YResolution"].value
        assert abs(xr[0] / xr[1] - 420 * 0.0254) < 0.01 and abs(yr[0] / yr[1] - 380 * 0.0254) < 0.01
        assert page.photometric.name == "PALETTE"


def test_outputs_and_palette_csv(geometric_run):
    job = geometric_run["job"]
    out = job.output_dir
    for f in ("design.tiff", "palette.csv", "preview.png", "qc.json", "report.json"):
        assert (out / f).exists()
    idx, rgb = read_indexed_tiff(out / "design.tiff")
    assert len(np.unique(idx)) <= job.machine_colors
    with open(out / "palette.csv") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == ["index", "yarn_code", "name", "R", "G", "B", "knots", "coverage_pct", "quantization_deltaE"]
    assert sum(int(r["knots"]) for r in rows) == idx.size
    rep = read_json(out / "report.json")
    for k in ("versions", "setup", "photo", "grid", "design_class", "class_auto", "class_margin", "colors", "fringe",
              "illumination", "internal_geometry", "palette_confirmed", "design_mode", "design_mode_requested",
              "design_mode_class_default", "edit", "regularize", "guard", "dither", "qc", "timings_s", "warnings",
              "partial_view", "non_planar", "non_planar_internal", "deterministic"):
        assert k in rep
    assert rep["edit"] is None and rep["deterministic"] is True and rep["design_mode"] == "quantize"


def test_consent_pair_exported(geometric_pair, tmp_root, monkeypatch):
    from knotvision import export as E
    from tests.conftest import make_job, run
    monkeypatch.setattr(E, "DATA_DIR", tmp_root / "data")
    job = make_job(geometric_pair["dir"], tmp_root / "runs", "geometric", job_id="geo_consent", consent_train=True)
    run(job)
    pair = tmp_root / "data" / "real" / "pairs" / "geo_consent"
    assert (pair / "design.tiff").exists() and (pair / "meta.json").exists() and (pair / "corners.json").exists()
    meta = read_json(pair / "meta.json")
    assert meta["reviewed"] is False and meta["design_class"] == "geometric" and meta["designer_score"] is None
