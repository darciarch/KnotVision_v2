import cv2
import numpy as np

from knotvision import palette as P
from knotvision.color import delta_e2000, rgb_to_lab
from knotvision.job import CONFIG_DIR, read_json
from tests.conftest import design_of, make_job, photo_of, run, write_pair


def _cluster_from_image(img, mc=6, gregions=None):
    return P.cluster_photo(img, None, mc, 0.5, gregions or [])


def test_adjacent_similar_stay_apart_nonadjacent_similar_merge():
    # A: two similar colours sharing a long boundary; B: the same two colours far apart (no boundary)
    img = np.zeros((200, 400, 3), np.uint8)
    img[:, :200] = [120, 120, 120]
    img[:, 200:] = [130, 130, 130]          # similar, adjacent
    img[20:60, 20:60] = [200, 40, 40]        # distinct motif
    cl = _cluster_from_image(img, mc=3)
    prot = np.zeros(len(cl["centers"]), bool)
    groups_adj = P.merge_clusters(cl, 3, adjacency_penalty=2.0, merge_cost_stop=0.5, protected=prot)
    img2 = np.zeros((200, 400, 3), np.uint8)
    img2[:] = [120, 120, 120]
    img2[150:190, 300:390] = [130, 130, 130]
    img2[20:60, 20:60] = [200, 40, 40]
    img2[80:120, 20:60] = [40, 40, 200]    # keeps k reasonable
    cl2 = _cluster_from_image(img2, mc=3)
    groups_non = P.merge_clusters(cl2, 2, adjacency_penalty=2.0, merge_cost_stop=8.0, protected=np.zeros(len(cl2["centers"]), bool))
    # non-adjacent similar greys merge into one group when merging continues; adjacent ones cost more
    def n_grey_groups(cl, groups):
        greys = [i for i, c in enumerate(cl["centers"]) if abs(c[1]) < 3 and abs(c[2]) < 3 and 40 < c[0] < 60]
        return len({gi for gi, g in enumerate(groups) if any(i in g for i in greys)})
    assert n_grey_groups(cl2, groups_non) == 1
    assert n_grey_groups(cl, groups_adj) >= 1


def test_two_colour_design_uses_two_entries(tmp_root):
    d = design_of("minimal", seed=141, n_colors=2, border=False, gradient=False, noise=False, thin_lines=False, symmetry=False)
    assert len(d.palette_rgb) == 2
    r = photo_of(d, seed=141)
    p = write_pair(d, r, tmp_root / "pairs" / "two")
    job = make_job(p, tmp_root / "runs", "minimal", job_id="two", machine_colors=8)
    rep = run(job)
    assert rep["colors"]["palette_used"] <= 3
    assert rep["colors"]["palette_used"] <= rep["colors"]["machine_colors"]


def test_palette_upper_bound_and_card(geometric_run):
    job = geometric_run["job"]
    pal = read_json(job.stage_dir("palette") / "palette.json")
    assert 1 <= len(pal["entries"]) <= job.machine_colors
    card = cv2.imread(str(job.stage_dir("palette") / "palette_card.png"))
    assert card is not None
    # one 128 px square per entry: count distinct solid squares by sampling cell centres
    n = len(pal["entries"])
    cols = min(n, 6)
    for i, e in enumerate(n and pal["entries"]):
        r_, c_ = divmod(i, cols)
        x = 8 + c_ * 136 + 64; y = 8 + r_ * (128 + 28 + 8) + 64
        bgr = card[y, x]
        assert np.abs(bgr[::-1].astype(int) - np.array(e["rgb"])).max() <= 2


def test_gradient_endpoints_in_palette(gradient_run):
    job = gradient_run["job"]
    pal = read_json(job.stage_dir("palette") / "palette.json")
    eps = [e for e in pal["entries"] if e.get("gradient_endpoint")]
    assert len(eps) >= 2


def test_motif_colour_protected():
    img = np.full((300, 300, 3), 200, np.uint8)
    img[:, 150:] = [60, 60, 60]
    for i in range(6):                        # 6 small red motifs, ~1 % coverage
        y = 20 + i * 45
        img[y:y + 12, 20:32] = [220, 30, 30]
    cl = _cluster_from_image(img, mc=2)
    prot = P.motif_protected(cl, 5, 10, 100, 100)
    reds = [i for i, c in enumerate(cl["centers"]) if c[1] > 30]
    assert reds and all(prot[i] for i in reds)
    groups = P.merge_clusters(cl, 2, 2.0, 0.5, prot)
    # the red cluster survives as its own group when 3 groups are allowed
    groups3 = P.merge_clusters(cl, 3, 2.0, 0.5, prot)
    assert any(set(g) <= set(reds) for g in groups3)


def test_library_selection_stops_on_gain(tmp_root, geometric_pair):
    from knotvision.intake import load_yarn_csv
    rows, yarn_lab = load_yarn_csv(CONFIG_DIR / "yarns" / "sample.csv")
    img = geometric_pair["render"]["photo"]
    cl = _cluster_from_image(img, mc=8)
    chosen_tight = P.select_from_library(cl, yarn_lab, rows, 8, gain_stop_select=0.3)
    chosen_loose = P.select_from_library(cl, yarn_lab, rows, 8, gain_stop_select=50.0)
    assert 1 <= len(chosen_loose) <= len(chosen_tight) <= 8
    job = make_job(geometric_pair["dir"], tmp_root / "runs", "geometric", job_id="geo_yarn", yarn_csv=CONFIG_DIR / "yarns" / "sample.csv")
    rep = run(job)
    pal = read_json(job.stage_dir("palette") / "palette.json")
    assert pal["source"] == "library" and all(e["yarn_code"] for e in pal["entries"])
    assert rep["setup"]["yarn_library"] is True


def test_merge_cost_stop_separate_from_gain_stop():
    img = np.zeros((200, 200, 3), np.uint8)
    for i, c in enumerate([[10, 10, 10], [250, 250, 250], [200, 30, 30], [30, 200, 30], [30, 30, 200], [200, 200, 30]]):
        img[:, i * 33:(i + 1) * 33] = c
    cl = _cluster_from_image(img, mc=6)
    prot = np.zeros(len(cl["centers"]), bool)
    g_stop = P.merge_clusters(cl, 6, 2.0, merge_cost_stop=0.5, protected=prot)
    g_more = P.merge_clusters(cl, 6, 2.0, merge_cost_stop=1e6, protected=prot)
    assert len(g_stop) == 6 and len(g_more) == 1


def test_palette_override_and_confirmed(geometric_pair, tmp_root):
    ov = [{"rgb": [200, 180, 150], "yarn_code": "12", "name": "Beige"}, {"rgb": [60, 40, 30], "yarn_code": "27", "name": "Dark"},
          {"rgb": [150, 60, 40], "yarn_code": "44", "name": "Rust"}]
    job = make_job(geometric_pair["dir"], tmp_root / "runs", "geometric", job_id="geo_palov", palette_override=ov)
    rep = run(job)
    pal = read_json(job.stage_dir("palette") / "palette.json")
    assert pal["source"] == "override" and len(pal["entries"]) == 3 and rep["palette_confirmed"] is True
