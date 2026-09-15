# KnotVision v6.6

Customer carpet photo → indexed Wilton production design (Texcelle TIFF). Spec (Turkish, authoritative):
`knotvision-pipeline-v6.6.md`. When code and spec disagree, the spec wins; when the spec is silent or its
literal rule breaks at a scale the tests use, pick the deterministic choice and record it under
"Spec deviations" in the module docstring (rectify.py and structure.py carry such lists). Code, comments
and identifiers are English.

## Environment

- `.venv/bin/python` (3.14). Core deps only; torch / diffusers / sam3 are optional and imported lazily.
  The modelless chain and every test run without them. No model weights, no archive, no real pairs and no
  `data/photo_survey.csv` exist yet: every job is `needs_review` by design (`cli.py setup-check` lists why).
- OpenCV is 5.x (`opencv-contrib-python-headless`): `HoughLinesP` returns `(N, 4)` — always
  `np.asarray(lines).reshape(-1, 4)`; `cv2.remap` maps must be `float32`.
- Tests: `.venv/bin/pytest` (≈ 2 min; session-scoped fixtures render the synthetic rugs once). `gpu` and
  `slow` markers exist and are skipped on CPU.
- Run dirs go under `runs/<job_id>/`; tests and experiments pass `runs_dir=` to keep the repo clean.
  Scratch files go to the session scratchpad, never into the repo.

## Pipeline shape

`intake → rectify → structure → palette → design → production → regularize → dither → qc → export`
(`knotvision/run.py`). One stage = one `run(job)` in its module, one input hash → one output dir
`stages/<nn>_<name>/`. Working resolution is fixed: 4 px/knot, `rectified.png` is exactly `4W × 4H`;
index maps are `(H, W)` int32 at grid resolution.

Stage contract:
1. `key = job.cache_key(stage, inputs, extra=...)`. The key hashes only the params the stage reads
   (`STAGE_PARAM_KEYS` in `job.py`; `design` passes `edit_param_keys(...)` in edit mode, `()` otherwise).
   A stage that starts reading a new param adds it there, otherwise a config change serves stale output.
2. `if job.cached(sd, key) and <main output exists>: job.timings[stage] = 0.0; return job.state`.
   A cache hit must leave `job.params` / `job.state` exactly as a real run would: `structure` re-applies
   class + routing from `layout.json["class"]` because `job.params["design_mode"]` is decided there.
3. Work, write outputs, `job.warn(...)` / `job.stop(...)`, append to `job.state["stages_ran"]`.
4. `job.mark_done(sd, key); job.save()`.

`job.warn(code)` is tagged with the current stage (`run_job` calls `begin_stage`); a cache hit restores
that stage's warnings from `stages/<nn>/warnings.json`, a rerun drops them first. Never write to
`job.warnings` directly. `job.stop(code)` raises `JobError` unless `--no-gate`, in which case it warns.

Parameter layers: loom profile ← class file ← agent file (`edit_*` keys only, edit mode only) ← job
override. `job.params` is the resolved set (`design_class` is always present); `job.params["agent"]`
holds the raw agent yaml. Loom-only fields (`LOOM_ONLY_FIELDS` in `job.py`) may not appear in class files.

## Rules that are easy to break

- **Single homes.** `is_thin`, `edge_fidelity`, `cc_*` live in `metrics.py`; alignment in `align.py`;
  colour maths in `color.py` (own Lab/ΔE2000, not skimage). Call them, never re-implement.
- **Determinism: ties → lowest index / smallest |slope| / smallest integer.** k-means `random_state=0`,
  fixed `n_init`; candidate seed `edit_seed + 1000·r + 10·k + t`; equal scores → lowest `(r, k)`.
- **Guard after every index-map change** (`guard.assert_no_motif_loss`), logged to `stages/guard.json`.
  No-op inside the gradient mask and for `distressed`.
- **Gradient-mask exemptions live in stage code, not class files** (no cleanup, no regularize, no
  adjacency penalty, dither only there).
- **Class files change by measurement, not by hand.** `design_mode: segment|edit` is written to
  `config/classes/*.yaml` only with an eval CSV reference in the commit. Profile default stays `quantize`.
- **Edit mode is the only resolution exception** and it stays inside `design`: backend works at
  ≤ 2 px/knot, output is upsampled back to `4W × 4H`. Reference for added/missing is the quantize
  baseline (`stages/05_design/baseline/ref_indexed.npy`), never raw coarse labels. The loop stops on the
  four `edit_*` thresholds only (`loop.violations`); speckle / jaggedness are feedback, not stop criteria.
- **QC model is small and fixed** (`qc.CORE_FEATURES`, priority order); `edit_*` values are hard gates,
  never model features. Any change to `agents/prompts.py` changes every agent hash and invalidates
  edit calibrations on purpose.
- **Licences:** local models/libs must be Apache/MIT/BSD/SAM. `ultralytics` (AGPL) is banned. API
  backends need `allow_api: true` in the loom profile and report `cost_usd` + `api_egress`.
- **Palette merging dedups first.** k-means splits one flat colour into fragments that look like motifs;
  `palette.dedup_groups` (ΔE < `DUP_DE`) runs before `motif_protected`, and protection only penalises a
  merge across different colours.

## Scale: measure in knots, not pixels

Real rugs are 300–900 knots a side; test rugs are ≈ 100 × 140. A 512 px analysis image is therefore
0.5 px/knot on a real rug and 4 px/knot on a test rug, and a designer's 1-knot dither reads as smooth
shading on one and as a hard checkerboard on the other. Every feature that feeds a threshold
(`edge_density`, `line_frac`, `thin_line_frac`, `noise_frac`, border thickness, the gradient-mask block
tests, `edge_width_knots`) is computed at grid resolution in knot units. Knot-scale texture (dither,
distressed speckle, pile ripple) is suppressed by working on the coarse label map (sampled at each knot's
centre 2 × 2 px) or on Lab smoothed at σ ≈ 2 knots, never by lowering a Canny threshold.

The outermost knot ring of `rectified.png` is the pile edge blended with the background; `rectify`
replaces it with its inner neighbour ring. Measurements that sit near the gradient-mask boundary ignore a
3-knot band around it.

## Synthetic data gotchas

- A 4-connected 1-knot diagonal is a two-level staircase (`|dy|+|dx| ∈ {s, s+1}`), so `is_thin` counts
  the generator's `ring` lattice and 2-wide chevron stripes as thin lines. `thin_lines=False` therefore
  widens both without consuming extra RNG (fixture seeds stay stable); `meta["thin_line_frac"]` is
  always the measured value on the final design.
- A gradient region can be painted over later (noise, symmetry); its endpoints are re-validated and the
  mask cleared where they no longer exist. Trust `meta` and `gradient_mask`, not the paint order.
- Edit-mode fixtures must be free of thin lines, otherwise routing (`edit_thin_lines`) sends the job to
  `quantize` before the mock backend ever runs.
- The `SHARP` render (0.6-knot blur at ≈ 2.8 px/knot, cubic upsample to 4 px/knot) measures ≈ 0.7–1.9
  knots of `edge_width_knots` depending on the design; `photo_px_warn` (1.5) sits inside that range,
  so `low_photo_resolution` on a sharp fixture is not a bug.
- Class-rule order is the spec's: an hv-symmetric lattice is `classic` (rule 5) before `geometric`
  (rule 6). `test_class_detection_pure_templates` tolerates one miss for that reason.

## Testing conventions

- Synthetic fixtures come from `data/synth/generate_designs.py` + `generate.py` (procedural, seeded);
  tests build small rugs and pass `corners_override` so SAM is never needed (`tests/conftest.py`).
- Edit-mode tests use `backend: mock` (`agents/backends/mock.py`, behaviour via `agent["mock"]`).
- Every hard gate / warning code in the spec has a test that triggers it. A threshold test asserts the
  spec's rule, not a number the fixture happened to produce.
- Model-dependent paths (`segment` inference, real backends) are tested through fallbacks and
  monkeypatched loaders on CPU; real inference sits behind `@pytest.mark.gpu`.

## Status (2026-09-15, uncommitted work in the tree)

Session baseline was 27 failing tests of ≈ 136; the last full run had 6. The final batch of edits
(adaptive Canny + p25 in `edge_width_px`, sigma-clipped illumination fit, broad-zone rule in the gradient
mask, first-peak border period, original-contour vertices and the arc-vs-polyline rule in `regularize`)
has **not** been verified by a full `pytest` run. Known open items:

- `tests/test_regularize.py`: `test_arc_fit_residual` and `test_jagged_diagonal_becomes_clean_stairs`
  broke with the last regularize change. Cause found: on a rasterised circle the arc rms (≈ 0.13) is not
  ≤ 0.5 × the DP-polyline rms (≈ 0.10–0.13), so no arc is ever accepted; use "arc rms ≤ polyline rms and
  sagitta > tol" (spec: equal residuals → line) instead of the 0.5 factor. Diamonds still stay diamonds
  because their polyline rms is ≈ 0. The diagonal test needs the same re-check after that.
- `tests/test_structure.py::test_thin_line_frac_and_edit_routing`: measured `thin_line_frac` (0.070) vs
  meta (0.097). The loss is diagonal lines that touch each other or a border motif in the coarse label
  map and merge into one non-thin component; not yet solved.
- `tests/test_structure.py::test_pile_blur_blocks_distressed` and the edit identity test
  (`test_identity_backend_passes_and_stops_first_round`) were fixed in the last batch but not re-run.
- Spec-listed tests still missing: `tests/test_design.py`, `tests/test_prepare.py`, `tests/test_tools.py`;
  `eval/synth_modern_ids.txt` is not generated (`generate_designs.py --n 100 --ids-file`, ≈ 2 s/design).
- Two test expectations were corrected to the spec's rule rather than the fixture number
  (`test_calibrate.py`: null-share feature drop and the precision-based `qc_accept`; `test_edit.py`:
  tiling only when the single pass is below `edit_px_per_knot_target`).
