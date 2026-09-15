# KnotVision v6.6

Customer carpet photo → indexed Wilton production design (Texcelle TIFF). Spec (Turkish, authoritative):
`knotvision-pipeline-v6.6.md`. When code and spec disagree, the spec wins; when the spec is silent, pick
the deterministic choice and note it in the module docstring. Code, comments and identifiers are English.

## Environment

- `.venv/bin/python` (3.14). Core deps only; torch / diffusers / sam3 are optional and imported lazily.
  The modelless chain and every test must run without them.
- OpenCV is 5.x (`opencv-contrib-python-headless`): `HoughLinesP` returns `(N, 4)` — always
  `np.asarray(lines).reshape(-1, 4)`; `cv2.remap` maps must be `float32`.
- Tests: `.venv/bin/pytest` (`-m "not gpu"` is implied on CPU; `gpu` and `slow` markers exist).
- Run dirs go under `runs/<job_id>/`; tests and experiments pass `runs_dir=` to keep the repo clean.

## Pipeline shape

`intake → rectify → structure → palette → design → production → regularize → dither → qc → export`
(`knotvision/run.py`). One stage = one `run(job)` in its module, one input hash → one output dir
`stages/<nn>_<name>/`. Working resolution is fixed: 4 px/knot, `rectified.png` is exactly `4W × 4H`;
index maps are `(H, W)` int32 at grid resolution.

Stage contract:
1. `key = job.cache_key(stage, inputs, extra=..., exclude_prefix="edit_" unless the stage reads edit params)`.
2. `if job.cached(sd, key) and <main output exists>: job.timings[stage] = 0.0; return job.state`.
3. Work, write outputs, `job.warn(...)` / `job.stop(...)`, append to `job.state["stages_ran"]`.
4. `job.mark_done(sd, key); job.save()`.

`job.warn(code)` is tagged with the current stage (`run_job` calls `begin_stage`); a cache hit restores
that stage's warnings from `stages/<nn>/warnings.json`, a rerun drops them first. Never write to
`job.warnings` directly. `job.stop(code)` raises `JobError` unless `--no-gate`, in which case it warns.

Parameter layers: loom profile ← class file ← agent file (`edit_*` keys only, edit mode only) ← job
override. `job.params` is the resolved set; `job.params["agent"]` holds the raw agent yaml. Loom-only
fields (`LOOM_ONLY_FIELDS` in `job.py`) may not appear in class files.

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
  baseline (`stages/05_design/baseline/ref_indexed.npy`), never raw coarse labels.
- **QC model is small and fixed** (`qc.CORE_FEATURES`, priority order); `edit_*` values are hard gates,
  never model features. Any change to `agents/prompts.py` changes every agent hash and invalidates
  edit calibrations on purpose.
- **Licences:** local models/libs must be Apache/MIT/BSD/SAM. `ultralytics` (AGPL) is banned. API
  backends need `allow_api: true` in the loom profile and report `cost_usd` + `api_egress`.

## Testing conventions

- Synthetic fixtures come from `data/synth/generate_designs.py` + `generate.py` (procedural, seeded);
  tests build small rugs (≈100×140 knots) and pass `corners_override` so SAM is never needed.
- Edit-mode tests use `backend: mock` (`agents/backends/mock.py`, behaviour via `agent["mock"]`).
- Every hard gate / warning code in the spec has a test that triggers it.
