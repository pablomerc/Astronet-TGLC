# TGLC Adaptation — Working Notes

> Running reference for adapting the Astronet **vetting** model
> (`/pdo/users/pablomer/Astronet-Triage`) from QLP light curves to **TGLC**
> (TESS–Gaia Light Curves). All adaptation work lives in
> `/pdo/users/pablomer/TGLC_adaptation`.
>
> Created 2026-06-25. Keep this updated — it is the orientation doc for this effort.

---

## 0. TL;DR

- We want the production vetting ensemble (`pablomer_final`, 10 models, Mar-2026) to work
  on **TGLC** photometry instead of (or in addition to) **QLP**.
- The governing law of this codebase is **"train/inference preprocessing MUST match"**
  (DEPLOYMENT.md gotcha #1 — when it didn't, 24–31% of top-class predictions flipped).
  QLP→TGLC is a *much* bigger change than the binning change that caused those flips, so
  **shipping the QLP-trained model on TGLC unchanged is not an option.**
- We **cannot** build a fully-TGLC training set right now: the QLP-pipeline TGLC
  reprocessing (S56–S93) is **blocked** on QLP I/O, and the only TGLC we have today
  (Te Han's S1–S56) **lacks QLP quality flags + QLP detrending**.
- **Recommended path:** Option 1 in spirit (make training match production), executed as
  Option 2's mechanism (warm-start fine-tune of the existing ensemble) on a *representative*
  mixed-source set — **not** a from-scratch retrain and **not** a few-example fine-tune.
  Details + reasoning in `TGLC_strategy_memo.pdf` (and `TGLC_strategy_memo.md`).

---

## 0.5 MAJOR UPDATE (2026-07-10): production reads TGLC from the DB → reprocessed set built

Two findings this session change the plan and **supersede parts of §3 and §7**. Full detail
in the new knowledge base: [`knowledge/PRODUCTION_LC_PATH.md`](knowledge/PRODUCTION_LC_PATH.md),
[`knowledge/DATA_LOCATIONS.md`](knowledge/DATA_LOCATIONS.md),
[`knowledge/FITS_SCHEMA_AND_PREPROCESSING.md`](knowledge/FITS_SCHEMA_AND_PREPROCESSING.md).

1. **Production stitches from the `lightcurvedb` DB, not from `/pdo/qlp-data/*.h5`.**
   `qlp estools astronet` (default `--source db`) → `default_io_backend.read_lightcurve(tic)`
   → `_build_lc_from_obs_data_list` concatenates per-observation baselines (orbit-ordered).
2. **TGLC (production photometry since S94) is ingested into the SAME DB tables as legacy QLP.**
   So `read_lightcurve(tic)` returns ONE stitched baseline: legacy-QLP for sectors ≤93 + TGLC
   for ≥94, automatically (no per-sector routing). Verified live on TIC 592638 (S5+S32 QLP,
   S98 TGLC). DB covers sectors 1–104. Both pipelines expose 3 apertures (small/primary/large);
   production uses `primary`.

**Consequence:** the file-glob mix-and-match idea (`h5_to_fits_hybrid.py`, §7, and its D1–D4
TGLC-file decisions in §3) is **obsolete** — the DB already does the QLP/TGLC merge. New tool:
- **`db_to_fits.py`** — reads `read_lightcurve(tic)`, cuts at `--sector`, writes FITS in the
  existing training schema (SAP_FLUX=primary, SML/MID/LAG=small/primary/large; rel-flux ~1,
  transits dip; QUALITY = CCD-wide|TGLC). Run with the QLP operator venv + a clean PYTHONPATH:
  `PYTHONPATH= /sw/qlp-environment/.venv/bin/python db_to_fits.py -i data/reobserved_s103_tics.txt
  -o /pdo/users/pablomer/mnt/tess/reprocessed_s103_qlptglc_fits_files -s 103 -n 8`.
- **`make_reobserved_tic_list.py`** — builds the input list + companion CSV.

**What we produced (this session):** the reprocessed **reobserved subset** — training targets with
an actual light curve in S94–S103 (`reobs103_lc` in `data/training_reobserved_s103.csv`):
**5,781 unique TICs** (companion 5,834 TCEs: Junk 2,267 · Planet 1,885 · EB 1,682), each rebuilt
through **S103** (QLP≤S93 + TGLC≥S94) →
`/pdo/users/pablomer/mnt/tess/reprocessed_s103_qlptglc_fits_files/`. This sits alongside the
existing 15k set as raw material for retraining/fine-tuning. **Scope of this task = DB→FITS light
curves only.** TFRecord generation (`generate_input_records_3.py`) is a later step in Pablo's env.

**Extended-mission rebinning is deferred:** the `cadences>40000 → 30 min` downsample happens in the
TFRecord-stage getter (`helpers._preprocess_lc_arrays`), not in the LC content. `db_to_fits.py`
keeps all cadences + a faithful `CADENCENO` so it's recoverable later.

**Deployment reality (verified):** `/sw/astronet` → `astronet-3.0.1`; the v3.1.0 cutover has NOT
happened. Live vetting model is still `AstroCNNModelVetting_cshallue_…` (single model), not the
`pablomer_final` ensemble. So "match production" for the new ensemble = match the code it was
trained with (`generate_input_records_3.py`), which becomes production at cutover.

> §3's TGLC-file inventory and §7's `h5_to_fits_hybrid.py` prototype are kept below for history but
> are **superseded** by the DB approach above.

---

## 1. What we're adapting (the current model in one screen)

- **Model:** `AstroCNNModel` (1-D CNN), 10-member ensemble `pablomer_final`, trained Mar-2026.
  - Live model dir: `/pdo/astronet-data/models/vetting/experimental/pablomer/march2026/20260305/pablomer_final-final-3k-ensemble/`
  - 4-class softmax `[disp_p, disp_e, disp_n, disp_j]`; planet prob = `pred[:,0]`.
- **Input per TCE:** a *bag of phase-folded views* (global / local / secondary / sample-segment,
  multiple backbone spacings + odd/even/half/double-period) + ~23 aux scalars.
- **One TCE = ONE light-curve FILE — but for most TCEs that file is a *multi-sector stitched* LC.**
  `read_and_process_light_curve()` opens exactly one FITS and folds it. Critically (verified
  2026-06-25), for **~76% of the training set that one FITS is a multi-sector STITCHED QLP light
  curve** (built by the h5→FITS converter; median span ~760 d ≈ many sectors); only ~24% are
  genuine single-sector files. **So the model is already trained mostly on stitched multi-sector
  QLP.** The "combine QLP-old + TGLC-new into one LC then fold" production plan is therefore the
  *same* stitched paradigm we already use — just swapping the per-sector photometry source. It is
  **not** a new pipeline capability. (See §4.)
- **Preprocessing:** `astronet/preprocess/generate_input_records_3.py` (cadence-aware binning +
  scatter weighting = "cad_scat"). This is the canonical `generate_input_records.py` that QLP
  imports in production.
- **Training driver:** `astronet/ensemble_train_vetting_2025_final.sh`; config
  `configurations_vetting.py::pablomer_final`. Vetting warm-starts from a frozen triage backbone.

## 2. The change: QLP → TGLC

- **QLP** = aperture photometry on FFI cutouts + QLP detrending + QLP quality flags. What the
  current model was trained on.
- **TGLC** = effective-PSF / forward-model photometry, Gaia-informed **decontamination** of
  blends. Better for faint / crowded fields. **Different** noise floor, systematics, and —
  critically for vetting — **different apparent transit depths and blended-EB signatures**
  (decontamination changes/erases blends). This directly touches the planet-vs-EB boundary.
- **Production LC plan (as described):** hybrid — keep already-preprocessed **QLP for old
  sectors**, add **TGLC for new sectors**, stitch, then fold. (Requires the multi-sector
  stitching capability noted above.)

## 3. Hard constraints (from the 2026-06 discussion — verbatim gist)

1. Te Han has processed the **main mission + EM1 (≤ S56)** as TGLC, **but without QLP quality
   flags and without QLP detrending.** → "raw" TGLC, not pipeline-consistent.
2. The **S56–S93 reprocessing through QLP** (TGLC fluxes + QLP flags/detrend) is **underway but
   blocked**: cutouts are done; **PSF and beyond are stuck on QLP being overloaded on I/O.**
3. Next steps under discussion:
   - **(a)** Pablo + David: determine whether the **S1–56 TGLC** data is usable and compatible
     with Astronet's expectations.
   - **(b)** Alternative: only use **the most recent sector(s)** as TGLC for retraining — i.e.
     in the training set make only the last 1–2 sectors TGLC.
4. Either path may be **blocked on QLP**; need Willie to confirm status/estimates. Meanwhile,
   still try to use the ≤ S56 data.

**Implication:** a uniformly QLP-pipeline-consistent TGLC dataset across all sectors does not
exist yet. Today's realistically-available TGLC = Te's raw ≤ S56 (no QLP flags/detrend).

**TGLC data inventory (located 2026-06-26 — Te Han = user `tehan`):**
- **New reprocessed products (have flags!):** `/pdo/users/tehan/tglc-gpu-production/hlsp_s%04d/`
  for **S56–S63** (the S56–S93 effort, partially done). TIC-bucketed dir tree. There are several
  variants per sector (`hlsp_s0056`, `…_twirl`, `…_fs_v1`, `…_fs_v2`, `…_v4`) — **canonical one TBD.**
- **v2.1 reference set:** `/pdo/users/tehan/tglc_v2.1_s56cam4/sector0056/lc/<cam>-<ccd>/`.
- **Format:** HLSP FITS `hlsp_tglc_tess_ffi_gaiaid-<GAIADR3>-s%04d-cam<C>-ccd<D>_tess_v2.1_llc.fits`.
  **Files are keyed by Gaia DR3 id, NOT TIC** (header carries both `TICID` + `GAIADR3`) → a hybrid
  stitcher needs a TIC→Gaia crossmatch (pyticdb) to find the file.
- **HDU1 columns:** `time` (BTJD = BJD-2457000, same system as QLP), `psf_flux`, `aperture_flux`,
  `cal_psf_flux` (detrended, ~1-normalized, **can be negative**), `cal_aper_flux`, `background`,
  `cadence_num`, **`TESS_flags`, `TGLC_flags`** (quality bitmasks), `aperture_flux_raw`.
- ⚠️ **Correction to the "raw TGLC has no QLP flags" caveat:** these NEW products DO carry quality
  flags; the no-flags concern applies to Te's *older main-mission* TGLC. So the new S56+ set is the
  more pipeline-friendly one (but is exactly the sector range gated by the QLP I/O blockage).
- **Open decisions for the stitcher** (in `h5_to_fits_hybrid.py`): D1 canonical variant dir;
  D2 flux column (`cal_psf_flux` vs `psf_flux`/aperture) + negative-flux handling (pseudo-mag clips
  to a faint floor, or refactor to carry flux directly); D3 pyticdb Gaia column name; D4 new-S56+
  vs old-main-mission set (Phase 0 decides).

## 4. Current training-set composition (the dec2025 labeled set)

Source CSV: `/pdo/users/pablomer/mnt/tess/astronet/tces-vetting-v01-tois-triageJs-nocentroid-dec2025-all.csv`
(this is the set behind the production model; `-train/-val/-test` are the splits).

- **N = 15,471 TCEs.** Labels (`first_letter`): **Planet 5,823 · EB 3,390 · Junk 6,258.**
- **Two kinds of light curve (verified by the `File` prefix + LC time-span):**
  - **`mk_*` → 3,768 (24%) genuine SINGLE-sector** QLP LCs (median span ~27 d), tagged with their
    **true** observation sector **S3–S34**; camera/CCD resolves 100% via tess-point.
  - **`astronet_*` → 11,703 (76%) MULTI-sector STITCHED** QLP LCs (median span ~760 d). The
    **`s0064`/`s0085` in the filename is a BATCH/PROCESSING tag, NOT a per-target observation
    sector** — these are the two bulk-TOI additions. A stitched LC spans several cam/CCDs, so a
    single camera/CCD is **undefined** for them.
  - **Precise meaning of the tag (verified 2026-06-25 against real sector boundary times):** the
    tag = the `-s` ("last sector to search") cutoff passed to the h5→FITS converter
    (`Astronet-Triage/h5_to_fits.py:57`). The converter (`:88-95`, `for sector in range(1,
    last_sector+1)`) pulls **every QLP sector the target was actually observed in, from S1 up to
    that cutoff**, and stitches them. Empirically: s64-batch targets reach max observed sector
    **≤ 63**; s85-batch targets reach **≤ 86** (the +1 is an edge effect / `-s 86`); span counts
    range from 1 to ~37 sectors per target. **So Pablo's reading is right:** a "P, s64" example
    means the team labeled the cumulative QLP curve up to the s64 cutoff; the target may only have
    been observed in some lower sectors. This is exactly what makes mix-and-match valid: rebuild
    that same span as **QLP(older sectors) + TGLC(recent sectors)** and the label still describes
    it — *provided* the TGLC swap doesn't change the class-defining signal (EB-depth caveat → Phase 0).
- ⚠️ **Consequence:** the "sector distribution" over the `File` tag is really *true sectors for
  the 24% single-sector set + two batch tags for the 76% stitched set*. The S64 bar (≈all Planet,
  4,257 P) and S85 bar (≈all Junk, 6,053 J) are **processing batches**, not sky-sectors.
- Figures (generated by `make_training_set_distributions.py`):
  - `figures/training_set_composition.png` — single-vs-stitched + File-tag-sector bars (colored).
  - `figures/training_set_camccd_singlesector.png` — camera / CCD / cam×ccd on the **single-sector
    subset only** (N=3,768; stitched LCs have no single cam/CCD).
  - `figures/training_set_sector_per_label.png` — P/EB/J overlaid vs File-tag sector (`*_log.png`).
  - Per-row enriched table: `data/training_set_enriched.csv` (adds `sector`, `cam`, `ccd`, `label`, `kind`).
- Method: sector from the QLP filename (`s%04d`); camera/CCD via `tess-point` (`tess_stars2px`,
  pointings through ~S96) on single-sector files only.

**Why this matters for the adaptation:**
1. The model **already trains predominantly (76%) on multi-sector stitched QLP**, so the production
   "stitch QLP-old + TGLC-new, then fold" plan is *consistent with how the model was built* — the
   adaptation is **swap the per-sector photometry source inside the existing stitch**, not a new
   architecture or a new stitching step.
2. Because each stitched example already mixes many sectors, "make recent sectors TGLC" becomes a
   **per-segment source mix *within* each example** — which *naturally avoids* a whole-example
   "source = label" shortcut (each example is a QLP+TGLC blend). Good.
3. The residual confound to watch is at the **batch** level (the s64 batch ≈ Planet, the s85 batch
   ≈ Junk were processed separately); keep that in mind when deciding which segments get TGLC.

## 5. The two options posed, and the recommendation

- **Option 1 — Mimic the new preprocessing in the training set, then retrain.** Correct in
  principle (it's the only thing that respects "train==inference"). Catch: a faithful mimic of the
  future hybrid is not buildable today (QLP-TGLC reprocessing blocked).
- **Option 2 — Take the trained model, fine-tune on a few recent TGLC examples, done.** As stated,
  the weakest: a handful of examples can't close a whole-distribution photometric shift, risks
  catastrophic forgetting, and doesn't match a *hybrid* production input. BUT warm-start
  fine-tuning is the right *mechanism* — the problem is "a few examples," not "fine-tune."

**Recommendation (synthesis):** Option 1's *goal* (training distribution = production distribution)
via Option 2's *mechanism* (warm-start continue-training the existing ensemble), scoped to what's
available, in three phases:

1. **Phase 0 — Characterize the gap first (cheap, do now).** On targets that have BOTH QLP and
   TGLC in overlapping sectors, run the current model on each and measure prediction drift
   (Δdisp_p, top-class flip rate) per class / magnitude / crowding, plus view-level diffs. Reuse
   the s96/s97 comparison methodology (`/pdo/users/pablomer/vetting_comparison_s9*`) and
   `rank_view_differences.py`. This answers constraint 3a (is raw ≤S56 TGLC usable?).
2. **Phase 1 — Build training that matches production, *inside the existing stitch*.** Because
   training is already 76% multi-sector stitched QLP, the move is to modify the stitcher/converter
   so chosen (recent) sectors are pulled from TGLC instead of QLP, run through the **same**
   `generate_input_records_3.py`; keep QLP for sectors where TGLC is unavailable; ensure TGLC gets
   QLP-equivalent flagging/detrending before preprocessing. **Do NOT switch to single-sector** —
   that would *break* train==inference (production stays stitched).
3. **Phase 2 — Warm-start fine-tune the ensemble** (low LR, keep QLP examples to avoid
   forgetting), then **recalibrate the decision threshold** (every source/preprocess change moves
   the operating point — s97 already pushed it from 0.75 toward ~0.9).

Full reasoning, risks, and decision gates: **`TGLC_strategy_memo.pdf`**.

## 6. Decision gates / open questions

- **G1. Is raw ≤S56 TGLC usable without QLP flags/detrend?** → Phase-0 drift analysis. If not,
  we are gated on the QLP-pipeline TGLC (blocked) or must add our own flag/detrend step.
- **G2. Which sectors get TGLC inside the stitch?** Training is *already* stitched multi-sector,
  so production should stay stitched and we swap per-sector source within it. The open choice is
  *which* segments are TGLC (recent only? everything ≤S56 too?) — driven by G1 + G4.
- **G3. How big is the TGLC→QLP shift (Phase 0)?** Small → light fine-tune. Large (esp. EB
  depths) → broader TGLC coverage + per-class attention + threshold recalibration.
- **G4. QLP reprocessing ETA?** Blocking constraint — confirm with Willie/Glen.

## 7. Continual retraining & mix-and-match — feasibility + v1 design (added 2026-06-25)

> Answers Pablo's "retrain after every sector with TEV labels + new-pipeline LCs" and "custom
> script that stitches old + TGLC data into a new training set" questions. Backed by a 6-agent
> codebase sweep + empirical verification (workflow `tglc-continual-retrain-feasibility`).

**Mix-and-match stitcher (Q3): feasible, ~moderate (~a few hundred lines).** The converter
`Astronet-Triage/h5_to_fits.py` is the place to do it:
- `:57` `--sector/-s` = "last sector to search" (the cutoff). `:88-95` `get_orbit_lc_locations`
  loops sectors 1..cutoff and resolves each to a QLP h5 (`:93`
  `/pdo/qlp-data/orbit-{orbit}/ffi/cam{cam}/ccd{ccd}/LC/{tic}.h5`). **Inject per-sector source
  routing at `:90`** (`if sector >= TGLC_CUTOFF: load TGLC else QLP`).
- `:99-144` `merge_lcs` is QLP-specific: reads QLP `qflag` (`:108`), `HDFLightCurve` (`:113`),
  normalizes 5 apertures by magnitude median (`:130-133`). A TGLC branch must replace these with
  TGLC's format + flux normalization, and **synthesize a quality flag** (raw TGLC has none →
  `light_curve_util/tess_io.py:92-94` filters on a `QUALITY`/flag column and silently mis-handles
  data without it).
- Downstream needs NO change: `generate_input_records_3.py:356-362 get_lightcurve` is one-TCE→
  one-FITS, source-agnostic; labels read at `:279-291`. (Add an optional `source_mix` CSV column
  for provenance.)
- **Validity rule:** the human label was assigned on a QLP cumulative curve. A hybrid example is
  only label-valid if the TGLC swap wouldn't flip the disposition → gate it on **Phase 0** drift
  (run current ensemble on QLP-only vs hybrid of the same targets; exclude/re-vet class-flippers).

**Prototype (SUPERSEDED 2026-07-10 — see §0.5; removed in favor of `db_to_fits.py`):**
`TGLC_adaptation/h5_to_fits_hybrid.py` — a faithful, syntax-checked (but UNTESTED)
clone of `h5_to_fits.py` with the per-sector source decision added (`decide_source()`,
`--tglc-cutoff` / `--tglc-sectors`). QLP path is byte-identical (and with no TGLC flag it
reproduces the original output, so it's a safe drop-in). The TGLC loader
(`load_tglc_sector_segment`) is a documented **stub** with three `TODO[TGLC]` items to fill once
Te Han's products are located: (1) path/format, (2) flux→pseudo-mag normalization (median→Tmag,
replicate into the 5 aperture channels), (3) **quality-flag synthesis** (raw TGLC has none; the
empty flag would let momentum-dumps fake transits via `tess_io.py:92-94`). Must run in the
QLP/FFITools env (needs `qlp`, `pyticdb`, `lightcurvedb`), not the astronet conda env.

**Per-sector continual loop (Q1):** mechanics mostly EXIST; the *continual* part is gated by
external deps, not the repo.

| # | step | status | difficulty | gate |
|---|------|--------|-----------|------|
| 1 | acquire LCs (QLP / TGLC) | EXISTS / **BLOCKED** | blocked-external | QLP works; TGLC S1–56 raw (no flags), S56+ stuck on QLP I/O |
| 2 | TEV labels for new sector | **BUILD** | moderate→hard | no TEV API today (manual Google-Sheets flow); schema + **label lag (weeks?)** unknown |
| 3 | stitch QLP+TGLC hybrid FITS | **BUILD** | moderate | `h5_to_fits.py:90`; TGLC format + flux norm + flag synthesis |
| 4 | TFRecords | EXISTS | easy | `gen_tfrecords_vetting_2025_sectors.sh` |
| 5 | warm-start fine-tune | EXISTS (mech) / BUILD (wrapper) | moderate | `train.py:154-189` warm-start; need low-LR/short/mix wrapper; **cost unmeasured** |
| 6 | validate + recalibrate threshold | EXISTS (method) / BUILD (auto) | moderate | `vetting_comparison_s97/`; sign-off (Glen/Katharine) |
| 7 | deploy | EXISTS (steps) / BLOCKED | blocked-external | PR-protected `main`; swadm `/sw` flip; threshold `0.75` hard-coded in external FFITools `estools.py` |

**"Is it easy?"** The *machinery* (stitch→tfrecords→warm-start→eval) is easy and largely exists.
The *continual* loop is **not** easy because of external gates: TEV label availability/lag, the
swadm-owned `/sw` cutover, PR-protected `main`, the FFITools-owned threshold, and the blocked
QLP→TGLC reprocessing. **v1 (semi-automated, one-command-per-sector with human gates): ~1–2 weeks
of build once the TEV + TGLC unknowns are resolved.** **v2 (fully autonomous): not realistic
near-term** (label lag + human-only deploy gates).

**Continual-specific risks:** catastrophic forgetting (keep ≥50% old examples in each fine-tune);
**data leakage — split by TIC, not row** (stitched LCs recur across sectors); threshold drift
(anchor recal to a *fixed* hold-out, e.g. s96/s97); train/inference mismatch (keep production
QLP-only until TGLC is in the serving path too — DEPLOYMENT gotcha #1); silent regression (abort
retrain if disp_p drops >~5% on the fixed hold-out); TEV label noise (QC vs existing sheet).

**Recommended v1:** a single `continual_retrain_sector.sh <N>` driver chaining steps 4→6 (the
EXISTS spine) with a warm-start fine-tune wrapper (low LR, ≥50% old-example mix, split-by-TIC) and
a hard validation gate, **stopping at every human gate (PR / `/sw` / threshold) with a checklist**.
Keep inference QLP-only. The TGLC hybrid stitcher and TEV client are follow-ons after Phase 0.

**3 things to nail down first:** (1) **TEV label availability + schema + lag** — decides whether
"after every sector" is even possible vs "periodic post-hoc retrain"; (2) **TGLC reality (Phase 0 /
gate G1)** — Te Han's S1–56 location/format/flux-units/flags + S56+ ETA from Willie/Glen; (3)
**threshold ownership + `/sw` cutover contract** — can `0.75` become a runtime config, who signs off.

## 8. Status / log

- **2026-06-25** — Effort kicked off. Generated training-set composition + cam/ccd + sector-per-label
  distributions. **Discovered the training set is 76% multi-sector *stitched* QLP LCs** (the
  `astronet_*` s64/s85 batches), not single-sector — corrected this note + the memo accordingly.
  Wrote this note + the strategy memo. No model/data changes yet.
- **2026-06-25 (later)** — Ran a 6-agent feasibility sweep on a per-sector continual-retrain loop +
  mix-and-match stitcher (§7). Verified the `sNN` tag = `-s` cutoff and the stitched LC = all
  observed QLP sectors up to it (Pablo's interpretation confirmed).
- **2026-06-26** — Scaffolded `h5_to_fits_hybrid.py` (per-sector QLP/TGLC source routing; QLP path
  faithful, TGLC loader stubbed with 3 TODOs). Next: locate TGLC products → fill the stub → run
  Phase 0 drift on QLP-only vs hybrid for overlap targets.
- **2026-07-09** — Confirmed production facts (also in `.cursor/rules/tglc-qlp-sector-status.mdc`):
  **TGLC introduced in S94** (no longer approximate). QLP sector vetting has been sporadic
  (report issues): **S95–S97 fully vetted; S89–S94 not.** Faint-star-search TOIs released for
  S89–S93, but only a few brighter than Tmag 10.5. Prefer S95–S97 for Phase 0 / eval pools.
- **2026-07-16** — **EB period revision pipeline (motivated by a real failure).** Discovered via the
  before/after diagnostics that TIC 364302118's training period (0.6728 d, from its single-sector S12
  file) is the aliased **half-period** — full-baseline BLS gives 1.34534 d and the stale period folds
  to garbage on the reprocessed curve. Built the 4-step revision pipeline for the EB subset
  (see README "EB period revision"): master table `data/reobserved_s103_revised.csv`
  (`Per/Epoc` overwritten when `revised=YES`, originals kept in `Per_original/Epoc_original`,
  provenance in `revision_source`). Results: **catalog crossmatch auto-revised 265/1,682** EB TCEs
  (VSX 160 · Villanova 59 · Gaia 4 · TOI 42; 71 true 2×/0.5× changes — incl. TIC 364302118 via
  Villanova at 1.3453456 d, matching our independent BLS); **risk scan of the remaining 1,417**:
  194 half-period suspects · 780 stale · 92 weak · 351 ok (**~69% suspicious** — the single-sector
  EB periods really don't survive the 7-yr baseline). Manual flag gallery at
  `figures/period_gallery/index.html` (risk-sorted candidate folds; export CSV →
  `apply_revisions.py`). Notable: ALL 1,417 unrevised EBs are single-sector `mk_` originals —
  consistent with §4 (the stitched s64/s85 batches were ≈all Planet/Junk). ⚠ Downstream rule: build
  TFRecords for EBs from the **revised** master, never the raw companion.
- **2026-07-10** — **Big one (see §0.5).** Established (code + live DB) that production stitches
  LCs from `lightcurvedb` and that **TGLC S94+ lives in that same DB**, so `read_lightcurve(tic)`
  returns a stitched QLP+TGLC baseline automatically. Wrote the `knowledge/` base (3 MD files).
  Built **`db_to_fits.py`** (DB→FITS, production-faithful, progress/ETA) + `make_reobserved_tic_list.py`.
  Generated the **reprocessed reobserved subset** (5,781 TICs, through S103) →
  `mnt/tess/reprocessed_s103_qlptglc_fits_files/`. Retired `h5_to_fits_hybrid.py` (file-glob
  approach obsolete). Confirmed `/sw/astronet`=3.0.1 (ensemble not yet deployed). Next: TFRecord
  gen from the new FITS in Pablo's env + Phase 0 drift.

## 9. Pointers

- Current-model orientation: `Astronet-Triage/CLAUDE.md`, `README_May2026.md`.
- Production / QLP integration + gotchas: `Astronet-Triage/DEPLOYMENT.md`.
- Legacy + recent data-prep + training pipeline: `Astronet_processing_training_steps-notion.md`
  (the `.txt` of the same name is the stale 2024 version).
- Comparison tooling to reuse for Phase 0: `/pdo/users/pablomer/vetting_comparison_s96cam1/`,
  `/pdo/users/pablomer/vetting_comparison_s97/`.
- Collaborators: **Glen Petitpas** (QLP + lightcurve DB), **Willie/swadm** (`/sw` cutover),
  **David** (S1–56 usability), **Te Han** (TGLC).
