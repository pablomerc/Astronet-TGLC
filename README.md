# TGLC_adaptation

Adapting the Astronet **vetting** model (`/pdo/users/pablomer/Astronet-Triage`) from QLP to
**TGLC** photometry. Start with **[NOTES.md](NOTES.md)** (the running orientation doc) and the
**[knowledge/](knowledge/)** base for the load-bearing facts.

## What's here

### Reprocessed training data (QLP+TGLC through S103) — the current focus
Production stitches light curves from the **`lightcurvedb` DB**, and TGLC (production photometry
since S94) is ingested into that same DB — so one `read_lightcurve(tic)` returns a stitched
legacy-QLP(≤S93)+TGLC(≥S94) baseline. We use this to regenerate the reobserved subset of the
training set through S103. (Details: [knowledge/PRODUCTION_LC_PATH.md](knowledge/PRODUCTION_LC_PATH.md).)

- **`make_reobserved_tic_list.py`** — selects training targets re-observed with a light curve since
  S94 (`reobs103_lc`, ~5,781 TICs) and writes:
  - `data/reobserved_s103_tics.txt` — the input list (one TIC/line)
  - `data/reobserved_s103_companion.csv` — the labeled dec2025 TCE rows for those TICs, with `File`
    rewritten to the new FITS names (drop-in for the later TFRecord step)
  ```bash
  python3 make_reobserved_tic_list.py            # needs only pandas/numpy
  ```

- **`db_to_fits.py`** — reads each TIC from the DB, cuts at `--sector`, and writes a FITS in the
  existing Astronet training schema (SAP_FLUX=primary; SML/MID/LAG=small/primary/large; rel-flux ~1,
  transits dip; QUALITY = CCD-wide|TGLC). Prints a live progress bar + ETA. Must run in the QLP
  operator venv with a **clean PYTHONPATH** (so the venv's qlp 0.14.x wins over the old /pdo/app one):
  ```bash
  PYTHONPATH= /sw/qlp-environment/.venv/bin/python db_to_fits.py \
      -i data/reobserved_s103_tics.txt \
      -o /pdo/users/pablomer/mnt/tess/reprocessed_s103_qlptglc_fits_files \
      -s 103 -n 8
  # writes ~5,781 FITS + a _run_summary.csv (per-TIC status/provenance) to the output dir
  ```
  Output FITS live at `/pdo/users/pablomer/mnt/tess/reprocessed_s103_qlptglc_fits_files/`
  (outside this repo; prefix `astronet_hlsp_qlptglc_…`, so they never collide with existing sets).

- **`inspect_reprocessed.ipynb`** — QA: schema check, full stitched curves with the TGLC (S94+)
  portion highlighted, distinct aperture channels, phase-folds, and NEW-vs-old-QLP comparison.

#### Before/after-TGLC inspection (does adding TGLC change the model input?)
Splits each reprocessed FITS at the **S94 cadence boundary** into *before* (QLP only, drop sectors
≥94) and *after* (+TGLC), runs the **real Astronet vetting preprocessing** on both, and compares the
full light curve + local/global views. Runs in your astronet dev env (e.g. conda
`daniel_env_cloned_v2`) — reads only the FITS files, no DB.
- **`before_after_core.py`** — shared logic (FITS read, S94 split, view-building, figure builder).
- **`inspect_before_after.ipynb`** — interactive: ~10 examples/class, figures inline. Open in your
  astronet kernel and run `show_class('p'/'e'/'j')`.
- **`diagnose_before_after.py`** — headless: renders the same figures and posts them to a Discord
  webhook (batched). `--dry-run` saves PNGs to `figures/before_after/` without posting.
  ```bash
  PY=/pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python3
  $PY diagnose_before_after.py --dry-run            # render + save locally
  $PY diagnose_before_after.py --n-per-class 10     # render + post to Discord
  ```
  Webhook is read from `--webhook`, `$DISCORD_WEBHOOK_URL`, or a git-ignored `.discord_webhook`
  file (keep it out of git — it's a secret). Note: rendering uses the real spline detrend on full
  multi-sector curves (~30–60 s per target), so a full 10/class run takes ~15–30 min. Runs cache
  computed arrays under `<outdir>/_cache/` (git-ignored); re-style instantly with `--replot`, or
  post an already-rendered folder to Discord instantly with `--post-existing <dir>`.

  **Rendered galleries (each has a `README.md` that renders on GitHub):**
  - `figures/before_after/` — 3-panel (full LC · local view · global view)
  - `figures/before_after_folded/` — 2×3, adds the **folded detrended scatter** (before over after)
    and a **repeated/old vs new-TGLC** provenance panel

> Not done here (on purpose): extended-mission cadence rebinning and TFRecord generation. The
> `cadences>40000 → 30 min` downsample belongs to the TFRecord stage (`generate_input_records_3.py`
> / the FFITools getter), run later in your own env. We keep all cadences + a faithful `CADENCENO`.
> Superseded: `h5_to_fits_hybrid.py` (a file-glob mix-and-match prototype) — the DB does the QLP/TGLC
> merge, so it was removed (see git history / NOTES §0.5).

### Reobservation & training-set analysis
- **`observed_sectors.py`** + **[OBSERVED_SECTORS_README.md](OBSERVED_SECTORS_README.md)** — TIC →
  observed sectors via tess-point; produced `data/training_reobserved_since94.csv`.
- **`make_training_set_distributions.py`** — training-set composition figures + `data/training_set_enriched.csv`.
- `data/training_reobserved_s103.csv` — per-TIC reobservation flags through S103 (`reobs103_lc`,
  `reobs103_geom`, …); the selection source for the reprocessed set.

### Knowledge base ([knowledge/](knowledge/))
- `PRODUCTION_LC_PATH.md` — how production reads/stitches LCs from the DB; QLP vs TGLC provenance.
- `DATA_LOCATIONS.md` — DB coverage (S1–104), filesystem paths, reobservation counts, deployment state.
- `FITS_SCHEMA_AND_PREPROCESSING.md` — FITS schema, tess_io / generate_input_records_3, aperture channels.

### Strategy / status
- `NOTES.md` (start here), `TGLC_strategy_memo.md` / `.pdf`, `tglc-qlp-sector-status.mdc`.

## Environment note
Anything that touches the DB (`db_to_fits.py`) needs the QLP operator stack (qlp 0.14.x +
lightcurvedb 3.0.0 + pyticdb; DB creds at `~/.config/lightcurvedb/db.conf`). Use
`PYTHONPATH= /sw/qlp-environment/.venv/bin/python`. The list/analysis/inspection scripts need only
pandas/numpy/astropy/matplotlib and run in any env.
