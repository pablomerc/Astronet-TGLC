# Observed-sectors lookup (TIC → which TESS sectors) — context for a fresh session

> Goal: given a TIC ID (or a list of ~15k), find which TESS sectors it was observed in — e.g. to
> ask "how much of our training set has been re-observed since TGLC went into production (S94?)".
> Created 2026-06-26. Nothing in the existing data was overwritten; all files here are new.

## TL;DR — yes, we can do this, and the tool is ready

- `tess-point` (`tess_stars2px`, installed in `daniel_env_cloned_v2`) knows the TESS pointing law
  through **~S96**. Given a target's RA/Dec it returns **every sector it falls on a CCD**, plus
  camera/CCD. TESS pointing is deterministic, so this is effectively "was it observed?" — it does
  **not** tell you whether QLP/TGLC actually *produced* a light curve (separate data-access question).
- **Tool:** `TGLC_adaptation/observed_sectors.py`. **Run with**
  `/pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python3`. No pip installs.

```bash
PY=/pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python3
cd /pdo/users/pablomer/TGLC_adaptation

# one TIC
$PY observed_sectors.py --tic 261136679 --since 94
# the whole training set (offline, ~2-3 min), summarize + write CSV:
$PY observed_sectors.py --from-training --since 94 --out data/training_reobserved_since94.csv
# an arbitrary list (uses MAST for TICs not in the training set):
$PY observed_sectors.py --tic-file mylist.txt --since 94 --out out.csv
# direct coordinates, no TIC resolution:
$PY observed_sectors.py --radec 84.291 -80.469 --since 94
```

Output columns: `tic, ra, dec, n_sectors, min_sector, max_sector, observed_sectors,
sectors_since_<N>, observed_since_<N>, resolver`.

## The TIC-ID question (this came up — resolved)

There was a worry that "the training-set CSV doesn't include TIC ID." **It does.** Verified on
`/pdo/users/pablomer/mnt/tess/astronet/tces-vetting-v01-tois-triageJs-nocentroid-dec2025-train.csv`:
the `TIC ID` column is **100% populated** and **100% consistent** with the TIC embedded in the
`File` name (`...-sNNNN-<16-digit TIC>_...`), 0 mismatches.

- Where it comes from: `Astronet-Triage/preprocessing/Preprocess-vetting-cross-validation.ipynb`
  builds the table from `mnt/tess/labels/vetting-v01.csv` + `vetting-new-events-withfilenames.csv`
  (indexed by **Astro ID**) and carries a `TIC ID` column through to the saved CSV
  (cell 21, `X.to_csv(..., index=True)`). So TIC ID is preserved; **no reproduction is needed.**
- Robust fallback: even if a future CSV lacked the column, the TIC is in the `File` string and is
  recoverable with `df['File'].str.extract(r'-s\d{4}-0*(\d+)_')`.

## Files in this directory you'll use

- `observed_sectors.py` — the lookup tool (this README's subject).
- `data/training_set_enriched.csv` — the dec2025 labeled set (N=15,471) with `TIC ID, RA, Dec,
  sector(tag), cam, ccd, label, kind`. The `local` resolver reads RA/Dec from here (offline).
- `data/training_set_tics.txt` — **15,314 unique training TICs**, one per line (handy to hand to
  `--tic-file`, or to send to Willie/Glen for a processed-data cross-check).
- `NOTES.md` — the overall TGLC-adaptation context (stitching facts, strategy, continual-loop
  design, TGLC data inventory). Read §4 and §7 for why "re-observed since TGLC" matters.

## Caveats / decisions

- **First TGLC *production* sector = S94 (confirmed 2026-07-09).** `--since` defaults to 94 — keep
  that. (Te Han's reprocessed TGLC products on disk currently cover S56–S63; S94+ is the production
  hybrid threshold.) Related: QLP sector vetting has been sporadic — S95–S97 fully vetted, S89–S94
  not; faint-star TOIs exist for S89–S93 but few are brighter than Tmag 10.5.
- **Predicted vs processed:** tess-point gives the pointing prediction (≈ observed). Whether a
  TGLC light curve exists/was processed for each TIC is the separate access question in NOTES §3.
- **Resolvers:** `local` (training CSV, offline, instant) → `mast` (astroquery, needs internet;
  confirmed reachable from this node, HTTP 200). `auto` tries local then MAST. For the 15k training
  set everything resolves `local`, so it's fast and offline.
- tess-point caps at **~S96** in this env — fine for "since S94", but a far-future sector would
  need a tess-point pointing update.

## Why this matters for the effort

If only a small fraction of the 15k training TCEs have been re-observed since TGLC production, then
the near-term TGLC-eligible training pool (for the mix-and-match stitched set, NOTES §7) is small —
which argues for Phase-0-on-overlap + the older ≤S56 TGLC, rather than waiting on new-sector
re-observations. Running `--from-training --since <first_TGLC_sector>` quantifies that pool directly.
