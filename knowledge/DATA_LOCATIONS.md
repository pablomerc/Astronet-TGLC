# TESS light-curve data: locations, DB coverage, and the reobservation counts

_Established 2026-07-10 (filesystem + live DB inspection)._

## The authoritative source: the production DB
- **`lightcurvedb` Postgres**, host `pdodb2.mit.edu`, db `lightcurvedb`, port 5432.
  Creds (already working) in `~/.config/lightcurvedb/db.conf` (`[Credentials]`,
  user `lcdb_writer`). Needs `lightcurvedb==3.0.0` (only in the QLP operator venv).
- Coverage: `tess_orbit` sectors **1–105**; `best_tess_orbit_lightcurve` rows through
  **sector 104**. Sectors ≤93 → legacy QLP apertures (`Aperture_00X`, `KSPMagnitude`);
  **sectors 94–104 → TGLC** (`TGLC Aperture Small/Primary/Large`, `QSPMagnitude`).
- `read_lightcurve(tic)` returns the full stitched QLP+TGLC baseline (see
  [PRODUCTION_LC_PATH.md](PRODUCTION_LC_PATH.md)). **This is what `db_to_fits.py` uses.**

## Filesystem (secondary; not used by db_to_fits, kept for reference)
| What | Path | Sectors | Format | Keyed by |
|---|---|---|---|---|
| QLP-tree h5 | `/pdo/qlp-data/orbit-{o}/ffi/cam{c}/ccd{d}/LC/{tic}.h5` | 1–~107 (orbits to 221) | h5 | TIC |
| QLP qflags | `/pdo/qlp-data/qflagpath/orbit{o}cam{c}ccd{d}_qflag.txt` | — | text | — |
| TGLC S94 h5 | `/pdo/users/tehan/tglc-s94-faint-allstar/orbit-{195,196}/…/LC/{tic}.h5` | **S94 only** | h5, 3 named aps | TIC |
| TGLC reproc FITS | `/pdo/users/tehan/tglc-gpu-production/hlsp_s00{56..63}/…` | S56–S63 | FITS | Gaia DR3 |
| TGLC v2.1 FITS | `/pdo/users/tehan/tglc_v2.1_s56cam4/…` | S56 | FITS | Gaia DR3 |

- Orbit↔sector: `sector = (orbit − 7)/2` (S94 = orbits 195/196 … S103 = orbits 213/214).
- Note: `/pdo/qlp-data/orbit-*` has h5 for S94–S103, but from S94 those are TGLC-format
  photometry (ingested into the DB as TGLC). Standalone TGLC *files* under `tehan` only
  exist for S94, S56–S63, S56 — which is another reason to go through the DB, not files.

## Existing training FITS datasets (for schema reference / QLP-only comparison)
- `/pdo/users/pablomer/mnt/tess/april2025_dataset_fits_files/` (~19.4k, `astronet_hlsp_qlp_…`)
- `/pdo/astronet-data/data/fits/oct2025_dataset_all_fits_files/` and `…_30min_bin_v2/`
  (the `_30min_bin` variants = EM pre-binned to 30 min; the precedent for handling
  extended-mission cadence at FITS creation time).

## Our new reprocessed output
- **`/pdo/users/pablomer/mnt/tess/reprocessed_s103_qlptglc_fits_files/`** — new, non-colliding
  (prefix `astronet_hlsp_qlptglc_…`). ~5,781 FITS + `_run_summary.csv`.

## Reobservation of the training set since TGLC (S94)
Source: `data/training_reobserved_s103.csv` (15,314 unique training TICs). The `*_lc`
flags come from `qlp find coverage --tic-list … -l 103 --levine-only` (`has_lightcurve`),
the `*_geom` flags from tess-point footprint.
- **`reobs103_lc` = True: 5,781** (has an actual light curve in S94–S103) ← what we reprocess.
- `reobs103_geom` = True: 5,912 (geometric footprint in S94–S103).
- `reobs96_*`: ~2,835 (lc) / ~2,920 (geom) — the S94–96 subset; ~2,992 more added in S97–103.
- Label breakdown of the reprocessed set (companion CSV, 5,834 TCEs): Junk 2,267 ·
  Planet 1,885 · EB 1,682.
- Full labeled set for context: 15,471 TCEs (P 5,823 · EB 3,390 · Junk 6,258),
  15,314 unique TICs; `.../astronet/tces-vetting-…-dec2025-all.csv`.

## Deployment state (verified 2026-07-10)
`/sw/astronet` → `astronet-3.0.1`; **no `astronet-3.1.0`** — the v3.1.0 cutover has not
happened. Live production vetting model = `AstroCNNModelVetting_cshallue_20250429_181612`
(single model), not the `pablomer_final` ensemble (which is in
`/pdo/astronet-data/models/vetting/experimental/`). So "match production" for the new
ensemble = match the code it was trained with (`generate_input_records_3.py`).
