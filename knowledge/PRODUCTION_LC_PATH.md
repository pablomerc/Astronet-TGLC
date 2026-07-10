# How QLP production reads & stitches light curves (and how TGLC fits in)

_Established 2026-07-10 by code reading + a live read-only DB query. This is the
single most important fact for building a training set that matches production._

## TL;DR
Production does **not** stitch light curves from `/pdo/qlp-data/*.h5` files. It reads
the **`lightcurvedb` Postgres DB** and concatenates per-observation baselines. **TGLC
(production photometry since Sector 94) is ingested into the same DB tables as legacy
QLP**, so a single `read_lightcurve(tic)` returns one stitched baseline that is
legacy-QLP for sectors ≤93 and TGLC for sectors ≥94 — automatically, with no
per-sector routing. This is why `db_to_fits.py` sources from the DB.

## The read path (FFITools)
- Entry: `qlp estools astronet` (`--triage`/`--vetting`) defaults to `--source db`
  (`FFITools/src/qlp/cli/mixins.py:40-48`, `estools/bin/astronet.py:38-42`).
- `io = default_io_backend = DataRepository(PostgresBackend())`
  (`FFITools/src/qlp/io/backends/__init__.py:33`).
- `io.read_lightcurve(tic)` (`io/backends/postgres.py:63`):
  1. `resolve_tmag(tic)` (from pyticdb).
  2. `resolve_tess_observations_for_target` (`contrib/lcdb/queries.py:546`) — selects
     **every** observation with any `BestTESSOrbitLightcurve` row for the target,
     **no pipeline/sector filter**, `ORDER BY orbit_id ASC`.
  3. `collect_all_lightcurve_data(...)` builds each observation's per-aperture baseline.
  4. `_build_lc_from_obs_data_list` (`postgres.py:103`) `np.concatenate`s them in orbit
     order → one `LightcurveObservation`. **No global cadence argsort** (unlike the old
     `h5_to_fits.py`).
- A sector cutoff is applied *after* the read: `lc.limit_to_sector(sector_info)`
  (`io/datastructures/lightcurve.py:578`) masks cadences `<= sector_info.cadence_range[1]`.

## What each cadence carries (DB path)
- **time** = `TargetSpecificTime.barycentric_julian_dates`, treated as **BTJD** (BJD−2457000).
- **cadence** = `observation.cadence_reference`.
- **quality** = `CCDWideQualityFlags | TGLCQualityFlags` (bitwise OR of CCD-wide and
  target-specific TGLC flags; `queries.py:1021-1048`). 0 = good.
- **magnitudes** per aperture × method, per-orbit aligned to Tmag
  (`values = raw − nanmedian(raw) + tmag`, `queries.py:1074`).
- **relative_flux** = `magnitude_to_flux(values, ref_median)` where `ref_median` uses only
  `quality==0` cadences (`build_baseline_qtable`, `queries.py:1252-1266`). Median ≈ 1,
  transits dip **below** 1.

## Apertures: 3, named small/primary/large (for BOTH QLP and TGLC)
The DB normalizes both pipelines to `small/primary/large` (`APERTURE_TYPES`,
`io/datastructures/_types.py:7`). The per-orbit "best" aperture is baked into
`BestTESSOrbitLightcurve`; **`primary` = that best aperture**. Provenance discriminator
(the ONLY QLP-vs-TGLC tag): the aperture/method recorded per observation —
`Aperture_00X`+`KSPMagnitude` (legacy QLP ≤S93) vs `TGLC Aperture *`+`QSPMagnitude`
(TGLC ≥S94), set in `BestTESSOrbitLightcurve.from_h5`
(`contrib/lcdb/models/lightcurve_lookup.py:67-123`).

## Production inference preprocessing (for reference; NOT what db_to_fits does)
`BackendLCGetter` (`reports/astronet/helpers.py:100-142`) → `preprocess_lightcurve`
(`helpers.py:35-67`, defaults `source_ap="primary"`, `detrend_method="raw"`):
`limit_to_quality_flag(0)` → `limit_to_finite` → take `primary/raw` magnitudes →
`_preprocess_lc_arrays` (`helpers.py:70-97`): **rebin extended-mission points
(`cadences>40000`) to 30-min**, then `flux = 10**(-(mag − nanmedian(mag))/2.5)`. This
getter feeds `create_astronet_tfrecords` (= `astronet .../generate_input_records.create`).
→ The **EM rebinning lives at the TFRecord stage, not in the light-curve content**, so
`db_to_fits.py` deliberately does not do it (it keeps all cadences + CADENCENO).

## Live confirmation
TIC 592638 `best_tess_orbit_lightcurve` rows: S5 & S32 → `Aperture_003`/`KSPMagnitude`
(legacy QLP); S98 → `TGLC Aperture Primary`/`QSPMagnitude` (TGLC). `read_lightcurve`
returned a continuous baseline BTJD 1438 (S5) → 4046 (S98). DB covers sectors 1–104,
with TGLC apertures for S94–S104.

## The old file-glob path (`Astronet-Triage/h5_to_fits.py`) — how it differs
Reads `/pdo/qlp-data/orbit-*/ffi/cam*/ccd*/LC/{tic}.h5` directly, 5 numbered apertures,
`get_bestap(tmag)` static best-ap, global cadence argsort, CCD-wide qflags only (no TGLC),
flux relative to Tmag. It produced the existing 15k training FITS. It cannot see TGLC and
diverges from production on aperture set, sorting, alignment median, and flags — which is
why we abandoned it for the DB path.

## The operator production run recipe (context)
Per sector/cam/ccd, operators run (in the sector `ffi/run` direnv, staging branch):
`qlp lctools bls … --source db`; `qlp estools tic-filter`; `qlp estools astronet --triage`;
`astronet-filter`; `difference-image`; `diffimage-centroid-filter`;
`qlp estools astronet --vetting`; `astronet-filter --score-col disp_p --cutoff 0.75`;
`qlp estools report`. The `--vetting` step is what the new ensemble will replace.
