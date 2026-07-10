# FITS schema + how the vetting training preprocessing consumes it

_Established 2026-07-10 by reading the Astronet-Triage preprocessing code._

## The FITS schema (what `db_to_fits.py` writes, matching the existing training FITS)
- HDU0 `PRIMARY` (no data; we add `TICID`, `SECTOR`, `DATE`, `ORIGIN`, `NTGLCCAD`).
- HDU1 BinTable **`LIGHTCURVE`** (read by *index 1*, not by name), columns in order:

  | col | fmt | meaning |
  |---|---|---|
  | `TIME` | D (f8) | BTJD = BJD−2457000 |
  | `CADENCENO` | J (i4) | cadence number (carried; **unused** by tess_io, needed only if EM-rebinning later) |
  | `SAP_FLUX` | E (f4) | relative flux ≈ 1, transits dip **down** |
  | `QUALITY` | J (i4) | 0 = good (CCD-wide \| TGLC bitmask) |
  | `SAP_FLUX_SML` | E | small-aperture relative flux |
  | `SAP_FLUX_MID` | E | mid-aperture relative flux |
  | `SAP_FLUX_LAG` | E | large-aperture relative flux |

- Filename: `astronet_hlsp_qlptglc_tess_ffi-s%04d-%016d_tess_v01_llc.fits`.

## Aperture mapping (DB gives 3; schema wants 4 flux columns)
DB exposes `small/primary/large`. Production vetting uses `primary`. We map:
`SAP_FLUX = SAP_FLUX_MID = primary`, `SAP_FLUX_SML = small`, `SAP_FLUX_LAG = large`.
So the 3 aperture channels are small/primary/large (all distinct); MID duplicates
SAP_FLUX. This is acceptable and mirrors the old QLP FITS, where `SAP_FLUX` = the
tmag-chosen best aperture and therefore also coincided with one of SML/MID/LAG.

## What the reader / preprocessing actually does
- `light_curve_util/tess_io.py::read_tess_light_curve(filename, flux_key)` reads HDU1
  `TIME`, one flux column (`flux_key`), and `QUALITY`; keeps `QUALITY==0` **only when
  `max(TIME) > 1354`** (else a hardcoded S1 bad-index list). Does **not** read `CADENCENO`.
- `astronet/preprocess/generate_input_records_3.py` (mode `vetting`) reads **all three**
  aperture columns via `aperture_key_map` (`s→SAP_FLUX_SML, m→…_MID, l→…_LAG`, none→SAP_FLUX)
  and builds `local_aperture_{s,m,l}` — these are **required, distinct** model channels
  (config `pablomer_final`, `configurations_vetting.py:915-941`, stacked as 3 conv channels).
  ⇒ the SML/MID/LAG columns must be meaningful, not copies of SAP_FLUX.
- Views (built 3× for spline bkspace ∈ {0.3, 5.0, None}): global (201 bins), local
  (61 bins, ±2·Dur), secondary, odd/even, local_aperture_{s,m,l}, sample_segments,
  half/double-period. TCE params come from the CSV: `Per`(period), `Epoc`(epoch),
  `Dur`(duration), `Depth`(aux scalar only), `Tmag/SMass/SRad/SRadEst` (aux scalars).
- **Binning is cadence-aware by absolute BTJD** (`median_filter2.py`: thresholds ~2036.2,
  2825.25 select 30/10/3.33-min half-widths). Extended-mission down-rebinning
  (`cadences>40000 → 30 min`) is a **separate** step done in the FFITools TFRecord getter
  (`helpers._preprocess_lc_arrays`), **not** in tess_io — see below.

## Extended-mission rebinning — deferred (decided)
Not done by `db_to_fits.py`. We keep the full stitched curve (all cadences) + a faithful
`CADENCENO`. When building TFRecords later, either (a) pre-bin the FITS to 30-min like the
`oct2025_..._30min_bin_v2` set, or (b) apply the rebin in the getter. tess_io itself does
not rebin.

## How the existing 15k training TFRecords were built (to mirror later)
`gen_tfrecords_vetting_2025tois_*.sh` → `generate_input_records_3.py --mode=vetting
--num_shards=50 --input_tce_csv_file=…tces-vetting-…-dec2025-{train,val,test}.csv
--tess_data_dir=<FITS dir>`. Training driver `ensemble_train_vetting_2025_final.sh` then
trains `AstroCNNModelVetting --config_name=pablomer_final` (10-member ensemble) on
`/pdo/astronet-data/data/tfrecords/dec2025_cad_scat_v5_aug/10x_0p1/`.
⚠ Both `generate_input_records*.py` have a hardcoded debug line in `main()`
(`tce_table = tce_table[tce_table["Astro ID"]==46637608501]`) — remove before a real
CLI generation run (harmless when imported as `create()`).

## Production vs training preprocessing equivalence
FFITools imports `create_astronet_tfrecords = astronet.preprocess.generate_input_records.create`;
that file is byte-identical to `generate_input_records_3.py` except one import-safe FLAGS
default. So the **same** views/features are built either way — the only variable is the
`get_lightcurve` callable (FITS reader for training vs `BackendLCGetter`/DB for production).
Caveat: what is *deployed* in `/sw/astronet` is 3.0.1, not this code (see DATA_LOCATIONS.md).
`create(..., mode="vetting", training=True)` can also build training TFRecords directly from
the DB via `BackendLCGetter`, bypassing FITS — an alternative to the FITS route.
