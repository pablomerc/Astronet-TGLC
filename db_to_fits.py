"""db_to_fits.py — generate reprocessed QLP+TGLC training light curves from the
production light-curve database, as FITS files matching the Astronet training schema.

WHY THE DB (not file globbing)
------------------------------
Production QLP builds stitched light curves by reading the `lightcurvedb` Postgres
DB (`default_io_backend.read_lightcurve(tic)` ->
`FFITools .../io/backends/postgres.py:_build_lc_from_obs_data_list`), NOT by globbing
`/pdo/qlp-data/*.h5`. TGLC (production photometry since Sector 94) is ingested into
the SAME DB tables as legacy QLP, so `read_lightcurve(tic)` returns ONE stitched
baseline that is legacy-QLP for sectors <=93 and TGLC for sectors >=94 — exactly what
production scores. This script therefore just reads that baseline, cuts it at the
requested sector, and serializes it to the existing FITS schema. No per-sector
mix-and-match logic is needed (the DB already did it). Supersedes h5_to_fits_hybrid.py.

WHAT IT WRITES  (schema consumed by light_curve_util/tess_io.py + generate_input_records_3.py)
---------------------------------------------------------------------------------------------
  HDU0 PRIMARY (no data; TICID/SECTOR/DATE/ORIGIN/NTGLCCAD in header)
  HDU1 BinTable "LIGHTCURVE":
    TIME (D, BTJD=BJD-2457000), CADENCENO (J), SAP_FLUX (E, rel flux ~1, transits dip down),
    QUALITY (J; 0=good, CCD-wide|TGLC bitmask), SAP_FLUX_SML/MID/LAG (E)
  Aperture mapping (DB exposes small/primary/large; production uses 'primary'):
    SAP_FLUX = SAP_FLUX_MID = primary ;  SAP_FLUX_SML = small ;  SAP_FLUX_LAG = large
  Filename: astronet_hlsp_qlptglc_tess_ffi-s%04d-%016d_tess_v01_llc.fits

NOT DONE HERE (deferred to the TFRecord step, on purpose)
---------------------------------------------------------
  Extended-mission cadence rebinning (cadences>40000 -> 30 min) happens in the
  FFITools TFRecord-stage getter (helpers._preprocess_lc_arrays), not in the LC
  content. We keep ALL cadences and a faithful CADENCENO so the EM boundary is
  recoverable downstream. tess_io does the QUALITY==0 filtering, so we do NOT filter.

ENVIRONMENT
-----------
Needs the QLP operator stack (qlp 0.14.x with qlp.io.backends + lightcurvedb 3.0.0
+ pyticdb; DB creds at ~/.config/lightcurvedb/db.conf). Run with the operator venv
and a CLEAN PYTHONPATH so the venv's qlp wins over the old /pdo/app one:

  PYTHONPATH= /sw/qlp-environment/.venv/bin/python db_to_fits.py \
      -i data/reobserved_s103_tics.txt \
      -o /pdo/users/pablomer/mnt/tess/reprocessed_s103_qlptglc_fits_files \
      -s 103 -n 4

  (equivalently, inside the sector run dir: `direnv allow && direnv exec . python ...`)
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool

import numpy as np
from astropy.io import fits

# Imported lazily-at-module-level so forked/pooled workers each get the backend.
from qlp.io.backends import default_io_backend as io

# ------------------------------------------------------------------ config globals
# Set once per worker by _init_worker (avoids re-reading sector info per TIC).
_CFG = {}


def construct_fitsfile(outdir: str, sector: int, tic: int) -> str:
    fname = "astronet_hlsp_qlptglc_tess_ffi-s%04d-%016d_tess_v01_llc.fits" % (sector, int(tic))
    return os.path.join(outdir, fname)


def _init_worker(outdir, sector, overwrite, min_points):
    """Runs once in each worker process. Reads sector metadata here (post-fork) so
    every worker opens its own DB connection."""
    _CFG["outdir"] = outdir
    _CFG["sector"] = sector
    _CFG["overwrite"] = overwrite
    _CFG["min_points"] = min_points
    _CFG["sector_info"] = io.read_sector_info(sector)
    # First cadence of S94 = the TGLC era boundary (for provenance counting).
    try:
        _CFG["s94_cad0"] = io.read_sector_info(94).cadence_range[0]
    except Exception:
        _CFG["s94_cad0"] = None


def _rel_flux(lc, aperture):
    """Relative flux (~1, transits dip down) for one aperture, raw method, masked."""
    return np.asarray(lc[aperture, "raw"]["relative_flux"], dtype=np.float64)


def process_one(tic: int) -> dict:
    """Read one TIC from the DB, cut to the sector, write a FITS. Returns a status dict."""
    tic = int(tic)
    outdir = _CFG["outdir"]
    sector = _CFG["sector"]
    fitspath = construct_fitsfile(outdir, sector, tic)

    if not _CFG["overwrite"] and os.path.exists(fitspath):
        return {"tic": tic, "status": "skipped", "npts": 0}

    try:
        lc = io.read_lightcurve(tic)
    except Exception as e:
        return {"tic": tic, "status": "no_data", "npts": 0, "msg": str(e)[:200]}

    try:
        lc = lc.limit_to_sector(_CFG["sector_info"])

        time_bjd = np.asarray(lc.bjd.value, dtype=np.float64)
        cadence = np.asarray(lc.cadences, dtype=np.int64)
        quality = np.asarray(lc.quality_flags)[lc.mask].astype(np.int64)
        primary = _rel_flux(lc, "primary")
        small = _rel_flux(lc, "small")
        large = _rel_flux(lc, "large")

        n = len(time_bjd)
        if n < _CFG["min_points"]:
            return {"tic": tic, "status": "too_few", "npts": n}

        # provenance: how many cadences fall in the TGLC era (S94+)
        s94_cad0 = _CFG["s94_cad0"]
        n_tglc = int(np.sum(cadence >= s94_cad0)) if s94_cad0 is not None else -1

        _write_fits(fitspath, tic, sector, time_bjd, cadence, quality,
                    primary, small, large, n_tglc)

        return {"tic": tic, "status": "written", "npts": n, "n_tglc": n_tglc,
                "bjd_min": float(np.nanmin(time_bjd)), "bjd_max": float(np.nanmax(time_bjd)),
                "cad_min": int(cadence.min()), "cad_max": int(cadence.max())}
    except Exception as e:
        return {"tic": tic, "status": "failed", "npts": 0, "msg": repr(e)[:300]}


def _write_fits(path, tic, sector, time_bjd, cadence, quality,
                primary, small, large, n_tglc):
    hdu0 = fits.PrimaryHDU()
    h = hdu0.header
    h["EXTNAME"] = "PRIMARY"
    h["TICID"] = (int(tic), "unique TESS target identifier")
    h["SECTOR"] = (int(sector), "last sector searched (cutoff)")
    h["DATE"] = datetime.today().strftime("%Y-%m-%d")
    h["ORIGIN"] = ("MIT/QLP+TGLC (lcdb)", "stitched from production lightcurvedb")
    h["NTGLCCAD"] = (int(n_tglc), "num cadences in TGLC era (sector>=94)")

    cols = [
        fits.Column(name="TIME", format="D", unit="BJD-2457000, days", array=time_bjd),
        fits.Column(name="CADENCENO", format="J", array=cadence),
        fits.Column(name="SAP_FLUX", format="E", array=primary.astype(np.float32)),
        fits.Column(name="QUALITY", format="J", array=quality),
        fits.Column(name="SAP_FLUX_SML", format="E", array=small.astype(np.float32)),
        fits.Column(name="SAP_FLUX_MID", format="E", array=primary.astype(np.float32)),
        fits.Column(name="SAP_FLUX_LAG", format="E", array=large.astype(np.float32)),
    ]
    hdu1 = fits.BinTableHDU.from_columns(cols)
    hdu1.header["EXTNAME"] = "LIGHTCURVE"
    hdu1.header["INHERIT"] = "T"
    tmp = path + ".tmp%d" % os.getpid()
    fits.HDUList([hdu0, hdu1]).writeto(tmp, overwrite=True)
    os.replace(tmp, path)  # atomic: a partial file never looks complete


def _fmt_dt(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inlist", "-i", required=True, help="file with 1 TIC per line")
    p.add_argument("--outdir", "-o", required=True, help="output directory for FITS")
    p.add_argument("--sector", "-s", type=int, default=103, help="cutoff sector (default 103)")
    p.add_argument("--nprocs", "-n", type=int, default=4, help="# worker processes (default 4)")
    p.add_argument("--overwrite", "-r", action="store_true", help="overwrite existing FITS")
    p.add_argument("--min-points", type=int, default=10,
                   help="skip targets with fewer than this many cadences (default 10)")
    p.add_argument("--limit", type=int, default=None,
                   help="only process the first N TICs (for quick tests)")
    p.add_argument("--summary-csv", default=None,
                   help="write a per-TIC provenance/status CSV here "
                        "(default: <outdir>/_run_summary.csv)")
    args = p.parse_args()

    logging.basicConfig(level="INFO", format="%(message)s")
    log = logging.getLogger("db_to_fits")

    tics = np.loadtxt(args.inlist, dtype=np.int64, ndmin=1)
    if args.limit:
        tics = tics[: args.limit]
    total = len(tics)
    os.makedirs(args.outdir, exist_ok=True)
    summary_csv = args.summary_csv or os.path.join(args.outdir, "_run_summary.csv")

    log.info(f"[db_to_fits] {total} TICs | cutoff s{args.sector} | nprocs {args.nprocs}")
    log.info(f"[db_to_fits] out: {args.outdir}")

    tally = {"written": 0, "skipped": 0, "too_few": 0, "no_data": 0, "failed": 0}
    rows = []
    t0 = time.time()

    def render(done):
        el = time.time() - t0
        rate = done / el if el > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else 0.0
        bar = (f"\r[db_to_fits] {done}/{total} ({done/total:5.1%}) | "
               f"ok {tally['written']} skip {tally['skipped']} few {tally['too_few']} "
               f"nodata {tally['no_data']} fail {tally['failed']} | "
               f"{rate:4.1f} tic/s | elapsed {_fmt_dt(el)} | eta {_fmt_dt(eta)}   ")
        sys.stderr.write(bar)
        sys.stderr.flush()

    with Pool(args.nprocs, initializer=_init_worker,
              initargs=(args.outdir, args.sector, args.overwrite, args.min_points)) as pool:
        for done, res in enumerate(pool.imap_unordered(process_one, tics, chunksize=1), 1):
            tally[res["status"]] = tally.get(res["status"], 0) + 1
            rows.append(res)
            if res["status"] in ("no_data", "failed") and res.get("msg"):
                sys.stderr.write("\n")
                log.warning(f"  TIC {res['tic']} {res['status']}: {res['msg']}")
            render(done)
    sys.stderr.write("\n")

    # provenance / status CSV
    import csv
    fields = ["tic", "status", "npts", "n_tglc", "bjd_min", "bjd_max",
              "cad_min", "cad_max", "msg"]
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    el = time.time() - t0
    log.info(f"[db_to_fits] DONE in {_fmt_dt(el)}: " +
             " ".join(f"{k}={v}" for k, v in tally.items()))
    n_tglc_targets = sum(1 for r in rows if r.get("n_tglc", 0) and r["n_tglc"] > 0)
    log.info(f"[db_to_fits] {n_tglc_targets} targets have TGLC-era (S94+) cadences")
    log.info(f"[db_to_fits] summary -> {summary_csv}")


if __name__ == "__main__":
    main()
