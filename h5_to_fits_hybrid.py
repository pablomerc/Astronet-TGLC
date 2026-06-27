"""Astronet H5 -> FITS tool  ·  QLP + TGLC MIX-AND-MATCH prototype
=================================================================

Prototype extension of ``Astronet-Triage/h5_to_fits.py`` that builds a SINGLE
stitched light-curve FITS per target where **each sector's photometry can come
from a different source**: QLP (the legacy aperture pipeline) for older sectors,
TGLC (TESS-Gaia effective-PSF) for the recent ones. The output FITS is identical
in schema to the existing converter's, so the rest of the Astronet pipeline
(``generate_input_records_3.py`` -> TFRecords -> model) needs NO change.

Why this is the right shape (see TGLC_adaptation/NOTES.md):
  * ~76% of the current training set is ALREADY multi-sector stitched QLP, so the
    model expects stitched curves. We only swap the per-sector *source*.
  * The ``-s`` arg is the "last sector to search" CUTOFF; the original pulls every
    QLP sector a target was observed in up to it. We keep that and add a per-sector
    source decision on top.

Design choices that keep the QLP path byte-identical to the original:
  * QLP segments are loaded EXACTLY as in the original ``merge_lcs`` (magnitudes,
    per-orbit self-normalized so median(rlc[qflag==0]) -> Tmag).
  * The TGLC segment loader returns the SAME representation (pseudo-magnitudes
    relative to Tmag), so the existing flux_from_mag writer is untouched and QLP
    and TGLC segments end up on the same relative-flux scale (~1 median).
  * With no TGLC cutoff given, this script reproduces the original QLP output.

STATUS: UNTESTED SCAFFOLD. The TGLC loader is a stub — three things must be filled
in once Te Han's TGLC products are located (all flagged ``TODO[TGLC]`` below):
  1. TGLC file PATH + on-disk format (HDF5? FITS? per-sector? columns?).
  2. Flux normalization + the conversion to the QLP-consistent pseudo-mag scale.
  3. QUALITY-flag synthesis (raw TGLC has no QLP qflag; tess_io.py:92-94 filters
     on a flag column, so we MUST provide one or transits get polluted).

Run in the QLP/FFITools environment (needs pyticdb, lightcurvedb, qlp) — the same
env the original h5_to_fits.py runs in, NOT the astronet conda env.

Example (QLP for s1-55, TGLC for s56+):
  python3 h5_to_fits_hybrid.py -i tic.ls -o out/ -s 64 --tglc-cutoff 56
"""

import argparse
import itertools as it
import logging
import os
from collections import namedtuple
from datetime import datetime
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple

os.environ.update(
    OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
    NUMEXPR_MAX_THREADS="1", MKL_NUM_THREADS="1",
)

import numpy as np
import pandas as pd  # noqa: F401  (kept for parity / future source-map CSV)
import pyticdb

from astropy.io import fits
from lightcurvedb import db
from qlp.lctools.hdf5lc import HDFLightCurve
from qlp.util.util import orbits_from_sector, time_correction

N_APERTURES = 5

# A "segment" is the unified per-source contribution for one sector: arrays of
# equal length plus the 5 aperture magnitude channels (pseudo-mag for TGLC).
Segment = namedtuple("Segment", ("bjd", "cadence", "flag", "mags"))  # mags: Dict[str, np.ndarray]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def cmd_parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inlist", "-i", help="inlist: file with 1 TIC per line")
    p.add_argument("--outdir", "-o", help="output directory")
    p.add_argument("--sector", "-s", type=int, help="last sector to search (cutoff)")
    p.add_argument("--nprocs", "-n", type=int, default=3, help="# processors (default 3)")
    p.add_argument("--tree", "-t", action="store_true", help="store in TIC-bucketed dir tree")
    p.add_argument("--overwrite", "-r", action="store_true", help="replace existing files")
    p.add_argument("--debug", action="store_true", help="debug logging")
    # --- mix-and-match controls (the new bit) ---
    p.add_argument("--tglc-cutoff", type=int, default=None,
                   help="sectors >= this value use TGLC; sectors below use QLP. "
                        "If omitted, all sectors use QLP (== original behavior).")
    p.add_argument("--tglc-sectors", type=str, default=None,
                   help="explicit comma-separated list of sectors to source from TGLC "
                        "(overrides --tglc-cutoff). e.g. '56,57,58'")
    p.add_argument("--require-tglc", action="store_true",
                   help="error out if a sector designated TGLC has no TGLC data "
                        "(default: fall back to QLP and warn).")
    return p.parse_args()


def construct_fitsfile(outdir: str, sector: int, tic: int, tree: bool = False) -> str:
    fitsfile = "astronet_hlsp_qlptglc_tess_ffi-s%.4d-%.16d_tess_v01_llc.fits" % (sector, tic)
    if tree:
        dirs = [f"{tic:016d}"[i:i + 4] for i in range(0, 16, 4)]
        full_dir = os.path.join(outdir, *dirs)
        os.makedirs(full_dir, exist_ok=True)
        return os.path.join(full_dir, fitsfile)
    return os.path.join(outdir, fitsfile)


# --------------------------------------------------------------------------- #
# Per-sector source decision  (THE injection point that h5_to_fits.py:90 lacked)
# --------------------------------------------------------------------------- #
def decide_source(sector: int, tglc_cutoff: Optional[int], tglc_sectors: Optional[set]) -> str:
    if tglc_sectors is not None:
        return "TGLC" if sector in tglc_sectors else "QLP"
    if tglc_cutoff is not None and sector >= tglc_cutoff:
        return "TGLC"
    return "QLP"


# --------------------------------------------------------------------------- #
# QLP loader  (faithful extraction of the original merge_lcs per-orbit body)
# --------------------------------------------------------------------------- #
QlpOrbitLoc = namedtuple("QlpOrbitLoc", ("sector", "orbit", "cam", "ccd", "lcfile"))


def get_qlp_orbit_locations(tic: int, sector: int) -> List[QlpOrbitLoc]:
    locs: List[QlpOrbitLoc] = []
    for orbit in orbits_from_sector(sector):
        for cam, ccd in it.product([1, 2, 3, 4], repeat=2):
            lcfile = f"/pdo/qlp-data/orbit-{orbit}/ffi/cam{cam}/ccd{ccd}/LC/{tic}.h5"
            if os.path.exists(lcfile):
                locs.append(QlpOrbitLoc(sector, orbit, cam, ccd, lcfile))
    return locs


def load_qlp_sector_segment(tic: int, sector: int, tmag: float, ra: float,
                            dec: float) -> Optional[Segment]:
    """All QLP orbits of one sector -> one Segment (magnitudes), or None if absent."""
    locs = get_qlp_orbit_locations(tic, sector)
    if not locs:
        return None
    bjd, cadence, qflag = [], [], []
    raw_mags: Dict[str, List[float]] = {f"Aperture_{i:03d}": [] for i in range(N_APERTURES)}
    for _, orbit, cam, ccd, lcfile in locs:
        qflag_file = np.loadtxt(f"/pdo/qlp-data/qflagpath/orbit{orbit}cam{cam}ccd{ccd}_qflag.txt")
        orbit_qflags = qflag_file[:, 1]
        orbit_lc = HDFLightCurve(name=lcfile)
        orbit_lc.load_basic_info()
        if orbit <= 34:  # Year-1 jd arrays need on-the-fly reconstruction (original HACK)
            with db:
                mid_tjd = db.query_frames_by_orbit(orbit, cam)["mid_tjd"]
            orbit_lc.data["jd"] = time_correction(orbit, mid_tjd, ra=ra, dec=dec)
        bjd += list(orbit_lc.data["jd"])
        cadence += list(orbit_lc.data["cadence"])
        qflag += list(orbit_qflags)
        for i in range(N_APERTURES):
            orbit_lc.load_from_file(ap=i, label="all")
            raw_mags[f"Aperture_{i:03d}"] += list(
                orbit_lc.data["rlc"]
                - (np.nanmedian(orbit_lc.data["rlc"][np.array(orbit_qflags) == 0]) - tmag)
            )
    return Segment(np.array(bjd), np.array(cadence), np.array(qflag),
                   {k: np.array(v) for k, v in raw_mags.items()})


# --------------------------------------------------------------------------- #
# TGLC loader  (SPECIFIED from real products; UNTESTED — see DECISIONS below)
# --------------------------------------------------------------------------- #
# Discovered 2026-06-26 from Te Han's products:
#   Root (new QLP-pipeline reprocessing, sectors S56-S63 available so far):
#     /pdo/users/tehan/tglc-gpu-production/hlsp_s%04d/...   (TIC-bucketed tree)
#   v2.1 reference set (S56 cam4): /pdo/users/tehan/tglc_v2.1_s56cam4/sector0056/lc/<cam>-<ccd>/
#   File: hlsp_tglc_tess_ffi_gaiaid-<GAIADR3>-s%04d-cam<C>-ccd<D>_tess_v2.1_llc.fits
#     -> FILES ARE KEYED BY GAIA DR3 ID, not TIC (header DOES carry TICID + GAIADR3).
#   HDU1 'LIGHTCURVE' columns: time(BTJD=BJD-2457000), psf_flux, aperture_flux,
#     cal_psf_flux, cal_aper_flux, background, cadence_num, TESS_flags, TGLC_flags,
#     aperture_flux_raw.  cal_psf_flux = detrended, ~1-normalized (CAN be negative).
#   NOTE: these NEW products DO carry quality flags (TESS_flags + TGLC_flags) —
#   the "raw TGLC has no flags" caveat applies to Te's OLDER main-mission S1-56 set.
TGLC_ROOT = "/pdo/users/tehan/tglc-gpu-production"   # DECISION: confirm canonical variant
TGLC_FILE_TMPL = "hlsp_s{sector:04d}"                #   (twirl / fs_v1 / fs_v2 / v4 ...)
TGLC_FLUX_COL = "cal_psf_flux"                        # DECISION: cal_psf vs psf vs aperture
_TINY = 1e-6


def _tic_to_gaia(tic: int) -> Optional[int]:
    """TIC -> Gaia DR3 source id (needed because TGLC files are gaia-keyed).
    DECISION: confirm the pyticdb column name for Gaia DR3 ('gaia' / 'GAIA' / 'gaiadr3')."""
    try:
        return int(pyticdb.query_by_id(tic, "gaia")[0][0])  # column name to confirm
    except Exception:
        return None


def _find_tglc_file(gaia: int, sector: int) -> Optional[str]:
    import glob
    sroot = os.path.join(TGLC_ROOT, TGLC_FILE_TMPL.format(sector=sector))
    hits = glob.glob(os.path.join(sroot, "**", f"*gaiaid-{gaia}-s{sector:04d}-*llc.fits"),
                     recursive=True)
    return hits[0] if hits else None


def load_tglc_sector_segment(tic: int, sector: int, tmag: float, ra: float,
                             dec: float) -> Optional[Segment]:
    """One sector of TGLC photometry -> Segment on the SAME pseudo-mag representation
    the QLP loader returns, so it concatenates seamlessly. Returns None if absent.

    !! UNTESTED — runs only in the QLP/FFITools env with the TGLC data present.
    DECISIONS to confirm before trusting output:
      (D1) canonical TGLC variant dir (TGLC_FILE_TMPL) and root;
      (D2) flux column (TGLC_FLUX_COL = cal_psf_flux: detrended & ~1-normalized but
           can go NEGATIVE -> the pseudo-mag log below clips negatives to faint mag;
           alternative: carry TGLC as flux directly and refactor the writer);
      (D3) the pyticdb Gaia-DR3 column name in _tic_to_gaia;
      (D4) whether to use these NEW (flagged, S56+) products or Te's OLDER main-mission
           S1-56 set (no QLP flags) — Phase 0 decides usability.
    """
    gaia = _tic_to_gaia(tic)
    if gaia is None:
        return None
    path = _find_tglc_file(gaia, sector)
    if path is None:
        return None
    with fits.open(path) as h:
        d = h[1].data
        time = np.asarray(d["time"], float)            # BTJD, same system as QLP
        cadence = np.asarray(d["cadence_num"], int)
        flux = np.asarray(d[TGLC_FLUX_COL], float)
        # quality: 0 = good; OR the TESS and TGLC flag bitmasks
        flag = (np.asarray(d["TESS_flags"], int) != 0) | (np.asarray(d["TGLC_flags"], int) != 0)
        flag = flag.astype(int)

    good = (flag == 0) & np.isfinite(flux)
    if good.sum() < 3:
        return None
    flux_norm = flux / np.nanmedian(flux[good])        # median(good) -> 1
    # -> pseudo-mag relative to Tmag (clip non-positive flux to a faint floor)
    pseudo_mag = tmag - 2.5 * np.log10(np.clip(flux_norm, _TINY, None))
    # TGLC is single-LC: replicate into all 5 aperture channels (degenerate but
    # schema-compatible; the model's multi-aperture local views collapse for TGLC sectors)
    mags = {f"Aperture_{i:03d}": pseudo_mag.copy() for i in range(N_APERTURES)}
    return Segment(time, cadence, flag, mags)


# --------------------------------------------------------------------------- #
# Hybrid merge
# --------------------------------------------------------------------------- #
def merge_lcs_hybrid(tic: int, last_sector: int, tmag: float, ra: float, dec: float,
                     tglc_cutoff: Optional[int], tglc_sectors: Optional[set],
                     require_tglc: bool) -> Tuple[Dict[str, Any], Dict[int, str]]:
    segments: List[Segment] = []
    used_source: Dict[int, str] = {}
    for sector in range(1, last_sector + 1):
        source = decide_source(sector, tglc_cutoff, tglc_sectors)
        seg = None
        if source == "TGLC":
            try:
                seg = load_tglc_sector_segment(tic, sector, tmag, ra, dec)
            except NotImplementedError:
                raise
            if seg is None:
                if require_tglc:
                    raise RuntimeError(f"TIC {tic} s{sector}: TGLC requested but no TGLC data.")
                logger.warning(f"TIC {tic} s{sector}: no TGLC data, falling back to QLP.")
                seg, source = load_qlp_sector_segment(tic, sector, tmag, ra, dec), "QLP(fallback)"
        else:
            seg = load_qlp_sector_segment(tic, sector, tmag, ra, dec)
        if seg is not None:
            segments.append(seg)
            used_source[sector] = source

    if not segments:
        raise RuntimeError(f"TIC {tic}: no data found in any sector up to {last_sector}.")

    # concatenate then sort chronologically by cadence (matches the original)
    bjd = np.concatenate([s.bjd for s in segments])
    cadence = np.concatenate([s.cadence for s in segments])
    flag = np.concatenate([s.flag for s in segments])
    mags = {f"Aperture_{i:03d}": np.concatenate([s.mags[f"Aperture_{i:03d}"] for s in segments])
            for i in range(N_APERTURES)}
    idx = np.argsort(cadence)
    data: Dict[str, Any] = {"bjd": bjd[idx], "cadence": cadence[idx], "flag": flag[idx]}
    for i in range(N_APERTURES):
        data[f"Aperture_{i:03d}"] = mags[f"Aperture_{i:03d}"][idx]
    return data, used_source


# --------------------------------------------------------------------------- #
# writer (identical schema to original h5_to_fits.py)
# --------------------------------------------------------------------------- #
def get_bestap(tmag: float) -> int:
    magbins = np.array([6, 7, 8, 9, 10, 11, 12])
    bestaps = np.array([4, 3, 3, 2, 2, 2, 1])
    index = np.searchsorted(magbins, tmag)
    if index == 0:
        return int(bestaps[0])
    if index >= len(magbins):
        return int(bestaps[-1])
    return int(bestaps[index] if tmag > magbins[index] - 0.5 else bestaps[index - 1])


def flux_from_mag(mag, ref_mag):
    return 10 ** (-0.4 * (mag - ref_mag))


def h5tofits(tic: int, job_no: int) -> None:
    logger.info(f"Job #{job_no}/{ntics}: TIC-{tic} (hybrid QLP/TGLC)")
    fitspath = construct_fitsfile(args.outdir, args.sector, tic, args.tree)
    if not args.overwrite and os.path.exists(fitspath):
        logger.warning(f"{tic} fits already exists. Use -r to overwrite.")
        return

    field_list = ["tmag", "ra", "dec"]
    tic_info = dict(zip(field_list, pyticdb.query_by_id(tic, *field_list)[0]))

    tglc_sectors = (set(int(x) for x in args.tglc_sectors.split(",")) if args.tglc_sectors else None)
    lcdata, used_source = merge_lcs_hybrid(
        tic, args.sector, tic_info["tmag"], tic_info["ra"], tic_info["dec"],
        args.tglc_cutoff, tglc_sectors, args.require_tglc)
    logger.info(f"TIC {tic} source-by-sector: {used_source}")

    hdu1 = fits.PrimaryHDU()
    hdu1.header.set("NEXTEND", value=1, comment="number of standard extensions")
    hdu1.header.set("EXTNAME", value="PRIMARY")
    hdu1.header.set("TICID", value=int(tic), comment="unique TESS target identifier")
    hdu1.header.set("SECTOR", value=int(args.sector), comment="last observed sector (cutoff)")
    hdu1.header.set("DATE", value=datetime.today().strftime("%Y-%m-%d"))
    # provenance: record which sectors came from TGLC (auditability for mix-and-match)
    tglc_used = sorted(s for s, src in used_source.items() if src.startswith("TGLC"))
    hdu1.header.set("TGLCSECS", value=",".join(map(str, tglc_used)) or "none",
                    comment="sectors sourced from TGLC")

    cols = [fits.Column(name="TIME", format="D", array=lcdata["bjd"], unit="BJD-2457000, days"),
            fits.Column(name="CADENCENO", format="J", array=lcdata["cadence"])]
    bestap = get_bestap(tic_info["tmag"])
    cols.append(fits.Column(name="SAP_FLUX", format="E",
                            array=flux_from_mag(lcdata[f"Aperture_{bestap:03d}"], tic_info["tmag"])))
    cols.append(fits.Column(name="QUALITY", format="J", array=lcdata["flag"]))
    for i, size in zip([1, 2, 3], ["SML", "MID", "LAG"]):
        cols.append(fits.Column(name=f"SAP_FLUX_{size}", format="E",
                                array=flux_from_mag(lcdata[f"Aperture_00{i}"], tic_info["tmag"])))

    hdu2 = fits.BinTableHDU().from_columns(fits.ColDefs(cols))
    hdu2.header.set("INHERIT", value="T")
    hdu2.header.set("EXTNAME", value="LIGHTCURVE")
    hdu2.header.set("BESTAP", value=bestap, comment="best aperture index (0 to 4)")
    fits.HDUList([hdu1, hdu2]).writeto(fitspath, overwrite=True)
    logger.info(f"Wrote {fitspath}")


if __name__ == "__main__":
    args = cmd_parse()
    logger = logging.getLogger(__name__)
    logging.basicConfig(level="DEBUG" if args.debug else "INFO")
    tics = np.loadtxt(args.inlist, dtype=int, ndmin=1)
    ntics = len(tics)
    logger.info(f"Read {ntics} TICs from {args.inlist}; cutoff s{args.sector}; "
                f"tglc_cutoff={args.tglc_cutoff} tglc_sectors={args.tglc_sectors}")
    tasks = [[tic, i + 1] for i, tic in enumerate(tics)]
    with Pool(args.nprocs) as pool:
        pool.starmap(h5tofits, tasks)
