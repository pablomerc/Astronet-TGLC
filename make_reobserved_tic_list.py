"""make_reobserved_tic_list.py — build the input list + companion CSV for the
reprocessed (QLP+TGLC-through-S103) training subset.

We only reprocess training targets that were actually re-observed with a light
curve since TGLC went into production at S94. That set is column ``reobs103_lc``
in ``data/training_reobserved_s103.csv`` (~5,781 unique TICs).

Outputs (both small, git-tracked, written to data/):
  1. reobserved_s103_tics.txt      — one TIC per line, the ``--inlist`` for db_to_fits.py
  2. reobserved_s103_companion.csv — the dec2025 labeled TCE rows for those TICs,
       with the ``File`` column rewritten to the new qlptglc FITS name, so the later
       TFRecord step (generate_input_records_3.py) can point straight at the new FITS.

This script needs only pandas/numpy — run it in any python (no qlp/DB needed):
  python3 make_reobserved_tic_list.py
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REOBS_CSV = os.path.join(HERE, "data", "training_reobserved_s103.csv")
# The labeled dec2025 TCE table (one row per TCE; multiple TCEs share a TIC).
DEC2025_ALL = ("/pdo/users/pablomer/mnt/tess/astronet/"
               "tces-vetting-v01-tois-triageJs-nocentroid-dec2025-all.csv")

# Must match db_to_fits.py::construct_fitsfile
FITS_TMPL = "astronet_hlsp_qlptglc_tess_ffi-s%04d-%016d_tess_v01_llc.fits"


def new_fits_name(tic: int, sector: int) -> str:
    return FITS_TMPL % (sector, int(tic))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reobs-csv", default=REOBS_CSV,
                   help="training_reobserved_s103.csv (has the reobs103_lc flag)")
    p.add_argument("--dec2025-all", default=DEC2025_ALL,
                   help="labeled dec2025 TCE table to build the companion CSV from")
    p.add_argument("--flag", default="reobs103_lc",
                   choices=["reobs103_lc", "reobs103_geom"],
                   help="which reobservation flag selects the subset (default reobs103_lc)")
    p.add_argument("--sector", "-s", type=int, default=103,
                   help="cutoff sector baked into the FITS filename (default 103)")
    p.add_argument("--outdir", default=os.path.join(HERE, "data"),
                   help="where to write the tic list + companion csv")
    p.add_argument("--no-splits", action="store_true",
                   help="skip writing train/val/test companion CSVs")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    reobs = pd.read_csv(args.reobs_csv)
    if args.flag not in reobs.columns:
        raise SystemExit(f"'{args.flag}' not in {args.reobs_csv} "
                         f"(columns: {list(reobs.columns)})")
    # robust truthiness (values may be True/False or 'True'/'False')
    sel = reobs[args.flag].astype(str).str.strip().str.lower().isin(("true", "1"))
    tics = sorted(int(t) for t in reobs.loc[sel, "tic"].unique())
    print(f"{args.flag}=True -> {len(tics)} unique TICs")

    tic_path = os.path.join(args.outdir, "reobserved_s103_tics.txt")
    np.savetxt(tic_path, np.array(tics, dtype=np.int64), fmt="%d")
    print(f"wrote {tic_path}  ({len(tics)} TICs)")

    # Companion CSV: dec2025 TCE rows for these TICs, File rewritten to new FITS.
    dec = pd.read_csv(args.dec2025_all)
    tic_col = "TIC ID" if "TIC ID" in dec.columns else "tic"
    tic_set = set(tics)
    comp = dec[dec[tic_col].astype("int64").isin(tic_set)].copy()
    comp["File"] = comp[tic_col].astype("int64").map(lambda t: new_fits_name(t, args.sector))
    comp_path = os.path.join(args.outdir, "reobserved_s103_companion.csv")
    comp.to_csv(comp_path, index=False)
    n_tce = len(comp)
    n_tic = comp[tic_col].nunique()
    print(f"wrote {comp_path}  ({n_tce} TCEs across {n_tic} TICs)")
    if "first_letter" in comp.columns:
        print("  label breakdown (first_letter):")
        print(comp["first_letter"].value_counts().to_string())
    # A few TICs in the list may be absent from dec2025-all (dedup/edge cases) — report.
    missing = tic_set - set(comp[tic_col].astype("int64").unique())
    if missing:
        print(f"  NOTE: {len(missing)} selected TICs had no row in dec2025-all "
              f"(no labels/params); they are still in the .txt list.")

    # Split the companion into train/val/test EXACTLY as in the original dec2025 split,
    # matched per-TCE by "Astro ID" (the original split is by TCE, not strictly by TIC:
    # ~52 TICs have TCEs in >1 split). This keeps the reprocessed set comparable to how
    # the current model was trained/evaluated. To de-leak (force a TIC into one split),
    # do that as a separate, explicit step.
    if not args.no_splits and "Astro ID" in comp.columns:
        comp_aid = comp.set_index("Astro ID")
        print("split (by original dec2025 assignment, matched on Astro ID):")
        covered = set()
        for sp in ("train", "val", "test"):
            sp_csv = args.dec2025_all.replace("-all.csv", f"-{sp}.csv")
            if not os.path.exists(sp_csv):
                print(f"  WARN: missing {sp_csv}; skipping {sp}")
                continue
            sp_aid = pd.read_csv(sp_csv, usecols=["Astro ID"])["Astro ID"]
            keep = comp_aid.index.intersection(sp_aid)
            out = comp_aid.loc[keep].reset_index()
            covered |= set(keep)
            out_path = os.path.join(args.outdir, f"reobserved_s103_companion_{sp}.csv")
            out.to_csv(out_path, index=False)
            print(f"  wrote {out_path}  ({len(out)} TCEs, {out[tic_col].nunique()} TICs)")
        leftover = set(comp["Astro ID"]) - covered
        if leftover:
            print(f"  NOTE: {len(leftover)} companion TCEs were in no split file.")


if __name__ == "__main__":
    main()
