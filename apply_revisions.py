"""apply_revisions.py — merge manual gallery picks into the master revision table.

Takes the period_revisions.csv exported by the flag gallery (columns:
astro_id,tic,choice,chosen_period) and applies it to data/reobserved_s103_revised.csv:

  choice in {P0, 2xP0, P0/2, BLS, 2xBLS, 4xBLS, BLS/2, custom}
      -> Per <- chosen_period, revised=YES, revision_source=manual_inspection
     (P0 = "original period confirmed fine": Per stays equal to Per_original, but the row
      still becomes revised=YES so we know a human checked it.)
  choice == unsure
      -> revised=DISCARD, revision_source=manual_inspection, Per untouched.
         Meaning: a human looked and could NOT determine a usable period — exclude these
         rows from training (filter revised!="DISCARD" downstream). A later export that
         picks a real period for the same target upgrades DISCARD -> YES.

Catalog-revised rows shown in the gallery's "auto-corrected" view: an ADP pick confirms the
adopted period (no change); any other pick OVERRIDES it and is labeled
revision_source=manual_override (so overridden catalog adoptions are auditable);
unsure on a catalog row -> revised=DISCARD with manual_override.
Epoc is left unchanged for manual picks (folding uses the period; the epoch still marks
one of the eclipses — at 2x it may be the secondary, which the TFRecord step can tolerate
or re-derive).

  PY=/pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python3
  $PY apply_revisions.py ~/Downloads/period_revisions.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.join(HERE, "data", "reobserved_s103_revised.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("revisions", help="period_revisions.csv exported from the gallery")
    p.add_argument("--master", default=MASTER)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    m = pd.read_csv(args.master)
    m["revised"] = m["revised"].fillna("NA")   # pandas reads the literal "NA" as NaN
    rev = pd.read_csv(args.revisions)
    need = {"astro_id", "choice", "chosen_period"}
    if not need.issubset(rev.columns):
        raise SystemExit(f"revisions file must have columns {need}, got {list(rev.columns)}")

    idx_by_aid = {int(a): i for i, a in enumerate(m["Astro ID"])}
    applied = confirmed = unsure = missing = bad = 0
    for _, r in rev.iterrows():
        aid = int(r["astro_id"])
        i = idx_by_aid.get(aid)
        if i is None:
            missing += 1
            continue
        choice = str(r["choice"])
        was_catalog = (m.at[i, "revised"] == "YES"
                       and str(m.at[i, "revision_source"]).startswith("catalog"))
        if choice == "unsure":
            # latest human decision wins: unsure always -> DISCARD. Labeled manual_override
            # when it displaces an existing YES (catalog or manual), so it's auditable.
            prev = m.at[i, "revised"]
            if prev != "DISCARD":
                m.at[i, "revised"] = "DISCARD"
                m.at[i, "revision_source"] = ("manual_override" if prev == "YES"
                                              else "manual_inspection")
            unsure += 1
            continue
        if choice == "ADP":                          # gallery confirm of the adopted catalog period
            confirmed += 1
            continue
        per = pd.to_numeric(r["chosen_period"], errors="coerce")
        if not (np.isfinite(per) and per > 0):
            bad += 1
            continue
        m.at[i, "Per"] = float(per)
        m.at[i, "revised"] = "YES"
        # special label when a human OVERRIDES a catalog-adopted period
        m.at[i, "revision_source"] = "manual_override" if was_catalog else "manual_inspection"
        applied += 1

    print(f"applied={applied}  confirmed(ADP)={confirmed}  unsure->DISCARD={unsure}  "
          f"missing_astro_id={missing}  bad_period={bad}")
    eb = m[m["first_letter"] == "e"]
    print(f"master now: YES={int((eb['revised']=='YES').sum())}  "
          f"DISCARD={int((eb['revised']=='DISCARD').sum())}  "
          f"NO={int((eb['revised']=='NO').sum())}  / {len(eb)} EB TCEs")
    if args.dry_run:
        print("(dry-run: master NOT written)")
    else:
        m.to_csv(args.master, index=False)
        print(f"-> {args.master}")


if __name__ == "__main__":
    main()
