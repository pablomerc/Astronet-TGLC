"""crossmatch_periods.py — Step 1 of the EB period revision: catalog crossmatch.

Creates/updates the master revision table data/reobserved_s103_revised.csv:
  * a full copy of data/reobserved_s103_companion.csv (all 5,834 TCE rows, all columns), plus
      revised          : NA (non-EB) | NO (EB, not yet corrected) | YES (corrected)
      Per_original / Epoc_original : preserved originals (every row)
      Per / Epoc       : OVERWRITTEN in place with revised values when revised==YES
      revision_source  : catalog:<vsx|villanova|gaia|toi> or manual_inspection
      single_sector    : True if the ORIGINAL training file was single-sector (mk_*)
      risk_score       : filled by scan_periods.py (step 2)
      catalog_name / catalog_period / catalog_ratio : best available catalog info (even if not adopted)
      P_candidates     : comma list of candidate periods for the manual gallery (P0, 2P0, P0/2 [+BLS])

Crossmatch (EBs only, first_letter=='e'), query order = coverage/authority:
  1. VSX      VizieR B/vsx/vsx           cone 5", keep Type =~ ^E (EA/EB/EW/E/ED/ESD/ELL), nearest
  2. Villanova/Prsa22  VizieR J/ApJS/258/16/tess-ebs   local join on TIC  (Per, BJD0)
  3. Gaia DR3 gaiadr3.vari_eclipsing_binary (+gaia_source)  positional join 5", period=1/frequency
  4. TOI      ExoArchive TAP toi          join on tid  (pl_orbper, pl_tranmid)

ADOPT RULE (user-confirmed): a catalog period is taken as ground truth iff it equals our P, 2P, or
P/2 within --ratio-tol (default 1%). Then Per <- catalog period, Epoc <- catalog epoch when provided
(converted to BTJD: full JD values get -2457000), revised=YES, revision_source=catalog:<name>.
An unrelated catalog period is recorded (catalog_*) but NOT adopted -> stays NO (manual gallery).

Raw catalog pulls are cached in data/catalog_cache/ so re-runs are offline. Re-running never
clobbers rows already revised==YES. Use --rebuild to recreate the master from the companion
(preserving nothing).

Env: conda daniel_env_cloned_v2 (astroquery/astropy/pandas). ~10-20 min first run (network).
  PY=/pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python3
  $PY crossmatch_periods.py
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
COMPANION = os.path.join(HERE, "data", "reobserved_s103_companion.csv")
MASTER = os.path.join(HERE, "data", "reobserved_s103_revised.csv")
DEC2025_ALL = ("/pdo/users/pablomer/mnt/tess/astronet/"
               "tces-vetting-v01-tois-triageJs-nocentroid-dec2025-all.csv")
CACHE = os.path.join(HERE, "data", "catalog_cache")
REPORT = os.path.join(HERE, "data", "period_crossmatch_report.txt")

NEW_COLS = ["revised", "Per_original", "Epoc_original", "revision_source",
            "single_sector", "risk_score", "catalog_name", "catalog_period",
            "catalog_ratio", "P_candidates"]


# --------------------------------------------------------------------------- master table
def build_master(companion=COMPANION, dec_all=DEC2025_ALL):
    comp = pd.read_csv(companion)
    m = comp.copy()
    m["Per_original"] = m["Per"]
    m["Epoc_original"] = m["Epoc"]
    m["revised"] = np.where(m["first_letter"] == "e", "NO", "NA")
    m["revision_source"] = ""
    # single_sector: the ORIGINAL dec2025 file name (companion File was rewritten to qlptglc)
    dec = pd.read_csv(dec_all, usecols=["Astro ID", "File"])
    orig_file = dec.set_index("Astro ID")["File"]
    m["single_sector"] = m["Astro ID"].map(orig_file).astype(str).str.startswith("mk_")
    m["risk_score"] = np.nan
    m["catalog_name"] = ""
    m["catalog_period"] = np.nan
    m["catalog_ratio"] = np.nan
    m["P_candidates"] = ""
    eb = m["first_letter"] == "e"
    m.loc[eb, "P_candidates"] = m.loc[eb, "Per"].map(
        lambda p: f"{p:.6f},{2*p:.6f},{0.5*p:.6f}" if np.isfinite(p) else "")
    return m


def load_or_build_master(rebuild=False):
    if rebuild or not os.path.exists(MASTER):
        m = build_master()
        print(f"[master] built fresh from companion ({len(m)} rows)")
    else:
        m = pd.read_csv(MASTER)
        m["revised"] = m["revised"].fillna("NA")   # pandas reads literal "NA" as NaN
        missing = [c for c in NEW_COLS if c not in m.columns]
        if missing:
            sys.exit(f"[master] existing file missing columns {missing}; run with --rebuild")
        print(f"[master] loaded existing ({len(m)} rows; "
              f"revised YES={int((m['revised']=='YES').sum())})")
    return m


# --------------------------------------------------------------------------- catalog pulls (cached)
def _cache_path(name):
    os.makedirs(CACHE, exist_ok=True)
    return os.path.join(CACHE, f"{name}.csv")


def _cached(name, fetch_fn, refresh=False):
    p = _cache_path(name)
    if os.path.exists(p) and not refresh:
        df = pd.read_csv(p)
        print(f"[{name}] cache hit: {len(df)} rows ({p})")
        return df
    t0 = time.time()
    df = fetch_fn()
    df.to_csv(p, index=False)
    print(f"[{name}] fetched {len(df)} rows in {time.time()-t0:.0f}s -> {p}")
    return df


def fetch_vsx(eb):
    """One batched VizieR cone query for all EB coordinates. Returns per-match rows with tidx."""
    from astroquery.vizier import Vizier
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    coords = SkyCoord(ra=eb["RA"].values * u.deg, dec=eb["Dec"].values * u.deg)
    v = Vizier(catalog="B/vsx/vsx", columns=["Name", "Type", "Period", "Epoch"], row_limit=-1)
    v.TIMEOUT = 600
    res = v.query_region(coords, radius=5 * u.arcsec)
    if not res:
        return pd.DataFrame(columns=["tidx", "Name", "Type", "Period", "Epoch", "_r"])
    t = res[0].to_pandas()
    t["tidx"] = t["_q"].astype(int) - 1          # 1-based index into the input coord vector
    keep = ["tidx", "Name", "Type", "Period", "Epoch"] + (["_r"] if "_r" in t.columns else [])
    return t[keep]


def fetch_villanova():
    from astroquery.vizier import Vizier
    v = Vizier(catalog="J/ApJS/258/16/tess-ebs", columns=["TIC", "Per", "e_Per", "BJD0", "Morph"],
               row_limit=-1)
    v.TIMEOUT = 600
    t = v.query_constraints()[0].to_pandas()
    return t


def fetch_gaia(eb, chunk=50):   # >~100 OR-conditions makes the TAP endpoint 500
    """Chunked positional join gaia_source x vari_eclipsing_binary (5 arcsec)."""
    import requests, io as _io
    rows = []
    r_deg = 5.0 / 3600.0
    recs = list(zip(eb["RA"].values, eb["Dec"].values))
    for i in range(0, len(recs), chunk):
        part = recs[i:i + chunk]
        conds = " OR ".join(
            f"1=CONTAINS(POINT('ICRS',g.ra,g.dec),CIRCLE('ICRS',{ra:.6f},{dec:.6f},{r_deg:.7f}))"
            for ra, dec in part)
        q = ("SELECT g.source_id,g.ra,g.dec,e.frequency "
             "FROM gaiadr3.gaia_source g JOIN gaiadr3.vari_eclipsing_binary e "
             "ON g.source_id=e.source_id WHERE " + conds)
        for attempt in range(3):
            try:
                resp = requests.post("https://gea.esac.esa.int/tap-server/tap/sync",
                                     data={"REQUEST": "doQuery", "LANG": "ADQL",
                                           "FORMAT": "csv", "QUERY": q}, timeout=300)
                if resp.status_code == 200:
                    part_df = pd.read_csv(_io.StringIO(resp.text))
                    rows.append(part_df)
                    break
                print(f"  [gaia] chunk {i//chunk}: HTTP {resp.status_code} (try {attempt+1})",
                      file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                print(f"  [gaia] chunk {i//chunk}: {e} (try {attempt+1})", file=sys.stderr)
            time.sleep(3)
        print(f"  [gaia] {min(i+chunk,len(recs))}/{len(recs)} targets queried", file=sys.stderr)
    if not rows:
        return pd.DataFrame(columns=["source_id", "ra", "dec", "frequency"])
    g = pd.concat(rows, ignore_index=True).drop_duplicates("source_id")
    return g


def fetch_toi():
    import requests, io as _io
    r = requests.get("https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
                     params={"query": "select tid,toi,pl_orbper,pl_tranmid,tfopwg_disp from toi",
                             "format": "csv"}, timeout=180)
    return pd.read_csv(_io.StringIO(r.text))


# --------------------------------------------------------------------------- matching helpers
def to_btjd(epoch):
    """Convert a catalog epoch to BTJD (BJD-2457000). Full-JD values get the offset removed;
    already-small values are assumed BTJD-like. Negative results are fine for folding."""
    try:
        e = float(epoch)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(e) or e == 0:
        return np.nan
    if e > 2_400_000:
        return e - 2_457_000.0
    if 0 < e < 20_000:
        return e
    return np.nan


def ratio_match(p_cat, p_ours, tol):
    """Return the matched multiple (1, 2, 0.5) if p_cat ~= mult*p_ours within tol, else None."""
    if not (np.isfinite(p_cat) and np.isfinite(p_ours)) or p_cat <= 0 or p_ours <= 0:
        return None
    for mult in (1.0, 2.0, 0.5):
        if abs(p_cat / (mult * p_ours) - 1.0) < tol:
            return mult
    return None


def build_lookups(eb, refresh=False):
    """Per-catalog lookup: eb-row-position -> (period, epoch_btjd). Also returns raw info map."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    lookups = {}

    # --- VSX (positional; prefer eclipsing types, then nearest)
    vsx = _cached("vsx", lambda: fetch_vsx(eb), refresh)
    lk = {}
    if len(vsx):
        vsx = vsx.copy()
        vsx["is_ecl"] = vsx["Type"].astype(str).str.match(r"^E")
        vsx["Period"] = pd.to_numeric(vsx["Period"], errors="coerce")
        vsx = vsx[np.isfinite(vsx["Period"]) & (vsx["Period"] > 0)]
        sortcols = ["is_ecl"] + (["_r"] if "_r" in vsx.columns else [])
        vsx = vsx.sort_values(sortcols, ascending=[False] + [True] * (len(sortcols) - 1))
        for tidx, grp in vsx.groupby("tidx"):
            best = grp.iloc[0]
            if bool(best["is_ecl"]):                      # only adopt eclipsing-type periods
                lk[int(tidx)] = (float(best["Period"]), to_btjd(best.get("Epoch")))
    lookups["vsx"] = lk

    # --- Villanova (TIC join)
    vil = _cached("villanova", fetch_villanova, refresh)
    vil["Per"] = pd.to_numeric(vil["Per"], errors="coerce")
    vil = vil[np.isfinite(vil["Per"]) & (vil["Per"] > 0)].drop_duplicates("TIC")
    vmap = vil.set_index(vil["TIC"].astype(np.int64))
    tics = eb["TIC ID"].astype(np.int64).values
    lk = {}
    for i, tic in enumerate(tics):
        if tic in vmap.index:
            r = vmap.loc[tic]
            lk[i] = (float(r["Per"]), to_btjd(r.get("BJD0")))
    lookups["villanova"] = lk

    # --- Gaia (positional; nearest within 5")
    gaia = _cached("gaia", lambda: fetch_gaia(eb), refresh)
    lk = {}
    if len(gaia):
        gc = SkyCoord(ra=gaia["ra"].values * u.deg, dec=gaia["dec"].values * u.deg)
        ec = SkyCoord(ra=eb["RA"].values * u.deg, dec=eb["Dec"].values * u.deg)
        idx, sep, _ = gc.match_to_catalog_sky(ec)          # for each gaia row: nearest EB target
        for gi, (ti, s) in enumerate(zip(idx, sep.arcsec)):
            if s <= 5.0:
                freq = float(gaia["frequency"].iloc[gi])
                if np.isfinite(freq) and freq > 0:
                    prev = lk.get(int(ti))
                    cand = (1.0 / freq, np.nan)
                    if prev is None or s < prev[2]:
                        lk[int(ti)] = (cand[0], cand[1], float(s))
        lk = {k: (v[0], v[1]) for k, v in lk.items()}
    lookups["gaia"] = lk

    # --- TOI (TIC join; period must ratio-gate anyway)
    toi = _cached("toi", fetch_toi, refresh)
    toi["pl_orbper"] = pd.to_numeric(toi["pl_orbper"], errors="coerce")
    toi = toi[np.isfinite(toi["pl_orbper"]) & (toi["pl_orbper"] > 0)]
    lk = {}
    by_tid = toi.groupby(toi["tid"].astype(np.int64))
    for i, tic in enumerate(tics):
        if tic in by_tid.groups:
            grp = by_tid.get_group(tic)
            lk[i] = [(float(r["pl_orbper"]), to_btjd(r.get("pl_tranmid"))) for _, r in grp.iterrows()]
    lookups["toi"] = lk
    return lookups


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rebuild", action="store_true", help="recreate the master CSV from the companion")
    p.add_argument("--refresh-catalogs", action="store_true", help="ignore catalog caches, refetch")
    p.add_argument("--ratio-tol", type=float, default=0.01, help="1x/2x/0.5x match tolerance (default 1%%)")
    args = p.parse_args()

    m = load_or_build_master(args.rebuild)
    eb_mask = m["first_letter"] == "e"
    eb = m[eb_mask].reset_index()                    # 'index' = row position in master
    print(f"[xmatch] {len(eb)} EB TCEs ({eb['TIC ID'].nunique()} TICs); "
          f"already revised: {int((eb['revised']=='YES').sum())}")

    lookups = build_lookups(eb, refresh=args.refresh_catalogs)

    order = ["vsx", "villanova", "gaia", "toi"]
    stats = {c: {"match": 0, "gated": 0, "adopted": 0} for c in order}
    matched_any, gated_any = set(), set()

    for i, row in eb.iterrows():
        midx = row["index"]
        p0 = float(row["Per_original"]) if np.isfinite(row["Per_original"]) else np.nan
        recorded = False
        for cat in order:
            lk = lookups[cat].get(i)
            if lk is None:
                continue
            cands = lk if isinstance(lk, list) else [lk]
            stats[cat]["match"] += 1
            matched_any.add(i)
            hit = None
            for (pcat, ecat) in cands:
                mult = ratio_match(pcat, p0, args.ratio_tol)
                if mult is not None:
                    hit = (pcat, ecat, mult)
                    break
            if not recorded:                          # record best-available info regardless
                pcat0 = cands[0][0]
                m.loc[midx, "catalog_name"] = cat
                m.loc[midx, "catalog_period"] = pcat0
                m.loc[midx, "catalog_ratio"] = (pcat0 / p0) if (np.isfinite(p0) and p0 > 0) else np.nan
                recorded = True
            if hit is None:
                continue
            stats[cat]["gated"] += 1
            gated_any.add(i)
            if m.loc[midx, "revised"] == "YES":       # never clobber an existing revision
                break
            pcat, ecat, mult = hit
            m.loc[midx, "Per"] = pcat
            if np.isfinite(ecat):
                m.loc[midx, "Epoc"] = ecat
            m.loc[midx, "revised"] = "YES"
            m.loc[midx, "revision_source"] = f"catalog:{cat}"
            m.loc[midx, "catalog_name"] = cat
            m.loc[midx, "catalog_period"] = pcat
            m.loc[midx, "catalog_ratio"] = pcat / p0 if (np.isfinite(p0) and p0 > 0) else np.nan
            stats[cat]["adopted"] += 1
            break                                     # priority order: stop at first gated catalog

    m.to_csv(MASTER, index=False)

    eb2 = m[eb_mask]
    n = len(eb2)
    lines = []
    lines.append("EB period crossmatch report — %s" % time.strftime("%Y-%m-%d %H:%M"))
    lines.append("EB TCEs: %d (%d unique TICs) | ratio tol ±%.1f%% at 1x/2x/0.5x"
                 % (n, eb2["TIC ID"].nunique(), 100 * args.ratio_tol))
    lines.append("")
    lines.append("%-11s %8s %8s %8s" % ("catalog", "matched", "gated", "adopted"))
    for c in order:
        s = stats[c]
        lines.append("%-11s %8d %8d %8d" % (c, s["match"], s["gated"], s["adopted"]))
    lines.append("")
    nyes = int((eb2["revised"] == "YES").sum())
    lines.append("any-catalog match : %d/%d (%.1f%%)" % (len(matched_any), n, 100 * len(matched_any) / n))
    lines.append("ratio-gated       : %d/%d (%.1f%%)" % (len(gated_any), n, 100 * len(gated_any) / n))
    lines.append("revised=YES       : %d/%d (%.1f%%)" % (nyes, n, 100 * nyes / n))
    lines.append("remaining manual  : %d  (of which single-sector: %d)"
                 % (int((eb2["revised"] == "NO").sum()),
                    int(((eb2["revised"] == "NO") & (eb2["single_sector"])).sum())))
    mult_changed = eb2[(eb2["revised"] == "YES") &
                       (np.abs(eb2["Per"] / eb2["Per_original"] - 1) > 0.02)]
    lines.append("adopted at 2x/0.5x (period actually changed >2%%): %d" % len(mult_changed))
    report = "\n".join(lines)
    print("\n" + report)
    with open(REPORT, "w") as f:
        f.write(report + "\n")
    print(f"\n[xmatch] master -> {MASTER}\n[xmatch] report -> {REPORT}")


if __name__ == "__main__":
    main()
