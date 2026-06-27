"""observed_sectors.py — given a TIC ID (or many), list the TESS sectors it was observed in.

Uses tess-point (`tess_stars2px`) which knows the TESS pointing law through ~S96, so given a
target's RA/Dec it returns EVERY sector the target falls on a CCD, with camera/CCD. This is a
*pointing prediction* — for "was it (re)observed?" it is authoritative (TESS pointing is
deterministic). It does NOT tell you whether a given pipeline (QLP/TGLC) actually produced a
light curve for it — that's a separate data-access question.

TIC -> RA/Dec is resolved with, in order of preference:
  1. local  : the training-set table data/training_set_enriched.csv (offline, instant; covers the
              ~15k training TICs only).
  2. mast   : astroquery MAST TIC catalog (batched; needs internet — confirmed reachable from the
              PDO node where this was written).

Env: /pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python3  (has tess_stars2px +
astroquery). NO pip installs needed. (pyticdb is NOT in this env; use the QLP env if you prefer it.)

Examples
--------
  PY=/pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python3
  # one TIC
  $PY observed_sectors.py --tic 261136679
  # several
  $PY observed_sectors.py --tic 261136679,77031414 --since 94
  # the whole training set (offline via the enriched CSV), summarize + write CSV
  $PY observed_sectors.py --tic-file my_15k_tics.txt --since 94 --out observed.csv
  # the training set straight from the enriched CSV (no separate TIC file needed)
  $PY observed_sectors.py --from-training --since 94 --out training_reobserved.csv

Output columns: tic, ra, dec, n_sectors, min_sector, max_sector, observed_sectors,
                sectors_since_<N>, observed_since_<N>, resolver
"""
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd

ENRICHED = "/pdo/users/pablomer/TGLC_adaptation/data/training_set_enriched.csv"
# NOTE: the first TGLC *production* sector is not yet confirmed (Pablo wrote 94). Override with
# --since. tess_stars2px in this env knows pointings up to ~S96.
DEFAULT_SINCE = 94


def load_local_coords():
    """TIC -> (ra,dec) from the enriched training-set table. Returns {} if absent."""
    if not os.path.exists(ENRICHED):
        return {}
    df = pd.read_csv(ENRICHED, usecols=["TIC ID", "RA", "Dec"]).dropna()
    return {int(t): (float(ra), float(dec))
            for t, ra, dec in zip(df["TIC ID"], df["RA"], df["Dec"])}


def resolve_mast(tics):
    """TIC -> (ra,dec) via astroquery MAST TIC catalog (batched). {} entries for misses."""
    from astroquery.mast import Catalogs
    out = {}
    tics = list(dict.fromkeys(int(t) for t in tics))
    CHUNK = 500
    for i in range(0, len(tics), CHUNK):
        chunk = tics[i:i + CHUNK]
        try:
            r = Catalogs.query_criteria(catalog="Tic", ID=chunk)
            for row in r:
                out[int(row["ID"])] = (float(row["ra"]), float(row["dec"]))
        except Exception as e:
            print(f"  [mast] chunk {i//CHUNK} failed: {e}", file=sys.stderr)
    return out


def resolve_coords(tics, resolver):
    coords, src = {}, {}
    if resolver in ("local", "auto"):
        local = load_local_coords()
        for t in tics:
            if int(t) in local:
                coords[int(t)] = local[int(t)]; src[int(t)] = "local"
    missing = [int(t) for t in tics if int(t) not in coords]
    if missing and resolver in ("mast", "auto"):
        print(f"Resolving {len(missing)} TIC(s) via MAST ...", file=sys.stderr)
        m = resolve_mast(missing)
        for t, rd in m.items():
            coords[t] = rd; src[t] = "mast"
    return coords, src


def observed_sectors(coords, chunk=1000, progress=True):
    """coords: {tic:(ra,dec)} -> DataFrame of per-(tic,sector) cam/ccd rows.

    The tess-point call is the slow part (~15s per 1000 targets, so a few minutes
    for the 15k training set). We process in chunks and print a progress line with
    an ETA to stderr so a big run isn't a silent multi-minute wait.
    """
    from tess_stars2px import tess_stars2px_function_entry as f
    tics_all = np.array(list(coords.keys()), dtype=np.int64)
    ra_all = np.array([coords[t][0] for t in tics_all])
    dec_all = np.array([coords[t][1] for t in tics_all])
    n = len(tics_all)
    parts, t0 = [], time.time()
    for i in range(0, n, chunk):
        ti, ra, dec = tics_all[i:i+chunk], ra_all[i:i+chunk], dec_all[i:i+chunk]
        oid, _, _, sec, cam, ccd, _, _, _ = f(ti, ra, dec)
        parts.append(pd.DataFrame({"tic": oid.astype(np.int64), "sector": sec.astype(int),
                                   "cam": cam.astype(int), "ccd": ccd.astype(int)}))
        if progress and n > chunk:
            done = min(i + chunk, n)
            el = time.time() - t0
            eta = el / done * (n - done)
            print(f"  [tess-point] {done:6d}/{n} ({done/n:4.0%})  "
                  f"elapsed {el:5.1f}s  eta {eta:5.1f}s", file=sys.stderr)
    return (pd.concat(parts, ignore_index=True)
            .drop_duplicates(["tic", "sector"]).sort_values(["tic", "sector"]))


def summarize(per_sec, coords, src, since):
    rows = []
    for tic, (ra, dec) in coords.items():
        s = sorted(per_sec.loc[per_sec.tic == tic, "sector"].tolist())
        since_list = [x for x in s if x >= since]
        rows.append({
            "tic": tic, "ra": ra, "dec": dec,
            "n_sectors": len(s), "min_sector": min(s) if s else None,
            "max_sector": max(s) if s else None,
            "observed_sectors": ",".join(map(str, s)),
            f"sectors_since_{since}": ",".join(map(str, since_list)),
            f"observed_since_{since}": bool(since_list),
            "resolver": src.get(tic, "?"),
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--tic", help="one TIC or comma-separated list")
    g.add_argument("--tic-file", help="file with one TIC per line")
    g.add_argument("--from-training", action="store_true",
                   help="use every TIC in the enriched training-set table")
    g.add_argument("--radec", nargs=2, type=float, metavar=("RA", "DEC"),
                   help="skip TIC resolution; query a single RA/Dec directly")
    p.add_argument("--since", type=int, default=DEFAULT_SINCE,
                   help=f"flag targets observed in sector >= this (default {DEFAULT_SINCE})")
    p.add_argument("--resolver", choices=["auto", "local", "mast"], default="auto",
                   help="TIC->coord backend (default auto: local CSV then MAST)")
    p.add_argument("--out", help="write per-TIC summary CSV here")
    args = p.parse_args()

    if args.radec:
        coords = {0: (args.radec[0], args.radec[1])}; src = {0: "radec"}
    else:
        if args.from_training:
            tics = list(load_local_coords().keys())
        elif args.tic_file:
            tics = [int(x) for x in np.loadtxt(args.tic_file, dtype=np.int64, ndmin=1)]
        else:
            tics = [int(x) for x in args.tic.split(",")]
        print(f"{len(tics)} TIC(s) requested", file=sys.stderr)
        coords, src = resolve_coords(tics, args.resolver)
        unresolved = [t for t in tics if int(t) not in coords]
        if unresolved:
            print(f"WARNING: {len(unresolved)} TIC(s) unresolved (no coords): "
                  f"{unresolved[:10]}{'...' if len(unresolved) > 10 else ''}", file=sys.stderr)
    if not coords:
        sys.exit("No coordinates resolved; nothing to do.")

    per_sec = observed_sectors(coords)
    summ = summarize(per_sec, coords, src, args.since)

    # console output
    pd.set_option("display.max_colwidth", 80)
    cols = ["tic", "n_sectors", "min_sector", "max_sector",
            f"observed_since_{args.since}", "observed_sectors"]
    print(summ[cols].to_string(index=False))
    n_since = int(summ[f"observed_since_{args.since}"].sum())
    print(f"\n{n_since}/{len(summ)} ({n_since/len(summ):.1%}) observed in sector >= {args.since}",
          file=sys.stderr)

    if args.out:
        summ.to_csv(args.out, index=False)
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
