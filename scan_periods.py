"""scan_periods.py — Step 2 of the EB period revision: risk-score unrevised EBs.

For every EB TCE still `revised == NO` in data/reobserved_s103_revised.csv, measure how suspicious
its catalog period P0 is on the FULL reprocessed baseline (S1->S103 FITS from db_to_fits.py), using a
FAST per-segment median detrend (seconds/target — triage quality, not the 40 s astronet spline):

  P_bls    : BLS refinement of P0 (narrow +/-2% window on the detrended flux) + robust epoch t0.
  snr_P0   : folded eclipse depth SNR at exactly P0        (stale period -> low)
  snr_bls  : folded eclipse depth SNR at P_bls             (snr_bls >> snr_P0 -> P0 imprecise)
  dpri,dsec: primary vs secondary depth folding at 2*P_bls (both real but DISTINCT -> true period
             is ~2x, i.e. catalog holds the aliased HALF period — the TIC 364302118 failure mode)

risk_score (0..1+, higher = look at it sooner in the gallery):
  +0.45 half-period signature (distinct primary/secondary at 2x)
  +0.35 stale (snr_bls/snr_P0 > 1.5)   |  +0.20 mildly stale (>1.2)
  +0.30 weak signal at both periods (needs human eyes)
  +0.30 catalog period exists but did NOT ratio-gate (unrelated catalog period)
  +0.25 single_sector original (short baseline -> aliasing-prone; gallery has a toggle)

Writes per-TCE metrics to data/period_scan.csv (resumable: re-run skips done astro_ids) and, at the
end, merges risk_score into the master CSV and appends P_bls to its P_candidates.

Run AFTER crossmatch_periods.py (needs the master):
  PY=/pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python3
  $PY scan_periods.py -n 8          # ~10-20 min for ~1,200 unrevised EBs
"""
import argparse
import csv
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FITS_DIR = "/pdo/users/pablomer/mnt/tess/reprocessed_s103_qlptglc_fits_files"
MASTER = os.path.join(HERE, "data", "reobserved_s103_revised.csv")
SCAN_OUT = os.path.join(HERE, "data", "period_scan.csv")
SECTOR = 103

FIELDS = ["astro_id", "tic", "n_pts", "P0", "P_bls", "t0_bls", "snr_P0", "snr_bls",
          "dpri", "dsec", "oe_z", "half_period_sig", "stale_ratio", "weak", "catalog_unrelated",
          "single_sector", "risk_score", "flag", "note"]
_CFG = {}


def fits_path(tic, fits_dir=FITS_DIR):
    return os.path.join(fits_dir, "astronet_hlsp_qlptglc_tess_ffi-s%04d-%016d_tess_v01_llc.fits"
                        % (SECTOR, int(tic)))


def read_flux(tic, fits_dir=FITS_DIR):
    from astropy.io import fits
    with fits.open(fits_path(tic, fits_dir)) as h:
        d = h[1].data
        q = d["QUALITY"] == 0
        t = np.asarray(d["TIME"][q], float)
        f = np.asarray(d["SAP_FLUX"][q], float)
    g = np.isfinite(t) & np.isfinite(f)
    t, f = t[g], f[g]
    o = np.argsort(t)
    return t[o], f[o]


def fast_detrend(t, f, gap=0.5, bin_days=1.0):
    """Per-segment (split on >gap-day gaps) running 1-day-bin median -> interp -> divide.
    Triage-quality: flattens cross-sector offsets + slow variability, preserves ~hours eclipses."""
    out = np.empty_like(f)
    edges = np.where(np.diff(t) > gap)[0]
    starts = np.r_[0, edges + 1]
    ends = np.r_[edges + 1, len(t)]
    for s, e in zip(starts, ends):
        ts, fs = t[s:e], f[s:e]
        med = np.nanmedian(fs)
        if e - s < 20 or ts[-1] - ts[0] < 2 * bin_days:
            out[s:e] = fs / (med if med > 0 else 1.0)
            continue
        nb = int(np.ceil((ts[-1] - ts[0]) / bin_days)) + 1
        bins = np.linspace(ts[0] - 1e-6, ts[-1] + 1e-6, nb + 1)
        idx = np.digitize(ts, bins) - 1
        bt, bf = [], []
        for b in range(nb):
            mb = idx == b
            if mb.sum() >= 5:
                bt.append(np.nanmedian(ts[mb]))
                bf.append(np.nanmedian(fs[mb]))
        trend = np.interp(ts, bt, bf) if len(bt) >= 2 else (med if med > 0 else 1.0)
        out[s:e] = fs / trend
    return out


def bin_series(t, f, cadence_days=1.0 / 48.0):
    """Bin to ~30-min medians (speeds BLS ~8x on TGLC-era data; eclipses are hours, they survive)."""
    nb = int(np.ceil((t[-1] - t[0]) / cadence_days)) + 1
    idx = ((t - t[0]) / cadence_days).astype(int)
    order = np.argsort(idx, kind="stable")
    idx_s, t_s, f_s = idx[order], t[order], f[order]
    cut = np.r_[0, np.flatnonzero(np.diff(idx_s)) + 1, len(idx_s)]
    bt = np.array([t_s[a:b].mean() for a, b in zip(cut[:-1], cut[1:])])
    bf = np.array([np.median(f_s[a:b]) for a, b in zip(cut[:-1], cut[1:])])
    return bt, bf


def _profile(t, f, P, t0, nbins=201):
    """Binned-median folded profile (primary ~ phase 0). Returns (centers, medians, bin scatter)."""
    ph = (((t - t0) / P + 0.5) % 1.0) - 0.5
    idx = np.clip(((ph + 0.5) * nbins).astype(int), 0, nbins - 1)
    order = np.argsort(idx, kind="stable")
    idx_s, f_s = idx[order], f[order]
    cut = np.r_[0, np.flatnonzero(np.diff(idx_s)) + 1, len(idx_s)]
    centers, meds = [], []
    for a, b in zip(cut[:-1], cut[1:]):
        if b - a >= 3:
            centers.append((idx_s[a] + 0.5) / nbins - 0.5)
            meds.append(np.median(f_s[a:b]))
    centers, meds = np.array(centers), np.array(meds)
    if len(meds) < 20:
        return centers, meds, np.nan
    out = np.abs(centers) > 0.25
    base = np.nanmedian(meds[out]) if out.sum() > 5 else np.nanmedian(meds)
    scat = np.nanmedian(np.abs(meds[out] - base)) * 1.4826 + 1e-9 if out.sum() > 5 else np.nan
    return centers, meds - base, scat        # profile relative to baseline


def fold_depth_snr(t, f, P, t0, dur):
    """Depth SNR of the deepest profile bin near phase 0 (bin-median profile beats window medians:
    a too-wide window would dilute a sharp eclipse to zero)."""
    if not (np.isfinite(P) and P > 0):
        return np.nan
    centers, prof, scat = _profile(t, f, P, t0)
    if not np.isfinite(scat) or not len(prof):
        return np.nan
    w = min(0.2, max((dur / P) if (np.isfinite(dur) and dur > 0) else 0.05, 0.02))
    near = np.abs(centers) < w
    if near.sum() < 2:
        return np.nan
    return float(-prof[near].min() / scat)


def odd_even(t, f, P, t0, dur):
    """Classic parity test at period P: if P is the aliased HALF of the true period, alternate
    epochs are really primary/secondary => unequal depths; if P is correct, odd == even.

    Robustness: the eclipse PHASE is located once on the combined profile (deepest bin near 0),
    then each parity's depth is the MEDIAN of its points at that same fixed phase. (Taking a
    per-parity profile MINIMUM instead is extreme-value-biased — half the points per parity means
    deeper random minima and spurious asymmetry.)"""
    if not (np.isfinite(P) and P > 0):
        return np.nan, np.nan, np.nan
    centers, prof, scat = _profile(t, f, P, t0)
    if not np.isfinite(scat) or not len(prof):
        return np.nan, np.nan, np.nan
    w = min(0.2, max((dur / P) if (np.isfinite(dur) and dur > 0) else 0.05, 0.02))
    near = np.abs(centers) < w
    if near.sum() < 2:
        return np.nan, np.nan, np.nan
    phi_min = float(centers[near][np.argmin(prof[near])])      # eclipse phase, from combined fold
    bw = max(w / 3.0, 1.0 / 201.0)

    ph = (((t - t0) / P + 0.5) % 1.0) - 0.5
    n = np.floor((t - t0) / P + 0.5).astype(np.int64)
    out = np.abs(ph) > 0.25
    base = np.nanmedian(f[out]) if out.sum() > 50 else np.nanmedian(f)
    sig_pt = (np.nanmedian(np.abs(f[out] - base)) * 1.4826 + 1e-9) if out.sum() > 50 else np.nan
    d, s = [], []
    for parity in (0, 1):
        sel = ((n % 2) == parity) & (np.abs(ph - phi_min) < bw)
        if sel.sum() < 10 or not np.isfinite(sig_pt):
            d.append(np.nan); s.append(np.nan)
            continue
        d.append(float(base - np.nanmedian(f[sel])))            # absolute flux depth
        s.append(float(1.2533 * sig_pt / np.sqrt(sel.sum())))   # sigma of that median
    if not (np.isfinite(d[0]) and np.isfinite(d[1])):
        return np.nan, np.nan, np.nan
    dpri = d[0] / s[0]                                           # per-parity depth significance
    dsec = d[1] / s[1]
    z_diff = abs(d[0] - d[1]) / np.sqrt(s[0] ** 2 + s[1] ** 2)   # significance of the DIFFERENCE
    return float(dpri), float(dsec), float(z_diff)


def _init_worker(fits_dir, min_pts):
    _CFG.update(fits_dir=fits_dir, min_pts=min_pts)


def process_tce(task):
    astro_id, tic, P0, dur, single_sector, catalog_unrelated = task
    row = {k: "" for k in FIELDS}
    row.update(astro_id=astro_id, tic=tic, P0=P0,
               single_sector=bool(single_sector), catalog_unrelated=bool(catalog_unrelated))
    try:
        if not (np.isfinite(P0) and P0 > 0):
            row["flag"] = "bad_period"
            row["risk_score"] = 1.0
            return row
        try:
            t, f = read_flux(tic, _CFG["fits_dir"])
        except Exception as e:  # noqa: BLE001
            row["flag"] = "no_fits"; row["note"] = repr(e)[:80]; row["risk_score"] = 0.5
            return row
        row["n_pts"] = len(t)
        if len(t) < _CFG["min_pts"]:
            row["flag"] = "low_data"; row["risk_score"] = 0.5
            return row
        fd = fast_detrend(t, f)
        bt, bf = bin_series(t, fd)               # ~30-min bins: BLS ~8x faster, eclipses survive

        from astropy.timeseries import BoxLeastSquares
        bls = BoxLeastSquares(bt, bf)
        dur_bls = float(np.clip(dur if (np.isfinite(dur) and dur > 0) else 0.08 * P0,
                                0.02, 0.4 * P0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # stage 1: coarse +/-2%; stage 2: fine around the coarse peak. The fine step is what
            # makes the fold phase-coherent over the ~2,700 d baseline (needs dP/P ~ 1e-6).
            g1 = np.linspace(0.98 * P0, 1.02 * P0, 500)
            g1 = g1[g1 > 0.05]
            r1 = bls.power(g1, dur_bls)
            p1 = float(g1[int(np.argmax(r1.power))])
            span = 2.0 * (g1[1] - g1[0])
            g2 = np.linspace(p1 - span, p1 + span, 800)
            g2 = g2[g2 > 0.05]
            r2 = bls.power(g2, dur_bls)
        j = int(np.argmax(r2.power))
        P_bls = float(g2[j]); t0 = float(r2.transit_time[j])

        snr_P0 = fold_depth_snr(bt, bf, P0, t0, dur)
        snr_bls = fold_depth_snr(bt, bf, P_bls, t0, dur)
        dpri, dsec, oe_z = odd_even(bt, bf, P_bls, t0, dur)  # parity split at P (odd vs even epochs)
        row.update(P_bls=round(P_bls, 6), t0_bls=round(t0, 5),
                   snr_P0=round(snr_P0, 1) if np.isfinite(snr_P0) else "",
                   snr_bls=round(snr_bls, 1) if np.isfinite(snr_bls) else "",
                   dpri=round(dpri, 1) if np.isfinite(dpri) else "",
                   dsec=round(dsec, 1) if np.isfinite(dsec) else "",
                   oe_z=round(oe_z, 1) if np.isfinite(oe_z) else "")

        risk = 0.0
        # parity depths unequal (>=5 sigma difference AND >=25% relative) => alternate eclipses
        # differ => P is the aliased half period. Both parity dips must themselves be real.
        half_sig = (np.isfinite(oe_z) and np.isfinite(dpri) and np.isfinite(dsec)
                    and max(dpri, dsec) > 7 and oe_z > 5.0
                    and min(dpri, dsec) / max(dpri, dsec) < 0.8)
        row["half_period_sig"] = bool(half_sig)
        if half_sig:
            risk += 0.45
        stale = (snr_bls / snr_P0) if (np.isfinite(snr_bls) and np.isfinite(snr_P0)
                                       and snr_P0 > 0) else np.nan
        row["stale_ratio"] = round(stale, 2) if np.isfinite(stale) else ""
        if np.isfinite(stale):
            if stale > 1.5:
                risk += 0.35
            elif stale > 1.2:
                risk += 0.20
        weak = not (np.isfinite(snr_P0) and snr_P0 > 5) and not (np.isfinite(snr_bls) and snr_bls > 5)
        row["weak"] = bool(weak)
        if weak:
            risk += 0.30
        if catalog_unrelated:
            risk += 0.30
        if single_sector:
            risk += 0.25
        row["risk_score"] = round(risk, 3)
        row["flag"] = ("half_period" if half_sig else
                       ("stale" if (np.isfinite(stale) and stale > 1.5) else
                        ("weak" if weak else "ok")))
        return row
    except Exception as e:  # noqa: BLE001
        row["flag"] = "error"; row["note"] = repr(e)[:120]; row["risk_score"] = 0.5
        return row


def _fmt_dt(s):
    s = int(s); h, r = divmod(s, 3600); m, s = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--master", default=MASTER)
    p.add_argument("--fits-dir", default=FITS_DIR)
    p.add_argument("--out", default=SCAN_OUT)
    p.add_argument("--nprocs", "-n", type=int, default=8)
    p.add_argument("--min-pts", type=int, default=300)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    m = pd.read_csv(args.master)
    m["revised"] = m["revised"].fillna("NA")   # pandas reads the literal "NA" as NaN
    todo = m[(m["first_letter"] == "e") & (m["revised"] == "NO")]
    tasks = [(int(r["Astro ID"]), int(r["TIC ID"]), float(r["Per_original"]),
              float(r["Dur"]) if np.isfinite(r["Dur"]) else np.nan,
              bool(r["single_sector"]),
              bool(pd.notna(r["catalog_period"])))     # catalog matched but didn't gate
             for _, r in todo.iterrows()]
    if args.limit:
        tasks = tasks[: args.limit]

    done = set()
    if os.path.exists(args.out):
        try:
            done = set(int(x) for x in pd.read_csv(args.out)["astro_id"])
        except Exception:  # noqa: BLE001
            done = set()
    tasks = [t for t in tasks if t[0] not in done]
    total = len(tasks)
    print(f"[scan] {total} unrevised EB TCEs to score ({len(done)} already scanned) | n={args.nprocs}")
    if total:
        new_file = not os.path.exists(args.out)
        fh = open(args.out, "a", newline="")
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        tally = {}
        t0 = time.time()
        with Pool(args.nprocs, initializer=_init_worker,
                  initargs=(args.fits_dir, args.min_pts)) as pool:
            for i, row in enumerate(pool.imap_unordered(process_tce, tasks, chunksize=4), 1):
                w.writerow(row)
                if i % 20 == 0:
                    fh.flush()
                tally[row["flag"]] = tally.get(row["flag"], 0) + 1
                el = time.time() - t0; rate = i / el if el else 0
                eta = (total - i) / rate if rate else 0
                sys.stderr.write(
                    f"\r[scan] {i}/{total} ({i/total:5.1%}) | half {tally.get('half_period',0)} "
                    f"stale {tally.get('stale',0)} weak {tally.get('weak',0)} ok {tally.get('ok',0)} "
                    f"| {rate:4.1f}/s elapsed {_fmt_dt(el)} eta {_fmt_dt(eta)}   ")
                sys.stderr.flush()
        fh.flush(); fh.close()
        sys.stderr.write("\n")
        print(f"[scan] DONE in {_fmt_dt(time.time()-t0)}: " +
              " ".join(f"{k}={v}" for k, v in sorted(tally.items())))

    # merge risk_score + P_bls candidate into the master
    scan = pd.read_csv(args.out).drop_duplicates("astro_id", keep="last").set_index("astro_id")
    m = pd.read_csv(args.master)
    is_todo = (m["first_letter"] == "e") & (m["revised"] == "NO")
    n_merged = 0
    for idx in m.index[is_todo]:
        aid = int(m.at[idx, "Astro ID"])
        if aid not in scan.index:
            continue
        s = scan.loc[aid]
        m.at[idx, "risk_score"] = s["risk_score"]
        pb = s["P_bls"]
        if pd.notna(pb) and str(pb) != "":
            cands = str(m.at[idx, "P_candidates"] or "")
            parts = [c for c in cands.split(",") if c]
            if len(parts) < 4:
                parts.append(f"{float(pb):.6f}")
                m.at[idx, "P_candidates"] = ",".join(parts)
        n_merged += 1
    m.to_csv(args.master, index=False)
    print(f"[scan] merged risk_score into master for {n_merged} rows -> {args.master}")
    flagged = scan[scan["flag"].isin(["half_period", "stale"])]
    print(f"[scan] suspicious (half_period/stale): {len(flagged)}; "
          f"weak: {int((scan['flag']=='weak').sum())}")


if __name__ == "__main__":
    main()
