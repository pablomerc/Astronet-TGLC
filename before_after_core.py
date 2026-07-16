"""before_after_core.py — shared diagnostics for the reprocessed QLP+TGLC light curves.

Question it answers: for a reobserved target, what does the model input look like with the
data we had BEFORE TGLC (sectors <= 93, QLP only) vs AFTER adding the new TGLC sectors
(<= 103)? We take the SAME reprocessed FITS (which carries all cadences + CADENCENO), split
it at the S94 cadence boundary to reconstruct the "before", run the REAL Astronet vetting
preprocessing on both, and compare the folded global/local views.

No database needed at run time: everything comes from the FITS files produced by db_to_fits.py.
Runs in the astronet dev env (e.g. conda `daniel_env_cloned_v2`), which can import
astronet.preprocess from /pdo/users/pablomer/Astronet-Triage.

Used by:
  - inspect_before_after.ipynb   (interactive)
  - diagnose_before_after.py     (headless + Discord)
"""
import os
import sys
import warnings

import numpy as np

# --- paths / constants ---------------------------------------------------------
ASTRONET_PATH = "/pdo/users/pablomer/Astronet-Triage"
FITS_DIR = "/pdo/users/pablomer/mnt/tess/reprocessed_s103_qlptglc_fits_files"
COMPANION = "/pdo/users/pablomer/TGLC_adaptation/data/reobserved_s103_companion.csv"
SUMMARY = os.path.join(FITS_DIR, "_run_summary.csv")

# TGLC became the production pipeline at Sector 94. First cadence of S94 in the QLP
# lightcurvedb (io.read_sector_info(94).cadence_range[0]); cadence increases monotonically
# with time, so CADENCENO >= this  <=>  sector >= 94  <=>  TGLC-era. (BTJD ~= 3856.26.)
S94_CADENCE_START = 1_135_641
SECTOR_CUTOFF = 103
CLASS_NAMES = {"p": "Planet", "e": "EB", "j": "Junk"}
MIN_PTS = 50  # need at least this many good points to attempt a view


def _ensure_astronet_on_path():
    # Use the NEW astronet code; drop any stale /pdo/app qlp that could shadow imports.
    sys.path[:] = [p for p in sys.path if "ffitools-versions" not in p and "FFITools-0.6.0" not in p]
    if ASTRONET_PATH not in sys.path:
        sys.path.insert(0, ASTRONET_PATH)


_ensure_astronet_on_path()
import astronet.preprocess.preprocess as pp  # noqa: E402
from light_curve_util import util, keplersplinev2  # noqa: E402


def _first(x):
    """global_view/local_view return (view, std, mask, ...); grab the view."""
    return x[0] if isinstance(x, tuple) else x


def fits_path(tic, fits_dir=FITS_DIR, sector=SECTOR_CUTOFF):
    return os.path.join(
        fits_dir, "astronet_hlsp_qlptglc_tess_ffi-s%04d-%016d_tess_v01_llc.fits" % (sector, int(tic))
    )


def read_curve(tic, fits_dir=FITS_DIR, flux_key="SAP_FLUX"):
    """Read one reprocessed FITS -> (time, flux, cadence, is_tglc), quality-filtered & time-sorted.

    is_tglc marks the newly-added S94+ (TGLC-era) cadences.
    """
    from astropy.io import fits

    with fits.open(fits_path(tic, fits_dir)) as h:
        d = h[1].data
        q = d["QUALITY"] == 0
        t = np.asarray(d["TIME"][q], float)
        f = np.asarray(d[flux_key][q], float)
        cad = np.asarray(d["CADENCENO"][q], np.int64)
    good = np.isfinite(t) & np.isfinite(f)
    t, f, cad = t[good], f[good], cad[good]
    order = np.argsort(t)
    t, f, cad = t[order], f[order], cad[order]
    is_tglc = cad >= S94_CADENCE_START
    return t, f, cad, is_tglc


def build_views(time, flux, period, epoch, duration, tic=0, is_tglc=None):
    """Run the REAL vetting preprocessing (bkspace=None): detrend -> fold -> global+local view.

    Mirrors astronet/preprocess/generate_input_records_3.py::_standard_views for the untagged view.
    The detrend is inlined from preprocess.detrend_and_filter so we can keep the `valid` mask and
    carry a per-point provenance tag (is_tglc) through to the fold (returned as fold_is_tglc).
    Returns folded arrays + global (201) / local (61) views.
    """
    period = float(period)
    if not np.isfinite(period) or period <= 0:
        raise ValueError(f"bad period {period}")
    if len(time) < MIN_PTS:
        raise ValueError(f"too few points ({len(time)})")

    input_mask = pp.get_spline_mask(time, period, float(epoch), float(duration))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spline_flux, metadata = keplersplinev2.choosekeplersplinev2(
            time, flux, input_mask=input_mask, fixed_bkspace=None, return_metadata=True, maxiter=10)
    y = spline_flux.copy()
    bad = ~metadata.light_curve_mask
    if np.sum(~bad) >= 2 and np.any(bad):
        y[bad] = np.interp(time[bad], time[~bad], y[~bad])
    elif np.sum(~bad) < 2:
        bad = ~input_mask
        if np.sum(~bad) >= 2 and np.any(bad):
            y[bad] = np.interp(time[bad], time[~bad], y[~bad])
    detr = flux / y
    valid = ~np.isnan(detr)
    dt, df = time[valid], detr[valid]
    if len(dt) < MIN_PTS:
        raise ValueError(f"too few points after detrend ({len(dt)})")

    ep = float(epoch)
    if ep < dt[0]:
        ep += period * np.ceil((dt[0] - ep) / period)

    fabs, _ = util.phase_fold_time(dt, period, ep)
    o = np.argsort(fabs)
    ft, ff = fabs[o], df[o]
    rt = dt[o]  # unfolded BTJD, fold-sorted (median_filter2 picks cadence width from absolute time)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gview = np.asarray(_first(pp.global_view(tic, ft, ff, period, all_30min=False, raw_time=rt, raw_flux=ff)), float)
        lview = np.asarray(_first(pp.local_view(tic, ft, ff, period, float(duration), all_30min=False, raw_time=rt, raw_flux=ff)), float)
    out = {"fold_t": ft, "fold_f": ff, "gview": gview, "lview": lview}
    if is_tglc is not None:
        out["fold_is_tglc"] = np.asarray(is_tglc)[valid][o]
    return out


def compute_example(tic, meta, fits_dir=FITS_DIR):
    """Do the EXPENSIVE work for one target (read + detrend + fold + views), no plotting.

    Returns a dict of arrays/scalars that figure_from_data() renders. Cache this (save_cache)
    so restyling the plot doesn't re-run the ~40 s/target spline detrend.
    """
    per, epo, dur = float(meta["Per"]), float(meta["Epoc"]), float(meta["Dur"])
    label = str(meta.get("first_letter", "?"))
    t, f, cad, is_tglc = read_curve(tic, fits_dir)
    dat = {"tic": int(tic), "label": label, "per": per, "dur": dur,
           "n_before": int((~is_tglc).sum()), "n_tglc": int(is_tglc.sum()),
           "t": t, "f": f, "is_tglc": is_tglc, "errors": {}}
    for tag, (tt, ff, itag) in [("before", (t[~is_tglc], f[~is_tglc], None)),
                                ("after", (t, f, is_tglc))]:
        try:
            v = build_views(tt, ff, per, epo, dur, tic=int(tic), is_tglc=itag)
            dat[f"{tag}_fold_t"] = v["fold_t"]; dat[f"{tag}_fold_f"] = v["fold_f"]
            dat[f"{tag}_gview"] = v["gview"]; dat[f"{tag}_lview"] = v["lview"]
            if "fold_is_tglc" in v:
                dat[f"{tag}_fold_is_tglc"] = v["fold_is_tglc"]
        except Exception as e:  # noqa: BLE001
            dat["errors"][tag] = repr(e)[:160]
    return dat


def save_cache(dat, path):
    import json

    arrs = {k: v for k, v in dat.items() if isinstance(v, np.ndarray)}
    scalars = {k: v for k, v in dat.items() if k != "errors" and not isinstance(v, np.ndarray)}
    np.savez_compressed(path, _scalars=json.dumps(scalars), _errors=json.dumps(dat["errors"]), **arrs)


def load_cache(path):
    import json

    z = np.load(path, allow_pickle=False)
    dat = json.loads(str(z["_scalars"]))
    dat["errors"] = json.loads(str(z["_errors"]))
    for k in z.files:
        if not k.startswith("_"):
            dat[k] = z[k]
    return dat


def _folded_ylim(ax, vals):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return
    med = np.nanmedian(vals)
    mad = np.nanmedian(np.abs(vals - med)) * 1.4826 + 1e-6
    lo = min(np.nanpercentile(vals, 0.5), med - 6 * mad)
    hi = max(np.nanpercentile(vals, 99.5), med + 6 * mad)
    pad = 0.12 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)


def figure_from_data(dat):
    """Render the before/after figure (2x3) from a computed/cached dict (cheap)."""
    import matplotlib.pyplot as plt

    tic = int(dat["tic"]); cls = CLASS_NAMES.get(dat["label"], dat["label"])
    per, dur = float(dat["per"]), float(dat["dur"])
    t, f, is_tglc = dat["t"], dat["f"], dat["is_tglc"].astype(bool)
    n_before, n_tglc = int(dat["n_before"]), int(dat["n_tglc"])
    has = lambda tag: f"{tag}_gview" in dat  # noqa: E731
    dur_w = dur if np.isfinite(dur) and dur > 0 else 0.05 * per
    win = min(per / 2, max(3.0 * dur_w, 0.03 * per))

    fig, axes = plt.subplots(2, 3, figsize=(19, 8))
    fig.suptitle(
        f"TIC {tic}  [{cls}]   P={per:.3f} d   |   before(<S94): {n_before} pts   "
        f"added TGLC(>=S94): {n_tglc} pts", fontsize=13, y=1.0)
    ax = axes.ravel()

    # ax0: full light curve
    if len(f):
        lo, hi = np.nanpercentile(f, [0.3, 99.7])
        ax[0].set_ylim(lo, hi)
    ax[0].plot(t[~is_tglc], f[~is_tglc], ".", ms=1.5, alpha=.45, color="0.55", label=f"before <S94 ({n_before})")
    ax[0].plot(t[is_tglc], f[is_tglc], ".", ms=1.5, alpha=.65, color="crimson", label=f"added TGLC >=S94 ({n_tglc})")
    ax[0].axvline(3856.26, color="crimson", ls=":", lw=1, alpha=.7)
    ax[0].set_title("Full light curve"); ax[0].set_xlabel("Time (BTJD)"); ax[0].set_ylabel("rel. flux (SAP)")
    ax[0].legend(markerscale=6, fontsize=8, loc="lower left")

    # ax1: folded before vs after -- draw AFTER (red) first, BEFORE (blue) on top & stronger
    allf = []
    if "after_fold_t" in dat:
        m = np.abs(dat["after_fold_t"]) < win
        ax[1].plot(dat["after_fold_t"][m], dat["after_fold_f"][m], ".", ms=1.5, alpha=.28, color="crimson", label="after (+TGLC)")
        if m.any():
            allf.append(dat["after_fold_f"][m])
    if "before_fold_t" in dat:
        m = np.abs(dat["before_fold_t"]) < win
        ax[1].plot(dat["before_fold_t"][m], dat["before_fold_f"][m], ".", ms=2.6, alpha=.6, color="C0", label="before (QLP only)")
        if m.any():
            allf.append(dat["before_fold_f"][m])
    _folded_ylim(ax[1], np.concatenate(allf) if allf else [])
    ax[1].axvline(0, color="k", ls=":", lw=.8, alpha=.5)
    ax[1].set_title(f"Folded detrended (±{win:.2f} d) — before over after")
    ax[1].set_xlabel("phase (days)"); ax[1].set_ylabel("detrended flux")
    ax[1].legend(markerscale=6, fontsize=8, loc="lower left")

    # ax2: folded AFTER, colored by provenance -- repeated/old vs new TGLC
    if "after_fold_is_tglc" in dat:
        tg = dat["after_fold_is_tglc"].astype(bool)
        m = np.abs(dat["after_fold_t"]) < win
        old, new = m & ~tg, m & tg
        ax[2].plot(dat["after_fold_t"][old], dat["after_fold_f"][old], ".", ms=1.5, alpha=.3, color="0.55", label=f"repeated/old ({int(old.sum())})")
        ax[2].plot(dat["after_fold_t"][new], dat["after_fold_f"][new], ".", ms=2.2, alpha=.6, color="crimson", label=f"new TGLC ({int(new.sum())})")
        _folded_ylim(ax[2], dat["after_fold_f"][m])
        ax[2].axvline(0, color="k", ls=":", lw=.8, alpha=.5)
    else:
        ax[2].text(0.5, 0.5, "(after view unavailable)", ha="center", va="center", transform=ax[2].transAxes)
    ax[2].set_title("Folded: repeated (old) vs new TGLC")
    ax[2].set_xlabel("phase (days)"); ax[2].set_ylabel("detrended flux")
    ax[2].legend(markerscale=6, fontsize=8, loc="lower left")

    # ax3: local view; ax4: global view
    for tag, color in [("before", "C0"), ("after", "crimson")]:
        if has(tag):
            lab = {"before": "before (QLP only)", "after": "after (+TGLC)"}[tag]
            ax[3].plot(dat[f"{tag}_lview"], color=color, lw=1.6, label=lab)
            ax[4].plot(dat[f"{tag}_gview"], color=color, lw=1.2, label=lab)
    ax[3].set_title("LOCAL view (61 bins)"); ax[3].set_xlabel("bin"); ax[3].set_ylabel("norm. flux"); ax[3].legend(fontsize=9)
    ax[4].set_title("GLOBAL view (201 bins)"); ax[4].set_xlabel("bin"); ax[4].set_ylabel("norm. flux"); ax[4].legend(fontsize=9)

    ax[5].axis("off")
    for a in ax[:5]:
        a.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def make_example_figure(tic, meta, fits_dir=FITS_DIR):
    """Compute + render one target's before/after figure. Returns (fig, info)."""
    dat = compute_example(tic, meta, fits_dir)
    info = {"tic": dat["tic"], "label": dat["label"], "n_before": dat["n_before"],
            "n_tglc": dat["n_tglc"], "period": dat["per"], "errors": dat["errors"]}
    return figure_from_data(dat), info


def load_tables(companion=COMPANION, summary=SUMMARY):
    import pandas as pd

    comp = pd.read_csv(companion)
    summ = pd.read_csv(summary) if os.path.exists(summary) else None
    return comp, summ


def pick_examples(comp, summ=None, n_per_class=10, seed=0):
    """Pick ~n examples per class: finite Per/Epoc/Dur, sensible period, most TGLC-era cadences."""
    import pandas as pd

    df = comp.copy()
    df = df[np.isfinite(df["Per"]) & np.isfinite(df["Epoc"]) & np.isfinite(df["Dur"])]
    df = df[(df["Per"] > 0.3) & (df["Per"] < 30)]
    df = df.drop_duplicates("TIC ID")
    if summ is not None:
        df = df.merge(summ[["tic", "n_tglc", "npts"]], left_on="TIC ID", right_on="tic", how="left")
    else:
        df["n_tglc"] = np.nan
    out = {}
    for L in ["p", "e", "j"]:
        sub = df[df["first_letter"] == L].copy()
        sub = sub.sort_values("n_tglc", ascending=False, na_position="last")
        out[L] = sub.head(n_per_class).to_dict("records")
    return out
