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
from light_curve_util import util  # noqa: E402


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


def build_views(time, flux, period, epoch, duration, tic=0):
    """Run the REAL vetting preprocessing (bkspace=None): detrend -> fold -> global+local view.

    Mirrors astronet/preprocess/generate_input_records_3.py::_standard_views for the untagged view.
    Returns a dict with detrended/folded arrays and the global (201) + local (61) views.
    """
    period = float(period)
    if not np.isfinite(period) or period <= 0:
        raise ValueError(f"bad period {period}")
    if len(time) < MIN_PTS:
        raise ValueError(f"too few points ({len(time)})")

    dt, df, dmask = pp.detrend_and_filter(tic, time, flux, period, float(epoch), float(duration), None)
    if len(dt) < MIN_PTS:
        raise ValueError(f"too few points after detrend ({len(dt)})")

    ep = float(epoch)
    if ep < dt[0]:
        ep += period * np.ceil((dt[0] - ep) / period)

    ft, ff, _, _ = pp.phase_fold_and_sort_light_curve(dt, df, dmask, period, ep)
    fabs, _ = util.phase_fold_time(dt, period, ep)
    o = np.argsort(fabs)
    rt, rf = dt[o], df[o]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gview = np.asarray(_first(pp.global_view(tic, ft, ff, period, all_30min=False, raw_time=rt, raw_flux=rf)), float)
        lview = np.asarray(_first(pp.local_view(tic, ft, ff, period, float(duration), all_30min=False, raw_time=rt, raw_flux=rf)), float)
    return {"dt_time": dt, "dt_flux": df, "fold_t": ft, "fold_f": ff, "gview": gview, "lview": lview}


def make_example_figure(tic, meta, fits_dir=FITS_DIR):
    """Build a before/after diagnostic figure for one target. Returns (fig, info).

    meta: dict-like with Per, Epoc, Dur, first_letter (label). info reports counts / any errors.
    """
    import matplotlib.pyplot as plt

    label = str(meta.get("first_letter", "?"))
    cls = CLASS_NAMES.get(label, label)
    per, epo, dur = float(meta["Per"]), float(meta["Epoc"]), float(meta["Dur"])

    t, f, cad, is_tglc = read_curve(tic, fits_dir)
    n_before, n_tglc = int((~is_tglc).sum()), int(is_tglc.sum())
    info = {"tic": int(tic), "label": label, "n_before": n_before, "n_tglc": n_tglc,
            "period": per, "errors": {}}

    def try_views(tt, hff, tag):
        try:
            return build_views(tt, hff, per, epo, dur, tic=int(tic))
        except Exception as e:  # noqa: BLE001
            info["errors"][tag] = repr(e)[:160]
            return None

    v_before = try_views(t[~is_tglc], f[~is_tglc], "before")
    v_after = try_views(t, f, "after")

    fig, ax = plt.subplots(1, 3, figsize=(18, 4.2))
    fig.suptitle(
        f"TIC {int(tic)}  [{cls}]   P={per:.3f} d   |   before(<S94): {n_before} pts   "
        f"added TGLC(>=S94): {n_tglc} pts", fontsize=13, y=1.02)

    # Panel 1: full light curve, before vs newly-added TGLC
    if len(f):
        lo, hi = np.nanpercentile(f, [0.3, 99.7])
        ax[0].set_ylim(lo, hi)
    ax[0].plot(t[~is_tglc], f[~is_tglc], ".", ms=1.5, alpha=.45, color="0.55", label=f"before <S94 ({n_before})")
    ax[0].plot(t[is_tglc], f[is_tglc], ".", ms=1.5, alpha=.65, color="crimson", label=f"added TGLC >=S94 ({n_tglc})")
    ax[0].axvline(3856.26, color="crimson", ls=":", lw=1, alpha=.7)
    ax[0].set_title("Full light curve"); ax[0].set_xlabel("Time (BTJD)"); ax[0].set_ylabel("rel. flux (SAP)")
    ax[0].legend(markerscale=6, fontsize=8, loc="lower left")

    # Panel 2: local view (transit) before vs after
    if v_before is not None:
        ax[1].plot(v_before["lview"], color="C0", lw=1.6, label="before (QLP only)")
    if v_after is not None:
        ax[1].plot(v_after["lview"], color="crimson", lw=1.6, label="after (+TGLC)")
    ax[1].set_title("LOCAL view (61 bins, transit)"); ax[1].set_xlabel("bin"); ax[1].set_ylabel("norm. flux")
    ax[1].legend(fontsize=9)

    # Panel 3: global view before vs after
    if v_before is not None:
        ax[2].plot(v_before["gview"], color="C0", lw=1.2, label="before (QLP only)")
    if v_after is not None:
        ax[2].plot(v_after["gview"], color="crimson", lw=1.2, label="after (+TGLC)")
    ax[2].set_title("GLOBAL view (201 bins)"); ax[2].set_xlabel("bin"); ax[2].set_ylabel("norm. flux")
    ax[2].legend(fontsize=9)

    for a in ax:
        a.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, info


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
