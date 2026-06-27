"""Training-set composition for the current production vetting model (dec2025
labeled set), with the IMPORTANT correction that the `File` column mixes two
kinds of light curve:

  * `mk_*`       -> genuine SINGLE-sector QLP light curves (median span ~27 d),
                    tagged with their TRUE observation sector (S3-S34).
                    camera/CCD is well-defined and resolves 100% via tess-point.
  * `astronet_*` -> MULTI-sector STITCHED QLP light curves built by the
                    h5->fits converter (median span ~760 d); the `s0064`/`s0085`
                    in the filename is a BATCH/PROCESSING tag, NOT an observation
                    sector. A stitched LC spans several cam/CCDs, so a single
                    camera/CCD is undefined for these.

So ~76% of the training set is multi-sector stitched QLP. Camera/CCD
distributions are therefore reported on the single-sector subset only.

Outputs (TGLC_adaptation/figures, /data):
  training_set_composition.png         single-vs-stitched + (batch-)sector bars
  training_set_camccd_singlesector.png camera / CCD / cam x ccd (single-sector)
  training_set_sector_per_label.png    overlaid P/EB/J vs (batch-)sector
  training_set_enriched.csv            per-row: sector(tag), cam, ccd, label, kind
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tess_stars2px import tess_stars2px_function_entry as ts2px

CSV = "/pdo/users/pablomer/mnt/tess/astronet/tces-vetting-v01-tois-triageJs-nocentroid-dec2025-all.csv"
OUTFIG = "/pdo/users/pablomer/TGLC_adaptation/figures"
OUTDAT = "/pdo/users/pablomer/TGLC_adaptation/data"
LABEL_MAP = {"p": "Planet (P)", "e": "EB", "j": "Junk (J)"}
LABEL_COLOR = {"Planet (P)": "#2c7fb8", "EB": "#d95f02", "Junk (J)": "#7570b3"}

def main():
    df = pd.read_csv(CSV)
    df["sector"] = df["File"].str.extract(r"s(\d{4})")[0].astype("Int64")
    df["prefix"] = df["File"].str.split("_").str[0]
    # kind: single-sector (true sector tag) vs multi-sector stitched (batch tag)
    df["kind"] = np.where(df["prefix"] == "astronet", "stitched multi-sector",
                          "single-sector")
    df["label"] = df["first_letter"].map(LABEL_MAP)
    n_single = (df["kind"] == "single-sector").sum()
    n_stitch = (df["kind"] == "stitched multi-sector").sum()
    print(f"N={len(df)}  single-sector={n_single}  stitched={n_stitch}")

    # camera/ccd ONLY meaningful for single-sector files; tess-point on those.
    sub = df[df["kind"] == "single-sector"].dropna(subset=["RA", "Dec"]).drop_duplicates("TIC ID")
    oid, _, _, osec, ocam, occd, _, _, _ = ts2px(sub["TIC ID"].to_numpy(),
                                                 sub["RA"].to_numpy(), sub["Dec"].to_numpy())
    point = pd.DataFrame({"TIC ID": oid.astype(np.int64),
                          "sector": pd.array(osec.astype(int), dtype="Int64"),
                          "cam": ocam.astype(int), "ccd": occd.astype(int)}
                         ).drop_duplicates(["TIC ID", "sector"])
    df = df.merge(point, on=["TIC ID", "sector"], how="left")
    sscov = df.loc[df["kind"] == "single-sector", "cam"].notna().mean()
    print(f"single-sector cam/ccd coverage: {sscov:.1%}")
    df.to_csv(os.path.join(OUTDAT, "training_set_enriched.csv"), index=False)

    # ---------- Figure 1: composition ----------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5),
                             gridspec_kw={"width_ratios": [1, 2.4]})
    fig.suptitle(f"Astronet vetting training set (dec2025, N={len(df)}) — composition",
                 fontsize=14)
    # single vs stitched
    ax = axes[0]
    kinds = ["single-sector", "stitched multi-sector"]
    vals = [n_single, n_stitch]
    cols = ["#3182bd", "#de9f43"]
    ax.bar(["single-sector\n(true sector,\nQLP)", "stitched\nmulti-sector\n(batch tag, QLP)"],
           vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v}\n({v/len(df):.0%})", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("# TCEs"); ax.set_title("LC type")
    ax.set_ylim(0, max(vals) * 1.18)

    # sector bars, colored by kind
    ax = axes[1]
    sc = df.groupby(["sector", "kind"]).size().unstack(fill_value=0)
    sc = sc.reindex(columns=kinds, fill_value=0).sort_index()
    x = np.arange(len(sc))
    ax.bar(x, sc["single-sector"], color="#3182bd", label="single-sector (true sector)")
    ax.bar(x, sc["stitched multi-sector"], bottom=sc["single-sector"],
           color="#de9f43", label="stitched multi-sector (batch tag, NOT obs. sector)")
    ax.set_xticks(x); ax.set_xticklabels(sc.index.astype(int), rotation=90, fontsize=7)
    ax.set_xlabel("File-tag sector"); ax.set_ylabel("# TCEs")
    ax.set_title("Distribution over File-tag sector")
    for i, s in enumerate(sc.index):
        v = int(sc.loc[s].sum())
        if v > sc.sum(axis=1).max() * 0.05:
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=7)
    ax.legend(fontsize=9)
    ax.text(0.99, 0.80, "S64 / S85 bars are the two bulk-TOI\nprocessing batches "
            "(stitched, many sectors each)", transform=ax.transAxes, ha="right",
            fontsize=8, color="#7a4f10",
            bbox=dict(boxstyle="round", fc="#fff6e8", ec="#de9f43"))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p1 = os.path.join(OUTFIG, "training_set_composition.png")
    fig.savefig(p1, dpi=130); print("wrote", p1)

    # ---------- Figure 2: camera/ccd on single-sector subset ----------
    ss = df[(df["kind"] == "single-sector")].dropna(subset=["cam", "ccd"])
    fig2, axx = plt.subplots(1, 3, figsize=(16, 4.6))
    fig2.suptitle(f"Camera / CCD — SINGLE-SECTOR subset only (N={len(ss)}; "
                  "stitched multi-sector LCs have no single cam/CCD)", fontsize=13)
    cc = ss["cam"].astype(int).value_counts().reindex([1, 2, 3, 4], fill_value=0)
    axx[0].bar([str(i) for i in cc.index], cc.values, color="#31a354")
    axx[0].set_title("Camera"); axx[0].set_xlabel("camera"); axx[0].set_ylabel("# TCEs")
    for x_, v in enumerate(cc.values):
        axx[0].text(x_, v, str(v), ha="center", va="bottom")
    dc = ss["ccd"].astype(int).value_counts().reindex([1, 2, 3, 4], fill_value=0)
    axx[1].bar([str(i) for i in dc.index], dc.values, color="#e6550d")
    axx[1].set_title("CCD"); axx[1].set_xlabel("CCD")
    for x_, v in enumerate(dc.values):
        axx[1].text(x_, v, str(v), ha="center", va="bottom")
    mat = (ss.assign(cam=ss["cam"].astype(int), ccd=ss["ccd"].astype(int))
             .pivot_table(index="cam", columns="ccd", values="TIC ID",
                          aggfunc="count", fill_value=0)
             .reindex(index=[1, 2, 3, 4], columns=[1, 2, 3, 4], fill_value=0))
    im = axx[2].imshow(mat.values, cmap="viridis", aspect="auto")
    axx[2].set_xticks(range(4)); axx[2].set_xticklabels([1, 2, 3, 4])
    axx[2].set_yticks(range(4)); axx[2].set_yticklabels([1, 2, 3, 4])
    axx[2].set_title("Camera x CCD"); axx[2].set_xlabel("CCD"); axx[2].set_ylabel("camera")
    for i in range(4):
        for j in range(4):
            axx[2].text(j, i, int(mat.values[i, j]), ha="center", va="center",
                        color="w", fontsize=9)
    fig2.colorbar(im, ax=axx[2], fraction=0.046, pad=0.04)
    fig2.tight_layout(rect=[0, 0, 1, 0.92])
    p2 = os.path.join(OUTFIG, "training_set_camccd_singlesector.png")
    fig2.savefig(p2, dpi=130); print("wrote", p2)

    # ---------- Figure 3: sector per label (batch-tag sector) ----------
    fig3, ax = plt.subplots(figsize=(14, 6))
    secs = sorted(df["sector"].dropna().unique().astype(int))
    bins = np.arange(min(secs) - 0.5, max(secs) + 1.5, 1.0)
    for lab in ["Planet (P)", "EB", "Junk (J)"]:
        vals = df.loc[df["label"] == lab, "sector"].dropna().astype(int)
        ax.hist(vals, bins=bins, alpha=0.55, label=f"{lab}  (n={len(vals)})",
                color=LABEL_COLOR[lab], edgecolor="white", linewidth=0.3)
    ax.set_xlabel("File-tag sector  (S64 & S85 = bulk-TOI batches, stitched multi-sector)")
    ax.set_ylabel("# TCEs")
    ax.set_title("Sector distribution per label — Astronet vetting training set (dec2025)")
    ax.legend(); ax.set_xticks(secs); ax.tick_params(axis="x", rotation=90, labelsize=7)
    fig3.tight_layout()
    p3 = os.path.join(OUTFIG, "training_set_sector_per_label.png")
    fig3.savefig(p3, dpi=130); print("wrote", p3)
    ax.set_yscale("log"); ax.set_ylabel("# TCEs (log)")
    fig3.savefig(os.path.join(OUTFIG, "training_set_sector_per_label_log.png"), dpi=130)

    print("\nsector x label:\n",
          df.pivot_table(index="sector", columns="label", values="TIC ID",
                         aggfunc="count", fill_value=0).to_string())

if __name__ == "__main__":
    main()
