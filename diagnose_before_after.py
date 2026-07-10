"""diagnose_before_after.py — headless before/after-TGLC diagnostics, posted to Discord.

Runs the SAME diagnostics as inspect_before_after.ipynb (shared code in before_after_core.py):
for ~N reobserved targets per class, split each reprocessed FITS at the S94 cadence boundary
into "before" (QLP-only, <=S93) and "after" (+TGLC, <=S103), run the real Astronet vetting
preprocessing on both, and plot the full light curve + local + global views. Then upload every
figure to a Discord webhook (batched).

Run in the astronet dev env (e.g. conda daniel_env_cloned_v2):
  python diagnose_before_after.py                 # 10/class, post to Discord
  python diagnose_before_after.py --n-per-class 5 --classes p,e
  python diagnose_before_after.py --dry-run       # render + save PNGs, do NOT post

Webhook resolution (first found): --webhook  >  $DISCORD_WEBHOOK_URL  >
  /pdo/users/pablomer/TGLC_adaptation/.discord_webhook  (one line; git-ignored — keep it secret).
"""
import argparse
import io as _io
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import before_after_core as core  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WEBHOOK_FILE = os.path.join(HERE, ".discord_webhook")
MAX_FILES_PER_MSG = 10


def resolve_webhook(arg):
    if arg:
        return arg.strip()
    if os.environ.get("DISCORD_WEBHOOK_URL"):
        return os.environ["DISCORD_WEBHOOK_URL"].strip()
    if os.path.exists(WEBHOOK_FILE):
        with open(WEBHOOK_FILE) as f:
            return f.read().strip()
    return None


def post_batch(webhook, content, named_pngs, retries=4):
    """POST up to 10 (name, bytes) PNGs to a Discord webhook. Handles 429 rate limits."""
    import requests

    for attempt in range(retries):
        files = {}
        for i, (name, data) in enumerate(named_pngs):
            files[f"files[{i}]"] = (name, data, "image/png")
        r = requests.post(webhook, data={"content": content[:1990]}, files=files, timeout=120)
        if r.status_code in (200, 204):
            return True
        if r.status_code == 429:  # rate limited
            wait = 2.0
            try:
                wait = float(r.json().get("retry_after", 2.0))
            except Exception:  # noqa: BLE001
                pass
            print(f"  [discord] 429 rate-limited, sleeping {wait:.1f}s", file=sys.stderr)
            time.sleep(wait + 0.5)
            continue
        print(f"  [discord] HTTP {r.status_code}: {r.text[:200]}", file=sys.stderr)
        return False
    return False


def fig_to_png_bytes(fig, dpi=90):
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-per-class", type=int, default=10)
    p.add_argument("--classes", default="p,e,j", help="comma list of p/e/j")
    p.add_argument("--fits-dir", default=core.FITS_DIR)
    p.add_argument("--companion", default=core.COMPANION)
    p.add_argument("--webhook", default=None, help="Discord webhook URL (else env/.discord_webhook)")
    p.add_argument("--dry-run", action="store_true", help="render + save PNGs, do not post")
    p.add_argument("--outdir", default=os.path.join(HERE, "figures", "before_after"),
                   help="where to also save PNGs")
    p.add_argument("--dpi", type=int, default=90)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    webhook = None if args.dry_run else resolve_webhook(args.webhook)
    if not args.dry_run and not webhook:
        sys.exit("No webhook found (use --webhook, $DISCORD_WEBHOOK_URL, or .discord_webhook), "
                 "or pass --dry-run.")

    comp, summ = core.load_tables(args.companion,
                                  os.path.join(args.fits_dir, "_run_summary.csv"))
    examples = core.pick_examples(comp, summ, n_per_class=args.n_per_class)
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    header = (f"**Before/After TGLC diagnostics** — reprocessed reobserved set (through S103)\n"
              f"before = QLP only (drop sectors ≥94) · after = +TGLC · "
              f"{args.n_per_class}/class · panels: full LC | local view | global view")
    print(header.replace("**", ""))
    if not args.dry_run:
        post_batch(webhook, header, [])  # text-only intro (empty file list is fine)

    for L in classes:
        rows = examples.get(L, [])
        cls = core.CLASS_NAMES.get(L, L)
        print(f"\n=== {cls}: {len(rows)} examples ===")
        batch = []
        n_ok = 0
        for r in rows:
            tic = int(r["TIC ID"])
            try:
                fig, info = core.make_example_figure(tic, r, fits_dir=args.fits_dir)
            except Exception as e:  # noqa: BLE001
                print(f"  TIC {tic}: FAILED {e!r}", file=sys.stderr)
                continue
            png = fig_to_png_bytes(fig, dpi=args.dpi)
            fname = f"before_after_{L}_{tic}.png"
            with open(os.path.join(args.outdir, fname), "wb") as fh:
                fh.write(png)
            n_ok += 1
            note = f" (view err: {list(info['errors'])})" if info["errors"] else ""
            print(f"  TIC {tic}: before={info['n_before']} tglc={info['n_tglc']} P={info['period']:.3f}{note}")
            batch.append((fname, png))
            if not args.dry_run and len(batch) == MAX_FILES_PER_MSG:
                post_batch(webhook, f"**{cls}** (batch)", batch)
                batch = []
                time.sleep(1.0)
        if not args.dry_run and batch:
            post_batch(webhook, f"**{cls}** — {n_ok} examples", batch)
            time.sleep(1.0)
        print(f"  {cls}: {n_ok} figures {'saved' if args.dry_run else 'posted'} -> {args.outdir}")

    print("\nDONE." + ("" if args.dry_run else " Sent to Discord."))


if __name__ == "__main__":
    main()
