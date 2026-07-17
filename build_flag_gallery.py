"""build_flag_gallery.py — Step 3 of the EB period revision: the manual flag gallery.

For every EB TCE still `revised == NO` in the master CSV, render ONE strip of candidate folds
side by side — [P0 | 2*P0 | P0/2 | P_bls] — from the fast-detrended, 30-min-binned reprocessed
curve (same machinery as scan_periods.py), and build a single static HTML page:

  figures/period_gallery/index.html   (+ pngs/<astro_id>.png)

Gallery features (all client-side, no server needed — open the file or `python -m http.server`):
  * sorted by risk_score, worst first; toggle to include/exclude the single-sector upweight
  * per target: click the fold that looks right -> that period is recorded (P0 = "original is
    fine"), or type a custom period, or mark 'unsure'
  * choices persist in localStorage; "Export revisions CSV" downloads period_revisions.csv
  * apply with:  python apply_revisions.py ~/Downloads/period_revisions.csv

Resumable: PNGs already rendered are skipped; re-run after a crash just continues.

  PY=/pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python3
  $PY build_flag_gallery.py -n 8          # ~6-10 min for ~1,400 targets
"""
import argparse
import base64
import html
import json
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

import scan_periods as sp   # reuse read_flux / fast_detrend / bin_series / _profile

HERE = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.join(HERE, "data", "reobserved_s103_revised.csv")
SCAN = os.path.join(HERE, "data", "period_scan.csv")
OUTDIR = os.path.join(HERE, "figures", "period_gallery")

_CFG = {}


def _init_worker(fits_dir, dpi, outdir):
    _CFG.update(fits_dir=fits_dir, dpi=dpi, outdir=outdir)
    import matplotlib
    matplotlib.use("Agg")


def render_target(task):
    """Render the candidate-fold strip for one target -> pngs/<astro_id>.png.
    `cands` is the explicit list of (kind, period) panels — same list the HTML buttons use."""
    astro_id, tic, cands, dur, t0 = task
    out = os.path.join(_CFG["outdir"], "pngs", f"{astro_id}.png")
    if os.path.exists(out):
        return {"astro_id": astro_id, "status": "skipped"}
    import matplotlib.pyplot as plt
    try:
        t, f = sp.read_flux(tic, _CFG["fits_dir"])
        fd = sp.fast_detrend(t, f)
        bt, bf = sp.bin_series(t, fd)
        if not (np.isfinite(t0)):
            t0 = bt[0]
        fig, axes = plt.subplots(1, len(cands), figsize=(3.8 * len(cands), 2.9))
        axes = np.atleast_1d(axes)
        ylim = np.nanpercentile(bf, [0.3, 99.7])
        for ax, (lab, P) in zip(axes, cands):
            if not (np.isfinite(P) and P > 0):
                ax.axis("off")
                continue
            ph = (((bt - t0) / P + 0.5) % 1.0) - 0.5
            ax.plot(ph, bf, ".", ms=1.2, alpha=0.35, color="0.25", rasterized=True)
            snr = sp.fold_depth_snr(bt, bf, P, t0, dur)
            ax.set_title(f"{lab}: {P:.5f} d  (snr {snr:.0f})" if np.isfinite(snr)
                         else f"{lab}: {P:.5f} d", fontsize=10)
            ax.set_ylim(*ylim)
            ax.set_xlim(-0.5, 0.5)
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=7)
        fig.suptitle(f"TIC {tic}  (Astro {astro_id})", fontsize=10, y=1.02)
        fig.tight_layout()
        fig.savefig(out, dpi=_CFG["dpi"], bbox_inches="tight")
        plt.close(fig)
        return {"astro_id": astro_id, "status": "ok"}
    except Exception as e:  # noqa: BLE001
        return {"astro_id": astro_id, "status": "fail", "msg": repr(e)[:120]}


HTML_HEAD = """<!doctype html><html><head><meta charset="utf-8">
<title>EB period flag gallery</title>
<style>
 body{font-family:sans-serif;margin:14px;background:#fafafa}
 .row{border:1px solid #ccc;border-radius:8px;margin:10px 0;padding:8px;background:#fff}
 .row.decided{background:#eef7ee}
 .meta{font-size:13px;margin-bottom:4px}
 .meta b{font-size:14px}
 .imgwrap{position:relative;display:block}
 .imgwrap img{max-width:100%;height:auto;display:block;cursor:pointer}
 button{margin:2px;padding:5px 10px;cursor:pointer;border:1px solid #888;border-radius:5px;background:#f2f2f2}
 button.sel{background:#2e7d32;color:#fff;border-color:#2e7d32}
 .flag{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;margin-left:6px;color:#fff}
 .f_half{background:#c62828}.f_stale{background:#ef6c00}.f_weak{background:#757575}.f_ok{background:#2e7d32}
 .f_cat{background:#6a1b9a}.f_auto{background:#1565c0}
 #hdr{position:sticky;top:0;background:#fafafa;padding:8px 0;z-index:5;border-bottom:2px solid #ddd}
 input[type=number]{width:110px}
 #fw{position:fixed;left:0;top:0;width:100vw;height:100vh;pointer-events:none;z-index:99}
 #cheer{position:fixed;left:50%;top:38%;transform:translate(-50%,-50%);z-index:100;
        font-size:34px;font-weight:bold;color:#ff9800;text-shadow:0 2px 12px rgba(0,0,0,.35);
        pointer-events:none;opacity:0;transition:opacity .3s}
</style></head><body>
<div id="hdr">
 <b>EB period flag gallery</b> &mdash; click the panel (or button) of the fold that looks right (P0 = original is fine).
 <label><input type="checkbox" id="ssToggle" checked> single-sector upweight</label>
 <label><input type="checkbox" id="hideDecided"> hide decided</label>
 <label><input type="checkbox" id="showAuto"> show auto-corrected (catalog)</label>
 <button onclick="exportCSV()">Export revisions CSV</button>
 <span id="counts"></span>
</div>
<div id="list"></div>
<canvas id="fw"></canvas>
<div id="cheer"></div>
<script>
const DATA = %DATA%;
const KEY = "eb_period_revisions_v1";
const CHEERS = [
  "Good job — you're saving planets! 🪐",
  "Fix those binaries! 🔧",
  "Another eclipse phased into place. 🌘",
  "The vetting model thanks you for its clean folds. 🤖",
  "Period pirate! You just plundered an alias. 🏴‍☠️",
  "Kepler would be proud. 📜",
  "That EB never saw it coming. 👀",
  "One small click for you, one giant leap for the training set. 🚀",
  "BLS could never. You're the real search algorithm. 🧠",
  "Certified binary whisperer. 🐴",
  "Phase folded, case closed. 🕵️",
  "+10 XP toward Grandmaster Vetter. 🏆"
];
let picks = JSON.parse(localStorage.getItem(KEY) || "{}");
function save(){ localStorage.setItem(KEY, JSON.stringify(picks)); refresh(); }
function celebrateMaybe(){ if (Math.random() < 0.10) { fireworks(); cheer(); } }
function choose(aid, kind, period){
  picks[aid] = {choice:kind, period:period};
  celebrateMaybe();
  save();
}
function imgClick(aid, ev){
  const d = DATA.find(x=>x.astro_id===aid); if(!d) return;
  const img = ev.currentTarget;
  const frac = ev.offsetX / img.clientWidth;
  const i = Math.max(0, Math.min(d.cands.length-1, Math.floor(frac * d.cands.length)));
  const c = d.cands[i];
  choose(aid, c.kind, c.period);
}
function custom(aid){
  const v = parseFloat(document.getElementById("cust"+aid).value);
  if (isFinite(v) && v>0) { picks[aid]={choice:"custom",period:v}; celebrateMaybe(); save(); }
}
function unsure(aid){ picks[aid]={choice:"unsure",period:null}; save(); }
function clearPick(aid){ delete picks[aid]; save(); }
function score(d){ const ss=document.getElementById("ssToggle").checked; return d.risk_base + (ss? d.ss_bonus:0); }
function refresh(){
  const list=document.getElementById("list"); list.innerHTML="";
  const hide=document.getElementById("hideDecided").checked;
  const showAuto=document.getElementById("showAuto").checked;
  const items=[...DATA].sort((a,b)=>score(b)-score(a));
  let done=0, shown=0;
  for(const d of items){
    const p=picks[d.astro_id];
    if(p) done++;
    if(d.auto && !showAuto) continue;
    if(hide && p) continue;
    shown++;
    const div=document.createElement("div");
    div.className="row"+(p?" decided":"");
    const flags = d.flags.map(f=>`<span class="flag f_${f.split(':')[0]}">${f}</span>`).join("");
    let btns="";
    for(const c of d.cands){
      const sel = p && p.choice===c.kind ? "sel":"";
      btns += `<button class="${sel}" onclick="choose(${d.astro_id},'${c.kind}',${c.period})">${c.kind} ${c.period.toFixed(5)}</button>`;
    }
    const selCust = p && p.choice==="custom" ? "sel":"";
    const selUns  = p && p.choice==="unsure" ? "sel":"";
    div.innerHTML = `<div class="meta"><b>TIC ${d.tic}</b> (Astro ${d.astro_id})
       risk=${score(d).toFixed(2)} ${flags}
       ${d.auto?('<span class="flag f_auto">auto '+d.auto+'</span>'):''}
       ${d.cat_note?('<span class="flag f_cat">'+d.cat_note+'</span>'):''}
       ${p?('<b> &rarr; '+p.choice+(p.period?(' @ '+p.period):'')+'</b> <button onclick="clearPick('+d.astro_id+')">undo</button>'):''}
      </div>
      <div class="imgwrap"><img loading="lazy" src="pngs/${d.astro_id}.png"
           onclick="imgClick(${d.astro_id},event)"></div>
      <div>${btns}
        <button class="${selCust}" onclick="custom(${d.astro_id})">custom:</button>
        <input type="number" step="any" id="cust${d.astro_id}" placeholder="period (d)">
        <button class="${selUns}" onclick="unsure(${d.astro_id})" title="no usable period — row becomes revised=DISCARD (excluded from training)">unsure&nbsp;&rarr;&nbsp;discard</button>
      </div>`;
    list.appendChild(div);
  }
  document.getElementById("counts").innerHTML = ` &nbsp; decided ${done}/${DATA.length} (showing ${shown})`;
}
function exportCSV(){
  let rows=[["astro_id","tic","choice","chosen_period"]];
  for(const d of DATA){
    const p=picks[d.astro_id];
    if(!p) continue;
    rows.push([d.astro_id,d.tic,p.choice,p.period===null?"":p.period]);
  }
  const csv=rows.map(r=>r.join(",")).join("\\n");
  const a=document.createElement("a");
  a.href=URL.createObjectURL(new Blob([csv],{type:"text/csv"}));
  a.download="period_revisions.csv"; a.click();
}
function cheer(){
  const el=document.getElementById("cheer");
  el.textContent=CHEERS[Math.floor(Math.random()*CHEERS.length)];
  el.style.opacity=1;
  setTimeout(()=>{ el.style.opacity=0; }, 2200);
}
function fireworks(){
  const cv=document.getElementById("fw");
  cv.width=innerWidth; cv.height=innerHeight;
  const ctx=cv.getContext("2d");
  const bursts=1+Math.floor(Math.random()*3), parts=[];
  for(let b=0;b<bursts;b++){
    const cx=innerWidth*(0.15+0.7*Math.random()), cy=innerHeight*(0.15+0.5*Math.random());
    const hue=Math.floor(Math.random()*360);
    for(let i=0;i<90;i++){
      const a=Math.random()*2*Math.PI, v=2+4*Math.random();
      parts.push({x:cx,y:cy,vx:Math.cos(a)*v,vy:Math.sin(a)*v,life:50+Math.random()*30,
                  c:"hsl("+(hue+Math.floor(Math.random()*40))+",95%,60%)"});
    }
  }
  let frame=0;
  (function tick(){
    ctx.clearRect(0,0,cv.width,cv.height);
    let alive=false;
    for(const p of parts){
      if(p.life<=0) continue;
      alive=true;
      p.x+=p.vx; p.y+=p.vy; p.vy+=0.06; p.vx*=0.985; p.life--;
      ctx.globalAlpha=Math.max(0,p.life/80);
      ctx.fillStyle=p.c;
      ctx.fillRect(p.x,p.y,3,3);
    }
    ctx.globalAlpha=1;
    if(alive && frame++<140) requestAnimationFrame(tick);
    else ctx.clearRect(0,0,cv.width,cv.height);
  })();
}
document.getElementById("ssToggle").onchange=refresh;
document.getElementById("hideDecided").onchange=refresh;
document.getElementById("showAuto").onchange=refresh;
refresh();
</script></body></html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--master", default=MASTER)
    p.add_argument("--scan", default=SCAN)
    p.add_argument("--fits-dir", default=sp.FITS_DIR)
    p.add_argument("--outdir", default=OUTDIR)
    p.add_argument("--nprocs", "-n", type=int, default=8)
    p.add_argument("--dpi", type=int, default=75)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-revised", action="store_true",
                   help="do NOT include the auto-corrected (catalog-revised) targets")
    args = p.parse_args()
    args.include_revised = not args.skip_revised
    outdir = args.outdir
    os.makedirs(os.path.join(outdir, "pngs"), exist_ok=True)

    m = pd.read_csv(args.master)
    m["revised"] = m["revised"].fillna("NA")   # pandas reads the literal "NA" as NaN
    scan = (pd.read_csv(args.scan).drop_duplicates("astro_id", keep="last").set_index("astro_id")
            if os.path.exists(args.scan) else None)
    todo = m[(m["first_letter"] == "e") & (m["revised"] == "NO")].copy()
    auto = m[(m["first_letter"] == "e") & (m["revised"] == "YES") &
             m["revision_source"].astype(str).str.startswith("catalog")].copy() \
        if args.include_revised else m.head(0)
    if args.limit:
        todo = todo.head(args.limit)
        auto = auto.head(args.limit)
    print(f"[gallery] {len(todo)} unrevised + {len(auto)} auto-corrected (catalog) EB TCEs")

    tasks, meta = [], []
    for _, r in todo.iterrows():
        aid = int(r["Astro ID"]); tic = int(r["TIC ID"])
        P0 = float(r["Per_original"]); dur = float(r["Dur"]) if np.isfinite(r["Dur"]) else np.nan
        s = scan.loc[aid] if (scan is not None and aid in scan.index) else None
        P_bls = float(s["P_bls"]) if (s is not None and pd.notna(s["P_bls"])) else np.nan
        t0 = float(s["t0_bls"]) if (s is not None and pd.notna(s["t0_bls"])) else np.nan
        risk = float(s["risk_score"]) if (s is not None and pd.notna(s["risk_score"])) else 0.5
        ss = bool(r["single_sector"])
        ss_bonus = 0.25 if ss else 0.0
        flags = []
        if s is not None:
            if str(s.get("flag", "")):
                flags.append(str(s["flag"]))
            if bool(s.get("half_period_sig", False)):
                if "half_period" not in flags:
                    flags.append("half_period")
        cat_note = ""
        if pd.notna(r.get("catalog_period")):
            cat_note = f"cat:{r['catalog_name']} P={float(r['catalog_period']):.4f} (ratio {float(r['catalog_ratio']):.2f})"
        cands = [{"kind": "P0", "period": P0},
                 {"kind": "2xP0", "period": 2 * P0},
                 {"kind": "P0/2", "period": 0.5 * P0}]
        if np.isfinite(P_bls):
            cands += [{"kind": "BLS", "period": P_bls},
                      {"kind": "2xBLS", "period": 2 * P_bls},
                      {"kind": "4xBLS", "period": 4 * P_bls},
                      {"kind": "BLS/2", "period": 0.5 * P_bls}]
        tasks.append((aid, tic, [(c["kind"], c["period"]) for c in cands], dur, t0))
        meta.append({"astro_id": aid, "tic": tic, "risk_base": round(risk - ss_bonus, 3),
                     "ss_bonus": ss_bonus, "flags": flags, "cat_note": cat_note,
                     "auto": "", "cands": cands})

    # auto-corrected cohort: verify/override the adopted catalog period
    for _, r in auto.iterrows():
        aid = int(r["Astro ID"]); tic = int(r["TIC ID"])
        P_adp = float(r["Per"]); P0 = float(r["Per_original"])
        dur = float(r["Dur"]) if np.isfinite(r["Dur"]) else np.nan
        cands = [{"kind": "ADP", "period": P_adp},
                 {"kind": "P0", "period": P0},
                 {"kind": "2xADP", "period": 2 * P_adp},
                 {"kind": "ADP/2", "period": 0.5 * P_adp}]
        tasks.append((aid, tic, [(c["kind"], c["period"]) for c in cands], dur, np.nan))
        meta.append({"astro_id": aid, "tic": tic, "risk_base": 0.0, "ss_bonus": 0.0,
                     "flags": [], "cat_note": "",
                     "auto": f"{r['revision_source']} @ {P_adp:.5f} d (was {P0:.5f})",
                     "cands": cands})

    # render PNG strips
    t0c = time.time()
    tally = {}
    with Pool(args.nprocs, initializer=_init_worker, initargs=(args.fits_dir, args.dpi, outdir)) as pool:
        for i, res in enumerate(pool.imap_unordered(render_target, tasks, chunksize=4), 1):
            tally[res["status"]] = tally.get(res["status"], 0) + 1
            if res["status"] == "fail":
                print(f"  {res['astro_id']}: {res.get('msg','')}", file=sys.stderr)
            el = time.time() - t0c
            rate = i / el if el else 0
            eta = (len(tasks) - i) / rate if rate else 0
            sys.stderr.write(f"\r[gallery] {i}/{len(tasks)} | ok {tally.get('ok',0)} "
                             f"skip {tally.get('skipped',0)} fail {tally.get('fail',0)} | "
                             f"{rate:4.1f}/s eta {int(eta)}s   ")
            sys.stderr.flush()
    sys.stderr.write("\n")

    # keep only targets whose PNG exists
    meta = [d for d in meta if os.path.exists(os.path.join(outdir, "pngs", f"{d['astro_id']}.png"))]
    page = HTML_HEAD.replace("%DATA%", json.dumps(meta))
    with open(os.path.join(outdir, "index.html"), "w") as f:
        f.write(page)
    print(f"[gallery] {len(meta)} targets -> {os.path.join(outdir,'index.html')}")
    print("[gallery] open it directly, or:  cd figures/period_gallery && python3 -m http.server 8811")


if __name__ == "__main__":
    main()
