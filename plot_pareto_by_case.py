"""
plot_pareto_by_case.py

Cikti yapisi:
    <output>/3-3/n5/case1.png, case2.png, ..., tum_caseler.png, n5_veriler.xlsx
    <output>/3-3/n6/case1.png, case2.png, ..., tum_caseler.png, n6_veriler.xlsx

n ve case degerleri otomatik kesfedilir.
Noktalar arasinda cizgi YOK, sadece bagimsiz markerlar.

Kullanim:
    python plot_pareto_by_case.py --config 3-3 --results-root results
    python plot_pareto_by_case.py --results-root results
"""

from __future__ import annotations
import argparse, re
from pathlib import Path

N_PRODUCTS_RE = re.compile(r"Sliced first (\d+) products")
CASE_RE = re.compile(r"product_set_id\s*=\s*(\w+)")
CONFIG_DIR_RE = re.compile(r"^(\d+)-(\d+)$")

STYLE = {
    "mip_route": dict(color="#2E7D32", marker="X", ms=13, label="MIP (route duration)"),
    "mip_wait":  dict(color="#1B5E20", marker="o", ms=9,  label="MIP (wait time)"),
    "nsga2_old": dict(color="#1E2761", marker="+", ms=13, label="NSGA-II Tasarim A"),
    "nsga2_new": dict(color="#E8734A", marker="_", ms=13, label="NSGA-II Tasarim B"),
}
NAVY = "#1E2761"

def _find_files(root, pat):
    return sorted(root.rglob(pat)) if root.exists() else []

def _n_case_from_log(lp):
    if not lp.exists(): return None, None
    t = lp.read_text(encoding="utf-8", errors="ignore")
    nm = N_PRODUCTS_RE.search(t); cm = CASE_RE.search(t)
    return (int(nm.group(1)) if nm else None, cm.group(1) if cm else None)

def _classify(name):
    lo = name.lower()
    if "wait" in lo: return "wait"
    if "route" in lo or "duration" in lo: return "route"
    return "unknown"

def _read_pareto_summary(xp, gap_threshold=0.05):
    """Read (f1, f2) points from a pareto_summary workbook, splitting them by
    whether the MIP proved optimality within `gap_threshold`.

    Returns (kept, dropped) where:
      - kept    = points whose mip_gap <= gap_threshold (proven near-optimal),
      - dropped = points whose mip_gap >  gap_threshold (time-limited, NOT
                  proven optimal — excluded from the reported exact frontier).

    Rationale: an epsilon-constraint node that hits the time limit returns a
    feasible incumbent, not a proven-optimal one. Such a point can sit above
    the true frontier (e.g. the stray wait-time circles at high f1) yet still
    appear "non-dominated" in a sparse region, so a pure dominance filter
    cannot remove it. Gating on the recorded mip_gap is the principled fix:
    only optimality-proven points define the exact frontier. If the workbook
    predates gap logging (no mip_gap column) every point is kept and flagged
    via gap=None, preserving backward compatibility.
    """
    import openpyxl
    wb = openpyxl.load_workbook(xp, data_only=True)
    if "pareto_summary" not in wb.sheetnames: return [], []
    ws = wb["pareto_summary"]; rows = list(ws.iter_rows(values_only=True))
    if len(rows)<2: return [], []
    hdr = rows[0]; idx = {str(h):i for i,h in enumerate(hdr) if h}
    if "route_duration_min" not in idx or "total_wait_min" not in idx: return [], []
    has_gap = "mip_gap" in idx
    kept, dropped = [], []
    for r in rows[1:]:
        f1,f2 = r[idx["route_duration_min"]], r[idx["total_wait_min"]]
        if f1 is None or f2 is None: continue
        gap = r[idx["mip_gap"]] if has_gap else None
        pt = (float(f1), float(f2))
        if gap is not None and float(gap) > gap_threshold:
            dropped.append(pt)
        else:
            kept.append(pt)
    return kept, dropped

def collect_mip(mip_dir, n, case, gap_threshold=0.05):
    """Collect MIP frontier points, split into route/wait sweeps.

    Returns (rp, wp, rp_drop, wp_drop) where *_drop are the points excluded
    for exceeding the optimality-gap threshold, kept only for the audit sheet.
    """
    if not mip_dir.exists(): return [],[],[],[]
    rp, wp, rp_drop, wp_drop = [], [], [], []
    for td in mip_dir.iterdir():
        if not td.is_dir(): continue
        kind = _classify(td.name)
        if kind=="unknown": continue
        cands = []
        # run_*.log HER DERINLIKTE olabilir -- eski yapida dogrudan
        # multiobj_.../run_....log, yeni yapida (run_mip_wait.py
        # duzeltmesinden sonra) case1_n5/multiobj_.../run_....log gibi
        # bir kat daha derin. rglob ikisini de bulur.
        for log_path in td.rglob("run_*.log"):
            run_dir = log_path.parent
            if case not in str(run_dir.relative_to(td)):
                continue
            rn, rc = _n_case_from_log(log_path)
            if rn == n and rc == case:
                cands.append(run_dir)
        if not cands: continue
        latest = max(cands, key=lambda d: d.stat().st_mtime)
        sums = sorted(latest.glob("pareto_summary_*.xlsx"))
        if not sums: continue
        kept, dropped = _read_pareto_summary(sums[0], gap_threshold=gap_threshold)
        if kind=="route":
            rp, rp_drop = kept, dropped
        else:
            wp, wp_drop = kept, dropped
    return rp, wp, rp_drop, wp_drop

def collect_nsga2(nsga2_dir, n, case, label):
    import openpyxl
    cands = []
    for lp in _find_files(nsga2_dir, "run_*.log"):
        txt = lp.read_text(encoding="utf-8", errors="ignore")
        if label not in txt: continue
        rn,rc = _n_case_from_log(lp)
        if rn!=n or rc!=case: continue
        cands.append(lp)
    if not cands: return []
    lp = max(cands, key=lambda p: p.stat().st_mtime)
    pts = []
    xls = sorted(lp.parent.glob("result_*.xlsx"))
    if not xls: return []
    wb = openpyxl.load_workbook(xls[0], data_only=True)
    if "pareto" not in wb.sheetnames: return []
    for r in list(wb["pareto"].iter_rows(values_only=True))[1:]:
        if r[0] is not None and r[1] is not None:
            pts.append((float(r[0]),float(r[1])))
    return sorted(set(pts))

def discover(mip_dir, nsga2_dir):
    ns, cs = set(), set()
    for d in (mip_dir, nsga2_dir):
        if not d.exists(): continue
        for lp in d.rglob("run_*.log"):
            n,c = _n_case_from_log(lp)
            if n: ns.add(n)
            if c: cs.add(c)
    return sorted(ns), sorted(cs)

def _pareto_filter(pts, eps=1e-9):
    """Return the non-dominated subset of (f1, f2) points.

    Both objectives are minimised (f1 = route duration, f2 = total wait).
    A point p is dominated if some other point q satisfies q.f1 <= p.f1 and
    q.f2 <= p.f2 with at least one strict inequality. The eps guard keeps
    near-identical points (numerically equal within 1e-9) as mutual
    non-dominated rather than letting floating-point noise delete one of a
    tied pair. Order of the input is irrelevant; the returned list is sorted
    by f2 ascending for a clean left-to-right frontier.
    """
    pts = list(pts)
    keep = []
    for i, (f1, f2) in enumerate(pts):
        dominated = False
        for j, (g1, g2) in enumerate(pts):
            if j == i:
                continue
            if (g1 <= f1 + eps and g2 <= f2 + eps and
                    (g1 < f1 - eps or g2 < f2 - eps)):
                dominated = True
                break
        if not dominated:
            keep.append((f1, f2))
    # de-duplicate exact repeats, then sort by f2 for a tidy frontier
    keep = sorted(set(keep), key=lambda p: (p[1], p[0]))
    return keep


def _split_by_source(combined_nd, rp, wp, eps=1e-6):
    """Given the merged non-dominated MIP frontier, tag each surviving point
    with the sweep it came from so the plot can keep the X / o distinction.

    A surviving point is attributed to 'route' if it matches (within eps) a
    point produced by the route-duration sweep, otherwise to 'wait'. Points
    present in BOTH sweeps are attributed to 'route' by convention (arbitrary
    but consistent). Returns (route_pts, wait_pts) each already
    non-dominated, so the union is the single true MIP frontier and no
    dominated marker (e.g. the stray wait-time circles) is drawn.
    """
    def _match(p, pool):
        return any(abs(p[0]-q[0]) <= eps and abs(p[1]-q[1]) <= eps for q in pool)
    r_out, w_out = [], []
    for p in combined_nd:
        if _match(p, rp):
            r_out.append(p)
        elif _match(p, wp):
            w_out.append(p)
        else:
            # Shouldn't happen (every surviving point came from one sweep),
            # but keep it visible under 'route' rather than dropping silently.
            r_out.append(p)
    return r_out, w_out


def _scatter(ax, key, pts):
    if not pts: return
    st = STYLE[key]
    xs = [p[1] for p in pts]; ys = [p[0] for p in pts]
    kw = dict(s=st["ms"]**2, marker=st["marker"], color=st["color"],
              label=st["label"], zorder=3)
    if key.startswith("mip"): kw["edgecolor"]="black"; kw["linewidth"]=0.8
    else: kw["linewidth"]=2.2
    ax.scatter(xs, ys, **kw)

def draw_single(ax, rp, wp, op, np_, title):
    _scatter(ax,"mip_route",rp); _scatter(ax,"mip_wait",wp)
    _scatter(ax,"nsga2_old",op); _scatter(ax,"nsga2_new",np_)
    ax.set_title(title, fontsize=12, fontweight="bold", color=NAVY)
    ax.set_xlabel("Total wait time (min) — f2", fontsize=9.5)
    ax.set_ylabel("Route duration (min) — f1", fontsize=9.5)
    ax.grid(True, alpha=0.3)
    if not any([rp,wp,op,np_]):
        ax.text(0.5,0.5,"veri yok",transform=ax.transAxes,ha="center",va="center",fontsize=10,color="gray")

def save_single_png(n, case, rp, wp, op, np_, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5,6))
    draw_single(ax,rp,wp,op,np_,case)
    h,l = ax.get_legend_handles_labels()
    fig.suptitle(f"MIP vs NSGA-II — {case}  |P|={n}",fontsize=13,fontweight="bold",color="white",y=1.0)
    fig.patches.append(matplotlib.patches.Rectangle((0,0.955),1,0.06,transform=fig.transFigure,facecolor=NAVY,zorder=-1,clip_on=False))
    if h: fig.legend(h,l,loc="lower center",ncol=2,fontsize=9,bbox_to_anchor=(0.5,-0.06),frameon=False)
    fig.tight_layout(rect=[0,0.05,1,0.94])
    fig.savefig(path,dpi=160,bbox_inches="tight"); plt.close(fig)

def save_grid_png(n, all_data, cases, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nc=len(cases); ncols=2 if nc>1 else 1; nrows=(nc+ncols-1)//ncols
    fig,axes=plt.subplots(nrows,ncols,figsize=(6.5*ncols,5.2*nrows))
    axes=[axes] if nc==1 else axes.flatten()
    hl=None
    for i,case in enumerate(cases):
        d=all_data.get(case,{})
        draw_single(axes[i],d.get("mip_route",[]),d.get("mip_wait",[]),d.get("nsga2_old",[]),d.get("nsga2_new",[]),case)
        if hl is None and axes[i].get_legend_handles_labels()[0]: hl=axes[i].get_legend_handles_labels()
    for j in range(nc,len(axes)): axes[j].axis("off")
    fig.suptitle(f"MIP vs NSGA-II — tum case'ler  |P|={n}",fontsize=15,fontweight="bold",color="white",y=0.995)
    fig.patches.append(matplotlib.patches.Rectangle((0,0.965),1,0.05,transform=fig.transFigure,facecolor=NAVY,zorder=-1,clip_on=False))
    if hl: fig.legend(*hl,loc="lower center",ncol=len(hl[0]),fontsize=10,bbox_to_anchor=(0.5,-0.02),frameon=False)
    fig.tight_layout(rect=[0,0.03,1,0.95])
    fig.savefig(path,dpi=160,bbox_inches="tight"); plt.close(fig)

def save_excel(n, all_data, cases, path):
    import openpyxl
    from openpyxl.styles import Font, PatternFill
    wb=openpyxl.Workbook(); ws=wb.active; ws.title="all"
    ws.append(["case","source","route_duration_min","total_wait_min"])
    for c in ws[1]: c.font=Font(bold=True,color="FFFFFF"); c.fill=PatternFill(start_color="1E2761",end_color="1E2761",fill_type="solid")
    labels={"mip_route":"MIP - route duration","mip_wait":"MIP - wait time","nsga2_old":"NSGA-II Tasarim A","nsga2_new":"NSGA-II Tasarim B"}
    for case in cases:
        d=all_data.get(case,{})
        wsc=wb.create_sheet(case[:31])
        wsc.append(["source","route_duration_min","total_wait_min"])
        for c in wsc[1]: c.font=Font(bold=True,color="FFFFFF"); c.fill=PatternFill(start_color="1E2761",end_color="1E2761",fill_type="solid")
        for sk,sl in labels.items():
            for f1,f2 in d.get(sk,[]):
                wsc.append([sl,f1,f2]); ws.append([case,sl,f1,f2])
    # ---- audit sheet: raw MIP points with a dominated flag ----
    # Documents exactly which raw MIP points were dropped by the combined
    # non-dominated filter, so the frontier cleanup is transparent (a
    # dropped point is not deleted from the record, only from the plotted
    # frontier). A run of dropped wait-time points at high f1 is the
    # signature of a time-limited, non-proven-optimal sweep.
    wsa = wb.create_sheet("mip_raw_audit")
    wsa.append(["case","sweep","route_duration_min","total_wait_min","kept_on_frontier"])
    for c in wsa[1]:
        c.font=Font(bold=True,color="FFFFFF")
        c.fill=PatternFill(start_color="1E2761",end_color="1E2761",fill_type="solid")
    for case in cases:
        d=all_data.get(case,{})
        kept={ (round(f1,6),round(f2,6)) for f1,f2 in (d.get("mip_route",[])+d.get("mip_wait",[])) }
        gapdrop={ (round(f1,6),round(f2,6))
                  for f1,f2 in (d.get("_mip_route_gapdrop",[])+d.get("_mip_wait_gapdrop",[])) }
        # audit both the proven-optimal raw points and the gap-dropped ones
        for sweep_key,gap_key,sweep_lbl in (
                ("_mip_route_raw","_mip_route_gapdrop","route sweep"),
                ("_mip_wait_raw","_mip_wait_gapdrop","wait sweep")):
            for f1,f2 in d.get(sweep_key,[]):
                flag = "yes" if (round(f1,6),round(f2,6)) in kept else "NO (dominated)"
                wsa.append([case,sweep_lbl,f1,f2,flag])
            for f1,f2 in d.get(gap_key,[]):
                wsa.append([case,sweep_lbl,f1,f2,"NO (gap>5%, not proven optimal)"])
    wb.save(path)

def process_config(mip_dir, nsga2_dir, config_label, output_root):
    ns, cases = discover(mip_dir, nsga2_dir)
    if not ns or not cases:
        print(f"[plot] {config_label}: veri bulunamadi, atlaniyor"); return False
    print(f"[plot] {config_label}: n={ns}  case'ler={cases}")
    for n in ns:
        folder = output_root / config_label / f"n{n}"
        folder.mkdir(parents=True, exist_ok=True)
        all_data = {}
        for case in cases:
            rp_raw, wp_raw, rp_gapdrop, wp_gapdrop = collect_mip(mip_dir, n, case)
            op = collect_nsga2(nsga2_dir, n, case, "Tasarim A")
            np_ = collect_nsga2(nsga2_dir, n, case, "Tasarim B")
            # Two-stage cleanup of the two epsilon-constraint sweeps:
            #   Stage 1 (inside collect_mip): drop points whose recorded
            #     mip_gap > 5% — these hit the time limit and are NOT proven
            #     optimal, so they must not define the exact frontier even if
            #     they look non-dominated in a sparse region.
            #   Stage 2 (here): merge the two proven-optimal sweeps and keep
            #     the non-dominated union, re-tagging by source so X / o
            #     markers still show provenance.
            combined_nd = _pareto_filter(rp_raw + wp_raw)
            rp, wp = _split_by_source(combined_nd, rp_raw, wp_raw)
            all_data[case] = {"mip_route":rp,"mip_wait":wp,"nsga2_old":op,"nsga2_new":np_,
                              "_mip_route_raw":rp_raw,"_mip_wait_raw":wp_raw,
                              "_mip_route_gapdrop":rp_gapdrop,"_mip_wait_gapdrop":wp_gapdrop}
            n_domdrop = (len(rp_raw) + len(wp_raw)) - len(combined_nd)
            n_gapdrop = len(rp_gapdrop) + len(wp_gapdrop)
            print(f"[plot]   n={n} {case}: MIP-route={len(rp_raw)}->{len(rp)} "
                  f"MIP-wait={len(wp_raw)}->{len(wp)} "
                  f"(gap>5% dropped={n_gapdrop}, dominated dropped={n_domdrop}) "
                  f"TasA={len(op)} TasB={len(np_)}")
            save_single_png(n, case, rp, wp, op, np_, folder / f"{case}.png")
        save_grid_png(n, all_data, cases, folder / "tum_caseler.png")
        save_excel(n, all_data, cases, folder / f"n{n}_veriler.xlsx")
        print(f"[plot]   -> {folder}")
    return True

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root",default="results")
    p.add_argument("--config",default=None)
    p.add_argument("--output-dir",default="pareto_charts")
    args=p.parse_args()
    root=Path(args.results_root); out=Path(args.output_dir)
    if args.config: configs=[args.config]
    else:
        configs=sorted(d.name for d in root.iterdir() if d.is_dir() and CONFIG_DIR_RE.match(d.name)) if root.exists() else []
    if not configs: print(f"[plot] {root} altinda yapilandirma bulunamadi."); return 1
    print(f"[plot] Yapilandirmalar: {configs}\n")
    ok=False
    for cfg in configs:
        cd=root/cfg; mip=cd/"mip"; nsga2=cd/"nsga2"
        if not mip.exists() and not nsga2.exists():
            print(f"[plot] {cfg}: mip/ ve nsga2/ yok, atlaniyor"); continue
        if process_config(mip, nsga2, cfg, out): ok=True
    if not ok: print("\n[plot] Hicbir veri bulunamadi."); return 1
    print(f"\n[plot] BITTI — ciktilar -> {out}"); return 0

if __name__=="__main__":
    import sys; sys.exit(main())
