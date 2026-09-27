"""Aggregate a campaign produced by run_all.py.

Outputs (in the campaign root):
  master_runs.csv     every method x seed x instance row, with HV_ratio_to_MIP
  master_summary.csv  per instance and method: mean, sample SD, n
  mip_summary.csv     per instance: MIP front size, total sweep time, max gap
  friedman.txt        Friedman test across instances on instance-mean HV_norm,
                      plus pairwise Wilcoxon signed-rank tests with Holm.

The unit of replication for the cross-instance tests is the INSTANCE; seeds
only enter through the instance mean. Within-instance seed tests live in each
experiment's comparisons.csv.
"""
import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from metrics import hypervolume_2d


def latest(paths):
    paths = sorted(paths)
    return paths[-1] if paths else None


def collect(root):
    runs, mips = [], []
    for inst_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        instance = inst_dir.name
        exp_dir = latest(inst_dir.glob("experiment/experiment_*"))
        ref = None
        if exp_dir and (exp_dir / "runs.csv").exists():
            eff = json.loads((exp_dir / "effective_config.json").read_text(encoding="utf-8"))
            ref = (float(eff["hv_ref_f1"]), float(eff["hv_ref_f2"]))
            df = pd.read_csv(exp_dir / "runs.csv")
            df.insert(0, "instance", instance)
            runs.append(df)
        summary = latest(inst_dir.glob("mip/**/pareto_summary*.xlsx"))
        if summary is not None:
            m = pd.read_excel(summary)
            feas = m.dropna(subset=["route_duration_min", "total_wait_min"])
            front = list(zip(feas["route_duration_min"], feas["total_wait_min"]))
            mips.append(dict(
                instance=instance, mip_points=len(front),
                mip_sweep_time_s=float(feas["runtime_s"].sum()),
                mip_max_gap=float(feas["mip_gap"].max()) if len(feas) else math.nan,
                mip_HV=hypervolume_2d(front, ref) if (ref and front) else math.nan,
            ))
    runs = pd.concat(runs, ignore_index=True) if runs else pd.DataFrame()
    return runs, pd.DataFrame(mips)


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adjusted, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adjusted[i] = running
    return adjusted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/campaign"))
    args = parser.parse_args()
    runs, mips = collect(args.root)
    if runs.empty:
        print("No experiment results found.")
        return
    if not mips.empty:
        runs = runs.merge(mips[["instance", "mip_HV"]], on="instance", how="left")
        runs["HV_ratio_to_MIP"] = runs["HV"] / runs["mip_HV"]
        mips.to_csv(args.root / "mip_summary.csv", index=False)
    runs.to_csv(args.root / "master_runs.csv", index=False)

    metrics = [c for c in ("HV_norm", "HV_ratio_to_MIP", "PF_size", "Spacing",
                           "Deb_Delta", "runtime_s") if c in runs]
    ok = runs[runs["status"] == "ok"]
    summary = ok.groupby(["instance", "method"])[metrics].agg(["mean", "std", "count"])
    summary.to_csv(args.root / "master_summary.csv")

    means = ok.groupby(["instance", "method"])["HV_norm"].mean().unstack("method")
    means = means.dropna()
    lines = [f"Instances with complete results: {len(means)}",
             f"Methods: {list(means.columns)}", ""]
    if len(means) >= 2 and means.shape[1] >= 3:
        if (means.nunique(axis=1) == 1).all():
            lines.append("Friedman: all methods tied on every instance (test undefined).")
        else:
            stat, p = friedmanchisquare(*[means[c] for c in means.columns])
            lines.append(f"Friedman chi2={stat:.4f}  p={p:.4g}  (instance-mean HV_norm)")
        lines.append("Mean ranks (1 = best, higher HV is better):")
        ranks = means.rank(axis=1, ascending=False).mean()
        lines += [f"  {m}: {r:.2f}" for m, r in ranks.sort_values().items()]
        pairs = list(combinations(means.columns, 2))
        raw = []
        for a, b in pairs:
            diff = (means[a] - means[b]).round(12)
            raw.append(1.0 if (diff == 0).all() else
                       wilcoxon(means[a], means[b], zero_method="wilcox").pvalue)
        lines.append("")
        lines.append("Pairwise Wilcoxon signed-rank over instances (Holm):")
        for (a, b), p_raw, p_adj in zip(pairs, raw, holm(raw)):
            lines.append(f"  {a} vs {b}: p={p_raw:.4g}  p_holm={p_adj:.4g}")
        lines.append("")
        lines.append(f"Note: with {len(means)} instances the smallest attainable "
                     f"two-sided exact Wilcoxon p is {2 / 2 ** len(means):.4g}.")
    else:
        lines.append("Friedman test needs >= 2 complete instances and >= 3 methods.")
    (args.root / "friedman.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWritten to {args.root}")


if __name__ == "__main__":
    main()
