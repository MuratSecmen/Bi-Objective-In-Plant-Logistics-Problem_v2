"""Repeated experiments on ONE instance; no cross-case pseudo-replication."""
import copy
import csv
import json
import math
import statistics
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

from metrics import (compute_nadir_reference, compute_true_extremes,
                     diversity_metrics, hv_fixed_ref_norm, hypervolume_2d,
                     nondominated_points, deduplicate_front)


def summarize_runs(rows):
    """Sample SD (ddof=1); expose missing/undefined values and failed runs."""
    summary = []
    for method in sorted({r["method"] for r in rows}):
        group = [r for r in rows if r["method"] == method]
        for metric in ("HV", "HV_norm", "Spacing", "Deb_Delta", "PF_size", "runtime_s"):
            values = [r[metric] for r in group if metric in r and
                      r[metric] is not None and math.isfinite(r[metric])]
            mean = statistics.mean(values) if values else None
            sd = statistics.stdev(values) if len(values) > 1 else None
            summary.append(dict(method=method, metric=metric, n_total=len(group),
                                n_defined=len(values),
                                n_failed=sum(r["status"] != "ok" for r in group),
                                mean=mean, std=sd,
                                mean_plus_minus_std=(f"{mean:.6g} +/- {sd:.6g}"
                                                     if sd is not None else "undefined")))
    return summary


def compare_methods_statistical(a, b, paired=True):
    """HV maps {seed: value}; positive rank-biserial favours method A.

    Wilcoxon assumes symmetric paired differences. Same seeds must be a
    predeclared blocking design, not retrofitted pairing. Independent runs use
    Mann-Whitney. These tests describe seed variability on this instance only.
    """
    from scipy.stats import mannwhitneyu, rankdata, wilcoxon
    if paired and set(a) != set(b):
        raise ValueError("Paired comparison requires identical seed sets")
    x = [a[k] for k in sorted(a)]
    y = [b[k] for k in sorted(b)]
    if min(len(x), len(y)) < 2 or not all(math.isfinite(v) for v in x + y):
        raise ValueError("Need at least two finite observations per method")
    if paired:
        # Round subtraction noise before ranking ties, not the original HV.
        diff = [round(v - w, 12) for v, w in zip(x, y)]
        nonzero = [d for d in diff if d != 0]
        if not nonzero:
            statistic, p_value, effect = 0.0, 1.0, 0.0
        else:
            result = wilcoxon(diff, zero_method="wilcox", alternative="two-sided",
                              method="auto")
            ranks = rankdata([abs(d) for d in nonzero], method="average")
            effect = sum(r if d > 0 else -r for r, d in zip(ranks, nonzero)) / sum(ranks)
            statistic, p_value = result.statistic, result.pvalue
        test = "wilcoxon"
    else:
        result = mannwhitneyu(x, y, alternative="two-sided", method="auto")
        statistic, p_value = result.statistic, result.pvalue
        effect = 2 * statistic / (len(x) * len(y)) - 1
        test = "mannwhitneyu"
    return dict(test=test, statistic=float(statistic), p_value=float(p_value),
                rank_biserial=float(effect), n_a=len(x), n_b=len(y))


def plot_hv_boxplot(by_method, output_path):
    """Save only; use an Agg canvas without modifying pyplot's global backend."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(7, 4), tight_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    labels = list(by_method)
    ax.boxplot([by_method[k] for k in labels])
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_ylabel("HV (shared fixed reference)")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    fig.clear()


def _write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: (None if isinstance(v, float) and not math.isfinite(v)
                              else v) for k, v in row.items()} for row in rows)


def run_experiment(args, cfg, run_once):
    count = args.n_seeds if args.n_seeds is not None else cfg["n_seeds"]
    if count < 1:
        raise ValueError("n_seeds must be >= 1")
    methods = args.methods or [cfg["run_mode"]]
    if len(methods) != len(set(methods)):
        raise ValueError("Duplicate methods")
    if not cfg["auto_verify"]:
        raise ValueError("Experiment metrics require auto_verify=true")
    base_seed = args.seed
    if base_seed is None:
        base_seed = cfg.get("experiment_seed")
    if base_seed is None:
        base_seed = cfg.get(methods[0] + "_seed", cfg.get("solomon_random_seed"))
    if base_seed is None:
        raise ValueError("Set an integer experiment_seed or --seed")
    seeds = list(range(int(base_seed), int(base_seed) + count))
    seeds_by_method = {method: [s + (i * count if args.independent else 0) for s in seeds]
                       for i, method in enumerate(methods)}
    cfg = copy.deepcopy(cfg)
    reference_fronts = []
    source = "pooled_verified_final_fronts"
    if args.reference_fronts:
        payload = json.loads(args.reference_fronts.read_text(encoding="utf-8"))
        if (payload["product_set_id"] != cfg["product_set_id"] or
                str(payload["num_products"]) != str(cfg["num_products"])):
            raise ValueError("Reference fronts must belong to this case/product count")
        reference_fronts = payload["fronts"]
        cfg["true_extremes"] = compute_true_extremes(
            p for front in reference_fronts for p in front)
        source = str(args.reference_fronts)
        if cfg["hv_ref_f1"] is None:
            cfg["hv_ref_f1"], cfg["hv_ref_f2"] = compute_nadir_reference(reference_fronts)
    ref = ((cfg["hv_ref_f1"], cfg["hv_ref_f2"])
           if cfg["hv_ref_f1"] is not None else None)
    if ref:
        hypervolume_2d([], ref)  # Validate before spending the experiment budget.
    root = Path(args.output_dir or cfg["output_dir"]) / (
        "experiment_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    root.mkdir(parents=True)
    records = []
    for method in methods:
        for seed in seeds_by_method[method]:
            local_cfg, local_args = copy.deepcopy(cfg), copy.copy(args)
            local_cfg["run_mode"] = method
            local_cfg["solomon_random_seed"] = seed
            for design in ("nsga2_old", "nsga2_new"):
                local_cfg[design + "_seed"] = seed
            local_args.output_dir = root / method / f"seed_{seed}"
            captured = []
            started = time.perf_counter()
            code = run_once(local_args, local_cfg, captured)
            record = captured[0] if captured else dict(front=[], conv_log=[], run_dir=None)
            record.update(method=method, seed=seed, exit_code=code,
                          status="ok" if code == 0 and record["front"] else "no_valid_front",
                          runtime_s=time.perf_counter() - started)
            records.append(record)
            # Preserve diagnostics even when verification's 'raise' stops a run.
            (root / "runs_raw.json").write_text(json.dumps(records, indent=2,
                                                          default=str), encoding="utf-8")
            if code not in (0, 3):
                return code  # Includes verify_on_fail=raise => exit 4, not success.
    fronts = reference_fronts + [r["front"] for r in records]
    if not any(fronts) and ref is None:
        missing = [dict(method=r["method"], seed=r["seed"], status=r["status"],
                        HV=None, HV_norm=None, PF_size=0, runtime_s=r["runtime_s"])
                   for r in records]
        _write_csv(root / "runs.csv", missing)
        _write_csv(root / "summary.csv", summarize_runs(missing))
        print(f"[experiment] No valid front; no reference can be estimated. See {root}")
        return 3
    if ref is None:
        ref = compute_nadir_reference(fronts)
    extremes = cfg.get("true_extremes")
    if extremes is None and any(fronts):
        extremes = compute_true_extremes(p for front in fronts for p in front)
    cfg.update(hv_ref_f1=ref[0], hv_ref_f2=ref[1], true_extremes=extremes,
               reference_source=source, seed_list=seeds,
               seeds_by_method=seeds_by_method, methods=methods)
    # Automatically filled effective config, not an invented pre-run nadir.
    (root / "effective_config.json").write_text(json.dumps(cfg, indent=2, default=str),
                                               encoding="utf-8")
    (root / "reference_fronts.json").write_text(json.dumps(dict(
        product_set_id=cfg["product_set_id"], num_products=cfg["num_products"],
        fronts=[r["front"] for r in records]), indent=2), encoding="utf-8")
    rows = []
    for record in records:
        front = nondominated_points(deduplicate_front(record["front"]))
        rows.append(dict(method=record["method"], seed=record["seed"],
                         status=record["status"], HV=hypervolume_2d(front, ref),
                         HV_norm=hv_fixed_ref_norm(front, ref), PF_size=len(front),
                         runtime_s=record["runtime_s"],
                         outside_reference=sum(p[0] > ref[0] or p[1] > ref[1] for p in front),
                         **diversity_metrics(front, extremes)))
        for row in record["conv_log"]:
            points = json.loads(row["front_points_json"])
            row.update(hv_fixed_ref=hypervolume_2d(points, ref),
                       hv_fixed_ref_norm=hv_fixed_ref_norm(points, ref),
                       hv_ref_f1=ref[0], hv_ref_f2=ref[1],
                       **diversity_metrics(points, extremes))
        if record["run_dir"]:
            from nsga2 import _conv_log_path
            log_cfg = dict(cfg, nsga2={"seed": record["seed"]})
            _write_csv(_conv_log_path(record["run_dir"], log_cfg, record["method"]),
                       record["conv_log"])
    _write_csv(root / "runs.csv", rows)
    _write_csv(root / "summary.csv", summarize_runs(rows))
    comparisons = []
    for a, b in combinations(methods, 2):
        groups = [[r for r in rows if r["method"] == m] for m in (a, b)]
        # Never silently discard failed seeds or fabricate a statistical result.
        if count < 2 or any(r["status"] != "ok" for g in groups for r in g):
            comparisons.append(dict(method_a=a, method_b=b, status="insufficient_valid_runs"))
            continue
        comparisons.append(dict(method_a=a, method_b=b, status="ok",
            **compare_methods_statistical(*[{r["seed"]: r["HV_norm"] for r in g}
                                            for g in groups], paired=not args.independent)))
    # Holm correction when testing more than one method pair.
    tested = sorted((r for r in comparisons if "p_value" in r), key=lambda r: r["p_value"])
    adjusted = 0.0
    for i, row in enumerate(tested):
        adjusted = max(adjusted, min(1.0, (len(tested) - i) * row["p_value"]))
        row["p_holm"] = adjusted
    _write_csv(root / "comparisons.csv", comparisons)
    plot_hv_boxplot({m: [r["HV"] for r in rows if r["method"] == m] for m in methods},
                    root / "hv_boxplot.png")
    print(f"[experiment] {len(rows)} runs; metrics and shared reference -> {root}")
    return 0 if all(r["status"] == "ok" for r in rows) else 3
