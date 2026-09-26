# Research version — Internal Logistics PD-VRP

Full research version of the Internal Logistics PD-VRP project: a bi-objective
pickup-and-delivery routing problem with release dates and shift-based time
windows, minimising total route duration (f₁) and total part waiting time (f₂).
It includes:

- The full MIP formulation (single-objective and ε-constraint multi-objective),
  extended to a depot-as-customer formulation — products may have the depot
  itself as origin or destination, not only work-centre-to-work-centre moves.
- The Solomon I1 construction heuristic, with 2-opt and Or-opt-on-pairs local
  search, plus the full repair ladder (bounded-depth backtracking + multi-start
  over alpha-biases + extended biases + random alpha-restarts + shuffled-order
  restarts).
- Two NSGA-II designs (Design A / Design B), kept side by side, each with
  per-generation convergence logging.
- A repeated-experiment driver: multi-seed runs, one shared frozen hypervolume
  reference per instance, and nonparametric comparison tests.

A teaching edition of an earlier stage of this codebase is published
separately at
[`MuratSecmen/Bi-Objective-In-Plant-Logistics-Problem`](https://github.com/MuratSecmen/Bi-Objective-In-Plant-Logistics-Problem).
It is intentionally scoped to the MIP and the Solomon heuristic for
coursework, and excludes NSGA-II and the experiment machinery found here.

## What's inside

```
Bi-Objective-In-Plant-Logistics-Problem_v2/
├── run_model.py              # MIP + heuristic + NSGA-II dispatch + result writers
├── solomon.py                # Solomon I1 + 2-opt + Or-opt + repair ladder
├── nsga2.py                  # NSGA-II Design A (nsga2_old) and Design B (nsga2_new)
├── metrics.py                # Hypervolume, reference point, Spacing, Deb's Δ
├── experiments.py            # Multi-seed driver, summaries, statistical tests
├── verify.py                 # Solution validator
├── visualize.py              # Gantt / route / wait / route-timings PNGs
├── plot_pareto_by_case.py    # MIP vs. NSGA-II A vs. NSGA-II B Pareto overlay
├── requirements.txt          # Runtime dependencies
├── README.md                 # This file
├── tests/
│   ├── test_research_metrics.py  # Unit tests for metrics and the experiment driver
│   └── smoke_run.py              # Small real-data integration check
└── inputs_new/
    ├── config.xlsx           # Config (MIP + Solomon + NSGA-II + experiment keys)
    ├── nodes.xlsx            # Node list (depot + work centres, N1–N60)
    ├── vehicles.xlsx         # Fleet definition (id, capacity, max_route, type, active)
    ├── products.xlsx         # Four sheets: case1, case2, case3, case4
    └── distances_minutes.xlsx  # OD travel-time matrix (long format)
```

## Run modes

Five run modes are exposed through `inputs_new/config.xlsx`:

| `run_mode`         | What it does |
|--------------------|--------------|
| `single_objective` | One MIP solve. Optimises `primary_obj` under `limit_on_constraint_obj`. |
| `multi_objective`  | Adaptive ε-constraint sweep over the Pareto front of `(route_duration, wait_time)`. |
| `heuristic`        | Solomon I1 construction + 2-opt + Or-opt; no MIP. Much faster than the MIP; returns one feasible solution. |
| `nsga2_old`        | NSGA-II Design A — two-layer chromosome (assignment + priority). |
| `nsga2_new`        | NSGA-II Design B — single-layer chromosome (assignment only), greedy `(rt_score, c₁)` decode. |

The `objective_method` parameter selects between Gurobi's `lexicographic`
multi-objective treatment and `augmented_eps`, an augmented ε-constraint
formulation (AUGMECON2-style slack augmentation). The two carry different
optimality guarantees; neither is a weighted-sum scalarisation.

## Quick start

```bash
cd Bi-Objective-In-Plant-Logistics-Problem_v2
pip install -r requirements.txt

# 1. (Optional) Edit inputs_new/config.xlsx to pick a case, n, and run_mode.
# 2. Run:
python run_model.py
```

`--inputs` / `--config` override the default `inputs_new/` location. MIP
outputs land in `results/<label>_<timestamp>/`:

- `result_*.xlsx` — summary sheet + variable sheets (`x_used`, `assignment_f`,
  `itinerary`, `wait_w`, `route_timings`, …; Pareto/history sheets for NSGA-II)
- `route.png`, `gantt.png`, `waits.png`, `route_timings.png` — auto-generated
  visualisations (when `auto_visualize=True`)
- `pareto_*.png` — Pareto front plot (`multi_objective` and NSGA-II modes)
- `gurobi.log` — Gurobi solver log (`single_objective` and `multi_objective` only)

In `heuristic`, `nsga2_old` and `nsga2_new` modes, a run is automatically
repeated `n_seeds` times (default 10) and written under
`results/experiment_<timestamp>/<method>/seed_<seed>/`. See
[Repeated experiments and metrics](#repeated-experiments-and-metrics).

## Key configuration knobs

### MIP

| Parameter | Purpose |
|---|---|
| `run_mode` | `single_objective`, `multi_objective`, `heuristic`, `nsga2_old`, or `nsga2_new` |
| `primary_obj` / `constraint_obj` | `route_duration` or `wait_time` |
| `objective_method` | `lexicographic` or `augmented_eps` |
| `limit_on_constraint_obj` | upper bound on secondary objective (`single_objective` mode) |
| `eps_step` | ε decrement between `multi_objective` iterations (minutes) |
| `product_set_id` | `case1`, `case2`, `case3`, or `case4` |
| `num_products` | integer slice, or `all` |
| `shift_duration_min` | `T_max` in shift-relative minutes (480 = 8 h) |
| `time_limit_seconds` | Gurobi wall-clock cap |
| `mip_gap` | Gurobi relative optimality tolerance |

Valid-inequality / LP-tightening flags (`add_*_cut`, `tight_*`, `mtz_type`,
`break_vehicle_symmetry`, `use_indicator_constraints`) are all on by default.
Toggle them one at a time to see how each affects the LP relaxation and solve
time.

### Solomon heuristic

| Parameter | Purpose |
|---|---|
| `heuristic_objective` | Bias label (`route_duration` or `wait_time`). |
| `alpha_1`, `alpha_2`, `alpha_3` | Weights in `c_1 = α₁·Δd + α₂·ΔT + α₃·ΔW`. Defaults are the strict-lex triple `(0.0001, 1, 0.01)` for the route-duration bias. |
| `lambda_c2` | Scaling on `c(h, o_p)` in `c_2`. |
| `heuristic_wait_limit` | ε bound on total wait. |
| `apply_2opt`, `apply_or_opt` | Toggle the two local-search phases. |
| `solomon_backtrack_max_depth` | Bounded-depth backtracking when greedy reaches `infeasible_unrouted`. n=20 instances typically need depth ≥ 15. |
| `solomon_multistart` | After backtracking fails, retry under the other stock biases (route_duration → wait_time → balanced → distance → distance+wait → distance+duration). |
| `solomon_random_restarts` | After the 6 stock biases fail, try this many log-uniform random alpha triples. |
| `solomon_shuffle_restarts` | After random alpha-restarts fail, try this many shuffled-order restarts (random insertion order, no c2 ranking, no backtracking). |
| `solomon_random_seed` | Seed for reproducible random/shuffle restarts. |

### NSGA-II

Keys prefixed `nsga2_old_` apply to Design A, `nsga2_new_` to Design B.
For a fair comparison, both designs must use identical `pop_size` and `n_gen`.

| Parameter | Purpose |
|---|---|
| `nsga2_n_workers` | Parallel decode pool size. Blank → CPU cores − 1. |
| `*_pop_size`, `*_n_gen` | Population size and number of generations. |
| `*_seed` | Random seed (overridden per repetition by the experiment driver). |
| `*_p_crossover` | Crossover probability. |
| `nsga2_old_p_assignment_mutation`, `nsga2_old_p_priority_mutation` | Design A mutation rates (assignment / priority layer). |
| `nsga2_new_p_mutation`, `nsga2_new_p_swap` | Design B mutation rates (transfer / swap). |
| `*_n_solomon_seeds` | Number of Solomon-seeded individuals in the initial population. |
| `*_time_limit_sec` | Wall-clock budget per run (seconds). |
| `*_results_every` | Write a decoder-only Pareto snapshot every N generations (0 disables). |

### Experiments and verification

| Parameter | Purpose |
|---|---|
| `n_seeds` | Repetitions per method (default 10). CLI: `--n-seeds`. |
| `experiment_seed` | First seed of the shared seed list. Blank → the first method's configured seed. CLI: `--seed`. |
| `hv_ref_f1`, `hv_ref_f2` | Fixed HV reference point for this instance. Blank → estimated once after the experiment (see below). |
| `auto_verify` | Validate solutions with `verify.py`. Must be true for experiments. |
| `verify_on_fail` | `raise` stops the run with exit code 4 if any solution fails verification. |

### Test cases

`products.xlsx` ships with four sheets:

- **case1** — distinct origin/destination pairs (basic).
- **case2** — concentrated locations (shared nodes).
- **case3** — shared-pickup cluster (tight precedence).
- **case4** — larger 100-product set (use small `num_products` slices).

Each case includes a small number of depot-touching products (`op="h"` or
`dp="h"`), exercising the depot-as-customer extension.

## Requirements

- Python 3.10+
- Packages in `requirements.txt` (`numpy`, `pandas`, `openpyxl`, `scipy`,
  `matplotlib`, `gurobipy`)
- A valid Gurobi licence for `single_objective` and `multi_objective`

The Solomon heuristic and both NSGA-II designs do **not** need Gurobi. NSGA-II
is a from-scratch implementation (no `pymoo` dependency).

## Repeated experiments and metrics

### Running

```bash
python run_model.py --methods nsga2_old nsga2_new --n-seeds 10 --seed 42
python run_model.py --methods nsga2_old nsga2_new --n-seeds 10 --seed 42 --independent
python run_model.py --methods nsga2_old nsga2_new --reference-fronts mip_front.json
```

All runs of one experiment belong to ONE instance (`product_set_id`,
`num_products`). Do not pool different cases or product counts into one
statistical sample.

### Outputs

| File | Content |
|---|---|
| `runs.csv` | One row per method × seed: HV, HV_norm, PF_size, runtime, Spacing, Deb_Delta, points outside the reference box. |
| `summary.csv` | Mean and sample SD (ddof = 1) per metric, with defined and failed counts. |
| `comparisons.csv` | Test statistic, p-value, Holm-adjusted p-value and rank-biserial effect size per method pair (positive favours method A). |
| `hv_boxplot.png` | HV distribution per method on the shared reference. |
| `effective_config.json` | Config actually used, including the filled reference point and seed lists. |
| `reference_fronts.json` | Verified final fronts; reusable via `--reference-fronts`. |
| `runs_raw.json` | Raw fronts and convergence logs. |

### Hypervolume

- Lower bound fixed at `(0, 0)`; reference point `(hv_ref_f1, hv_ref_f2)`.
- If no reference is supplied, it is estimated ONCE after all runs as the
  nadir of the union of all verified final fronts (all methods, all seeds)
  plus a 10% margin, then frozen; every saved generation front is rescored on
  that scale. This is an empirical estimate, not a proven nadir.
- `HV_norm = HV / (hv_ref_f1 · hv_ref_f2)`. Because the box starts at the
  origin, normalised values can be small; compare them only within one
  instance and one frozen reference.
- To anchor the reference to exact results, pass a verified MIP front in this
  schema (illustrative numbers only):

```json
{"product_set_id": "case1", "num_products": 5,
 "fronts": [[[100, 40], [140, 10]]]}
```

- Reuse one frozen reference across separately launched experiments; never
  compare independently estimated scales.

### Diversity metrics

- `Spacing` — adjacent-Euclidean spread with `d_f = d_l = 0` (dimensionless).
  It is NOT Schott's nearest-neighbour spacing.
- `Deb_Delta` — Deb et al.'s Δ; requires two endpoint pairs, taken from
  verified MIP endpoints or the best-known union front. A best-known endpoint
  is an estimate, not a proven Pareto boundary.
- Both are computed on raw minutes; for empty or single-point fronts they are
  undefined (blank/NaN), not zero.

### Statistical comparison

- Default: paired Wilcoxon signed-rank on HV_norm over seeds; `--independent`
  switches to Mann–Whitney U with disjoint seed ranges per method.
- Sharing seed numbers does not by itself create meaningful pairing across
  different algorithms; declare the design before running.
- Holm correction is applied across method pairs.
- Failed or empty fronts are kept with HV = 0 and a status; if any run of a
  method failed, its comparisons are marked `insufficient_valid_runs`.
- These tests describe seed variability on a single instance, not general
  superiority across instances.
- The heuristic is deterministic and returns one point; report it as a
  reference, not as a front in the statistical comparison.

## Analysis: comparing MIP vs. NSGA-II A vs. NSGA-II B

`plot_pareto_by_case.py` overlays MIP (ε-constraint sweep), NSGA-II Design A
and NSGA-II Design B Pareto points on one chart per `(config, n, case)`, and
writes a summary workbook per `n`.

```bash
python plot_pareto_by_case.py --results-root results --config 3-3
python plot_pareto_by_case.py --results-root results
```

It expects a `results/<config>/{mip,nsga2}/` layout. Compatibility with the
`experiment_<timestamp>/<method>/seed_<seed>/` layout of the experiment
driver has not yet been verified.

## Tests

```bash
python -m unittest discover -s tests -v   # metrics and experiment driver
python tests/smoke_run.py                  # 3 products, pop 6, 2 generations
```

The smoke run is an integration check, not a benchmark. When Gurobi is
absent, it stubs only the unused import and forbids solver calls.

## Notes for the researcher

- Read `verify.py` for the solution-correctness contract. Every MIP and every
  Solomon solve is validated immediately. NSGA-II is validated in two stages:
  (i) intermediate generations use penalty rejection without repair — an
  infeasible decode receives f₁ = f₂ = 1e9 and is eliminated by
  non-dominated sorting; (ii) after the last generation, every rank-1
  individual's route labels are canonicalised (a pure relabelling; f₁ and f₂
  are unchanged), converted with `fleet_to_solution()` and checked with
  `validate_solution()`. Failed individuals are removed from the reported
  front and listed in the `pareto_rejected` sheet and `result_*.validation.txt`.
- Both designs decode each chromosome under the three weight triples in
  `nsga2.ALPHA_COMBOS_NEW`: distance-, duration- and wait-weighted.
- Generation fronts and snapshots are decoder-feasible diagnostics
  (`decoder_only`); only final fronts are independently verified.
- `crossover_feasibility_rate` / `clone_feasibility_rate` measure the share of
  offspring with at least one feasible phenotype; they are not acceptance
  rates.
- Reported fronts are de-duplicated in objective space at six decimals after
  verification; chromosomes and selection are unchanged.
- The two NSGA-II designs are kept side by side deliberately, not merged.

## Version notes

**2026-09-16**

- The third decode weight triple changed from `(0.5, 0.5, 0.0001)` to the
  wait-weighted `(0.0001, 0.01, 1)`. NSGA-II results produced before this
  date used a different decode family and are not comparable with new runs.
- Final NSGA-II fronts are now verified with `verify.py`.
- Hypervolume uses one frozen reference per instance; earlier running-ideal
  HV series are not comparable.
- `delta_metric` was split into `Spacing` and `Deb_Delta`.
- Multi-seed experiment driver and statistical comparison added.
- Operator diagnostics renamed to feasibility rates; objective-space
  de-duplication moved to six decimals.
## License

Released under the MIT License — see [`LICENSE`](LICENSE).
