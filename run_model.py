from __future__ import annotations
import argparse
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, time
from itertools import permutations
from pathlib import Path
from typing import Optional
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum
# =============================================================================
# Logging
# =============================================================================
log = logging.getLogger("internal_logistics")
def configure_logging(level: int = logging.INFO,
                      log_file: Optional[Path] = None,
                      verbose: bool = False) -> None:
    """INFO+ -> log file only by default (console stays clean for Gurobi's
    own output). verbose=True also streams to stdout."""
    handlers: list[logging.Handler] = []
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    if verbose:
        handlers.append(logging.StreamHandler(sys.stdout))
    if not handlers:
        handlers.append(logging.NullHandler())
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )
# =============================================================================
# Errors
# =============================================================================
class DataConsistencyError(Exception):
    """Raised when input data fails validation against the model's assumptions."""
class ConfigError(Exception):
    """Raised when config.xlsx is malformed or missing required keys."""
# =============================================================================
# Constants
# =============================================================================
OBJECTIVES = {"route_duration", "wait_time"}
RUN_MODES = {"single_objective", "multi_objective", "heuristic",
            "nsga2_old", "nsga2_new"}
OBJ_METHODS = {"augmented_eps", "lexicographic"}
PRODUCT_SET_TO_SHEET = {
    # Legacy fallback mapping. Only consulted if the actual sheet name in
    # products.xlsx doesn't match the product_set_id directly or via its
    # short prefix (e.g. "case1" for "case1_all_distinct").
    "case1_all_distinct": "Sheet1",
    "case2_loc_c": "Sheet2",
    "case3_shared_pickup": "Sheet3",
    "case4_operational_data": "Sheet4",
}
READY_TIME_FORMATS = {"clock", "relative_to_shift"}
REQUIRED_CONFIG_KEYS = {
    "run_mode", "primary_obj", "constraint_obj", "limit_on_constraint_obj",
    "augmentation_weight", "product_set_id", "num_products",
    "shift_start_clock_min", "shift_duration_min",
    "strict_shift_window", "time_limit_seconds", "mip_gap",
    "output_dir", "output_prefix", "write_full_var_sheets",
    "tight_big_M", "break_vehicle_symmetry",
    "add_work_lb_cut", "tight_time_var_bounds",
    "tight_route_activation",
    "use_indicator_constraints",
    "add_product_lb_cut",
    "add_pair_lb_cut", "pair_lb_threshold_min",
    "add_wait_lb_cut",
    "add_reverse_arc_cut", "add_endpoints_cut", "add_adjacency_cut",
    "mtz_type",
    "objective_method",
    "auto_verify", "verify_on_fail", "auto_visualize",
}
VERIFY_FAIL_MODES = {"raise", "warn"}
# Optional parameters: keys allowed in config.xlsx but not required.
# Each "_override" entry defaults to a value computed from the input data.
# Gurobi solver-tuning parameters are intentionally NOT in this list — Gurobi
# uses its own defaults so we never silently impose a tuning choice.
OPTIONAL_CONFIG_DEFAULTS = {
    "n_seeds": 10,
    "experiment_seed": None,  # None => first method's configured seed.
    "hv_ref_f1": None,
    "hv_ref_f2": None,
    # Input file names (default: standard names under --inputs dir).
    # Paths are taken relative to --inputs (so don't include the folder).
    "nodes_file":     "nodes.xlsx",
    "vehicles_file":  "vehicles.xlsx",
    "products_file":  "products.xlsx",
    "distances_file": "distances_minutes.xlsx",
    # Ready-time interpretation in products.xlsx.
    #   "clock"             : value is clock time (e.g. "07:10", 430)
    #   "relative_to_shift" : value is minutes after shift start (e.g. 10 = 07:10)
    "ready_time_format": "clock",
    # Per-objective time budget for the lex multi-objective mode. Total
    # TimeLimit (above) still applies to the whole solve; this caps the
    # second stage (secondary objective) so the sweep never burns the
    # full budget on diminishing returns to the second objective.
    "second_obj_time_limit_seconds": 60,
    # Epsilon-constraint sweep step. After each multi-objective solve the
    # constraint on the secondary objective is tightened by this amount
    # (achieved_secondary - eps_step). Smaller -> denser Pareto front,
    # more iterations.
    "eps_step": 1.0,
    # Big-M overrides (default: computed from input data)
    "C_max_minutes_override": None,    # default: max(c_ij)         used in M16
    "e_min_minutes_override": None,    # default: min(e_p_relative)  used in M22
    "Q_max_override": None,            # default: max(q_k)           used in M24, M25
    # Solomon I1 heuristic parameters (used when run_mode = heuristic)
    # Label only — drives the strict-lex alpha-triple defaults below and
    # the bias label reported in the result xlsx.
    "heuristic_objective": "route_duration",  # or "wait_time"
    "alpha_1": 0.0001,                # weight on Δ-distance in c_1
    "alpha_2": 1.0,                   # weight on Δ-fleet-route-duration in c_1
    "alpha_3": 0.01,                  # weight on Δ-fleet-total-wait in c_1
    "lambda_c2": 1.0,                 # scaling on c(h, o_p) in selection c_2
    "heuristic_wait_limit": 9999.0,   # epsilon bound on total wait time
    "apply_2opt": True,               # run 2-opt local search after construction
    "apply_or_opt": True,             # run Or-opt-on-pairs after 2-opt
    # Bounded-depth backtracking when greedy reaches infeasible_unrouted.
    # Empirically n=20 instances need depth >= 15-17; 20 is the safe default.
    "solomon_backtrack_max_depth": 20,
    # When backtracking still fails, retry under the other two stock alpha
    # biases (route_duration / wait_time / balanced).
    "solomon_multistart": True,
    # After the 6 stock biases (3 core + 3 extended diverse) all fail,
    # try this many random alpha-triple restarts (log-uniform in [1e-4,1]^3).
    "solomon_random_restarts": 20,
    # After random alpha-restarts also fail, try this many shuffled-order
    # restarts: each picks a random insertion order over |P| parts and takes
    # each part's best feasible insertion. Defeats tight single-route
    # instances where the deterministic greedy creates an irrecoverable
    # early commitment.
    "solomon_shuffle_restarts": 100,
    # Seed for the random alpha-restarts AND shuffled-order restarts.
    "solomon_random_seed": 42,
    # NSGA-II parameters (used when run_mode = nsga2_old / nsga2_new)
    "nsga2_old_pop_size":               100,
    "nsga2_old_n_gen":                  200,
    "nsga2_old_seed":                    42,
    "nsga2_old_p_crossover":            0.90,
    "nsga2_old_p_assignment_mutation":  0.15,
    "nsga2_old_p_priority_mutation":    0.15,
    "nsga2_old_n_solomon_seeds":           3,
    "nsga2_old_results_every":           25,  # Retained: now writes snapshots.
    "nsga2_old_time_limit_sec":         3600,
    "nsga2_new_pop_size":               100,
    "nsga2_new_n_gen":                  200,
    "nsga2_new_seed":                    42,
    "nsga2_new_p_crossover":            0.90,
    "nsga2_new_p_mutation":             0.15,
    "nsga2_new_p_swap":                 0.10,
    "nsga2_new_n_solomon_seeds":           3,
    "nsga2_new_results_every":           25,
    "nsga2_new_time_limit_sec":         3600,
    # NSGA-II parallel decode pool (hem nsga2_old hem nsga2_new icin ortak)
    # Bos/None ise kod otomatik 'CPU cekirdek sayisi - 1' kullanir. Sadece
    # kac islemciye is dagitildigini belirler, algoritmayi/sonucu etkilemez
    # (nsga2.py: paralel calisma sirali calismayla sonuc olarak birebir ayni).
    "nsga2_n_workers": None,
    "nsga2_parallel":  True,
}
# =============================================================================
# Config loader
# =============================================================================
def _truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"true", "1", "yes", "y", "t"}
def load_config(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    df = pd.read_excel(path, sheet_name="config")
    cols = {c.lower(): c for c in df.columns}
    if "parameter" not in cols or "value" not in cols:
        raise ConfigError(
            f"config sheet must have columns 'parameter' and 'value' "
            f"(got {df.columns.tolist()})"
        )
    raw = {}
    for _, row in df.iterrows():
        k = row[cols["parameter"]]
        if pd.isna(k):
            continue
        k = str(k).strip()
        if not k or k.startswith("#"):
            continue
        raw[k] = row[cols["value"]]
    cfg: dict = {}
    missing = [k for k in REQUIRED_CONFIG_KEYS
               if k not in raw or pd.isna(raw[k])]
    if missing:
        raise ConfigError(f"missing required config keys: {missing}")
    for k in REQUIRED_CONFIG_KEYS:
        cfg[k] = raw[k]
    for k, default in OPTIONAL_CONFIG_DEFAULTS.items():
        cfg[k] = raw[k] if k in raw and not pd.isna(raw[k]) else default
    # Coercions: these keys previously existed only in the documentation.
    for key in ("hv_ref_f1", "hv_ref_f2"):
        if cfg[key] is not None:
            cfg[key] = float(cfg[key])
            if not math.isfinite(cfg[key]) or cfg[key] <= 0:
                raise ConfigError(f"{key} must be finite and positive")
    if (cfg["hv_ref_f1"] is None) != (cfg["hv_ref_f2"] is None):
        raise ConfigError("Set both HV reference coordinates, or neither")
    cfg["n_seeds"] = int(cfg["n_seeds"])
    if cfg["n_seeds"] < 1:
        raise ConfigError("n_seeds must be >= 1")
    if cfg["experiment_seed"] is not None:
        cfg["experiment_seed"] = int(cfg["experiment_seed"])
    for key in ("nsga2_old_results_every", "nsga2_new_results_every"):
        cfg[key] = int(cfg[key])
        if cfg[key] < 0:
            raise ConfigError(f"{key} must be >= 0 (0 disables snapshots)")
    cfg["run_mode"] = str(cfg["run_mode"]).strip()
    cfg["primary_obj"] = str(cfg["primary_obj"]).strip()
    cfg["constraint_obj"] = str(cfg["constraint_obj"]).strip()
    cfg["objective_method"] = str(cfg["objective_method"]).strip()
    cfg["product_set_id"] = str(cfg["product_set_id"]).strip()
    cfg["output_dir"] = str(cfg["output_dir"]).strip()
    cfg["output_prefix"] = str(cfg["output_prefix"]).strip()
    cfg["strict_shift_window"] = _truthy(cfg["strict_shift_window"])
    cfg["write_full_var_sheets"] = _truthy(cfg["write_full_var_sheets"])
    cfg["tight_big_M"] = _truthy(cfg["tight_big_M"])
    cfg["break_vehicle_symmetry"] = _truthy(cfg["break_vehicle_symmetry"])
    cfg["add_work_lb_cut"] = _truthy(cfg["add_work_lb_cut"])
    cfg["tight_time_var_bounds"] = _truthy(cfg["tight_time_var_bounds"])
    cfg["tight_route_activation"] = _truthy(cfg["tight_route_activation"])
    cfg["use_indicator_constraints"] = _truthy(cfg["use_indicator_constraints"])
    cfg["add_product_lb_cut"] = _truthy(cfg["add_product_lb_cut"])
    cfg["add_pair_lb_cut"] = _truthy(cfg["add_pair_lb_cut"])
    cfg["pair_lb_threshold_min"] = float(cfg["pair_lb_threshold_min"])
    cfg["add_wait_lb_cut"] = _truthy(cfg["add_wait_lb_cut"])
    cfg["add_reverse_arc_cut"] = _truthy(cfg["add_reverse_arc_cut"])
    cfg["add_endpoints_cut"] = _truthy(cfg["add_endpoints_cut"])
    cfg["add_adjacency_cut"] = _truthy(cfg["add_adjacency_cut"])
    cfg["mtz_type"] = str(cfg["mtz_type"]).strip().lower()
    if cfg["mtz_type"] == "ss_lifted":
        raise ConfigError(
            "mtz_type='ss_lifted' has been removed: the coefficients "
            "produced an invalid lifting that cut off feasible route "
            "endpoints (u_first = 1, u_last = U). Use 'dl_lifted' "
            "(Desrochers-Laporte), which is the strongest known "
            "polynomial single-constraint lift of MTZ."
        )
    if cfg["mtz_type"] not in {"base", "dl_lifted"}:
        raise ConfigError(
            f"mtz_type must be one of base, dl_lifted "
            f"(got {cfg['mtz_type']!r})"
        )
    cfg["auto_verify"] = _truthy(cfg["auto_verify"])
    cfg["auto_visualize"] = _truthy(cfg["auto_visualize"])
    cfg["verify_on_fail"] = str(cfg["verify_on_fail"]).strip().lower()
    # Input file names (strip whitespace; tolerate Excel quirks)
    for fk in ("nodes_file", "vehicles_file", "products_file", "distances_file"):
        cfg[fk] = str(cfg[fk]).strip()
    # Ready-time format
    cfg["ready_time_format"] = str(cfg["ready_time_format"]).strip().lower()
    if cfg["ready_time_format"] not in READY_TIME_FORMATS:
        raise ConfigError(
            f"ready_time_format must be one of {sorted(READY_TIME_FORMATS)}, "
            f"got {cfg['ready_time_format']!r}"
        )
    cfg["second_obj_time_limit_seconds"] = int(cfg["second_obj_time_limit_seconds"])
    if cfg["second_obj_time_limit_seconds"] < 1:
        raise ConfigError(
            "second_obj_time_limit_seconds must be >= 1 "
            f"(got {cfg['second_obj_time_limit_seconds']})"
        )
    cfg["eps_step"] = float(cfg["eps_step"])
    if cfg["eps_step"] <= 0:
        raise ConfigError(
            f"eps_step must be > 0 (got {cfg['eps_step']})"
        )
    np_val = cfg["num_products"]
    cfg["num_products"] = (
        None if str(np_val).strip().lower() in {"all", "none", ""}
        else int(np_val)
    )
    for k in ("shift_start_clock_min",
              "shift_duration_min", "time_limit_seconds"):
        cfg[k] = int(cfg[k])
    for k in ("limit_on_constraint_obj", "augmentation_weight", "mip_gap"):
        cfg[k] = float(cfg[k])
    for k in ("C_max_minutes_override", "e_min_minutes_override",
              "Q_max_override"):
        cfg[k] = float(cfg[k]) if cfg[k] is not None and not pd.isna(cfg[k]) else None
    # Validations
    if cfg["run_mode"] not in RUN_MODES:
        raise ConfigError(f"run_mode must be in {RUN_MODES}, got '{cfg['run_mode']}'")
    if cfg["primary_obj"] not in OBJECTIVES:
        raise ConfigError(f"primary_obj must be in {OBJECTIVES}, got '{cfg['primary_obj']}'")
    if cfg["constraint_obj"] not in OBJECTIVES:
        raise ConfigError(f"constraint_obj must be in {OBJECTIVES}, got '{cfg['constraint_obj']}'")
    if cfg["primary_obj"] == cfg["constraint_obj"]:
        raise ConfigError("primary_obj and constraint_obj must be different")
    if cfg["objective_method"] not in OBJ_METHODS:
        raise ConfigError(
            f"objective_method must be in {OBJ_METHODS}, "
            f"got '{cfg['objective_method']}'"
        )
    if cfg["verify_on_fail"] not in VERIFY_FAIL_MODES:
        raise ConfigError(
            f"verify_on_fail must be in {VERIFY_FAIL_MODES}, "
            f"got '{cfg['verify_on_fail']}'"
        )
    # product_set_id is no longer validated against PRODUCT_SET_TO_SHEET.
    # The sheet lookup at load_instance time is permissive and accepts
    # exact name, short prefix (case1), and the legacy mapping.
    # default_max_routes is no longer a config parameter — per-vehicle
    # max_route values are read from vehicles.xlsx.
    if cfg["limit_on_constraint_obj"] < 0:
        raise ConfigError("limit_on_constraint_obj must be >= 0")
    if cfg["shift_duration_min"] <= 0:
        raise ConfigError("shift_duration_min must be > 0")
    if cfg["augmentation_weight"] < 0:
        raise ConfigError("augmentation_weight must be >= 0")
    # Solomon heuristic coercions / validations
    cfg["alpha_1"]   = float(cfg["alpha_1"])
    cfg["alpha_2"]   = float(cfg["alpha_2"])
    cfg["alpha_3"]   = float(cfg["alpha_3"])
    cfg["lambda_c2"] = float(cfg["lambda_c2"])
    cfg["heuristic_wait_limit"] = float(cfg["heuristic_wait_limit"])
    cfg["apply_2opt"] = _truthy(cfg["apply_2opt"])
    cfg["apply_or_opt"] = _truthy(cfg["apply_or_opt"])
    cfg["solomon_backtrack_max_depth"] = int(cfg["solomon_backtrack_max_depth"])
    if cfg["solomon_backtrack_max_depth"] < 0:
        raise ConfigError(
            "solomon_backtrack_max_depth must be >= 0 "
            f"(got {cfg['solomon_backtrack_max_depth']})"
        )
    cfg["solomon_multistart"] = _truthy(cfg["solomon_multistart"])
    cfg["solomon_random_restarts"] = int(cfg["solomon_random_restarts"])
    if cfg["solomon_random_restarts"] < 0:
        raise ConfigError(
            "solomon_random_restarts must be >= 0 "
            f"(got {cfg['solomon_random_restarts']})"
        )
    cfg["solomon_shuffle_restarts"] = int(cfg["solomon_shuffle_restarts"])
    if cfg["solomon_shuffle_restarts"] < 0:
        raise ConfigError(
            "solomon_shuffle_restarts must be >= 0 "
            f"(got {cfg['solomon_shuffle_restarts']})"
        )
    seed_val = cfg["solomon_random_seed"]
    if seed_val in (None, "", "none", "None") or (
            isinstance(seed_val, float) and pd.isna(seed_val)):
        cfg["solomon_random_seed"] = None
    else:
        cfg["solomon_random_seed"] = int(seed_val)
    cfg["heuristic_objective"] = str(cfg["heuristic_objective"]).strip().lower()
    if cfg["heuristic_objective"] not in {"route_duration", "wait_time"}:
        raise ConfigError(
            "heuristic_objective must be 'route_duration' or 'wait_time' "
            f"(got {cfg['heuristic_objective']!r})"
        )
    return cfg
# =============================================================================
# Time helpers
# =============================================================================
def ready_to_clock_min(v) -> int:
    """Convert ready_time entry to clock minutes from midnight (no shift offset)."""
    if pd.isna(v):
        raise DataConsistencyError("ready_time is NaN")
    if isinstance(v, (pd.Timestamp, datetime)):
        return int(v.hour) * 60 + int(v.minute)
    if isinstance(v, time):
        return int(v.hour) * 60 + int(v.minute)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return int(v)
    s = str(v).strip()
    dt = pd.to_datetime(s, errors="coerce")
    if pd.notna(dt):
        return int(dt.hour) * 60 + int(dt.minute)
    if ":" in s:
        hh, mm = s.split(":")[:2]
        return int(hh) * 60 + int(mm)
    return int(float(s))
def minutes_to_hhmm(minutes) -> str:
    if minutes is None or pd.isna(minutes):
        return ""
    # Round the TOTAL minutes first, then split via divmod. Splitting first
    # (floor for h, then round the remainder for m) can push m to exactly 60
    # when the remainder is e.g. 59.9999... (float noise), producing an
    # invalid string like "07:60" instead of "08:00". Rounding before the
    # split makes that overflow impossible.
    total_min = int(round(float(minutes)))
    h, m = divmod(total_min, 60)
    return f"{h:02d}:{m:02d}"
# =============================================================================
# Instance
# =============================================================================
@dataclass
class Instance:
    N: list           # incl. depot 'h'
    Nw: list          # work-centers (N \ {h})
    K: list           # vehicle ids
    R: list           # 1..max(max_routes[k]) — global ceiling across fleet
    P: list           # product ids
    c: dict           # (i,j) -> minutes
    e: dict           # p -> shift-relative ready time
    s_load: dict
    s_unload: dict
    q_p: dict
    o: dict
    d: dict
    q_k: dict
    max_routes: dict        # vehicle_id -> max route count (from vehicles.xlsx)
    T_max: float
    C_max: float
    e_min: float
    Q_max: float
    M16: float          # value used in constraint (16); depends on tight_big_M
    M20: float          # value used in constraint (20)
    M22_p: dict         # PRODUCT-INDEXED Big-M used in constraint (22)
    M24: float
    M25: float
    M16_pdf: float      # PDF nominal (T_max + C_max), kept for reporting
    M22_pdf: float      # PDF nominal (T_max - e_min), kept for reporting
    big_m_mode: str     # "tight" or "pdf"
    vehicle_symmetry_groups: list   # list of lists of vehicle ids that are identical
    min_complete: dict              # p -> min single-product route completion time
    pair_lb_cuts: list              # list of (p, q, min_pair_complete) above threshold
    U: int
    config: dict
    @property
    def shift_offset(self) -> int:
        return self.config["shift_start_clock_min"]
    def routes_of(self, k) -> list:
        """Route indices available to vehicle k: [1, 2, ..., max_routes[k]]."""
        return list(range(1, int(self.max_routes[k]) + 1))
    def last_route_of(self, k) -> int:
        """The last (highest-indexed) route of vehicle k."""
        return int(self.max_routes[k])
    @property
    def KR_pairs(self) -> list:
        """All valid (vehicle, route) pairs across the fleet."""
        return [(k, r) for k in self.K for r in self.routes_of(k)]
# =============================================================================
# Helpers for min-completion-time computation (pair / singleton cuts)
# =============================================================================
def _min_route_through_nodes(perm, node_unloads, node_loads, node_ready, c):
    """Simulate a route h -> perm -> h, returning total elapsed minutes.
    At each node: first do unloads (sum_unload), then wait until any pickup's
    ready time, then loads (sum_load). Captures constraints (17)-(19).
    If op or dp is the depot, "h" can appear inside perm; treat repeated
    visits to h as a zero-cost non-move rather than a self-loop arc (c has
    no (h,h) entry).
    """
    time = 0.0
    prev = "h"
    for node in perm:
        time += 0.0 if node == prev else c[(prev, node)]
        ts = max(time + node_unloads.get(node, 0.0),
                 node_ready.get(node, 0.0))
        time = ts + node_loads.get(node, 0.0)
        prev = node
    time += 0.0 if prev == "h" else c[(prev, "h")]
    return time
def _min_complete_one(o, d, e, sl, su, c):
    """Minimum route completion time for one product alone."""
    if o == d:
        return _min_route_through_nodes(
            [o], {o: su}, {o: sl}, {o: e}, c
        )
    return _min_route_through_nodes(
        [o, d], {d: su}, {o: sl}, {o: e}, c
    )
def _min_complete_pair(o1, d1, e1, sl1, su1,
                        o2, d2, e2, sl2, su2, c):
    """Minimum route completion time for a pair (1, 2) on the same route.

    Handles all node-coincidence cases (shared origin, shared destination,
    origin = destination across products, etc.) by enumerating permutations
    of the UNIQUE work-centre node set.
    """
    unique_nodes = list({o1, d1, o2, d2})
    node_unloads = {n: 0.0 for n in unique_nodes}
    node_loads = {n: 0.0 for n in unique_nodes}
    node_ready = {n: 0.0 for n in unique_nodes}
    # Aggregate events at each visited node.
    node_loads[o1] += sl1
    node_loads[o2] += sl2
    node_unloads[d1] += su1
    node_unloads[d2] += su2
    node_ready[o1] = max(node_ready[o1], e1)
    node_ready[o2] = max(node_ready[o2], e2)
    best = float("inf")
    for perm in permutations(unique_nodes):
        pos = {n: i for i, n in enumerate(perm)}
        # Pickup-before-delivery per product (trivial when o == d).
        if o1 != d1 and pos[o1] > pos[d1]:
            continue
        if o2 != d2 and pos[o2] > pos[d2]:
            continue
        t = _min_route_through_nodes(
            perm, node_unloads, node_loads, node_ready, c
        )
        if t < best:
            best = t
    return best
# =============================================================================
# Data loader (with strict validation — no silent skips)
# =============================================================================
def resolve_products_sheet(xlsx_path: Path, product_set_id: str) -> str:
    """Match product_set_id to a sheet: exact -> prefix before '_' ->
    legacy PRODUCT_SET_TO_SHEET -> case-insensitive. Raises with the
    available sheet list if nothing matches."""
    xl = pd.ExcelFile(xlsx_path)
    available = list(xl.sheet_names)
    # 1. Exact
    if product_set_id in available:
        return product_set_id
    # 2. Short prefix
    short = product_set_id.split("_", 1)[0] if "_" in product_set_id else product_set_id
    if short != product_set_id and short in available:
        return short
    # 3. Legacy mapping
    legacy = PRODUCT_SET_TO_SHEET.get(product_set_id)
    if legacy and legacy in available:
        return legacy
    # 4. Case-insensitive matching
    avail_lower = {s.lower(): s for s in available}
    for cand in (product_set_id, short):
        if cand.lower() in avail_lower:
            return avail_lower[cand.lower()]
    raise DataConsistencyError(
        f"could not find a sheet for product_set_id={product_set_id!r} in "
        f"{xlsx_path.name}. Available sheets: {available}"
    )
def load_instance(inputs_dir: Path, cfg: dict) -> Instance:
    log.info("Loading inputs from %s", inputs_dir)
    nodes_path     = inputs_dir / cfg["nodes_file"]
    vehicles_path  = inputs_dir / cfg["vehicles_file"]
    products_path  = inputs_dir / cfg["products_file"]
    distances_path = inputs_dir / cfg["distances_file"]
    log.info("Input files: nodes=%s, vehicles=%s, products=%s, distances=%s",
             cfg["nodes_file"], cfg["vehicles_file"],
             cfg["products_file"], cfg["distances_file"])
    # Nodes
    if not nodes_path.exists():
        raise DataConsistencyError(f"nodes file not found: {nodes_path}")
    nodes_df = pd.read_excel(nodes_path)
    if "node_id" not in nodes_df.columns:
        raise DataConsistencyError(
            f"{cfg['nodes_file']} must contain column 'node_id'"
        )
    nodes_df["node_id"] = nodes_df["node_id"].astype(str).str.strip()
    if (nodes_df["node_id"] == "").any() or nodes_df["node_id"].isna().any():
        raise DataConsistencyError(
            f"{cfg['nodes_file']} contains empty node_id rows"
        )
    if nodes_df["node_id"].duplicated().any():
        dups = nodes_df.loc[nodes_df["node_id"].duplicated(), "node_id"].tolist()
        raise DataConsistencyError(
            f"duplicate node_id(s) in {cfg['nodes_file']}: {dups}"
        )
    if "h" not in nodes_df["node_id"].tolist():
        raise DataConsistencyError(
            f"{cfg['nodes_file']} must contain depot node 'h'"
        )
    N = nodes_df["node_id"].tolist()
    Nw = [n for n in N if n != "h"]
    # Vehicles
    if not vehicles_path.exists():
        raise DataConsistencyError(f"vehicles file not found: {vehicles_path}")
    veh_df = pd.read_excel(vehicles_path)
    required = {"vehicle_id", "capacity_m2", "max_route", "active"}
    if not required.issubset(set(veh_df.columns)):
        raise DataConsistencyError(
            f"{cfg['vehicles_file']} must have columns {sorted(required)} "
            f"(got {veh_df.columns.tolist()})"
        )
    veh_df["vehicle_id"] = veh_df["vehicle_id"].astype(str).str.strip()
    if veh_df["vehicle_id"].duplicated().any():
        dups = veh_df.loc[veh_df["vehicle_id"].duplicated(), "vehicle_id"].tolist()
        raise DataConsistencyError(f"duplicate vehicle_id(s): {dups}")
    if (veh_df["capacity_m2"].astype(float) <= 0).any():
        raise DataConsistencyError("all vehicle capacity_m2 must be > 0")
    # active: per-vehicle on/off flag. Accepts 1/0, true/false, yes/no.
    # Inactive vehicles are dropped from the model entirely: they do not
    # appear in K, no variables/constraints reference them, no symmetry
    # group includes them. At least one vehicle must be active.
    if veh_df["active"].isna().any():
        bad = veh_df.loc[veh_df["active"].isna(), "vehicle_id"].tolist()
        raise DataConsistencyError(
            f"{cfg['vehicles_file']} column 'active' has NaN value(s) "
            f"for vehicle(s): {bad}"
        )
    active_mask = veh_df["active"].apply(_truthy).tolist()
    inactive_ids = [v for v, a in zip(veh_df["vehicle_id"], active_mask) if not a]
    if inactive_ids:
        log.info("Inactive vehicles (excluded from model): %s", inactive_ids)
        print(f"[load] Inactive vehicles excluded: {inactive_ids}")
    veh_df = veh_df.loc[active_mask].reset_index(drop=True)
    if veh_df.empty:
        raise DataConsistencyError(
            f"all vehicles in {cfg['vehicles_file']} are marked inactive — "
            "the model has no fleet to dispatch"
        )
    # max_route: positive integer per active vehicle.
    max_route_raw = veh_df["max_route"]
    if max_route_raw.isna().any():
        raise DataConsistencyError(
            f"{cfg['vehicles_file']} column 'max_route' has NaN values"
        )
    try:
        max_route_int = max_route_raw.astype(int)
    except Exception as exc:
        raise DataConsistencyError(
            f"{cfg['vehicles_file']} column 'max_route' must be integer "
            f"(got: {max_route_raw.tolist()}): {exc}"
        )
    if (max_route_int < 1).any():
        bad = [(vid, mr) for vid, mr in zip(veh_df["vehicle_id"], max_route_int)
               if mr < 1]
        raise DataConsistencyError(
            f"max_route must be >= 1 for every active vehicle; offenders: {bad}"
        )
    K = veh_df["vehicle_id"].tolist()
    q_k = dict(zip(K, veh_df["capacity_m2"].astype(float)))
    max_routes = dict(zip(K, max_route_int.astype(int).tolist()))
    R = list(range(1, max(max_routes.values()) + 1))
    log.info("Active vehicles (%d): %s", len(K), K)
    log.info("Per-vehicle max_route: %s; global R = 1..%d",
             max_routes, R[-1])
    # Detect identical-vehicle groups for symmetry breaking
    # "identical" = all non-id columns match (auto-covers future columns).
    sym_cols = [c for c in veh_df.columns if c != "vehicle_id"]
    veh_df["_attr_signature"] = veh_df[sym_cols].apply(
        lambda row: tuple(row.tolist()), axis=1
    )
    vehicle_symmetry_groups = []
    for sig, grp in veh_df.groupby("_attr_signature", sort=False):
        ids = grp["vehicle_id"].tolist()
        if len(ids) > 1:
            vehicle_symmetry_groups.append(ids)
    log.info("Vehicle symmetry groups (by columns %s): %s",
             sym_cols, vehicle_symmetry_groups or "none")
    # Products
    if not products_path.exists():
        raise DataConsistencyError(f"products file not found: {products_path}")
    sheet = resolve_products_sheet(products_path, cfg["product_set_id"])
    products_df = pd.read_excel(products_path, sheet_name=sheet)
    log.info("%s::%s loaded with %d rows  (product_set_id=%s)",
             cfg["products_file"], sheet, len(products_df),
             cfg["product_set_id"])
    required_p = {"product_id", "origin", "destination", "ready_time",
                  "load_time", "unload_time", "area_m2"}
    if not required_p.issubset(set(products_df.columns)):
        raise DataConsistencyError(
            f"products.xlsx::{sheet} must have columns {sorted(required_p)} "
            f"(got {products_df.columns.tolist()})"
        )
    if cfg["num_products"] is not None:
        n = int(cfg["num_products"])
        if n > len(products_df):
            raise DataConsistencyError(
                f"num_products={n} but sheet has only {len(products_df)} rows"
            )
        products_df = products_df.head(n).copy()
        log.info("Sliced first %d products (num_products=%d)", len(products_df), n)
    for col in ("product_id", "origin", "destination"):
        products_df[col] = products_df[col].astype(str).str.strip()
    if products_df["product_id"].duplicated().any():
        dups = products_df.loc[products_df["product_id"].duplicated(), "product_id"].tolist()
        raise DataConsistencyError(f"duplicate product_id(s): {dups}")
    bad_o = sorted(set(products_df["origin"]) - set(N))
    if bad_o:
        raise DataConsistencyError(
            f"product origins not in nodes.xlsx: {bad_o}"
        )
    bad_d = sorted(set(products_df["destination"]) - set(N))
    if bad_d:
        raise DataConsistencyError(
            f"product destinations not in nodes.xlsx: {bad_d}"
        )
    for col in ("load_time", "unload_time", "area_m2"):
        if products_df[col].isna().any():
            raise DataConsistencyError(f"products column '{col}' has NaN values")
        if (products_df[col].astype(float) < 0).any():
            raise DataConsistencyError(f"products column '{col}' has negative values")
    P = products_df["product_id"].tolist()
    # Interpret ready_time according to ready_time_format:
    #   "clock"             : value is clock time of day. Numeric like 430,
    #                         string "07:10", or a timestamp/time object —
    #                         all converted to clock minutes from midnight,
    #                         then made shift-relative by subtracting
    #                         shift_start_clock_min.
    #   "relative_to_shift" : value is already minutes-after-shift-start.
    #                         No subtraction performed. Strings like "10"
    #                         and numbers like 10 both accepted.
    fmt = cfg["ready_time_format"]
    if fmt == "clock":
        e_clock = {p: ready_to_clock_min(v)
                   for p, v in zip(P, products_df["ready_time"])}
        e_relative = {p: e_clock[p] - cfg["shift_start_clock_min"] for p in P}
    elif fmt == "relative_to_shift":
        def _to_rel(v):
            if pd.isna(v):
                raise DataConsistencyError("ready_time is NaN")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return int(v)
            # Reject clock-style values explicitly: datetime / time objects
            # and strings containing ':' are almost certainly clock format,
            # not minutes-after-shift-start.
            if isinstance(v, (pd.Timestamp, datetime, time)):
                raise DataConsistencyError(
                    f"ready_time value {v!r} looks like clock time but "
                    f"ready_time_format='relative_to_shift'. "
                    f"Set ready_time_format='clock' in config.xlsx, "
                    f"or change the column to integer minutes-after-shift-start."
                )
            s = str(v).strip()
            if ":" in s or "-" in s:
                raise DataConsistencyError(
                    f"ready_time value {s!r} looks like clock time but "
                    f"ready_time_format='relative_to_shift'. "
                    f"Set ready_time_format='clock' in config.xlsx, "
                    f"or change the column to integer minutes-after-shift-start."
                )
            try:
                return int(float(s))
            except ValueError as exc:
                raise DataConsistencyError(
                    f"could not parse ready_time {s!r} as an integer "
                    f"(minutes after shift start): {exc}"
                ) from exc
        e_relative = {p: _to_rel(v)
                      for p, v in zip(P, products_df["ready_time"])}
    else:
        raise ConfigError(f"unknown ready_time_format: {fmt!r}")
    out_of_window = {p: e_relative[p]
                     for p in P
                     if e_relative[p] < 0
                     or e_relative[p] > cfg["shift_duration_min"]}
    if out_of_window:
        msg = (f"products outside shift window [0, {cfg['shift_duration_min']}] "
               f"min after shift start: {out_of_window}")
        if cfg["strict_shift_window"]:
            raise DataConsistencyError(msg)
        log.warning(msg)
    s_load = dict(zip(P, products_df["load_time"].astype(float)))
    s_unload = dict(zip(P, products_df["unload_time"].astype(float)))
    q_p = dict(zip(P, products_df["area_m2"].astype(float)))
    o = dict(zip(P, products_df["origin"]))
    d = dict(zip(P, products_df["destination"]))
    # Restrict N / Nw to nodes referenced by the selected products
    # The nodes file may declare more work stations than the current product
    # subset actually uses. Carrying unused nodes inflates the model, the
    # result.xlsx (per-node sheets), and the visualisations. We restrict the
    # active node set to {h} ∪ {o_p, d_p : p ∈ P}, preserving the original
    # order from nodes.xlsx so downstream displays stay stable.
    used_nodes = {"h"} | set(o.values()) | set(d.values())
    full_N = list(N)
    N = [n for n in full_N if n in used_nodes]
    Nw = [n for n in N if n != "h"]
    unused = [n for n in full_N if n not in used_nodes]
    msg = (
        f"Active nodes: {len(N)} of {len(full_N)} "
        f"(skipping {len(unused)} unused work stations"
        + (f": {', '.join(unused)}" if 0 < len(unused) <= 12
           else f": {', '.join(unused[:12])}, ..." if unused
           else "")
        + ")"
    )
    log.info(msg)
    print(f"[load] {msg}")
    # Distances
    # Primary path comes from cfg["distances_file"]; a Turkish-named legacy
    # file is also tolerated when the user hasn't overridden the config.
    dist_path = distances_path
    if not dist_path.exists():
        legacy = inputs_dir / "distances - dakika.xlsx"
        if legacy.exists():
            log.warning("distance file %s not found; using legacy %s",
                        cfg["distances_file"], legacy.name)
            dist_path = legacy
        else:
            raise DataConsistencyError(
                f"distances file not found: {dist_path} "
                f"(also tried legacy 'distances - dakika.xlsx')"
            )
    dist_df = pd.read_excel(dist_path)
    if not {"from_node", "to_node", "duration_min"}.issubset(set(dist_df.columns)):
        raise DataConsistencyError(
            f"{dist_path.name} must have columns from_node, to_node, duration_min"
        )
    dist_df["from_node"] = dist_df["from_node"].astype(str).str.strip()
    dist_df["to_node"] = dist_df["to_node"].astype(str).str.strip()
    dist_df["duration_min"] = pd.to_numeric(dist_df["duration_min"], errors="coerce")
    if dist_df["duration_min"].isna().any():
        raise DataConsistencyError("distances has non-numeric duration_min")
    if (dist_df["duration_min"] < 0).any():
        raise DataConsistencyError("distances has negative duration_min")
    dist_df = dist_df[
        (dist_df["from_node"].isin(N))
        & (dist_df["to_node"].isin(N))
        & (dist_df["from_node"] != dist_df["to_node"])
    ]
    if dist_df.duplicated(subset=["from_node", "to_node"]).any():
        raise DataConsistencyError("distances has duplicate (from,to) rows")
    have = set(zip(dist_df["from_node"], dist_df["to_node"]))
    need = {(i, j) for i in N for j in N if i != j}
    missing = need - have
    if missing:
        raise DataConsistencyError(
            f"distances missing {len(missing)} pair(s); "
            f"first 10: {sorted(missing)[:10]}"
        )
    c = {(r["from_node"], r["to_node"]): float(r["duration_min"])
         for _, r in dist_df.iterrows()}
    # Derived parameters
    T_max = float(cfg["shift_duration_min"])
    C_max = (float(cfg["C_max_minutes_override"])
             if cfg["C_max_minutes_override"] is not None else max(c.values()))
    e_min = (float(cfg["e_min_minutes_override"])
             if cfg["e_min_minutes_override"] is not None else min(e_relative.values()))
    Q_max = (float(cfg["Q_max_override"])
             if cfg["Q_max_override"] is not None else max(q_k.values()))
    # Big-M coefficients
    # Shift-relative time means T_start = 0 in the model.
    # PDF nominal values (eq. 37-41):
    M16_pdf = T_max + C_max          # eq. 37
    M22_pdf = T_max - e_min          # eq. 39 (uniform across products)
    # Tight values (provably valid; argument in MODEL_CODE_AUDIT.md):
    #   M16 = T_max          (the +C_max term in (16) is x-dependent and vanishes when x=0)
    #   M22(p) = T_max + s_unload[p] - e_p   (per-product, replaces the uniform value)
    M16_tight = T_max
    M22_p_tight = {p: T_max + s_unload[p] - e_relative[p] for p in P}
    # Pick the active set based on the config flag.
    big_m_mode = "tight" if cfg["tight_big_M"] else "pdf"
    if cfg["tight_big_M"]:
        M16 = M16_tight
        M22_p = M22_p_tight
    else:
        M16 = M16_pdf
        M22_p = {p: M22_pdf for p in P}
    M20 = T_max          # eq. 38, no tightening proposed in this round
    M24 = Q_max          # eq. 40
    M25 = Q_max          # eq. 41
    U = len(Nw)
    log.info("Sets: |N|=%d, |Nw|=%d, |K|=%d, |R|=%d, |P|=%d",
             len(N), len(Nw), len(K), len(R), len(P))
    log.info("Derived: T_max=%.0f, C_max=%.2f, e_min=%.2f, Q_max=%.2f, U=%d",
             T_max, C_max, e_min, Q_max, U)
    log.info("Big-M mode: %s", big_m_mode)
    log.info("  M16(pdf)=%.2f  M16(used)=%.2f", M16_pdf, M16)
    log.info("  M22(pdf,uniform)=%.2f", M22_pdf)
    log.info("  M22(used) min=%.2f, max=%.2f, mean=%.2f",
             min(M22_p.values()), max(M22_p.values()),
             sum(M22_p.values()) / len(M22_p))
    log.info("  M20=%.2f  M24=%.2f  M25=%.2f", M20, M24, M25)
    # Self-loop product check (o_p == d_p disallowed)
    self_loops = [p for p in P if o[p] == d[p]]
    if self_loops:
        raise DataConsistencyError(
            f"products with origin == destination are not allowed: {self_loops}"
        )
    # Per-product min completion time (used by product_lb_cut)
    min_complete = {
        p: _min_complete_one(o[p], d[p], e_relative[p],
                              s_load[p], s_unload[p], c)
        for p in P
    }
    # Pre-compute pair_lb cuts (filtered by threshold)
    # delta(p, q) = min_pair_complete(p, q) - max(min_complete(p), min_complete(q))
    # Emit a cut only when delta > pair_lb_threshold_min.
    pair_lb_cuts: list = []
    pair_lb_threshold = float(cfg["pair_lb_threshold_min"])
    if cfg["add_pair_lb_cut"]:
        bucket_counts = {"<=0": 0, "(0, 2]": 0, "(2, 5]": 0,
                          "(5, 10]": 0, "> 10": 0}
        P_list = list(P)
        for i in range(len(P_list)):
            for j in range(i + 1, len(P_list)):
                p1, p2 = P_list[i], P_list[j]
                mp = _min_complete_pair(
                    o[p1], d[p1], e_relative[p1], s_load[p1], s_unload[p1],
                    o[p2], d[p2], e_relative[p2], s_load[p2], s_unload[p2],
                    c,
                )
                delta = mp - max(min_complete[p1], min_complete[p2])
                # Bucket for histogram
                if delta <= 0:
                    bucket_counts["<=0"] += 1
                elif delta <= 2:
                    bucket_counts["(0, 2]"] += 1
                elif delta <= 5:
                    bucket_counts["(2, 5]"] += 1
                elif delta <= 10:
                    bucket_counts["(5, 10]"] += 1
                else:
                    bucket_counts["> 10"] += 1
                if delta > pair_lb_threshold:
                    pair_lb_cuts.append((p1, p2, mp))
        log.info("Pair-LB delta histogram (threshold = %.2f):",
                 pair_lb_threshold)
        for label, count in bucket_counts.items():
            log.info("  delta %s : %d pairs", label, count)
        log.info("Pair-LB will emit %d pairs (filtered).",
                 len(pair_lb_cuts))
    return Instance(
        N=N, Nw=Nw, K=K, R=R, P=P,
        c=c, e=e_relative, s_load=s_load, s_unload=s_unload,
        q_p=q_p, o=o, d=d, q_k=q_k, max_routes=max_routes,
        T_max=T_max, C_max=C_max, e_min=e_min, Q_max=Q_max,
        M16=M16, M20=M20, M22_p=M22_p, M24=M24, M25=M25,
        M16_pdf=M16_pdf, M22_pdf=M22_pdf,
        big_m_mode=big_m_mode,
        vehicle_symmetry_groups=vehicle_symmetry_groups,
        min_complete=min_complete,
        pair_lb_cuts=pair_lb_cuts,
        U=U,
        config=cfg,
    )
# =============================================================================
# Model builder
# =============================================================================
def build_model_v2(inst: Instance, *, primary: str, constraint: str,
                limit: float, weight: float, method: str):
    """PD-VRP MIP: per-vehicle route sets Rk, active-node restriction to N,
    tightened route-activation/time-window/Big-M's, lifted MTZ, 8 optional
    valid inequalities. Delta/y load-flow is defined over all of N (depot
    included), so products with op or dp = h are load-tracked correctly.

    primary, constraint in {'route_duration', 'wait_time'}.
    method in {'augmented_eps', 'lexicographic'}.
    Epsilon constraint (constraint_expr <= limit) is added in both methods.
    """
    if primary == constraint:
        raise ValueError("primary_obj and constraint_obj must differ")
    if method not in OBJ_METHODS:
        raise ValueError(f"method must be in {OBJ_METHODS}, got '{method}'")

    m = gp.Model(f"InternalLogistics_{primary}_primary")
    # Decision variables (per-vehicle route index sets)
    # Each vehicle k has its own route range 1..max_routes[k] from
    # vehicles.xlsx. Variables are created only for valid (k, r) pairs,
    # so a vehicle with max_route = 1 contributes one route's worth of
    # variables while a vehicle with max_route = 3 contributes three.
    # Tight upper bounds on time variables (recommendation B). Under the
    # shift-window assumption (already implicit in the PDF Big-Ms), every
    # feasible time variable lies in [0, T_max]. Adding ub=T_max is free
    # tightening of the LP polyhedron and does not cut any feasible point.
    time_ub = inst.T_max if inst.config["tight_time_var_bounds"] else GRB.INFINITY
    KR = inst.KR_pairs  # list of (k, r) pairs across the whole fleet
    arc_keys = [(i, j, k, r) for (k, r) in KR
                for i in inst.N for j in inst.N if i != j]
    f_keys     = [(p, k, r) for (k, r) in KR for p in inst.P]
    y_keys     = [(j, k, r) for (k, r) in KR for j in inst.N]
    ta_keys    = [(j, k, r) for (k, r) in KR for j in inst.N]
    ts_keys    = [(j, k, r) for (k, r) in KR for j in inst.Nw]
    u_keys     = [(j, k, r) for (k, r) in KR for j in inst.Nw]
    delta_keys = [(j, k, r) for (k, r) in KR for j in inst.N]
    x     = m.addVars(arc_keys, vtype=GRB.BINARY, name="x")
    f     = m.addVars(f_keys,   vtype=GRB.BINARY, name="f")
    w     = m.addVars(inst.P,   vtype=GRB.CONTINUOUS, lb=0.0, name="w")
    y     = m.addVars(y_keys,   vtype=GRB.CONTINUOUS, lb=0.0, name="y")
    ta    = m.addVars(ta_keys,  vtype=GRB.CONTINUOUS, lb=0.0,
                      ub=time_ub, name="ta")
    td    = m.addVars(ta_keys,  vtype=GRB.CONTINUOUS, lb=0.0,
                      ub=time_ub, name="td")
    ts    = m.addVars(ts_keys,  vtype=GRB.CONTINUOUS, lb=0.0,
                      ub=time_ub, name="ts")
    u     = m.addVars(u_keys,   vtype=GRB.INTEGER, lb=0, ub=inst.U, name="u")
    delta = m.addVars(delta_keys, vtype=GRB.CONTINUOUS,
                      lb=-GRB.INFINITY, name="delta")
    log.info("Per-vehicle route counts: %s; total |KR|=%d",
             {k: inst.max_routes[k] for k in inst.K}, len(KR))
    # Shift-relative model: T_start = 0, so route_duration is the sum over
    # vehicles of the arrival time at the depot in EACH vehicle's last route.
    # (Each vehicle's "last route" is its max_routes[k], which can differ.)
    route_duration_expr = quicksum(ta["h", k, inst.last_route_of(k)]
                                   for k in inst.K)
    total_wait_expr = quicksum(w[p] for p in inst.P)
    obj_exprs = {"route_duration": route_duration_expr,
                 "wait_time": total_wait_expr}
    # (1)/(2) Objective: augmented or lexicographic
    if method == "augmented_eps":
        # Single objective: primary + 10^-3 · secondary (PDF formulation)
        m.setObjective(obj_exprs[primary] + weight * obj_exprs[constraint],
                       GRB.MINIMIZE)
    else:  # method == "lexicographic"
        # Two objectives, priority(primary) > priority(secondary).
        # Gurobi solves primary to optimality first, then minimizes secondary
        # subject to primary being held at its optimum.
        #
        # abstol=0 and reltol=0 force STRICT lex ordering: the primary may not
        # worsen by even a numerical tolerance during secondary optimization.
        # Defaults are 1e-6 / 0.0; we set both to zero so each Pareto point
        # we report is on the true Pareto frontier with no slack.
        m.ModelSense = GRB.MINIMIZE
        m.NumObj = 2
        m.setObjectiveN(obj_exprs[primary],    index=0, priority=2,
                        abstol=0.0, reltol=0.0,
                        name=f"min_{primary}")
        m.setObjectiveN(obj_exprs[constraint], index=1, priority=1,
                        abstol=0.0, reltol=0.0,
                        name=f"min_{constraint}")
    # (3) Epsilon constraint
    m.addConstr(obj_exprs[constraint] <= limit, name=f"c3_eps_on_{constraint}")
    # (4)-(9) Route structure
    for k in inst.K:
        for r in inst.routes_of(k):
            out_h = quicksum(x["h", j, k, r] for j in inst.Nw)
            in_h = quicksum(x[j, "h", k, r] for j in inst.Nw)
            m.addConstr(out_h == in_h, name=f"c4[{k},{r}]")
            m.addConstr(out_h <= 1, name=f"c5[{k},{r}]")
            # (6) route activation: tight form uses 2*(products on route) + 1
            # (the +1 piggybacks on (5)); PDF form uses (2|P|+1). Both are 0
            # when no product is assigned.
            total_arcs = quicksum(x[i, j, k, r] for i in inst.N
                                  for j in inst.N if i != j)
            total_f = quicksum(f[p, k, r] for p in inst.P)
            if inst.config["tight_route_activation"]:
                m.addConstr(
                    total_arcs <= 2 * total_f + out_h,
                    name=f"c6[{k},{r}]",
                )
            else:
                m.addConstr(
                    total_arcs <= (2 * len(inst.P) + 1) * total_f,
                    name=f"c6[{k},{r}]",
                )
    for j in inst.Nw:
        for k in inst.K:
            for r in inst.routes_of(k):
                inflow = quicksum(x[i, j, k, r] for i in inst.N if i != j)
                outflow = quicksum(x[j, i, k, r] for i in inst.N if i != j)
                relevant = quicksum(
                    f[p, k, r] for p in inst.P
                    if inst.o[p] == j or inst.d[p] == j
                )
                m.addConstr(inflow <= relevant, name=f"c7[{j},{k},{r}]")
                m.addConstr(inflow == outflow, name=f"c8[{j},{k},{r}]")
                m.addConstr(inflow <= 1, name=f"c9[{j},{k},{r}]")
    # (10)-(12) Product assignment + visit pickup/delivery
    for p in inst.P:
        m.addConstr(
            quicksum(f[p, k, r] for (k, r) in inst.KR_pairs) == 1,
            name=f"c10[{p}]",
        )
        op, dp = inst.o[p], inst.d[p]
        for k in inst.K:
            for r in inst.routes_of(k):
                m.addConstr(
                    quicksum(x[i, op, k, r] for i in inst.N if i != op)
                    >= f[p, k, r],
                    name=f"c11[{p},{k},{r}]",
                )
                m.addConstr(
                    quicksum(x[i, dp, k, r] for i in inst.N if i != dp)
                    >= f[p, k, r],
                    name=f"c12[{p},{k},{r}]",
                )
    # Symmetry breaking across identical vehicles: within each group, force
    # the product count to be non-increasing so the solver doesn't explore
    # relabelings of the same solution.
    if inst.config["break_vehicle_symmetry"]:
        for group in inst.vehicle_symmetry_groups:
            for k_a, k_b in zip(group, group[1:]):
                m.addConstr(
                    quicksum(f[p, k_a, r] for p in inst.P for r in inst.routes_of(k_a))
                    >= quicksum(f[p, k_b, r] for p in inst.P for r in inst.routes_of(k_b)),
                    name=f"sym[{k_a}>={k_b}]",
                )
    # Work-LB: route duration >= travel + service time (wait >= 0), chained
    # via (14)+(21) up to the vehicle's last route.
    if inst.config["add_work_lb_cut"]:
        for k in inst.K:
            travel_expr = quicksum(
                inst.c[(i, j)] * x[i, j, k, r]
                for i in inst.N for j in inst.N
                if i != j
                for r in inst.routes_of(k)
            )
            service_expr = quicksum(
                (inst.s_load[p] + inst.s_unload[p]) * f[p, k, r]
                for p in inst.P for r in inst.routes_of(k)
            )
            m.addConstr(
                ta["h", k, inst.last_route_of(k)] >= travel_expr + service_expr,
                name=f"work_lb[{k}]",
            )
    # Product-LB: min_complete(p) lower-bounds the last depot arrival of
    # whichever vehicle carries p. Off by default (interacts poorly with
    # Gurobi's heuristics on small instances); worth trying on 20+ products.
    if inst.config["add_product_lb_cut"]:
        for p in inst.P:
            for k in inst.K:
                last_route_idx = inst.last_route_of(k)
                m.addConstr(
                    ta["h", k, last_route_idx]
                    >= inst.min_complete[p]
                       * quicksum(f[p, k, r] for r in inst.routes_of(k)),
                    name=f"product_lb[{p},{k}]",
                )
    # Pair-LB: same idea as product-LB but for pairs sharing a vehicle,
    # filtered to pairs where the combined bound exceeds each singleton.
    if inst.config["add_pair_lb_cut"] and inst.pair_lb_cuts:
        emitted = 0
        for (p1, p2, mp_val) in inst.pair_lb_cuts:
            q_pair = inst.q_p[p1] + inst.q_p[p2]
            for k in inst.K:
                if q_pair > inst.q_k[k]:
                    continue   # cannot share a route, skip
                last_route_idx = inst.last_route_of(k)
                m.addConstr(
                    ta["h", k, last_route_idx]
                    >= mp_val * (
                        quicksum(f[p1, k, r] for r in inst.routes_of(k))
                        + quicksum(f[p2, k, r] for r in inst.routes_of(k))
                        - 1
                    ),
                    name=f"pair_lb[{p1},{p2},{k}]",
                )
                emitted += 1
        log.info("Pair-LB cuts emitted: %d", emitted)
    # Wait-LB: w_p >= load_time + shortest_path(o_p,d_p) + unload_time,
    # a constant per-product floor independent of (k,r). Uses Floyd-Warshall
    # since c may not satisfy the triangle inequality.
    if inst.config["add_wait_lb_cut"]:
        # Floyd-Warshall over the existing arc matrix. |N|^3 = trivial at
        # our scale. Use only when the cut is actually requested.
        node_list = list(inst.N)
        n_idx = {n: i for i, n in enumerate(node_list)}
        n_nodes = len(node_list)
        INF = float("inf")
        sp = [[INF] * n_nodes for _ in range(n_nodes)]
        for i in range(n_nodes):
            sp[i][i] = 0.0
        for (a, b), val in inst.c.items():
            sp[n_idx[a]][n_idx[b]] = float(val)
        for k_idx in range(n_nodes):
            row_k = sp[k_idx]
            for i in range(n_nodes):
                dik = sp[i][k_idx]
                if dik == INF:
                    continue
                row_i = sp[i]
                for j in range(n_nodes):
                    via = dik + row_k[j]
                    if via < row_i[j]:
                        row_i[j] = via
        wait_lb_total = 0.0
        relaxed = 0
        for p in inst.P:
            op_i, dp_i = n_idx[inst.o[p]], n_idx[inst.d[p]]
            travel_lb = sp[op_i][dp_i]
            direct = inst.c.get((inst.o[p], inst.d[p]), travel_lb)
            if travel_lb + 1e-9 < direct:
                relaxed += 1
            lb_p = inst.s_load[p] + travel_lb + inst.s_unload[p]
            m.addConstr(w[p] >= lb_p, name=f"wait_lb[{p}]")
            wait_lb_total += lb_p
        log.info("Wait-LB cuts emitted: %d (Σ w_p >= %.2f min); "
                 "%d cuts loosened by triangle-inequality violations",
                 len(inst.P), wait_lb_total, relaxed)
    # Three precedence cuts on x/f directly (no big-M), targeting fractional
    # patterns that (20) and (32) leave too loose to catch on their own.
    # Reverse-arc: if p ships on (k,r), the truck can't go d_p -> o_p directly.
    if inst.config["add_reverse_arc_cut"]:
        emitted = 0
        for p in inst.P:
            op, dp = inst.o[p], inst.d[p]
            if op in inst.Nw and dp in inst.Nw and op != dp:
                for k in inst.K:
                    for r in inst.routes_of(k):
                        m.addConstr(
                            x[dp, op, k, r] + f[p, k, r] <= 1,
                            name=f"rev_arc[{p},{k},{r}]",
                        )
                        emitted += 1
        log.info("Reverse-arc cuts emitted: %d", emitted)
    # Endpoints: if p ships on (k,r), d_p can't be the first stop and o_p
    # can't be the last stop before returning to the depot.
    if inst.config["add_endpoints_cut"]:
        emitted = 0
        for p in inst.P:
            op, dp = inst.o[p], inst.d[p]
            if op not in inst.Nw or dp not in inst.Nw:
                continue
            for k in inst.K:
                for r in inst.routes_of(k):
                    m.addConstr(
                        x["h", dp, k, r] + f[p, k, r] <= 1,
                        name=f"endpt_first[{p},{k},{r}]",
                    )
                    m.addConstr(
                        x[op, "h", k, r] + f[p, k, r] <= 1,
                        name=f"endpt_last[{p},{k},{r}]",
                    )
                    emitted += 2
        log.info("Endpoints cuts emitted: %d", emitted)
    # Adjacency: forbid the 1-hop reverse detour d_p -> v -> o_p. Usually
    # the highest-yield precedence cut of the three.
    if inst.config["add_adjacency_cut"]:
        emitted = 0
        for p in inst.P:
            op, dp = inst.o[p], inst.d[p]
            if op not in inst.Nw or dp not in inst.Nw:
                continue
            for v in inst.Nw:
                if v == op or v == dp:
                    continue
                for k in inst.K:
                    for r in inst.routes_of(k):
                        m.addConstr(
                            x[dp, v, k, r] + x[v, op, k, r] + f[p, k, r] <= 2,
                            name=f"adj1[{p},{v},{k},{r}]",
                        )
                        emitted += 1
        log.info("1-vertex adjacency cuts emitted: %d", emitted)
    # (13)-(15) Route timing
    for k in inst.K:
        m.addConstr(td["h", k, 1] == 0, name=f"c13[{k}]")
    for k in inst.K:
        for r in inst.routes_of(k)[1:]:
            m.addConstr(td["h", k, r] >= ta["h", k, r - 1], name=f"c14[{k},{r}]")
            m.addConstr(ta["h", k, r] >= ta["h", k, r - 1], name=f"c15[{k},{r}]")
    # (16) Time consistency on traversed arcs
    use_ind = inst.config["use_indicator_constraints"]
    for (i, j, k, r) in arc_keys:
        if use_ind:
            m.addGenConstrIndicator(
                x[i, j, k, r], True,
                ta[j, k, r] - td[i, k, r] >= inst.c[(i, j)],
                name=f"c16[{i},{j},{k},{r}]",
            )
        else:
            m.addConstr(
                ta[j, k, r]
                >= td[i, k, r] + inst.c[(i, j)] * x[i, j, k, r]
                   - inst.M16 * (1 - x[i, j, k, r]),
                name=f"c16[{i},{j},{k},{r}]",
            )
    # (17)-(19) Service times
    for j in inst.Nw:
        for k in inst.K:
            for r in inst.routes_of(k):
                unload = quicksum(inst.s_unload[p] * f[p, k, r]
                                  for p in inst.P if inst.d[p] == j)
                load = quicksum(inst.s_load[p] * f[p, k, r]
                                for p in inst.P if inst.o[p] == j)
                m.addConstr(ts[j, k, r] >= ta[j, k, r] + unload,
                            name=f"c17[{j},{k},{r}]")
                m.addConstr(td[j, k, r] >= ts[j, k, r] + load,
                            name=f"c19[{j},{k},{r}]")
    for p in inst.P:
        op = inst.o[p]
        if op == "h":
            continue  # ts is defined only on Nw
        for k in inst.K:
            for r in inst.routes_of(k):
                m.addConstr(
                    ts[op, k, r] >= inst.e[p] * f[p, k, r],
                    name=f"c18[{p},{k},{r}]",
                )
    # (20)-(22) Pickup/delivery precedence, depot timing, waiting
    for p in inst.P:
        op, dp = inst.o[p], inst.d[p]
        for k in inst.K:
            for r in inst.routes_of(k):
                if use_ind:
                    m.addGenConstrIndicator(
                        f[p, k, r], True,
                        ta[dp, k, r] - td[op, k, r] >= 0,
                        name=f"c20[{p},{k},{r}]",
                    )
                else:
                    m.addConstr(
                        ta[dp, k, r] >= td[op, k, r] - inst.M20 * (1 - f[p, k, r]),
                        name=f"c20[{p},{k},{r}]",
                    )
    for k in inst.K:
        for r in inst.routes_of(k):
            m.addConstr(ta["h", k, r] >= td["h", k, r], name=f"c21[{k},{r}]")
    for p in inst.P:
        dp = inst.d[p]
        ep = inst.e[p]
        sp = inst.s_unload[p]
        m22_p = inst.M22_p[p]   # per-product Big-M (constraint 22)
        for k in inst.K:
            for r in inst.routes_of(k):
                if use_ind:
                    m.addGenConstrIndicator(
                        f[p, k, r], True,
                        w[p] - ta[dp, k, r] >= sp - ep,
                        name=f"c22[{p},{k},{r}]",
                    )
                else:
                    m.addConstr(
                        w[p] >= ta[dp, k, r] + sp - ep
                                - m22_p * (1 - f[p, k, r]),
                        name=f"c22[{p},{k},{r}]",
                    )
    # (23)-(27) Load flow
    for j in inst.N:
        for k in inst.K:
            for r in inst.routes_of(k):
                load_in = quicksum(inst.q_p[p] * f[p, k, r]
                                   for p in inst.P if inst.o[p] == j)
                load_out = quicksum(inst.q_p[p] * f[p, k, r]
                                    for p in inst.P if inst.d[p] == j)
                m.addConstr(delta[j, k, r] == load_in - load_out,
                            name=f"c23[{j},{k},{r}]")
    for i in inst.N:
        for j in inst.N:
            if i == j:
                continue
            for k in inst.K:
                for r in inst.routes_of(k):
                    if use_ind:
                        m.addGenConstrIndicator(
                            x[i, j, k, r], True,
                            y[j, k, r] - y[i, k, r] - delta[j, k, r] == 0,
                            name=f"c24_25[{i},{j},{k},{r}]",
                        )
                    else:
                        m.addConstr(
                            y[j, k, r] >= y[i, k, r] + delta[j, k, r]
                                           - inst.M24 * (1 - x[i, j, k, r]),
                            name=f"c24[{i},{j},{k},{r}]",
                        )
                        m.addConstr(
                            y[j, k, r] <= y[i, k, r] + delta[j, k, r]
                                           + inst.M25 * (1 - x[i, j, k, r]),
                            name=f"c25[{i},{j},{k},{r}]",
                        )
    # (28) Route monotonicity
    for k in inst.K:
        for r in inst.routes_of(k)[:-1]:
            m.addConstr(
                quicksum(x["h", j, k, r] for j in inst.Nw)
                >= quicksum(x["h", j, k, r + 1] for j in inst.Nw),
                name=f"c28[{k},{r}]",
            )
    # MTZ subtour elimination, two variants via mtz_type:
    #   base:      u_j >= u_i + 1 - U(1 - x_ij)
    #   dl_lifted: adds + (U-2) x_ji (Desrochers-Laporte lift; valid because
    #              x_ji=1 already forces u_j <= u_i - 1 at integer points, so
    #              the lift removes no feasible solution, only tightens LP).
    mtz_type = inst.config["mtz_type"]
    for i in inst.Nw:
        for j in inst.Nw:
            if i == j:
                continue
            for k in inst.K:
                for r in inst.routes_of(k):
                    xij = x[i, j, k, r]
                    xji = x[j, i, k, r]
                    if mtz_type == "base":
                        rhs = u[i, k, r] + 1 - inst.U * (1 - xij)
                    else:  # dl_lifted
                        rhs = (u[i, k, r] + 1
                               - inst.U * (1 - xij)
                               + (inst.U - 2) * xji)
                    m.addConstr(u[j, k, r] >= rhs,
                                name=f"c29[{i},{j},{k},{r}]")
    # u calibration: rank is 0 if unvisited, >=1 if visited (ties u to the
    # actual arc-flow).
    for j in inst.Nw:
        for k in inst.K:
            for r in inst.routes_of(k):
                indeg = quicksum(x[i, j, k, r] for i in inst.N if i != j)
                m.addConstr(u[j, k, r] <= inst.U * indeg,
                            name=f"c30[{j},{k},{r}]")
                m.addConstr(u[j, k, r] >= indeg,
                            name=f"c31[{j},{k},{r}]")
    # Pickup-before-delivery on visit order (rank-scale companion to (20),
    # smaller big-M so LP-stronger).
    for p in inst.P:
        op, dp = inst.o[p], inst.d[p]
        if op in inst.Nw and dp in inst.Nw:
            for k in inst.K:
                for r in inst.routes_of(k):
                    m.addConstr(
                        u[dp, k, r] >= u[op, k, r] + 1
                                        - inst.U * (1 - f[p, k, r]),
                        name=f"c32[{p},{k},{r}]",
                    )
    vars_ = {"x": x, "f": f, "w": w, "y": y, "ta": ta, "td": td,
             "ts": ts, "u": u, "delta": delta,
             "route_duration_expr": route_duration_expr,
             "total_wait_expr": total_wait_expr}
    return m, vars_
# =============================================================================
# Solve
# =============================================================================
def solve_model(m: gp.Model, *, cfg: dict, gurobi_log: Path,
                compute_iis_on_infeasible: bool = True) -> None:
    """Sadece TimeLimit/MIPGap/LogFile set eder, gerisi Gurobi default.
    compute_iis_on_infeasible=False: multiobj sweep'in tightening
    iterasyonlarinda infeasible beklenen sonuc, IIS gereksiz log kalabaligi
    yaratir."""
    m.setParam("TimeLimit", cfg["time_limit_seconds"])
    m.setParam("MIPGap", cfg["mip_gap"])
    m.setParam("LogFile", str(gurobi_log))
    # m.Params.MIPFocus = 3  # focus on improving the bound
    # m.Params.Cuts = 3  # aggressive cut generation
    # m.Params.Symmetry = 2  # detect symmetry beyond what we've broken statically
    # m.Params.VarBranch = 3  # strong branching (slower per node but tighter)

    # Multi-objective (lex) mode: cap the SECONDARY objective's solve time.
    # Each objective in a multi-obj model has its own parameter environment
    # accessible via getMultiobjEnv(idx); setting TimeLimit there bounds the
    # time spent on that particular stage. We leave the primary stage
    # (index 0) unbounded so it can fully resolve the objective-1 optimum,
    # then cap stage 2 (index 1, the secondary objective) to avoid burning
    # the rest of the global TimeLimit on diminishing returns.
    if getattr(m, "NumObj", 1) > 1:
        try:
            second_env = m.getMultiobjEnv(1)
            second_env.setParam(
                "TimeLimit",
                int(cfg["second_obj_time_limit_seconds"]),
            )
        except gp.GurobiError as exc:
            log.warning("Could not set per-objective time limit: %s", exc)

    m.update()
    m.optimize()
    # On infeasibility, compute and report the IIS
    if m.Status == GRB.INFEASIBLE and not compute_iis_on_infeasible:
        log.info("Model is INFEASIBLE; IIS computation skipped "
                 "(compute_iis_on_infeasible=False).")
        return  # caller is responsible for handling the infeasibility
    # IIS = smallest set of constraints/bounds that together are infeasible;
    # reading it tells you which constraint(s) are actually in conflict.
    if m.Status == GRB.INFEASIBLE:
        try:
            log.warning(
                "Model reported INFEASIBLE — computing IIS for diagnosis..."
            )
            print("[solve] Model is INFEASIBLE — computing IIS...")
            m.computeIIS()
            ilp_path = Path(str(gurobi_log)).with_suffix(".ilp")
            m.write(str(ilp_path))
            iis_constrs = [c.constrName for c in m.getConstrs() if c.IISConstr]
            iis_var_lb  = [v.varName for v in m.getVars() if v.IISLB]
            iis_var_ub  = [v.varName for v in m.getVars() if v.IISUB]
            log.info("IIS written to %s", ilp_path)
            log.info("IIS constraint count: %d", len(iis_constrs))
            log.info("IIS lb-bound count:   %d", len(iis_var_lb))
            log.info("IIS ub-bound count:   %d", len(iis_var_ub))
            # Group constraints by their family (the prefix before [) so the
            # report is readable even when 200+ are in conflict.
            from collections import Counter
            family = Counter(
                c.split("[", 1)[0] for c in iis_constrs
            )
            print(f"[iis] {len(iis_constrs)} constraints in conflict:")
            for fam, n in family.most_common():
                print(f"      {fam:<30s} {n}")
            if iis_var_lb:
                print(f"[iis] {len(iis_var_lb)} variable lower bounds in IIS:")
                for v in iis_var_lb[:20]:
                    print(f"      {v}")
                if len(iis_var_lb) > 20:
                    print(f"      ... and {len(iis_var_lb) - 20} more")
            if iis_var_ub:
                print(f"[iis] {len(iis_var_ub)} variable upper bounds in IIS:")
                for v in iis_var_ub[:20]:
                    print(f"      {v}")
                if len(iis_var_ub) > 20:
                    print(f"      ... and {len(iis_var_ub) - 20} more")
            print(f"[iis] Full ILP file written to {ilp_path}")
            print(f"[iis] Open it in a text editor — every constraint there "
                  f"is part of the infeasibility.")
        except gp.GurobiError as exc:
            # IIS computation can fail in multi-obj or with some attributes;
            # log but don't re-raise so the run still terminates cleanly.
            log.error("IIS computation failed: %s", exc)
            print(f"[iis] IIS computation failed: {exc}")
# =============================================================================
# Rota Plani sheet — shared by MIP / heuristic / NSGA-II reporters
# =============================================================================
# Agreed format (bkz. proje notlari): tek bir navy baslik bandi
# ("ROTA PLANLARI"), ardindan HER rota icin ayri, alternatif yesil/gold
# renkli bir alt-tablo (SIRA / LOKASYON / AÇIKLAMA). AÇIKLAMA sutunu
# bilerek bos birakilir, sonradan elle doldurulur.
#

def _location_label(node: str, inst: "Instance") -> str:
    """Node -> LOKASYON hucresi metni. 'h' (depo) icin sabit "DEPO"
    etiketi doner -- gercek kapi numaralandirmasi olmadigi icin
    uydurma bir "<no> KAPI" degeri YAZILMAZ. N1..N17 gibi tesis kodlari
    oldugu gibi gosterilir, onlarda belirsizlik yok."""
    if node == "h":
        return "DEPO"
    return node
def _build_route_sequences_from_solution(sol, inst: "Instance") -> dict:
    """{(k,r): [sirali dugum listesi]} -- sol.route_order() kullanarak
    (heuristic ve NSGA-II reporter'lari icin ortak yardimci)."""
    seqs = {}
    for k in inst.K:
        for r in inst.routes_of(k):
            if not sol.z_used.get((k, r), False):
                continue
            try:
                seqs[(k, r)] = sol.route_order(k, r)
            except ValueError:
                continue
    return seqs
def write_route_plan_sheet(workbook_or_writer, route_sequences: dict,
                            inst: "Instance", sheet_name: str = "ROTA PLANLARI",
                            sheet_prefix: str = "") -> None:
    """ROTA PLANLARI sekmesini yazar. route_sequences = {(k,r): [dugum
    listesi]}. workbook_or_writer: openpyxl Workbook ya da pandas
    ExcelWriter (.book uzerinden gercek Workbook'a erisilir)."""
    from openpyxl.styles import Font, PatternFill, Alignment

    wb = getattr(workbook_or_writer, "book", workbook_or_writer)
    full_name = f"{sheet_prefix}{sheet_name}"[:31]  # openpyxl sheet-adi siniri
    ws = wb.create_sheet(full_name)

    NAVY  = PatternFill(start_color="1E2761", end_color="1E2761", fill_type="solid")
    GREEN = PatternFill(start_color="C6E0B4", end_color="C6E0B4", fill_type="solid")
    GOLD  = PatternFill(start_color="FFE699", end_color="FFE699", fill_type="solid")
    band_font   = Font(bold=True, color="FFFFFF", size=13)
    header_font = Font(bold=True)

    ws.cell(row=1, column=1, value="ROTA PLANLARI")
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=3)
    band_cell = ws.cell(row=1, column=1)
    band_cell.fill = NAVY
    band_cell.font = band_font
    band_cell.alignment = Alignment(horizontal="center")

    row_i = 3
    route_i = 0
    for (k, r), seq in route_sequences.items():
        route_i += 1
        fill = GREEN if route_i % 2 == 1 else GOLD
        ws.cell(row=row_i, column=1, value=f"ROTA {route_i}  ({k}, R{r})")
        ws.merge_cells(start_row=row_i, start_column=1, end_row=row_i, end_column=3)
        header_cell = ws.cell(row=row_i, column=1)
        header_cell.fill = fill
        header_cell.font = header_font
        row_i += 1
        for col, label in enumerate(["SIRA", "LOKASYON", "AÇIKLAMA"], start=1):
            c = ws.cell(row=row_i, column=col, value=label)
            c.font = header_font
        row_i += 1
        for order, node in enumerate(seq, start=1):
            ws.cell(row=row_i, column=1, value=order)
            ws.cell(row=row_i, column=2, value=_location_label(node, inst))
            ws.cell(row=row_i, column=3, value="")  # AÇIKLAMA -- sonradan doldurulur
            row_i += 1
        row_i += 1  # alt-tablolar arasi bosluk
    for col, width in zip("ABC", (8, 20, 40)):
        ws.column_dimensions[col].width = width
def draw_route_diagrams(route_sequences: dict, ta: dict, td: dict,
                        inst: "Instance", output_dir: Path,
                        prefix: str = "") -> dict:
    """Her rota icin bir PNG: dugumler sirayla kutu, aralarinda ok
    (seyahat suresi, dk), altinda varis/cikis saat damgasi.

    ta/td: duz sozlukler {(node,k,r): dakika} -- Gurobi degiskeni degil,
    MIP (.X ile cikarilmis) ve heuristic/NSGA-II (sol.ta/sol.td) ile
    ayni sekilde cagrilabilir. Zaman damgasi yoksa saat satiri bos.

    Doner: {(k,r): png_path}.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    NAVY, ICE, GOLD, TEAL = "#1E2761", "#CADCFC", "#E8A838", "#0D7A8C"
    off = inst.shift_offset
    def stamp(v):
        return minutes_to_hhmm(v + off) if v is not None else "—"
    paths = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for (k, r), seq in route_sequences.items():
        n = len(seq)
        if n < 2:
            continue
        fig_w = max(6.0, n * 1.9)
        fig, ax = plt.subplots(figsize=(fig_w, 3.4))
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_ylim(-1.1, 1.2)
        ax.axis("off")

        for i, node in enumerate(seq):
            arr = ta.get((node, k, r))
            dep = td.get((node, k, r))
            label = _location_label(node, inst)
            ax.add_patch(FancyBboxPatch(
                (i - 0.38, 0.05), 0.76, 0.55,
                boxstyle="round,pad=0.05,rounding_size=0.08",
                facecolor=ICE, edgecolor=NAVY, linewidth=1.5,
            ))
            ax.text(i, 0.33, label, ha="center", va="center",
                    fontsize=10, fontweight="bold", color=NAVY)
            # Rotanin ilk kutusu (depodan CIKIS) icin "Varis" anlamsiz --
            # arac henuz yola cikmadi. Son kutusu (depoya VARIS) icin
            # "Cikis" anlamsiz -- rota orada bitiyor. Sadece gecerli
            # olan satiri goster.
            lines = []
            if i > 0:
                lines.append(f"Varış {stamp(arr)}")
            if i < n - 1:
                lines.append(f"Çıkış {stamp(dep)}")
            for li, text in enumerate(lines):
                ax.text(i, -0.20 - li * 0.22, text, ha="center",
                        va="top", fontsize=7.5)
            if i < n - 1:
                nxt = seq[i + 1]
                travel = inst.c.get((node, nxt), 0.0)
                ax.annotate(
                    "", xy=(i + 1 - 0.38, 0.33), xytext=(i + 0.38, 0.33),
                    arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=2,
                                   mutation_scale=14),
                )
                ax.text(i + 0.5, 0.72, f"{travel:.1f} dk", ha="center",
                        fontsize=8, color=TEAL, fontweight="bold")

        ax.set_title(f"ROTA — {k}, R{r}", fontsize=13, fontweight="bold",
                    color=NAVY, pad=10)
        fig.tight_layout()
        png_path = output_dir / f"{prefix}route_{k}_R{r}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths[(k, r)] = png_path
    return paths
# =============================================================================
# Reporter
# =============================================================================
def write_results(m: gp.Model, vars_: dict, inst: Instance,
                  *, output_path: Path) -> Optional[dict]:
    if m.SolCount == 0:
        log.error("No feasible solution; status=%s", m.status)
        return None
    x = vars_["x"]; f = vars_["f"]; w = vars_["w"]; y = vars_["y"]
    ta = vars_["ta"]; td = vars_["td"]; ts = vars_["ts"]
    u = vars_["u"]; delta = vars_["delta"]
    route_duration_val = sum(ta["h", k, inst.last_route_of(k)].X for k in inst.K)
    total_wait_val = sum(w[p].X for p in inst.P)
    off = inst.shift_offset
    def stamp(v):
        return minutes_to_hhmm(v + off) if v is not None and not pd.isna(v) else ""
    # Per-objective bounds and gap (Gurobi exposes ObjBound / MIPGap as scalars
    # only in single-objective mode; in multi-objective mode they must be
    # queried via the ObjNumber parameter). Wrap every read in try/except so
    # both modes work without diverging the reporter.
    def _safe_attr(model, name, default=float("nan")):
        try:
            return getattr(model, name)
        except Exception:
            return default
    is_multi_obj = (getattr(m, "NumObj", 1) > 1)
    if is_multi_obj:
        # In lex mode use the primary objective's value as obj_value
        # (the one Gurobi minimized first, with priority 2).
        if inst.config["primary_obj"] == "route_duration":
            obj_value = route_duration_val
        else:
            obj_value = total_wait_val
        # Per-objective bounds via ObjNumber are not portable; report NaN here
        # and rely on the route_duration_min / total_wait_min columns instead.
        best_bound = float("nan")
        mip_gap = float("nan")
    else:
        obj_value = _safe_attr(m, "ObjVal")
        best_bound = _safe_attr(m, "ObjBound")
        mip_gap = _safe_attr(m, "MIPGap")
    summary = {
        "product_set_id": inst.config["product_set_id"],
        "num_products": len(inst.P),
        "num_vehicles": len(inst.K),
        "num_routes_per_vehicle": "; ".join(
            f"{k}:{inst.last_route_of(k)}" for k in inst.K
        ),
        "primary_obj": inst.config["primary_obj"],
        "constraint_obj": inst.config["constraint_obj"],
        "objective_method": inst.config["objective_method"],
        "limit_on_constraint_obj": inst.config["limit_on_constraint_obj"],
        "augmentation_weight": (inst.config["augmentation_weight"]
                                if inst.config["objective_method"] == "augmented_eps"
                                else "(unused: lexicographic)"),
        "route_duration_min": route_duration_val,
        "total_wait_min": total_wait_val,
        "obj_value": obj_value,
        "best_bound": best_bound,
        "mip_gap": mip_gap,
        "runtime_s": _safe_attr(m, "Runtime"),
        "status": _safe_attr(m, "Status"),
        "T_max": inst.T_max, "C_max": inst.C_max,
        "e_min": inst.e_min, "Q_max": inst.Q_max,
        "big_m_mode": inst.big_m_mode,
        "M16_pdf": inst.M16_pdf, "M16_used": inst.M16,
        "M20": inst.M20,
        "M22_pdf_uniform": inst.M22_pdf,
        "M22_used_min": min(inst.M22_p.values()),
        "M22_used_max": max(inst.M22_p.values()),
        "M24": inst.M24, "M25": inst.M25, "U": inst.U,
        "shift_start_clock_min": off,
        "shift_duration_min": inst.config["shift_duration_min"],
        "break_vehicle_symmetry": inst.config["break_vehicle_symmetry"],
        "vehicle_symmetry_groups": (
            "; ".join(",".join(g) for g in inst.vehicle_symmetry_groups)
            if inst.vehicle_symmetry_groups else "none"
        ),
        "add_work_lb_cut": inst.config["add_work_lb_cut"],
        "tight_time_var_bounds": inst.config["tight_time_var_bounds"],
        "tight_route_activation": inst.config["tight_route_activation"],
        "use_indicator_constraints": inst.config["use_indicator_constraints"],
        "add_product_lb_cut": inst.config["add_product_lb_cut"],
        "add_pair_lb_cut": inst.config["add_pair_lb_cut"],
        "pair_lb_threshold_min": inst.config["pair_lb_threshold_min"],
        "pair_lb_cuts_emitted_count": len(inst.pair_lb_cuts),
        "add_wait_lb_cut": inst.config["add_wait_lb_cut"],
        "add_reverse_arc_cut": inst.config["add_reverse_arc_cut"],
        "add_endpoints_cut": inst.config["add_endpoints_cut"],
        "add_adjacency_cut": inst.config["add_adjacency_cut"],
        "mtz_type": inst.config["mtz_type"],
    }
    full = inst.config.get("write_full_var_sheets", True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="summary", index=False)

        # Per-product Big-M values used in (22) — for transparency
        rows = [{"product": p,
                 "ready_relative": inst.e[p],
                 "s_unload": inst.s_unload[p],
                 "M22_pdf_uniform": inst.M22_pdf,
                 "M22_used": inst.M22_p[p]}
                for p in inst.P]
        pd.DataFrame(rows).to_excel(writer, sheet_name="big_m_per_product", index=False)
        # Vehicle symmetry groups detected (informational; constraint only added if break_vehicle_symmetry=true)
        sym_rows = []
        for gi, group in enumerate(inst.vehicle_symmetry_groups, start=1):
            for vid in group:
                sym_rows.append({"group": f"G{gi}", "vehicle_id": vid,
                                 "capacity_m2": inst.q_k[vid]})
        if not sym_rows:
            sym_rows = [{"group": "(none)", "vehicle_id": "", "capacity_m2": ""}]
        pd.DataFrame(sym_rows).to_excel(writer, sheet_name="symmetry_groups", index=False)
        # Active arcs
        rows = [{"i": i, "j": j, "k": k, "r": r,
                 "c_ij_min": inst.c[(i, j)]}
                for (i, j, k, r), v in x.items() if v.X > 0.5]
        pd.DataFrame(rows).to_excel(writer, sheet_name="x_used", index=False)
        # Assignments
        rows = [{"product": p, "k": k, "r": r,
                 "origin": inst.o[p], "destination": inst.d[p]}
                for (p, k, r), v in f.items() if v.X > 0.5]
        pd.DataFrame(rows).to_excel(writer, sheet_name="assignment_f", index=False)
        # Wait times
        rows = [{"product": p,
                 "ready_relative": inst.e[p],
                 "ready_clock": stamp(inst.e[p]),
                 "wait_min": w[p].X}
                for p in inst.P]
        pd.DataFrame(rows).to_excel(writer, sheet_name="wait_w", index=False)
        # Itinerary (used routes only)
        itin = []
        for k in inst.K:
            for r in inst.routes_of(k):
                used = any(x["h", j, k, r].X > 0.5 for j in inst.Nw)
                if not used:
                    continue
                cur = "h"
                visited = {"h"}
                order = 1
                itin.append({"k": k, "r": r, "order": order, "node": "h",
                             "ta": ta["h", k, r].X, "td": td["h", k, r].X,
                             "ta_clock": stamp(ta["h", k, r].X),
                             "td_clock": stamp(td["h", k, r].X),
                             "y_after": y["h", k, r].X})
                while True:
                    nxt = None
                    for j in inst.N:
                        if j != cur and (cur, j, k, r) in x and x[cur, j, k, r].X > 0.5:
                            nxt = j
                            break
                    if nxt is None:
                        break
                    order += 1
                    itin.append({"k": k, "r": r, "order": order, "node": nxt,
                                 "ta": ta[nxt, k, r].X, "td": td[nxt, k, r].X,
                                 "ta_clock": stamp(ta[nxt, k, r].X),
                                 "td_clock": stamp(td[nxt, k, r].X),
                                 "y_after": y[nxt, k, r].X})
                    if nxt == "h" or nxt in visited:
                        break
                    visited.add(nxt)
                    cur = nxt
        pd.DataFrame(itin).to_excel(writer, sheet_name="itinerary", index=False)
        # route_timings sheet: one row per traversed arc
        # For each used (vehicle, route), walk the route in order and emit
        # one row per arc with depart-from-i / arrive-at-j clock stamps,
        # travel time, and the load / unload service times happening at
        # the endpoints. A TOTAL row per route summarises travel + service.
        timing_rows = []
        route_sequences = {}
        for k in inst.K:
            for r in inst.routes_of(k):
                if not any(x["h", j, k, r].X > 0.5 for j in inst.Nw):
                    continue
                seq = ["h"]
                cur = "h"
                visited = {"h"}
                while True:
                    nxt = None
                    for j in inst.N:
                        if j != cur and (cur, j, k, r) in x and x[cur, j, k, r].X > 0.5:
                            nxt = j
                            break
                    if nxt is None:
                        break
                    seq.append(nxt)
                    if nxt == "h" or nxt in visited:
                        break
                    visited.add(nxt)
                    cur = nxt
                route_sequences[(k, r)] = seq
                travel_sum = 0.0
                service_sum = 0.0
                for leg, (i_node, j_node) in enumerate(
                        zip(seq[:-1], seq[1:]), start=1):
                    depart_i = td[i_node, k, r].X if (i_node, k, r) in td else 0.0
                    arrive_j = ta[j_node, k, r].X if (j_node, k, r) in ta else 0.0
                    travel = inst.c.get((i_node, j_node), 0.0)
                    load_at_i = sum(
                        inst.s_load[p] for p in inst.P
                        if inst.o[p] == i_node
                        and (p, k, r) in f and f[p, k, r].X > 0.5
                    )
                    unload_at_j = sum(
                        inst.s_unload[p] for p in inst.P
                        if inst.d[p] == j_node
                        and (p, k, r) in f and f[p, k, r].X > 0.5
                    )
                    travel_sum += travel
                    service_sum += load_at_i + unload_at_j
                    timing_rows.append({
                        "vehicle": k,
                        "route": r,
                        "leg": leg,
                        "from": i_node,
                        "to": j_node,
                        "depart_min": round(depart_i, 3),
                        "depart_clock": stamp(depart_i),
                        "arrive_min": round(arrive_j, 3),
                        "arrive_clock": stamp(arrive_j),
                        "travel_min": round(travel, 3),
                        "load_min_at_from": round(load_at_i, 3),
                        "unload_min_at_to": round(unload_at_j, 3),
                    })
                # Per-route totals row
                timing_rows.append({
                    "vehicle": k,
                    "route": r,
                    "leg": "TOTAL",
                    "from": "—",
                    "to": "—",
                    "depart_min": "",
                    "depart_clock": "",
                    "arrive_min": "",
                    "arrive_clock": "",
                    "travel_min": round(travel_sum, 3),
                    "load_min_at_from": "",
                    "unload_min_at_to": "",
                })
                timing_rows.append({
                    "vehicle": k,
                    "route": r,
                    "leg": "SERVICE",
                    "from": "—",
                    "to": "—",
                    "depart_min": "",
                    "depart_clock": "",
                    "arrive_min": "",
                    "arrive_clock": "",
                    "travel_min": round(service_sum, 3),
                    "load_min_at_from": "",
                    "unload_min_at_to": "",
                })
        if timing_rows:
            pd.DataFrame(timing_rows).to_excel(
                writer, sheet_name="route_timings", index=False
            )
        write_route_plan_sheet(writer, route_sequences, inst)
        if full:
            rows = [{"node": j, "k": k, "r": r,
                     "ta": ta[j, k, r].X, "ta_clock": stamp(ta[j, k, r].X),
                     "td": td[j, k, r].X, "td_clock": stamp(td[j, k, r].X),
                     "y_after": y[j, k, r].X}
                    for (k, r) in inst.KR_pairs for j in inst.N]
            pd.DataFrame(rows).to_excel(writer, sheet_name="node_times", index=False)

            rows = [{"node": j, "k": k, "r": r,
                     "ts": ts[j, k, r].X, "ts_clock": stamp(ts[j, k, r].X),
                     "u": u[j, k, r].X, "delta": delta[j, k, r].X}
                    for (k, r) in inst.KR_pairs for j in inst.Nw]
            pd.DataFrame(rows).to_excel(writer, sheet_name="service_u_delta", index=False)

    log.info("Results written to %s", output_path)
    ta_vals = {key: v.X for key, v in ta.items()}
    td_vals = {key: v.X for key, v in td.items()}
    draw_route_diagrams(route_sequences, ta_vals, td_vals, inst, output_path.parent)
    return summary
# =============================================================================
# Heuristic-result reporter (mirrors write_results' sheet layout so the
# verifier and visualisers can read a Solomon solution exactly as they read a
# MIP solution).
# =============================================================================
def write_heuristic_results(sol, inst: Instance, summary: dict,
                             *, output_path: Path) -> dict:
    """Write a `result.xlsx` for a heuristic solution that mirrors the MIP
    reporter's sheet layout, so all downstream tools (route_timings,
    verifier, visualisations) can read it the same way."""
    off = inst.shift_offset
    def stamp(v):
        return minutes_to_hhmm(v + off) if v is not None and not pd.isna(v) else ""
    summary_row = {
        "model":                "heuristic_solomon_i1",
        "status":               summary["status"],
        "route_duration_min":   summary["route_duration"],
        "total_wait_min":       summary["total_wait"],
        "obj_value":            summary["route_duration"],
        "mip_gap":              float("nan"),
        "runtime_s":            summary.get("runtime_s", float("nan")),
        "iterations":           summary["iterations"],
        "primary_obj":          "route_duration",
        "constraint_obj":       "wait_time",
        "limit_on_constraint":  inst.config["heuristic_wait_limit"],
        "product_set_id":       inst.config["product_set_id"],
        "num_products":         len(inst.P),
        "|N|":                  len(inst.N),
        "|Nw|":                 len(inst.Nw),
        "|K|":                  len(inst.K),
        "alpha_1":              inst.config["alpha_1"],
        "alpha_2":              inst.config["alpha_2"],
        "lambda_c2":            inst.config["lambda_c2"],
        "shift_start_clock_min": off,
        "shift_duration_min":   inst.config["shift_duration_min"],
    }
    # x_used arcs
    x_rows = [{"i": i, "j": j, "k": k, "r": r, "val": 1}
              for (i, j, k, r) in sorted(sol.x_used)]
    # assignment
    f_rows = [{"p": p, "k": k, "r": r, "val": 1}
              for p, (k, r) in sorted(sol.f_assigned.items())]
    # waits
    w_rows = [{"p": p, "w": sol.w[p]} for p in sorted(inst.P)]
    # itinerary in route order
    itin = []
    for k in inst.K:
        for r in inst.routes_of(k):
            if not sol.z_used.get((k, r), False):
                continue
            try:
                seq = sol.route_order(k, r)
            except ValueError:
                continue
            for order, node in enumerate(seq, start=1):
                itin.append({
                    "k": k, "r": r, "order": order, "node": node,
                    "ta": sol.ta.get((node, k, r), 0.0),
                    "td": sol.td.get((node, k, r), 0.0),
                    "ta_clock": stamp(sol.ta.get((node, k, r), 0.0)),
                    "td_clock": stamp(sol.td.get((node, k, r), 0.0)),
                    "y_after": sol.y.get((node, k, r), 0.0),
                })
    # route_timings: one row per arc with depart/arrive/travel + service.
    timing_rows = []
    for k in inst.K:
        for r in inst.routes_of(k):
            if not sol.z_used.get((k, r), False):
                continue
            try:
                seq = sol.route_order(k, r)
            except ValueError:
                continue
            travel_sum = 0.0
            service_sum = 0.0
            for leg, (a, b) in enumerate(zip(seq[:-1], seq[1:]), start=1):
                depart = sol.td.get((a, k, r), 0.0)
                arrive = sol.ta.get((b, k, r), 0.0)
                travel = inst.c.get((a, b), 0.0)
                load_at_a = sum(inst.s_load[p] for p in inst.P
                                if inst.o[p] == a and sol.f_assigned.get(p) == (k, r))
                unload_at_b = sum(inst.s_unload[p] for p in inst.P
                                  if inst.d[p] == b and sol.f_assigned.get(p) == (k, r))
                travel_sum += travel
                service_sum += load_at_a + unload_at_b
                timing_rows.append({
                    "vehicle": k, "route": r, "leg": leg,
                    "from": a, "to": b,
                    "depart_min": round(depart, 3),
                    "depart_clock": stamp(depart),
                    "arrive_min": round(arrive, 3),
                    "arrive_clock": stamp(arrive),
                    "travel_min": round(travel, 3),
                    "load_min_at_from": round(load_at_a, 3),
                    "unload_min_at_to": round(unload_at_b, 3),
                })
            timing_rows.append({
                "vehicle": k, "route": r, "leg": "TOTAL",
                "from": "—", "to": "—",
                "depart_min": "", "depart_clock": "",
                "arrive_min": "", "arrive_clock": "",
                "travel_min": round(travel_sum, 3),
                "load_min_at_from": "", "unload_min_at_to": "",
            })
            timing_rows.append({
                "vehicle": k, "route": r, "leg": "SERVICE",
                "from": "—", "to": "—",
                "depart_min": "", "depart_clock": "",
                "arrive_min": "", "arrive_clock": "",
                "travel_min": round(service_sum, 3),
                "load_min_at_from": "", "unload_min_at_to": "",
            })
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame([summary_row]).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(x_rows).to_excel(writer, sheet_name="x_used", index=False)
        pd.DataFrame(f_rows).to_excel(writer, sheet_name="assignment_f", index=False)
        pd.DataFrame(w_rows).to_excel(writer, sheet_name="wait_w", index=False)
        pd.DataFrame(itin).to_excel(writer, sheet_name="itinerary", index=False)
        if timing_rows:
            pd.DataFrame(timing_rows).to_excel(
                writer, sheet_name="route_timings", index=False
            )
        write_route_plan_sheet(writer, _build_route_sequences_from_solution(sol, inst), inst)
    draw_route_diagrams(_build_route_sequences_from_solution(sol, inst),
                        sol.ta, sol.td, inst, output_path.parent)
    log.info("Heuristic results written to %s", output_path)
    print(f"[run] results -> {output_path}")
    return summary_row
def _solution_detail_rows(sol, inst: Instance) -> dict:
    """write_heuristic_results()'in kullandigi satirlari (x/f/wait/
    itinerary/route_timings) dosyaya yazmadan liste olarak doner --
    NSGA-II reporter'i coklu Pareto cozumu icin bunu tekrar kullanir."""
    off = inst.shift_offset
    def stamp(v):
        return minutes_to_hhmm(v + off) if v is not None and not pd.isna(v) else ""

    x_rows = [{"i": i, "j": j, "k": k, "r": r, "val": 1}
              for (i, j, k, r) in sorted(sol.x_used)]
    f_rows = [{"p": p, "k": k, "r": r, "val": 1}
              for p, (k, r) in sorted(sol.f_assigned.items())]
    w_rows = [{"p": p, "w": sol.w[p]} for p in sorted(inst.P)]

    itin = []
    timing_rows = []
    for k in inst.K:
        for r in inst.routes_of(k):
            if not sol.z_used.get((k, r), False):
                continue
            try:
                seq = sol.route_order(k, r)
            except ValueError:
                continue
            for order, node in enumerate(seq, start=1):
                itin.append({
                    "k": k, "r": r, "order": order, "node": node,
                    "ta": sol.ta.get((node, k, r), 0.0),
                    "td": sol.td.get((node, k, r), 0.0),
                    "ta_clock": stamp(sol.ta.get((node, k, r), 0.0)),
                    "td_clock": stamp(sol.td.get((node, k, r), 0.0)),
                    "y_after": sol.y.get((node, k, r), 0.0),
                })
            travel_sum = 0.0
            service_sum = 0.0
            for leg, (a, b) in enumerate(zip(seq[:-1], seq[1:]), start=1):
                depart = sol.td.get((a, k, r), 0.0)
                arrive = sol.ta.get((b, k, r), 0.0)
                travel = inst.c.get((a, b), 0.0)
                load_at_a = sum(inst.s_load[p] for p in inst.P
                                if inst.o[p] == a and sol.f_assigned.get(p) == (k, r))
                unload_at_b = sum(inst.s_unload[p] for p in inst.P
                                  if inst.d[p] == b and sol.f_assigned.get(p) == (k, r))
                travel_sum += travel
                service_sum += load_at_a + unload_at_b
                timing_rows.append({
                    "vehicle": k, "route": r, "leg": leg,
                    "from": a, "to": b,
                    "depart_min": round(depart, 3), "depart_clock": stamp(depart),
                    "arrive_min": round(arrive, 3), "arrive_clock": stamp(arrive),
                    "travel_min": round(travel, 3),
                    "load_min_at_from": round(load_at_a, 3),
                    "unload_min_at_to": round(unload_at_b, 3),
                })
            timing_rows.append({
                "vehicle": k, "route": r, "leg": "TOTAL",
                "from": "—", "to": "—", "depart_min": "", "depart_clock": "",
                "arrive_min": "", "arrive_clock": "", "travel_min": round(travel_sum, 3),
                "load_min_at_from": "", "unload_min_at_to": "",
            })
            timing_rows.append({
                "vehicle": k, "route": r, "leg": "SERVICE",
                "from": "—", "to": "—", "depart_min": "", "depart_clock": "",
                "arrive_min": "", "arrive_clock": "", "travel_min": round(service_sum, 3),
                "load_min_at_from": "", "unload_min_at_to": "",
            })
    return {
        "x_rows": x_rows, "f_rows": f_rows, "w_rows": w_rows,
        "itin": itin, "timing_rows": timing_rows,
    }
def write_nsga2_detail_sheets(wb, pareto: list, inst: Instance,
                               max_solutions: int = 15,
                               run_dir: Optional[Path] = None) -> None:
    """Her Pareto cozumu icin write_heuristic_results() sekme duzeninde
    detay sayfalari ekler (sadece f1/f2 ozeti degil)."""
    from solomon import fleet_to_solution
    seen: dict = {}
    for ind in pareto:
        if ind.fleet is None:
            continue
        key = (round(ind.f1, 6), round(ind.f2, 6))
        if key not in seen:
            seen[key] = {"ind": ind, "count": 0}
        seen[key]["count"] += 1
    unique_keys = sorted(seen.keys())  # sorted by f1 asc, then f2 asc
    n_unique = len(unique_keys)
    written_keys = unique_keys[:max_solutions]
    if n_unique > max_solutions:
        log.warning(
            "NSGA-II detail sheets: %d distinct Pareto points found, "
            "only writing the first %d (by f1) -- raise "
            "nsga2_detail_max_solutions in config.xlsx to write more.",
            n_unique, max_solutions,
        )
    idx_ws = wb.create_sheet("pareto_solutions_index")
    idx_ws.append(["solution_id", "route_duration_min", "total_wait_min",
                   "n_duplicate_individuals", "detail_sheets_written"])
    for key in unique_keys:
        f1, f2 = key
        written = key in written_keys
        sid = f"s{written_keys.index(key)+1:02d}" if written else ""
        idx_ws.append([sid or "(truncated)", f1, f2, seen[key]["count"], written])
    for i, key in enumerate(written_keys, start=1):
        ind = seen[key]["ind"]
        sid = f"s{i:02d}"
        sol = fleet_to_solution(ind.fleet)
        rows = _solution_detail_rows(sol, inst)
        ws = wb.create_sheet(f"{sid}_summary")
        ws.append(["route_duration_min", "total_wait_min", "n_duplicate_individuals"])
        ws.append([ind.f1, ind.f2, seen[key]["count"]])
        ws = wb.create_sheet(f"{sid}_x_used")
        ws.append(["i", "j", "k", "r", "val"])
        for row in rows["x_rows"]:
            ws.append([row["i"], row["j"], row["k"], row["r"], row["val"]])
        ws = wb.create_sheet(f"{sid}_assignment_f")
        ws.append(["p", "k", "r", "val"])
        for row in rows["f_rows"]:
            ws.append([row["p"], row["k"], row["r"], row["val"]])
        ws = wb.create_sheet(f"{sid}_wait_w")
        ws.append(["p", "w"])
        for row in rows["w_rows"]:
            ws.append([row["p"], row["w"]])
        ws = wb.create_sheet(f"{sid}_itinerary")
        ws.append(["k", "r", "order", "node", "ta", "td", "ta_clock", "td_clock", "y_after"])
        for row in rows["itin"]:
            ws.append([row["k"], row["r"], row["order"], row["node"],
                       row["ta"], row["td"], row["ta_clock"], row["td_clock"],
                       row["y_after"]])
        if rows["timing_rows"]:
            ws = wb.create_sheet(f"{sid}_route_timings")
            ws.append(["vehicle", "route", "leg", "from", "to",
                       "depart_min", "depart_clock", "arrive_min", "arrive_clock",
                       "travel_min", "load_min_at_from", "unload_min_at_to"])
            for row in rows["timing_rows"]:
                ws.append([row["vehicle"], row["route"], row["leg"], row["from"],
                           row["to"], row["depart_min"], row["depart_clock"],
                           row["arrive_min"], row["arrive_clock"], row["travel_min"],
                           row["load_min_at_from"], row["unload_min_at_to"]])
        route_seqs = _build_route_sequences_from_solution(sol, inst)
        write_route_plan_sheet(wb, route_seqs, inst, sheet_prefix=f"{sid}_")
        if run_dir is not None:
            try:
                paths = draw_route_diagrams(route_seqs, sol.ta, sol.td, inst,
                                            run_dir, prefix=f"{sid}_")
                if paths:
                    log.info("NSGA-II %s: rota diyagramlari -> %s",
                            sid, list(paths.values()))
            except Exception as exc:
                log.warning("NSGA-II %s rota diyagrami basarisiz: %s", sid, exc)

    log.info("NSGA-II detail sheets written for %d/%d distinct Pareto points",
             len(written_keys), n_unique)
# =============================================================================
# Post-solve: verify + visualize
# =============================================================================
def postprocess(m, vars_, inst, cfg, result_xlsx, run_dir, timestamp: str = ""):
    """Verify + visualize after a solve. Writes validation.txt,
    gantt/routes/wait PNGs to run_dir. verify_on_fail=raise re-raises
    AFTER the report is written."""
    verify_failed = False
    if cfg["auto_verify"]:
        try:
            from verify import (
                extract_solution, validate_solution, format_report, passed,
            )
        except ImportError as exc:
            print(f"[verify] could not import verify.py: {exc}")
            log.error("verify.py import failed: %s", exc)
            return
        sol = extract_solution(m, vars_, inst)
        findings = validate_solution(sol, inst)
        report_text = format_report(findings)
        report_path = run_dir / "validation.txt"
        report_path.write_text(report_text, encoding="utf-8")
        log.info("Verification report -> %s", report_path)
        try:
            import openpyxl
            wb = openpyxl.load_workbook(result_xlsx)
            if "validation" in wb.sheetnames:
                del wb["validation"]
            ws = wb.create_sheet("validation")
            ws.append(["category", "test_id", "passed", "detail"])
            for f in findings:
                ws.append([f.category, f.test_id, f.passed, f.detail])
            wb.save(result_xlsx)
        except Exception as exc:
            log.warning("could not append validation sheet to xlsx: %s", exc)
        if not passed(findings):
            verify_failed = True
            print()
            print("[verify] VERIFICATION FAILED — see report:")
            print(report_text)
        else:
            n = len(findings)
            print(f"[verify] {n}/{n} checks passed")
            log.info("verification: all %d checks passed", n)
    else:
        log.info("verification skipped (auto_verify=false)")
    if cfg["auto_visualize"]:
        try:
            from verify import extract_solution
            from visualize import render_all
            sol_for_plots = extract_solution(m, vars_, inst)
            paths = render_all(sol_for_plots, inst, run_dir, suffix=timestamp)
            for kind, p in paths.items():
                print(f"[visualize] {kind} -> {p}")
            log.info("visualizations rendered: %s", list(paths.values()))
        except ImportError as exc:
            print(f"[visualize] could not import visualize.py: {exc}")
            log.error("visualize.py import failed: %s", exc)
        except Exception as exc:
            print(f"[visualize] FAILED: {exc}")
            log.error("visualization failed: %s", exc, exc_info=True)
    else:
        log.info("visualization skipped (auto_visualize=false)")

    if verify_failed and cfg["verify_on_fail"] == "raise":
        raise RuntimeError(
            "Solution failed verification (verify_on_fail=raise). "
            "See report above and in the .validation.txt file."
        )
# =============================================================================
# Multi-objective ε-constraint sweep
# =============================================================================
def _get_constraint_value(summary: dict, constraint_obj: str) -> float:
    """Pull the secondary-objective value from a per-iteration summary."""
    if constraint_obj == "wait_time":
        return float(summary["total_wait_min"])
    if constraint_obj == "route_duration":
        return float(summary["route_duration_min"])
    raise ValueError(f"unknown constraint_obj: {constraint_obj}")
def run_multi_objective_sweep(cfg, inst, main_dir: Path,
                                timestamp: str = "") -> list:
    """Adaptive eps-constraint sweep. Starts at limit_on_constraint_obj,
    tightens by eps_step each iteration until infeasible (= Pareto front
    enumerated). Per-iter outputs -> main_dir/eps_<value>/; aggregate
    pareto_summary.xlsx + pareto_frontier.png -> main_dir."""
    constraint_obj = cfg["constraint_obj"]
    primary_obj = cfg["primary_obj"]
    initial_limit = float(cfg["limit_on_constraint_obj"])
    main_dir.mkdir(parents=True, exist_ok=True)
    aggregate_rows: list = []
    current_limit = initial_limit
    print(f"[multiobj] start sweep with limit_on_constraint_obj = {initial_limit}")
    print(f"[multiobj] primary = {primary_obj}, constraint = {constraint_obj}")
    print(f"[multiobj] outputs -> {main_dir}")
    print()
    sub_cfg = dict(cfg)
    sub_cfg["verify_on_fail"] = "warn"  # never abort sweep on validation failure
    it = 0
    while True:
        sub_label = f"eps_{current_limit:.2f}"
        sub_dir = main_dir / sub_label
        sub_dir.mkdir(parents=True, exist_ok=True)
        sfx = f"_{timestamp}" if timestamp else ""
        gurobi_log = sub_dir / f"gurobi{sfx}.log"
        result_xlsx = sub_dir / f"result{sfx}.xlsx"
        print(f"[multiobj] --- iteration {it}: eps = {current_limit:.2f} ---")
        log.info("Iteration %d: building model with limit=%.4f", it, current_limit)
        m, vars_ = build_model_v2(
            inst,
            primary=primary_obj,
            constraint=constraint_obj,
            limit=current_limit,
            weight=cfg["augmentation_weight"],
            method=cfg["objective_method"],
        )
        # IIS only on iter 0 (unconstrained) -- later infeasibility is just
        # the sweep's natural end, not a data problem.
        solve_model(
            m, cfg=cfg, gurobi_log=gurobi_log,
            compute_iis_on_infeasible=(it == 0),
        )
        # Detect infeasibility
        if m.SolCount == 0:
            print(f"[multiobj] iter {it}: INFEASIBLE at eps={current_limit:.2f} — stopping sweep")
            log.info("Iteration %d infeasible; terminating sweep", it)
            aggregate_rows.append({
                "iter": it,
                "epsilon_used": current_limit,
                "status": "infeasible",
                "subfolder": sub_label,
            })
            break
        summary = write_results(m, vars_, inst, output_path=result_xlsx)
        if summary is None:
            print(f"[multiobj] iter {it}: no solution captured; stopping sweep")
            break
        # Run verifier + visualizations inside the iteration's subfolder
        try:
            postprocess(m, vars_, inst, sub_cfg, result_xlsx, sub_dir,
                        timestamp=timestamp)
        except Exception as exc:
            log.warning("postprocess raised on iteration %d: %s", it, exc)
            print(f"[multiobj] iter {it}: postprocess warning: {exc}")
        achieved_secondary = _get_constraint_value(summary, constraint_obj)
        row = {
            "iter": it,
            "epsilon_used": current_limit,
            "route_duration_min": summary["route_duration_min"],
            "total_wait_min": summary["total_wait_min"],
            "obj_value": summary["obj_value"],
            "mip_gap": summary["mip_gap"],
            "runtime_s": summary["runtime_s"],
            "status": summary["status"],
            "subfolder": sub_label,
        }
        aggregate_rows.append(row)
        print(
            f"[multiobj] iter {it}: route_duration={summary['route_duration_min']:.2f}  "
            f"total_wait={summary['total_wait_min']:.2f}  "
            f"runtime={summary['runtime_s']:.2f}s"
        )
        # Tighten epsilon for next iteration. Step size controlled by
        # cfg["eps_step"] (default 1.0). Smaller step = denser Pareto sweep
        # but more iterations.
        step = float(cfg.get("eps_step", 1.0))
        new_limit = achieved_secondary - step
        if new_limit < 0:
            print(f"[multiobj] next eps = {new_limit:.2f} < 0 — stopping sweep")
            break
        current_limit = new_limit
        it += 1
    # Aggregate spreadsheet
    sfx = f"_{timestamp}" if timestamp else ""
    sheet_path = main_dir / f"pareto_summary{sfx}.xlsx"
    try:
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "pareto_summary"
        cols = ["iter", "epsilon_used", "route_duration_min", "total_wait_min",
                "obj_value", "mip_gap", "runtime_s", "status", "subfolder"]
        ws.append(cols)
        for row in aggregate_rows:
            ws.append([row.get(c) for c in cols])
        # echo config as a second sheet for reproducibility
        ws_cfg = wb.create_sheet("config_used")
        ws_cfg.append(["parameter", "value"])
        for k, v in sorted(cfg.items()):
            ws_cfg.append([k, str(v)])
        wb.save(sheet_path)
        print(f"[multiobj] aggregate -> {sheet_path}")
    except Exception as exc:
        log.error("could not write pareto_summary.xlsx: %s", exc)
        print(f"[multiobj] failed to write summary xlsx: {exc}")
    # Pareto frontier plot
    plot_path = main_dir / f"pareto_frontier{sfx}.png"
    try:
        from visualize import plot_pareto_frontier
        plot_pareto_frontier(
            aggregate_rows, plot_path,
            x_label=(f"{primary_obj} (min)"),
            y_label=(f"{constraint_obj} (min)"),
            title=f"Pareto frontier — {cfg['product_set_id']} (|P|={len(inst.P)})",
        )
        print(f"[multiobj] frontier -> {plot_path}")
    except Exception as exc:
        log.error("could not generate pareto_frontier.png: %s", exc)
        print(f"[multiobj] failed to plot frontier: {exc}")
    return aggregate_rows
# =============================================================================
# Main
# =============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="Internal Logistics MIP runner")
    parser.add_argument("--inputs", type=Path, default=Path("inputs_new"))
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to config.xlsx (default: <inputs>/config.xlsx)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output_dir from config")
    parser.add_argument("--verbose", action="store_true",
                        help="Also stream Python log records to the console")
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="First experiment seed")
    parser.add_argument("--methods", nargs="+", choices=["nsga2_old", "nsga2_new", "heuristic"])
    parser.add_argument("--reference-fronts", type=Path, default=None,
                        help="Verified same-instance fronts JSON (e.g. MIP endpoints)")
    parser.add_argument("--independent", action="store_true",
                        help="Use Mann-Whitney instead of seed-blocked Wilcoxon")
    args = parser.parse_args()
    config_path = args.config or (args.inputs / "config.xlsx")
    cfg = load_config(config_path)
    if args.methods or cfg["run_mode"] in ("nsga2_old", "nsga2_new", "heuristic"):
        from experiments import run_experiment
        return run_experiment(args, cfg, _run_once)
    return _run_once(args, cfg)


def _run_once(args, cfg, records=None) -> int:
    """Single solve; the experiment driver owns seed scheduling and aggregation."""
    config_path = args.config or (args.inputs / "config.xlsx")
    out_dir = (Path(args.output_dir) if args.output_dir
               else Path(cfg["output_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    # 2026-08-15: %S eklendi -- ayni dakikada baslayan run'lar klasor
    # adini paylasmasin diye.
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # One folder per run, named by run_mode (multiobj_* skips a redundant
    # run_* sibling).
    if cfg["run_mode"] == "single_objective":
        label = (f"{cfg['output_prefix']}_{cfg['product_set_id']}"
                 f"_{cfg['primary_obj']}_{timestamp}")
    elif cfg["run_mode"] == "heuristic":
        label = (f"heuristic_{cfg['product_set_id']}_{timestamp}")
    elif cfg["run_mode"] in ("nsga2_old", "nsga2_new"):
        label = (f"{cfg['run_mode']}_{cfg['product_set_id']}_{timestamp}")
    else:  # multi_objective
        label = (f"multiobj_{cfg['product_set_id']}"
                 f"_{cfg['primary_obj']}_{timestamp}")
    run_dir = out_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)
    # Every file inside the run folder carries the same timestamp suffix
    # as the folder itself, so artefacts from different runs (which may
    # share the same folder during testing) stay distinguishable when
    # files are copied out for archiving or sharing.
    log_path = run_dir / f"run_{timestamp}.log"
    gurobi_log = run_dir / f"gurobi_{timestamp}.log"
    result_xlsx = run_dir / f"result_{timestamp}.xlsx"
    configure_logging(log_file=log_path, verbose=args.verbose)
    print(f"[run] {label}")
    print(f"[run] config  : {config_path}")
    print(f"[run] inputs  : {args.inputs}")
    print(f"[run] output  : {out_dir}")
    print(f"[run] script log -> {log_path}")
    if cfg["run_mode"] == "single_objective":
        print(f"[run] gurobi log -> {gurobi_log}")
    print()
    log.info("Run label : %s", label)
    log.info("Config    : %s", config_path)
    log.info("Inputs    : %s", args.inputs)
    log.info("Output    : %s", out_dir)
    try:
        inst = load_instance(args.inputs, cfg)
    except DataConsistencyError as exc:
        print(f"[run] DATA VALIDATION FAILED: {exc}")
        log.error("DATA VALIDATION FAILED: %s", exc)
        return 2
    if cfg["run_mode"] == "single_objective":
        m, vars_ = build_model_v2(
            inst,
            primary=cfg["primary_obj"],
            constraint=cfg["constraint_obj"],
            limit=cfg["limit_on_constraint_obj"],
            weight=cfg["augmentation_weight"],
            method=cfg["objective_method"],
        )
        solve_model(m, cfg=cfg, gurobi_log=gurobi_log)
        summary = write_results(m, vars_, inst, output_path=result_xlsx)
        if summary:
            print()
            print(
                f"[run] DONE  route_duration={summary['route_duration_min']:.2f}  "
                f"total_wait={summary['total_wait_min']:.2f}  "
                f"obj={summary['obj_value']:.4f}  "
                f"runtime={summary['runtime_s']:.2f}s"
            )
            print(f"[run] results -> {result_xlsx}")
            log.info(
                "DONE - route_duration=%.2f total_wait=%.2f obj=%.4f runtime=%.2fs",
                summary["route_duration_min"],
                summary["total_wait_min"],
                summary["obj_value"],
                summary["runtime_s"],
            )
            postprocess(m, vars_, inst, cfg, result_xlsx, run_dir,
                        timestamp=timestamp)
        else:
            print("[run] no feasible solution found")
        return 0
    if cfg["run_mode"] == "heuristic":
        from solomon import construct_with_multistart, fleet_to_solution
        print(f"[run] heuristic mode -> {run_dir}")
        if cfg.get("solomon_multistart", True):
            fleet, status, summary = construct_with_multistart(inst, cfg)
        else:
            from solomon import construct
            fleet, status, summary = construct(inst, cfg)
        # One line per Solomon improvement phase, then the FINAL line.
        for ph in summary.get("phases", []):
            extra = ""
            if "swaps" in ph:
                extra = f"  swaps={ph['swaps']}"
            elif "moves" in ph:
                extra = f"  moves={ph['moves']}"
            elif "iterations" in ph:
                extra = f"  iters={ph['iterations']}"
            print(
                f"[run] Solomon[{ph['name']:>8}]  "
                f"f1={ph['f1']:.3f}  f2={ph['f2']:.3f}  "
                f"runtime={ph['runtime_s']:.4f}s{extra}"
            )
        print(
            f"[run] heuristic FINAL  status={status}  "
            f"f1={summary['route_duration']:.2f}  "
            f"f2={summary['total_wait']:.2f}  "
            f"total_runtime={summary.get('runtime_s', float('nan')):.4f}s"
        )
        if status != "feasible":
            print(f"[run] HEURISTIC INFEASIBLE: unrouted parts "
                  f"{summary.get('unrouted')}")
            log.error("Heuristic failed to route all parts: %s",
                      summary.get("unrouted"))
            return 3
        # Convert to a Solution and reuse the MIP-side verifier + visualisers.
        sol = fleet_to_solution(fleet)
        write_heuristic_results(sol, inst, summary, output_path=result_xlsx)
        # Verifier and visualisations on the heuristic solution.
        failed = []
        if cfg["auto_verify"]:
            from verify import validate_solution, format_report
            findings = validate_solution(sol, inst)
            report_path = result_xlsx.parent / f"{result_xlsx.stem}.validation.txt"
            report = format_report(findings)
            report_path.write_text(report, encoding="utf-8")
            failed = [f for f in findings if not f.passed]
            if failed:
                print(f"[verify] HEURISTIC SOLUTION FAILED — see {report_path}")
                if cfg["verify_on_fail"] == "raise":
                    return 4
            else:
                print(f"[verify] heuristic solution passed all checks")
        if records is not None:
            records.append(dict(front=[] if failed else [
                (summary["route_duration"], summary["total_wait"])],
                conv_log=[], run_dir=str(run_dir)))
        if cfg["auto_visualize"]:
            from visualize import render_all
            paths = render_all(sol, inst, run_dir, suffix=timestamp)
            for kind, p in paths.items():
                print(f"[visualize] {kind} -> {p}")
        return 0
    if cfg["run_mode"] in ("nsga2_old", "nsga2_new"):
        import nsga2 as ns
        from visualize import plot_pareto_frontier
        mode   = cfg["run_mode"]
        prefix = mode + "_"   # e.g. "nsga2_old_pop_size"
        cfg2 = dict(cfg)
        cfg2["nsga2"] = {
            "mode":            mode,
            "pop_size":        int(cfg.get(prefix + "pop_size", 100)),
            "n_generations":   int(cfg.get(prefix + "n_gen", 200)),
            "seed":            int(cfg.get(prefix + "seed", 42)),
            "n_solomon_seeds": int(cfg.get(prefix + "n_solomon_seeds", 3)),
            "results_every":  int(cfg.get(prefix + "results_every", 25)),
            "p_crossover":     float(cfg.get(prefix + "p_crossover", 0.90)),
            "time_limit_sec":  float(cfg.get(prefix + "time_limit_sec", 3600)),
        }
        if mode == "nsga2_old":
            cfg2["nsga2"]["p_assignment_mutation"] = float(
                cfg.get(prefix + "p_assignment_mutation", 0.15))
            cfg2["nsga2"]["p_priority_mutation"] = float(
                cfg.get(prefix + "p_priority_mutation", 0.15))
        else:
            cfg2["nsga2"]["p_mutation"] = float(cfg.get(prefix + "p_mutation", 0.15))
            cfg2["nsga2"]["p_swap"]     = float(cfg.get(prefix + "p_swap", 0.10))
        nc = cfg2["nsga2"]
        print(f"[run] {mode} mode -> {run_dir}")
        print(f"[run] pop_size={nc['pop_size']}  n_generations={nc['n_generations']}  "
             f"seed={nc['seed']}  n_solomon_seeds={nc['n_solomon_seeds']}")
        # 2026-08-15: nsga2() artik (pareto, history, conv_log) donuyor.
        # run_dir verilirse convergence_log_{mode}_{case}_{n}_{seed}.csv
        # dosyasini nsga2.py kendi yaziyor (incremental).
        pareto, history, conv_log = ns.nsga2(inst, cfg2, run_dir=run_dir)
        print(f"[run] {mode} DONE  |Pareto|={len(pareto)}")
        # ------------------------------------------------------------------
        # 2026-09-16: FINAL Pareto cephesinin verify.py ile dogrulanmasi.
        #
        # README "her NSGA-II Pareto cozumu verify.py ile bagimsiz dogrulanir"
        # diyordu, ancak bu dal validate_solution()'i hic cagirmiyordu
        # (yalnizca heuristic dalinda cagriliyordu). Asagidaki blok, heuristic
        # dalindaki cagriyla BIREBIR ayni arayuzu kullanir:
        #     sol      = fleet_to_solution(fleet)
        #     findings = validate_solution(sol, inst)   # -> list[Finding]
        #     failed   = [f for f in findings if not f.passed]
        #
        # Kapsam: dogrulama YALNIZCA final cephede yapilir, her nesilde degil.
        # Gerekce: her nesilde 3N fenotip uretiliyor; validate_solution
        # 9 kontrol grubunu tum rotalar uzerinde calistiriyor ve nesil basina
        # maliyeti belirgin artirirdi. Ara nesillerde "repair" mekanizmasi
        # yoktur; decode_*_single insertion/kapasite/zaman-penceresi ihlalinde
        # bireye PENALTY (f1=f2=1e9) atar ve birey is_feasible()=False olarak
        # secilimde otomatik elenir (penalty-rejection). Final dogrulama bu
        # decode-ici kontrolun bagimsiz, denklem-bazli teyididir.
        #
        # Politika: dogrulamayi GECEMEYEN bireyler cepheden CIKARILIR ve
        # "pareto_rejected" sayfasinda ayri raporlanir (feasible olarak
        # sunulmaz). Cikarma tercih edildi cunku cephe, result.xlsx / pareto
        # PNG / plot_pareto_by_case.py tarafindan "feasible Pareto" olarak
        # tuketiliyor; isaretleyip birakmak bu tuketicilerin hepsinde ayri
        # filtre gerektirirdi. verify_on_fail=raise ise en az bir ret varsa
        # kod 4 ile cikilir (heuristic daliyla ayni cikis kodu).
        # ------------------------------------------------------------------
        n_pareto_raw = len(pareto)
        rejected: list = []          # (ind, findings)
        n_relabelled = 0
        if cfg["auto_verify"]:
            from verify import validate_solution, format_report
            from solomon import fleet_to_solution, FleetState
            accepted: list = []

            def _canonical_route_labels(fleet):
                """Ayni aracin kullanilan rotalarini r=1..m olarak sikistirir.

                verify._check_route_monotonicity, MIP'in simetri-kirma
                kuralini ("r+1 kullaniliyorsa r de kullanilmali") kontrol
                eder. NSGA-II kromozomu bir parcayi (k,2)'ye atayip (k,1)'i
                bos birakabilir; bu FIZIKSEL olarak (k,1)'e atanmis cozumle
                BIREBIR aynidir: solomon.simulate_route bos rota icin
                depot_arrival = T_start dondurur, dolayisiyla sonraki rotanin
                baslangic zamani kaymaz -- f1/f2 degismez (asagida assert ile
                teyit edilir). Bu yuzden dogrulama oncesi rota etiketleri
                kanonik hale getirilir; kromozom/operator/decode koduna
                dokunulmaz, sadece raporlanan cozumun etiketi duzelir.
                """
                new = FleetState(inst=fleet.inst)
                changed = False
                for k in fleet.inst.K:
                    routes = fleet.routes_of_vehicle(k)
                    ordered = ([rt for rt in routes if not rt.is_empty()]
                               + [rt for rt in routes if rt.is_empty()])
                    for r_new, rt in zip(fleet.inst.routes_of(k), ordered):
                        if rt.r != r_new:
                            changed = True
                        clone = rt.deepcopy()
                        clone.r = r_new
                        new.routes[(k, r_new)] = clone
                return new, changed

            for ind in pareto:
                if ind.fleet is None:
                    rejected.append((ind, None))
                    continue
                canon, changed = _canonical_route_labels(ind.fleet)
                if changed:
                    # Etiket degisikligi amac degerlerini degistirmemeli.
                    assert abs(canon.total_route_duration() - ind.f1) < 1e-6 \
                        and abs(canon.total_wait() - ind.f2) < 1e-6, \
                        "route relabelling changed f1/f2 -- not a pure relabel"
                    ind.fleet = canon
                    n_relabelled += 1
                sol = fleet_to_solution(ind.fleet)
                findings = validate_solution(sol, inst)
                failed = [f for f in findings if not f.passed]
                if failed:
                    rejected.append((ind, findings))
                else:
                    accepted.append(ind)
            pareto = accepted
            report_path = result_xlsx.parent / f"{result_xlsx.stem}.validation.txt"
            lines = [f"NSGA-II ({mode}) final Pareto verification",
                     f"checked={n_pareto_raw} passed={len(accepted)} "
                     f"rejected={len(rejected)} "
                     f"route_labels_canonicalised={n_relabelled}", ""]
            for i, (ind, findings) in enumerate(rejected, start=1):
                lines.append(f"--- rejected #{i}: f1={ind.f1:.4f} f2={ind.f2:.4f}")
                lines.append("fleet is None (decode returned no FleetState)"
                             if findings is None else
                             format_report([f for f in findings if not f.passed]))
                lines.append("")
            report_path.write_text("\n".join(lines), encoding="utf-8")
            if rejected:
                print(f"[verify] NSGA-II: {len(rejected)}/{n_pareto_raw} Pareto "
                      f"solutions FAILED verification -- see {report_path}")
                log.error("NSGA-II Pareto verification: %d/%d rejected",
                          len(rejected), n_pareto_raw)
                if cfg["verify_on_fail"] == "raise":
                    return 4
            else:
                print(f"[verify] NSGA-II: all {n_pareto_raw} Pareto solutions "
                      f"passed verification")
        else:
            log.info("NSGA-II Pareto verification skipped (auto_verify=false)")
        # Deduplicate AFTER verification so an invalid representative cannot
        # hide a valid route plan with identical objectives. No genetic changes.
        n_passed = len(pareto)
        unique = {}
        for ind in sorted(pareto, key=lambda ind: (ind.f1, ind.f2)):
            unique.setdefault((round(ind.f1, 6), round(ind.f2, 6)), ind)
        pareto = list(unique.values())
        if records is not None:
            records.append(dict(front=[(ind.f1, ind.f2) for ind in pareto],
                                conv_log=conv_log, run_dir=str(run_dir)))
        import openpyxl
        wb = openpyxl.Workbook()
        ws1 = wb.active
        ws1.title = "pareto"
        ws1.append(["route_duration_min", "total_wait_min"])
        for ind in pareto:
            ws1.append([ind.f1, ind.f2])
        # Dogrulamada reddedilenler ayri sayfada; cephe sayfasina girmezler.
        wsr = wb.create_sheet("pareto_rejected")
        wsr.append(["route_duration_min", "total_wait_min", "reason"])
        for ind, findings in rejected:
            reason = ("fleet is None" if findings is None else
                      "; ".join(f"{f.category}/{f.test_id}" for f in findings if not f.passed))
            wsr.append([ind.f1, ind.f2, reason])
        wsr.append(["n_checked", n_pareto_raw if cfg["auto_verify"] else 0, ""])
        wsr.append(["n_passed", n_passed if cfg["auto_verify"] else 0, ""])
        wsr.append(["n_reported_unique", len(pareto), ""])
        wsr.append(["n_duplicates_removed", n_passed - len(pareto), ""])
        wsr.append(["n_rejected", len(rejected), ""])
        wsr.append(["n_route_labels_canonicalised", n_relabelled, ""])
        ws2 = wb.create_sheet("history")
        ws2.append(["generation", "n_feasible", "best_f1", "best_f2", "front0_size"])
        for row in history:
            ws2.append(list(row))
        # Explicit metric names; singleton/empty diversity is undefined.
        from metrics import diversity_metrics
        wsm = wb.create_sheet("metrics")
        wsm.append(["metric", "value"])
        for name, value in diversity_metrics(
                [(ind.f1, ind.f2) for ind in pareto], cfg.get("true_extremes")).items():
            wsm.append([name, value if math.isfinite(value) else None])
        max_sols = int(cfg.get("nsga2_detail_max_solutions", 15))
        write_nsga2_detail_sheets(wb, pareto, inst, max_solutions=max_sols,
                                  run_dir=run_dir)
        wb.save(result_xlsx)
        print(f"[run] results -> {result_xlsx}")
        pareto_rows = [{"route_duration_min": ind.f1, "total_wait_min": ind.f2}
                       for ind in pareto]
        png_path = run_dir / f"pareto_{timestamp}.png"
        plot_pareto_frontier(pareto_rows, png_path,
                             title=f"{mode} — {cfg['product_set_id']} "
                                   f"(|P|={len(inst.P)})")
        print(f"[run] Pareto grafigi -> {png_path}")
        return 0
    # multi_objective mode: adaptive epsilon-constraint sweep.
    # The already-created `run_dir` is the multiobj folder (its label was set
    # accordingly above), so we reuse it as `main_dir` for the sweep — no
    # redundant `run_*` sibling is created.
    print(f"[run] multi_objective mode -> {run_dir}")
    log.info("Multi-objective sweep folder: %s", run_dir)
    run_multi_objective_sweep(cfg, inst, run_dir, timestamp=timestamp)
    return 0
if __name__ == "__main__":
    sys.exit(main())
