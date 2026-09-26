"""
nsga2.py — NSGA-II for the Internal Logistics PD-VRP.

Mode secimi cfg["nsga2"]["mode"] ile: "nsga2_old" (Tasarim A) veya
"nsga2_new" (Tasarim B). Ikisi de N kromozomu 3 farkli (alpha_1,alpha_2,
alpha_3) agirligiyla ayri ayri cozup 3N fenotip uretir, sonra non-dominated
sort + crowding ile N'e duser.

Tasarim A: assignment vektoru (parca->(k,r)) + GLOBAL priority vektoru.
Decode: priority sirasina gore Solomon-greedy + 2-opt/Or-opt.
Crossover: assignment->uniform, priority->OX. Mutasyon: rastgele yeniden
atama (assignment), %70 swap / %30 inversion (priority).

Tasarim B: sadece assignment (L2 yok), Solomon-greedy sirayi RT(n) ile
kendi bulur, ardindan 2-opt/Or-opt. Crossover: uniform. Mutasyon: transfer + swap.

Kullanim
--------
    from nsga2 import nsga2
    pareto, history = nsga2(inst, cfg)
"""
from __future__ import annotations
import logging
import math
import os as _os
import random
import multiprocessing as _mp
import time as _time
import csv as _csv
import json as _json
from metrics import (hypervolume_2d, hv_fixed_ref_norm, hv_instance_norm,
                     compute_nadir_reference, compute_spacing_metric,
                     compute_deb_delta, compute_true_extremes, diversity_metrics,
                     deduplicate_front)
from pathlib import Path as _Path
from dataclasses import dataclass, field
from solomon import (
    FleetState,
    best_insert_into_route,
    apply_insertion,
    node_rt,
    two_opt_fleet,
    or_opt_fleet,
)
try:
    from solomon import construct, _STOCK_BIASES
    _SOLOMON_AVAILABLE = True
except ImportError:
    construct = None
    _SOLOMON_AVAILABLE = False
    _STOCK_BIASES = [
        ("route_duration", {"alpha_1": 0.0001, "alpha_2": 1.0,   "alpha_3": 0.01  }),
        ("wait_time",      {"alpha_1": 0.0001, "alpha_2": 0.01,  "alpha_3": 1.0   }),
        ("balanced",       {"alpha_1": 0.0001, "alpha_2": 0.5,   "alpha_3": 0.5   }),
        ("distance",       {"alpha_1": 1.0,    "alpha_2": 0.0001, "alpha_3": 0.0001}),
        ("dist_wait",      {"alpha_1": 0.5,    "alpha_2": 0.0001, "alpha_3": 0.5  }),
        ("dist_duration",  {"alpha_1": 0.5,    "alpha_2": 0.5,   "alpha_3": 0.0001}),
    ]
log = logging.getLogger("internal_logistics.nsga2")
PENALTY = 1e9
EPS     = 1e-9
# Fenotip agirliklari (alpha_1, alpha_2, alpha_3) = (mesafe, rota suresi,
# bekleme) -- solomon.best_insert_into_route c1 = a1*dd + a2*dT + a3*dW.
# HEM Tasarim A (decode_old_phenotypes) HEM Tasarim B (decode_new_phenotypes)
# her kromozomu bu 3 kombinasyonla ayri ayri cozer (3N fenotip).
#
# 2026-09-16 -- ALPHA_COMBOS_OLD -> ALPHA_COMBOS_NEW:
#   Eski uclu {(1,.01,.0001), (.01,1,.0001), (.5,.5,.0001)} icinde alpha_3
#   (bekleme) ucunde de ~0 idi; f2 ekseninde cesitlilik decode seviyesinde
#   hic aranmiyor, yalnizca atama katmanindan geliyordu. Ucuncu kombinasyon
#   bekleme-agirlikli (0.0001, 0.01, 1) yapildi (solomon._STOCK_BIASES
#   "wait_time" ile ayni). Ilk iki kombinasyon degismedi.
#   KARSILASTIRILABILIRLIK: Bu degisiklikten onceki tum NSGA-II sonuclari
#   (Pareto, HV, delta, convergence_log_*.csv) farkli bir decode ailesiyle
#   uretildi; yeni kosularla dogrudan karsilastirilamaz, deneyler bu
#   sabitle yeniden kosulmalidir. Tasarim A/B adalet acisindan fark yok:
#   iki tasarim da ayni listeyi kullanir.
ALPHA_COMBOS_NEW = [
    (1.0,    0.01,  0.0001),   # mesafe-agirlikli
    (0.01,   1.0,   0.0001),   # rota-suresi-agirlikli (f1)
    (0.0001, 0.01,  1.0),      # bekleme-agirlikli     (f2)  [degisti]
]
# Geriye donuk uyumluluk: dis kod eski adi import ediyorsa ayni listeyi alir.
ALPHA_COMBOS_OLD = ALPHA_COMBOS_NEW
# =============================================================================
# Convergence-logging yardimcilari (2026-08-15 eklendi -- ADDITIVE, mevcut
# hicbir fonksiyona/algoritmaya dokunmuyor; sadece olcum/loglama icin).
# =============================================================================
# =============================================================================
# Yardimci: FleetState -> gene cevirici
# =============================================================================
def gene_from_fleet(fleet: FleetState, mode: str = "nsga2_new") -> dict:
    """FleetState'i nsga2_new (tek katmanli) gene'e donustur.

    Sadece atama kodlanir, sira onemli degil — decode_new kendi sirasini
    RT(n) ile kurar. nsga2_old ayri bir kromozom yapisi kullanir; bkz.
    gene_from_fleet_old.
    """
    gene = {kr: list(route.parts) for kr, route in fleet.routes.items()}
    return gene
def gene_from_fleet_old(fleet: FleetState, inst) -> dict:
    """FleetState'i Tasarim A (nsga2_old) kromozomuna donustur:
    assignment vektoru (parca -> (arac, rota)) + GLOBAL priority vektoru.

    Priority sirasi, (k, r) ciftleri sabit bir sirayla gezilerek ve her
    rotanin kendi pickup sirasi ic ice eklenerek olusturulur (deterministik).
    """
    assignment: dict = {}
    priority: list = []
    for kr in sorted(fleet.routes.keys()):
        route   = fleet.routes[kr]
        ordered = []
        seen    = set()
        for node in route.nodes[1:-1]:
            for p in route.parts:
                if p not in seen and inst.o[p] == node:
                    ordered.append(p)
                    seen.add(p)
        for p in route.parts:
            if p not in seen:
                ordered.append(p)
                seen.add(p)
        for p in ordered:
            assignment[p] = kr
        priority.extend(ordered)
    return {"assignment": assignment, "priority": priority}
# =============================================================================
# Individual
# =============================================================================
@dataclass
class Individual:
    
    """Tek bir NSGA-II cozumu (fenotip). nsga2_old'da her kromozom 3
    fenotipe (3 farkli alpha agirligiyla) cozulur — hepsi ayni .gene'i
    paylasir, sadece f1/f2/fleet farklidir."""

    gene: dict
    f1: float = PENALTY
    f2: float = PENALTY
    rank: int = 0
    crowding: float = 0.0
    fleet: object = field(default=None, repr=False)
    def dominates(self, other: "Individual") -> bool:
        return (
            self.f1 <= other.f1 + EPS and
            self.f2 <= other.f2 + EPS and
            (self.f1 < other.f1 - EPS or self.f2 < other.f2 - EPS)
        )
    def is_feasible(self) -> bool:
        return self.f1 < PENALTY - EPS
def _clone(ind: Individual) -> Individual:
    return Individual(gene={kr: list(lst) for kr, lst in ind.gene.items()})
def _clone_old(ind: Individual) -> Individual:
    return Individual(gene={
        "assignment": dict(ind.gene["assignment"]),
        "priority":   list(ind.gene["priority"]),
    })
# =============================================================================
# Chromosome yardimcilari
# =============================================================================
def random_gene(inst, rng: random.Random) -> dict:
    products = list(inst.P)
    rng.shuffle(products)
    kr_pairs = list(inst.KR_pairs)
    gene     = {kr: [] for kr in kr_pairs}
    for i, p in enumerate(products):
        gene[kr_pairs[i % len(kr_pairs)]].append(p)
    for kr in gene:
        rng.shuffle(gene[kr])
    return gene
def random_gene_old(inst, rng: random.Random) -> dict:
    """Tasarim A icin rastgele kromozom: her parca rastgele bir (k,r)'ye
    atanir, priority global rastgele bir permutasyondur."""
    kr_pairs   = list(inst.KR_pairs)
    assignment = {p: rng.choice(kr_pairs) for p in inst.P}
    priority   = list(inst.P)
    rng.shuffle(priority)
    return {"assignment": assignment, "priority": priority}
def gene_is_valid(gene: dict, inst) -> bool:
    all_p = [p for lst in gene.values() for p in lst]
    return set(all_p) == set(inst.P) and len(all_p) == len(inst.P)
def gene_is_valid_old(gene: dict, inst) -> bool:
    assigned = set(gene["assignment"].keys())
    prio     = gene["priority"]
    # Coverage of P by both layers (original checks).
    coverage_ok = (
        assigned == set(inst.P)
        and set(prio) == set(inst.P)
        and len(prio) == len(set(prio))
    )
    # Additional invariant: every assignment target must be a LEGAL (k, r)
    # slot. crossover_old_combined mixes two parents' assignment dicts; if
    # the two parents ever drew from different KR spaces a stale (k, r)
    # could leak in and later index a non-existent route slot. Checking it
    # here keeps the invariant enforced in code rather than merely assumed.
    legal_slots = set(inst.KR_pairs)
    targets_ok = all(kr in legal_slots for kr in gene["assignment"].values())
    return coverage_ok and targets_ok
# =============================================================================
# Decoder — iki mod
# =============================================================================
def decode_old_single(gene: dict, inst, cfg: dict, alphas: tuple) -> Individual:
    """Tasarim A: TEK bir fenotip uret.

    Priority vektorune gore, her parcayi KENDI atandigi (k,r)'ye Solomon-tarzi
    en-iyi-ekleme (c1) ile yerlestirir; verilen (alpha_1, alpha_2, alpha_3)
    agirliklari c1 hesabinda kullanilir. Insertion bittikten sonra 2-opt +
    Or-opt yerel aramasi calistirilir (bkz. slayt: "Run 2-opt & Or-opt on
    each constructed route").
    """
    if not gene_is_valid_old(gene, inst):
        return Individual(gene=gene, f1=PENALTY, f2=PENALTY, fleet=None)
    cfg_local = dict(cfg)
    cfg_local["alpha_1"], cfg_local["alpha_2"], cfg_local["alpha_3"] = alphas
    fleet = FleetState.empty(inst)
    for p in gene["priority"]:
        kr    = gene["assignment"][p]
        route = fleet.routes[kr]        # her adimda taze al
        baseline_dur  = fleet.total_route_duration()
        baseline_wait = fleet.total_wait()
        plan = best_insert_into_route(
            p, route, fleet, cfg_local, baseline_dur, baseline_wait
        )
        if not math.isfinite(plan.c1):
            return Individual(gene=gene, f1=PENALTY, f2=PENALTY, fleet=None)
        apply_insertion(p, plan, fleet)
    two_opt_fleet(fleet, cfg_local)
    or_opt_fleet(fleet, cfg_local)
    return Individual(
        gene=gene,
        f1=fleet.total_route_duration(),
        f2=fleet.total_wait(),
        fleet=fleet,
    )
def decode_old_phenotypes(ind: Individual, inst, cfg: dict) -> list:
    """Tasarim A: TEK kromozomdan UC fenotip uretir (ALPHA_COMBOS_NEW).

    "Decoded (3N)" adiminin karsiligi: her genotip, 3 farkli agirlikla
    ayri ayri cozulur, sonucta uretilen 3 fenotip birbirinden bagimsiz
    bireyler olarak secilim havuzuna girer.
    """
    return [decode_old_single(ind.gene, inst, cfg, alphas)
            for alphas in ALPHA_COMBOS_NEW]
def decode_new_single(gene: dict, inst, cfg: dict, alphas: tuple) -> Individual:
    """Tasarim B: tek fenotip. Her adimda kalan parcalar RT(n) skoruna gore
    onceliklendirilir (esitlikte c1), Solomon greedy ile eklenir, ardindan
    2-opt + Or-opt calistirilir. alphas c1'in agirliklarini belirler."""
    all_parts = [p for lst in gene.values() for p in lst]
    if len(all_parts) != len(set(all_parts)) or set(all_parts) != set(inst.P):
        return Individual(gene=gene, f1=PENALTY, f2=PENALTY, fleet=None)
    cfg_local = dict(cfg)
    cfg_local["alpha_1"], cfg_local["alpha_2"], cfg_local["alpha_3"] = alphas
    fleet = FleetState.empty(inst)
    for k in inst.K:
        for r in inst.routes_of(k):
            remaining = list(gene.get((k, r), []))
            while remaining:
                route     = fleet.routes[(k, r)]   # her adimda taze al
                best_p    = None
                best_plan = None
                best_score = (math.inf, math.inf)  # (rt_score, c1)
                baseline_dur  = fleet.total_route_duration()
                baseline_wait = fleet.total_wait()
                current_parts = route.parts
                for p in remaining:
                    plan = best_insert_into_route(
                        p, route, fleet, cfg_local, baseline_dur, baseline_wait
                    )
                    if not math.isfinite(plan.c1):
                        continue
                    op, dp = inst.o[p], inst.d[p]
                    trial_parts = current_parts | {p}
                    if op == "h":
                        # pickup is automatic (T_start); the real scheduling
                        # question is when dp can be reached.
                        rt_score = node_rt(dp, trial_parts, inst)
                    elif dp == "h" and route.node_position(op) >= 0:
                        # op already on the route -> attaching p costs
                        # nothing (H_present in best_insert_into_route,
                        # delivery happens automatically on return), so
                        # prioritize it immediately.
                        rt_score = 0.0
                    else:
                        o_in_route = route.node_position(op) >= 0
                        target_node = dp if o_in_route else op
                        rt_score = node_rt(target_node, trial_parts, inst)
                    score = (rt_score, plan.c1)
                    if score < best_score:
                        best_score = score
                        best_p    = p
                        best_plan = plan
                if best_p is None:
                    return Individual(gene=gene, f1=PENALTY, f2=PENALTY,
                                      fleet=None)
                apply_insertion(best_p, best_plan, fleet)
                remaining.remove(best_p)
    two_opt_fleet(fleet, cfg_local)
    or_opt_fleet(fleet, cfg_local)
    return Individual(
        gene=gene,
        f1=fleet.total_route_duration(),
        f2=fleet.total_wait(),
        fleet=fleet,
    )
def decode_new_phenotypes(ind: Individual, inst, cfg: dict) -> list:
    """Tasarim B: TEK kromozomdan UC fenotip uretir (ALPHA_COMBOS_NEW ile
    ayni 3 agirlik kombinasyonu, Tasarim A ile tutarlilik icin paylasiliyor).
    """
    return [decode_new_single(ind.gene, inst, cfg, alphas)
            for alphas in ALPHA_COMBOS_NEW]
# =============================================================================
# Non-dominated Sort (Deb et al., 2002)
# =============================================================================
def non_dominated_sort(pop: list) -> list:
    n           = len(pop)
    S           = [[] for _ in range(n)]
    n_dominated = [0] * n
    fronts      = [[]]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if pop[i].dominates(pop[j]):
                S[i].append(j)
            elif pop[j].dominates(pop[i]):
                n_dominated[i] += 1
        if n_dominated[i] == 0:
            pop[i].rank = 1
            fronts[0].append(i)
    cur = 0
    while fronts[cur]:
        nxt = []
        for i in fronts[cur]:
            for j in S[i]:
                n_dominated[j] -= 1
                if n_dominated[j] == 0:
                    pop[j].rank = cur + 2
                    nxt.append(j)
        cur += 1
        fronts.append(nxt)
    return [f for f in fronts if f]
# =============================================================================
# Crowding Distance
# =============================================================================
def crowding_distance(pop: list, front: list) -> None:
    for i in front:
        pop[i].crowding = 0.0
    for key in ("f1", "f2"):
        srt   = sorted(front, key=lambda i: getattr(pop[i], key))
        pop[srt[0]].crowding  = math.inf
        pop[srt[-1]].crowding = math.inf
        f_min = getattr(pop[srt[0]],  key)
        f_max = getattr(pop[srt[-1]], key)
        span  = (f_max - f_min) if abs(f_max - f_min) > EPS else 1.0
        for k in range(1, len(srt) - 1):
            prev = getattr(pop[srt[k - 1]], key)
            nxt  = getattr(pop[srt[k + 1]], key)
            pop[srt[k]].crowding += (nxt - prev) / span
# =============================================================================
# Selection — Binary Tournament
# =============================================================================
def tournament_select(pop: list, rng: random.Random) -> Individual:
    i, j = rng.sample(range(len(pop)), 2)
    a, b = pop[i], pop[j]
    if a.rank != b.rank:
        return a if a.rank < b.rank else b
    return a if a.crowding >= b.crowding else b
# =============================================================================
# Crossover
# =============================================================================
def _ox_lists(lst_a: list, lst_b: list, rng: random.Random) -> list:
    """Order Crossover (OX): segment korunur, kalan pozisyonlar diger
    ebeveyinden soldan-saga doldurulur."""
    n = len(lst_a)
    if n <= 1:
        return list(lst_a)
    lo, hi  = sorted(rng.sample(range(n), 2))
    child   = [None] * n
    child[lo: hi + 1] = lst_a[lo: hi + 1]
    seg_set = set(child[lo: hi + 1])
    fill    = [x for x in lst_b if x not in seg_set]
    j = 0
    for i in range(n):
        if lo <= i <= hi:
            continue
        child[i] = fill[j]; j += 1
    return child
def crossover_old_combined(p1: Individual, p2: Individual,
                          inst, rng: random.Random) -> tuple:
    """Tasarim A crossover: assignment vektoru -> UNIFORM, priority vektoru
    -> OX (Order Crossover) — ikisi de AYNI kromozomda, tek bir crossover
    cagrisinda uygulanir (bkz. slayt "NSGA-II - Crossover")."""
    products = list(inst.P)
    def _build_child(primary: Individual, secondary: Individual) -> Individual:
        assignment_c = {}
        for p in products:
            src = primary if rng.random() < 0.5 else secondary
            assignment_c[p] = src.gene["assignment"][p]
        priority_c = _ox_lists(primary.gene["priority"],
                              secondary.gene["priority"], rng)
        return Individual(gene={"assignment": assignment_c,
                                "priority": priority_c})
    c1 = _build_child(p1, p2)
    c2 = _build_child(p2, p1)
    assert gene_is_valid_old(c1.gene, inst), "crossover_old_combined c1 invalid"
    assert gene_is_valid_old(c2.gene, inst), "crossover_old_combined c2 invalid"
    return c1, c2
def uniform_crossover(p1: Individual, p2: Individual,
                      inst, rng: random.Random) -> tuple:
    """nsga2_new crossover: Uniform, L2 yok, sira onemli degil."""
    kr_pairs = list(inst.KR_pairs)
    def _build_child(primary: Individual, secondary: Individual) -> Individual:
        gene_c   = {kr: [] for kr in kr_pairs}
        assigned = set()

        for kr in kr_pairs:
            for p in primary.gene.get(kr, []):
                if rng.random() < 0.5:
                    gene_c[kr].append(p)
                    assigned.add(p)
        for kr in kr_pairs:
            for p in secondary.gene.get(kr, []):
                if p not in assigned:
                    gene_c[kr].append(p)
                    assigned.add(p)
        missing = [p for p in inst.P if p not in assigned]
        for p in missing:
            lightest = min(kr_pairs, key=lambda kr: len(gene_c[kr]))
            gene_c[lightest].append(p)
        return Individual(gene=gene_c)
    c1 = _build_child(p1, p2)
    c2 = _build_child(p2, p1)
    assert gene_is_valid(c1.gene, inst), "uniform_crossover c1 invalid"
    assert gene_is_valid(c2.gene, inst), "uniform_crossover c2 invalid"
    return c1, c2
# =============================================================================
# Mutasyon operatorleri
# =============================================================================
def mutate_assignment_old(ind: Individual, inst, rng: random.Random,
                          p: float = 0.15) -> Individual:
    """Tasarim A: assignment vektorunde bir parcayi rastgele baska bir
    (arac, rota)'ya yeniden ata."""
    if rng.random() > p:
        return ind
    ind = _clone_old(ind)
    product = rng.choice(list(inst.P))
    ind.gene["assignment"][product] = rng.choice(list(inst.KR_pairs))
    return ind
def mutate_priority_old(ind: Individual, rng: random.Random,
                        p: float = 0.15) -> Individual:
    """Tasarim A: priority vektorunde %70 swap, %30 inversion."""
    if rng.random() > p:
        return ind
    ind = _clone_old(ind)
    lst = ind.gene["priority"]
    if len(lst) < 2:
        return ind
    if rng.random() < 0.70:
        i, j = rng.sample(range(len(lst)), 2)
        lst[i], lst[j] = lst[j], lst[i]
    else:
        lo, hi = sorted(rng.sample(range(len(lst)), 2))
        lst[lo: hi + 1] = lst[lo: hi + 1][::-1]
    return ind
def transfer_mutation(ind: Individual, inst, rng: random.Random,
                      p: float = 0.10) -> Individual:
    """nsga2_new: parcayi farkli (k,r)'ye tasi."""
    if rng.random() > p:
        return ind
    ind       = _clone(ind)
    non_empty = [kr for kr, lst in ind.gene.items() if lst]
    if not non_empty:
        return ind
    src_kr  = rng.choice(non_empty)
    tgt_kr  = rng.choice(list(inst.KR_pairs))
    if tgt_kr == src_kr:
        return ind
    src_lst = ind.gene[src_kr]
    p_idx   = rng.randint(0, len(src_lst) - 1)
    product = src_lst.pop(p_idx)
    # Duplicate kontrolu: tgt_kr'de zaten varsa ekleme
    if product not in ind.gene[tgt_kr]:
        tgt_lst = ind.gene[tgt_kr]
        ins_pos = rng.randint(0, len(tgt_lst))
        tgt_lst.insert(ins_pos, product)
    else:
        # Geri koy
        src_lst.insert(p_idx, product)
    return ind
def swap_mutation_new(ind: Individual, inst, rng: random.Random,
                      p: float = 0.10) -> Individual:
    """nsga2_new: iki farkli (k,r) arasinda birer parca takas et."""
    if rng.random() > p:
        return ind
    ind       = _clone(ind)
    non_empty = [kr for kr, lst in ind.gene.items() if lst]
    if len(non_empty) < 2:
        return ind
    kr_a, kr_b = rng.sample(non_empty, 2)
    lst_a = ind.gene[kr_a]
    lst_b = ind.gene[kr_b]
    idx_a = rng.randint(0, len(lst_a) - 1)
    idx_b = rng.randint(0, len(lst_b) - 1)
    pa, pb = lst_a[idx_a], lst_b[idx_b]
    # Duplicate olmamasi icin kontrol
    if pa != pb:
        lst_a[idx_a], lst_b[idx_b] = pb, pa
    return ind
# =============================================================================
# Ilk populasyon
# =============================================================================
def _solomon_seeds(inst, cfg, n: int, mode: str = "nsga2_new") -> list:
    """nsga2_new icin Solomon-tohumlu bireyler uretir."""
    if not _SOLOMON_AVAILABLE or construct is None:
        log.warning("Solomon bulunamadi; rastgele bireyler kullanilacak.")
        return []
    seeds = []
    seed_backtrack_depth = int(cfg.get("nsga2_seed_backtrack_max_depth", 5))
    for name, alphas in _STOCK_BIASES[:n]:
        trial_cfg = dict(cfg)
        trial_cfg.update(alphas)
        # Tohumlama icin derin backtrack gereksiz; sığ derinlik yeterli.
        trial_cfg["solomon_backtrack_max_depth"] = seed_backtrack_depth
        try:
            fleet, status, _ = construct(inst, trial_cfg)
            if status == "feasible":
                gene = gene_from_fleet(fleet, mode=mode)
                ind  = Individual(gene=gene)
                ind.fleet = fleet
                ind.f1    = fleet.total_route_duration()
                ind.f2    = fleet.total_wait()
                seeds.append(ind)
                log.info("Solomon seed [%s]: f1=%.2f  f2=%.2f",
                         name, ind.f1, ind.f2)
            else:
                log.warning("Solomon seed [%s]: infeasible, atlandi.", name)
        except Exception as exc:
            log.warning("Solomon seed [%s] basarisiz: %s", name, exc)
    return seeds
def _solomon_seeds_old(inst, cfg, n: int) -> list:
    """Tasarim A icin Solomon-tohumlu kromozomlar (gene dict listesi,
    henuz Individual/fenotip degil — decode_old_phenotypes'e verilecek)."""
    if not _SOLOMON_AVAILABLE or construct is None:
        log.warning("Solomon bulunamadi; rastgele kromozomlar kullanilacak.")
        return []
    genes = []
    seed_backtrack_depth = int(cfg.get("nsga2_seed_backtrack_max_depth", 5))
    for name, alphas in _STOCK_BIASES[:n]:
        trial_cfg = dict(cfg)
        trial_cfg.update(alphas)
        # Tohumlama icin sig backtrack derinligi.
        trial_cfg["solomon_backtrack_max_depth"] = seed_backtrack_depth
        try:
            fleet, status, _ = construct(inst, trial_cfg)
            if status == "feasible":
                genes.append(gene_from_fleet_old(fleet, inst))
                log.info("Solomon seed (old) [%s]: fleet f1=%.2f f2=%.2f",
                         name, fleet.total_route_duration(),
                         fleet.total_wait())
            else:
                log.warning("Solomon seed (old) [%s]: infeasible, atlandi.",
                            name)
        except Exception as exc:
            log.warning("Solomon seed (old) [%s] basarisiz: %s", name, exc)
    return genes
def initial_population(inst, cfg, pop_size: int,
                        rng: random.Random,
                        n_solomon_seeds: int = 3,
                        mode: str = "nsga2_new") -> list:
    """Tasarim B icin baslangic populasyonu: her kromozom (Solomon-tohumlu
    veya rastgele) 3 fenotibe cozulur (ALPHA_COMBOS_NEW ile); N fenotibe
    ulasilinca kesilir — Tasarim A ile ayni sema."""
    phenotypes = []
    for seed_ind in _solomon_seeds(inst, cfg, n_solomon_seeds, mode=mode):
        phenotypes.extend(decode_new_phenotypes(seed_ind, inst, cfg))
    while len(phenotypes) < pop_size:
        gene = random_gene(inst, rng)
        ind  = Individual(gene=gene)
        phenotypes.extend(decode_new_phenotypes(ind, inst, cfg))
    phenotypes = phenotypes[:pop_size]
    log.info("Baslangic pop (Tasarim B): %d fenotip, %d feasible",
             len(phenotypes), sum(1 for i in phenotypes if i.is_feasible()))
    return phenotypes
def initial_population_old(inst, cfg, pop_size: int,
                          rng: random.Random,
                          n_solomon_seeds: int = 3) -> list:
    """Tasarim A icin baslangic populasyonu: her kromozom (Solomon-tohumlu
    veya rastgele) 3 fenotibe cozulur; N fenotibe ulasilinca kesilir."""
    phenotypes = []
    for gene in _solomon_seeds_old(inst, cfg, n_solomon_seeds):
        seed_ind = Individual(gene=gene)
        phenotypes.extend(decode_old_phenotypes(seed_ind, inst, cfg))
    while len(phenotypes) < pop_size:
        gene = random_gene_old(inst, rng)
        ind  = Individual(gene=gene)
        phenotypes.extend(decode_old_phenotypes(ind, inst, cfg))
    phenotypes = phenotypes[:pop_size]
    log.info("Baslangic pop (Tasarim A): %d fenotip, %d feasible",
             len(phenotypes), sum(1 for i in phenotypes if i.is_feasible()))
    return phenotypes
_worker_inst = None
_worker_cfg = None
def _pool_init(inst, cfg) -> None:
    """Pool initializer: her worker surecinde inst/cfg BIR KEZ set edilir,
    boylece her nesilde ayni degismeyen nesneler tekrar tekrar pickle'lanip
    IPC uzerinden gonderilmez (onceki tasarim: her chromosome-task kendi
    (gene, inst, cfg) kopyasini tasiyordu)."""
    global _worker_inst, _worker_cfg
    _worker_inst = inst
    _worker_cfg = cfg
def _decode_old_worker(gene) -> list:
    ind = Individual(gene=gene)
    return decode_old_phenotypes(ind, _worker_inst, _worker_cfg)
def _decode_new_worker(gene) -> list:
    ind = Individual(gene=gene)
    return decode_new_phenotypes(ind, _worker_inst, _worker_cfg)
def _make_pool(cfg: dict, inst=None):
    """cfg["nsga2_parallel"]=False ise None doner (sirali calisma).
    Varsayilan worker: CPU cekirdek - 1 (en az 1). cfg["nsga2_n_workers"]
    ile ezilebilir. inst/cfg worker'lara initializer ile bir kez gonderilir."""
    if not cfg.get("nsga2_parallel", True):
        return None
    n_workers = cfg.get("nsga2_n_workers")
    if n_workers is None:
        n_workers = max(1, (_os.cpu_count() or 2) - 1)
    n_workers = int(n_workers)
    if n_workers <= 1:
        return None
    # Windows uyumlulugu icin spawn.
    ctx = _mp.get_context("spawn")
    return ctx.Pool(processes=n_workers, initializer=_pool_init,
                     initargs=(inst, cfg))
# =============================================================================
# NSGA-II Ana Dongu — tek giris noktasi
# =============================================================================
def nsga2(inst, cfg: dict, run_dir=None) -> tuple:
    """
    NSGA-II calistir. (pareto_front, history, conv_log) doner.

    Mod secimi cfg["nsga2"]["mode"] ile yapilir:
        "nsga2_old"  —  Tasarim A: assignment+priority, 3 fenotip/kromozom
        "nsga2_new"  —  Tasarim B: tek katmanli, RT(n) decode

    history formati (geriye-donuk uyumlu):
        [(nesil, n_feasible, best_f1, best_f2, |front_0|), ...]

    conv_log formati (2026-08-15): dict listesi, her generation icin
    genisletilmis metrikler --
        generation, n_feasible, pareto_size, best_f1, best_f2,
        hv_fixed_ref, hv_fixed_ref_norm, Spacing, Deb_Delta (if endpoints supplied),
        crossover_feasibility_rate, clone_feasibility_rate, feasibility_rate, elapsed_s
    Bkz. hypervolume_2d() / compute_spacing_metric() docstring'leri metrik
    tanimlari icin.

    run_dir: verilirse (str/Path), her generation'da
        convergence_log_{mode}_{product_set_id}_{num_products}_{seed}.csv
        dosyasina append edilir (run kesilse bile veri kaybolmaz). None
        ise CSV yazilmaz, conv_log sadece bellek-ici liste olarak doner.
    """
    solomon_log = logging.getLogger("internal_logistics.solomon")
    _prev_solomon_level = solomon_log.level
    if not cfg.get("nsga2_verbose_solomon_log", False):
        solomon_log.setLevel(logging.WARNING)
    try:
        nc   = cfg.get("nsga2", {})
        mode = nc.get("mode", "nsga2_old")
        if mode == "nsga2_old":
            return _nsga2_old(inst, cfg, run_dir=run_dir)
        return _nsga2_new(inst, cfg, run_dir=run_dir)
    finally:
        solomon_log.setLevel(_prev_solomon_level)
def _resolve_hv_reference(cfg: dict):
    """No per-run fallback. Missing reference means deferred scoring."""
    r1 = cfg.get("hv_ref_f1")
    r2 = cfg.get("hv_ref_f2")
    if r1 is not None and r2 is not None:
        log.info("HV referans noktasi: config'ten (hv_ref_f1=%.2f, hv_ref_f2=%.2f)",
                 float(r1), float(r2))
        return (float(r1), float(r2))
    if r1 is not None or r2 is not None:
        raise ValueError("Set both hv_ref_f1 and hv_ref_f2, or neither")
    log.info("HV deferred until a shared reference is available; saving fronts")
    return None
def _conv_log_path(run_dir, cfg: dict, mode: str):
    if run_dir is None:
        return None
    nc   = cfg.get("nsga2", {})
    pset = cfg.get("product_set_id", "case_unknown")
    npr  = cfg.get("num_products", "n_unknown")
    seed = nc.get("seed", "seed_unknown")
    fname = f"convergence_log_{mode}_{pset}_{npr}_{seed}.csv"
    return _Path(run_dir) / fname
def _append_conv_log_row(path, row: dict, write_header: bool) -> None:
    """conv_log satirini CSV'ye INCREMENTAL yazar (run kesilse bile o ana
    kadarki veri diskte kalir). path None ise no-op (CSV istenmemis)."""
    if path is None:
        return
    file_mode = "w" if write_header else "a"
    with open(path, file_mode, newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_pareto_snapshot(run_dir, mode, generation, points):
    """Decoder-feasible intermediate diagnostic; NOT a verified final result."""
    if run_dir is None:
        return
    path = _Path(run_dir) / f"snapshot_{mode}_gen_{generation}.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = _csv.writer(stream)
        writer.writerow(["route_duration_min", "total_wait_min", "validation_status"])
        writer.writerows((f1, f2, "decoder_only") for f1, f2 in points)
def _nsga2_old(inst, cfg: dict, run_dir=None) -> tuple:
    """Tasarim A ana dongusu. Her N kromozom 3 fenotibe cozulur (3N havuz),
    mevcut N fenotiple birlestirilir, non-dominated sort + crowding ile
    tekrar N'e dusurulur."""
    nc = cfg.get("nsga2", {})
    pop_size   = int(nc.get("pop_size",       100))
    n_gen      = int(nc.get("n_generations",  200))
    results_every = int(nc.get("results_every", 25))
    time_limit = nc.get("time_limit_sec")   # None = sinirsiz (eski davranis)
    time_limit = float(time_limit) if time_limit is not None else None
    p_cross    = float(nc.get("p_crossover", 0.90))
    n_seeds    = int(nc.get("n_solomon_seeds",  3))
    seed       = nc.get("seed", 42)
    p_assign   = float(nc.get("p_assignment_mutation", 0.15))
    p_priority = float(nc.get("p_priority_mutation",   0.15))
    rng = random.Random(seed)
    log.info("NSGA-II (Tasarim A) BASLIYOR | pop=%d gen=%d seed=%d",
             pop_size, n_gen, seed)
    population = initial_population_old(inst, cfg, pop_size, rng,
                                       n_solomon_seeds=n_seeds)
    fronts = non_dominated_sort(population)
    for front in fronts:
        crowding_distance(population, front)
    history = []
    # convergence-logging kurulumu (2026-08-15)
    conv_log  = []
    t_start   = _time.perf_counter()
    hv_ref    = _resolve_hv_reference(cfg)
    conv_path = _conv_log_path(run_dir, cfg, "nsga2_old")
    n_phen    = len(ALPHA_COMBOS_NEW)
    # /convergence-logging kurulumu
    pool = _make_pool(cfg, inst)
    if pool is not None:
        log.info("NSGA-II (Tasarim A): paralel decode acik, %d worker",
                 pool._processes)
    if time_limit is not None:
        log.info("NSGA-II (Tasarim A): zaman siniri aktif -- %.0f s", time_limit)
    try:
        for gen in range(1, n_gen + 1):
            # zaman siniri kontrolu (2026-08-15)
            # Nesil ICINDE degil, nesil BASINDA kontrol edilir -- yarim
            # kalmis bir nesil (tutarsiz next_pop) asla uretilmez, en kotu
            # ihtimalle bir nesil fazladan surer.
            if time_limit is not None and (_time.perf_counter() - t_start) > time_limit:
                log.warning(
                    "NSGA-II (Tasarim A): ZAMAN SINIRI ASILDI (%.0f s) -- "
                    "nesil %d/%d'de DURDURULDU. Elde edilen populasyonla "
                    "devam ediliyor (n_gen'e tam ulasilmadi).",
                    time_limit, gen - 1, n_gen,
                )
                break
            # N cocuk kromozom
            child_genotypes = []
            used_crossover = []  # conv-log: crossover/clone etiketi, karar
                                  # mantigini degistirmez.
            while len(child_genotypes) < pop_size:
                p1 = tournament_select(population, rng)
                p2 = tournament_select(population, rng)
                did_cross = rng.random() < p_cross
                if did_cross:
                    c1, c2 = crossover_old_combined(p1, p2, inst, rng)
                else:
                    c1, c2 = _clone_old(p1), _clone_old(p2)
                for child in (c1, c2):
                    child = mutate_assignment_old(child, inst, rng, p_assign)
                    child = mutate_priority_old(child, rng, p_priority)
                    if len(child_genotypes) < pop_size:
                        child_genotypes.append(child)
                        used_crossover.append(did_cross)
            # her kromozom -> 3 fenotip (paralel/sirali sonuc ayni --
            # pool.map girdi sirasini korur)
            if pool is not None:
                tasks = [child.gene for child in child_genotypes]
                # IPC maliyetini dusurmek icin chunksize ayari.
                _chunksize = max(1, len(tasks) // (pool._processes * 2))
                results = pool.map(_decode_old_worker, tasks,
                                   chunksize=_chunksize)
                offspring_phenotypes = [ind for sub in results for ind in sub]
            else:
                offspring_phenotypes = []
                for child in child_genotypes:
                    offspring_phenotypes.extend(
                        decode_old_phenotypes(child, inst, cfg))
            combined = population + offspring_phenotypes
            fronts = non_dominated_sort(combined)
            for front in fronts:
                crowding_distance(combined, front)
            # N'e dus (elitism)
            next_pop = []
            for front in fronts:
                if len(next_pop) + len(front) <= pop_size:
                    next_pop.extend(combined[i] for i in front)
                else:
                    remaining     = pop_size - len(next_pop)
                    best_of_front = sorted(
                        front,
                        key=lambda i: combined[i].crowding,
                        reverse=True,
                    )[:remaining]
                    next_pop.extend(combined[i] for i in best_of_front)
                    break
            population = next_pop
            feasible   = [ind for ind in population if ind.is_feasible()]
            n_feas     = len(feasible)
            best_f1    = min((ind.f1 for ind in feasible), default=PENALTY)
            best_f2    = min((ind.f2 for ind in feasible), default=PENALTY)
            pareto_now = [ind for ind in population if ind.rank == 1
                          and ind.is_feasible()]
            front_pts = deduplicate_front((ind.f1, ind.f2) for ind in pareto_now)
            history.append((gen, n_feas, best_f1, best_f2, len(front_pts)))
            if results_every > 0 and gen % results_every == 0:
                _write_pareto_snapshot(run_dir, nc["mode"], gen, front_pts)

            # convergence-logging
            hv_fixed = hypervolume_2d(front_pts, hv_ref) if hv_ref else None
            hv_norm = hv_fixed_ref_norm(front_pts, hv_ref) if hv_ref else None
            diversity = diversity_metrics(front_pts, cfg.get("true_extremes"))
            # TODO: true acceptance needs before/after variation + selection
            # instrumentation. These measure decode feasibility, not acceptance.
            cx_total = cx_ok = clone_total = clone_ok = 0
            for i, is_cx in enumerate(used_crossover):
                group = offspring_phenotypes[i * n_phen:(i + 1) * n_phen]
                feasible_any = any(ind.is_feasible() for ind in group)
                if is_cx:
                    cx_total += 1
                    cx_ok += int(feasible_any)
                else:
                    clone_total += 1
                    clone_ok += int(feasible_any)
            row = {
                "generation": gen,
                "n_feasible": n_feas,
                "pareto_size": len(front_pts),
                "best_f1": best_f1,
                "best_f2": best_f2,
                "hv_fixed_ref": hv_fixed,
                "hv_fixed_ref_norm": hv_norm,
                "hv_ref_f1": hv_ref[0] if hv_ref else None,
                "hv_ref_f2": hv_ref[1] if hv_ref else None,
                "front_points_json": _json.dumps(front_pts),
                **diversity,
                "crossover_feasibility_rate": (cx_ok / cx_total) if cx_total else None,
                "clone_feasibility_rate": (clone_ok / clone_total) if clone_total else None,
                "feasibility_rate": (cx_ok + clone_ok) / len(child_genotypes),
                "elapsed_s": _time.perf_counter() - t_start,
            }
            conv_log.append(row)
            _append_conv_log_row(conv_path, row, write_header=(gen == 1))
            # /convergence-logging

            if gen % 10 == 0 or gen == 1:
                log.info(
                    "Nesil %3d | uygun=%d/%d | f1=%.2f | f2=%.2f | |PF|=%d",
                    gen, n_feas, pop_size, best_f1, best_f2, len(pareto_now)
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    pareto = [ind for ind in population
              if ind.rank == 1 and ind.is_feasible()]
    log.info("NSGA-II (Tasarim A) BITTI | |Pareto|=%d", len(pareto))
    return pareto, history, conv_log
def _nsga2_new(inst, cfg: dict, run_dir=None) -> tuple:
    """Tasarim B ana dongusu, Tasarim A ile ayni semada: her N kromozom
    3 fenotibe cozulur, 3N havuzu mevcut N ile birlestirilir, non-dominated
    sort + crowding ile N'e dusurulur."""
    nc = cfg.get("nsga2", {})
    pop_size   = int(nc.get("pop_size",       100))
    n_gen      = int(nc.get("n_generations",  200))
    results_every = int(nc.get("results_every", 25))
    time_limit = nc.get("time_limit_sec")   # None = sinirsiz (eski davranis)
    time_limit = float(time_limit) if time_limit is not None else None
    p_cross    = float(nc.get("p_crossover", 0.90))
    n_seeds    = int(nc.get("n_solomon_seeds",  3))
    seed       = nc.get("seed", 42)
    p_transfer = float(nc.get("p_mutation", 0.15))
    p_swap_new = float(nc.get("p_swap",     0.10))
    rng = random.Random(seed)
    log.info("NSGA-II (Tasarim B) BASLIYOR | pop=%d gen=%d seed=%d",
             pop_size, n_gen, seed)
    population = initial_population(inst, cfg, pop_size, rng,
                                    n_solomon_seeds=n_seeds, mode="nsga2_new")
    fronts = non_dominated_sort(population)
    for front in fronts:
        crowding_distance(population, front)
    history = []
    # convergence-logging kurulumu (2026-08-15)
    conv_log  = []
    t_start   = _time.perf_counter()
    hv_ref    = _resolve_hv_reference(cfg)
    conv_path = _conv_log_path(run_dir, cfg, "nsga2_new")
    n_phen    = len(ALPHA_COMBOS_NEW)
    # /convergence-logging kurulumu
    pool = _make_pool(cfg, inst)
    if pool is not None:
        log.info("NSGA-II (Tasarim B): paralel decode acik, %d worker",
                 pool._processes)
    if time_limit is not None:
        log.info("NSGA-II (Tasarim B): zaman siniri aktif -- %.0f s", time_limit)
    try:
        for gen in range(1, n_gen + 1):
            # zaman siniri kontrolu (2026-08-15)
            if time_limit is not None and (_time.perf_counter() - t_start) > time_limit:
                log.warning(
                    "NSGA-II (Tasarim B): ZAMAN SINIRI ASILDI (%.0f s) -- "
                    "nesil %d/%d'de DURDURULDU. Elde edilen populasyonla "
                    "devam ediliyor (n_gen'e tam ulasilmadi).",
                    time_limit, gen - 1, n_gen,
                )
                break

            child_genotypes = []
            used_crossover = []  # conv-log: bkz. _nsga2_old
            while len(child_genotypes) < pop_size:

                p1 = tournament_select(population, rng)
                p2 = tournament_select(population, rng)

                did_cross = rng.random() < p_cross
                if did_cross:
                    c1, c2 = uniform_crossover(p1, p2, inst, rng)
                else:
                    c1, c2 = _clone(p1), _clone(p2)

                for child in (c1, c2):
                    child = transfer_mutation(child, inst, rng, p_transfer)
                    child = swap_mutation_new(child, inst, rng, p_swap_new)
                    if len(child_genotypes) < pop_size:
                        child_genotypes.append(child)
                        used_crossover.append(did_cross)
            if pool is not None:
                tasks = [child.gene for child in child_genotypes]
                _chunksize = max(1, len(tasks) // (pool._processes * 2))
                results = pool.map(_decode_new_worker, tasks,
                                   chunksize=_chunksize)
                offspring_phenotypes = [ind for sub in results for ind in sub]
            else:
                offspring_phenotypes = []
                for child in child_genotypes:
                    offspring_phenotypes.extend(
                        decode_new_phenotypes(child, inst, cfg))
            combined = population + offspring_phenotypes
            fronts = non_dominated_sort(combined)
            for front in fronts:
                crowding_distance(combined, front)
            next_pop = []
            for front in fronts:
                if len(next_pop) + len(front) <= pop_size:
                    next_pop.extend(combined[i] for i in front)
                else:
                    remaining     = pop_size - len(next_pop)
                    best_of_front = sorted(
                        front,
                        key=lambda i: combined[i].crowding,
                        reverse=True,
                    )[:remaining]
                    next_pop.extend(combined[i] for i in best_of_front)
                    break
            population = next_pop
            feasible   = [ind for ind in population if ind.is_feasible()]
            n_feas     = len(feasible)
            best_f1    = min((ind.f1 for ind in feasible), default=PENALTY)
            best_f2    = min((ind.f2 for ind in feasible), default=PENALTY)
            pareto_now = [ind for ind in population if ind.rank == 1
                          and ind.is_feasible()]
            front_pts = deduplicate_front((ind.f1, ind.f2) for ind in pareto_now)
            history.append((gen, n_feas, best_f1, best_f2, len(front_pts)))
            if results_every > 0 and gen % results_every == 0:
                _write_pareto_snapshot(run_dir, nc["mode"], gen, front_pts)

            # convergence-logging
            hv_fixed = hypervolume_2d(front_pts, hv_ref) if hv_ref else None
            hv_norm = hv_fixed_ref_norm(front_pts, hv_ref) if hv_ref else None
            diversity = diversity_metrics(front_pts, cfg.get("true_extremes"))
            # TODO: true acceptance needs before/after variation + selection
            # instrumentation. These measure decode feasibility, not acceptance.
            cx_total = cx_ok = clone_total = clone_ok = 0
            for i, is_cx in enumerate(used_crossover):
                group = offspring_phenotypes[i * n_phen:(i + 1) * n_phen]
                feasible_any = any(ind.is_feasible() for ind in group)
                if is_cx:
                    cx_total += 1
                    cx_ok += int(feasible_any)
                else:
                    clone_total += 1
                    clone_ok += int(feasible_any)
            row = {
                "generation": gen,
                "n_feasible": n_feas,
                "pareto_size": len(front_pts),
                "best_f1": best_f1,
                "best_f2": best_f2,
                "hv_fixed_ref": hv_fixed,
                "hv_fixed_ref_norm": hv_norm,
                "hv_ref_f1": hv_ref[0] if hv_ref else None,
                "hv_ref_f2": hv_ref[1] if hv_ref else None,
                "front_points_json": _json.dumps(front_pts),
                **diversity,
                "crossover_feasibility_rate": (cx_ok / cx_total) if cx_total else None,
                "clone_feasibility_rate": (clone_ok / clone_total) if clone_total else None,
                "feasibility_rate": (cx_ok + clone_ok) / len(child_genotypes),
                "elapsed_s": _time.perf_counter() - t_start,
            }
            conv_log.append(row)
            _append_conv_log_row(conv_path, row, write_header=(gen == 1))
            # /convergence-logging

            if gen % 10 == 0 or gen == 1:
                log.info(
                    "Nesil %3d | uygun=%d/%d | f1=%.2f | f2=%.2f | |PF|=%d",
                    gen, n_feas, pop_size, best_f1, best_f2, len(pareto_now)
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    pareto = [ind for ind in population
              if ind.rank == 1 and ind.is_feasible()]
    log.info("NSGA-II (Tasarim B) BITTI | |Pareto|=%d", len(pareto))
    return pareto, history, conv_log
