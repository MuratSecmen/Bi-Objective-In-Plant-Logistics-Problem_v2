import argparse
import csv
import json
import math
import tempfile
import unittest
import warnings
from pathlib import Path

import experiments as ex
import metrics as m


class MetricsTests(unittest.TestCase):
    def test_hv_analytical(self):
        front = [(1, 4), (3, 2), (1, 4), (4, 4)]
        self.assertEqual(m.hypervolume_2d(front, (5, 5)), 8)
        self.assertAlmostEqual(m.hv_fixed_ref_norm(front, (5, 5)), 0.32)
        self.assertEqual(m.hypervolume_2d([], (5, 5)), 0)
        self.assertEqual(m.hypervolume_2d([(6, 1)], (5, 5)), 0)
        self.assertEqual(m.hypervolume_2d([(0, 0)], (5, 5)), 25)

    def test_reference_union_nadir(self):
        self.assertEqual(m.compute_nadir_reference([[(1, 4)], [(3, 2), (9, 9)]]),
                         (3.3, 4.4))
        self.assertEqual(m.compute_nadir_reference([[(0, 0)]]), (0.1, 0.1))
        for fronts in ([], [[]]):
            with self.assertRaises(ValueError):
                m.compute_nadir_reference(fronts)
        for ref in ((0, 3), (math.inf, 3), (-1, 2)):
            with self.assertRaises(ValueError):
                m.hv_fixed_ref_norm([], ref)

    def test_hv_alias_and_scaling(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(m.hv_instance_norm([(1, 2)], (5, 5)), 0.48)
        self.assertEqual(len(caught), 1)
        first, later = [(3, 3)], [(3, 3), (1, 4)]
        self.assertGreater(m.hv_fixed_ref_norm(later, (5, 5)),
                           m.hv_fixed_ref_norm(first, (5, 5)))

    def test_diversity_and_endpoints(self):
        front = [(1, 3), (2, 2), (3, 1)]
        self.assertEqual(m.compute_spacing_metric(front), 0)
        self.assertEqual(m.compute_true_extremes(front), ((1, 3), (3, 1)))
        self.assertEqual(m.compute_deb_delta(front, ((1, 3), (3, 1))), 0)
        self.assertAlmostEqual(m.compute_deb_delta(front, ((0, 4), (4, 0))), 0.5)
        with self.assertRaises(ValueError):
            m.compute_deb_delta(front)
        for points in ([], [(1, 1)], [(1, 1), (1, 1)]):
            self.assertTrue(math.isnan(m.compute_spacing_metric(points)))
        self.assertEqual(len(m.deduplicate_front([(1, 2), (1.00000001, 2)])), 1)

    def test_statistics(self):
        a, b = dict(enumerate([4, 5, 6, 7])), dict(enumerate([1, 2, 3, 4]))
        self.assertEqual(ex.compare_methods_statistical(a, a)["p_value"], 1)
        self.assertEqual(ex.compare_methods_statistical(a, b)["rank_biserial"], 1)
        self.assertEqual(ex.compare_methods_statistical(b, a)["rank_biserial"], -1)
        self.assertEqual(ex.compare_methods_statistical({1: 5, 2: 6}, {3: 1, 4: 2},
                                                       paired=False)["rank_biserial"], 1)
        with self.assertRaises(ValueError):
            ex.compare_methods_statistical(a, {99: 1, 100: 2})
        result = ex.summarize_runs([dict(method="a", HV=x, status="ok") for x in [1, 3]])
        hv = next(r for r in result if r["metric"] == "HV")
        self.assertEqual(hv["mean"], 2)
        self.assertAlmostEqual(hv["std"], math.sqrt(2))

    def test_experiment_rescores_all_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(n_seeds=3, methods=["nsga2_old", "nsga2_new"],
                seed=42, reference_fronts=None, output_dir=tmp, independent=False)
            cfg = dict(n_seeds=10, auto_verify=True, run_mode="nsga2_new",
                       hv_ref_f1=None, hv_ref_f2=None, output_dir=tmp,
                       product_set_id="case1", num_products=5)
            seen = []
            def fake_solve(local_args, local_cfg, captured):
                method = local_cfg["run_mode"]
                seed = local_cfg[method + "_seed"]
                seen.append((method, seed, local_cfg["solomon_random_seed"]))
                front = [(seed - 40, 5), (8, 1)]
                local_args.output_dir.mkdir(parents=True)
                captured.append(dict(front=front, run_dir=str(local_args.output_dir),
                    conv_log=[dict(generation=1, front_points_json=json.dumps(front))]))
                return 0
            self.assertEqual(ex.run_experiment(args, cfg, fake_solve), 0)
            self.assertEqual(len(seen), 6)
            self.assertTrue(all(seed == solomon for _, seed, solomon in seen))
            root = next(Path(tmp).glob("experiment_*"))
            rows = list(csv.DictReader((root / "runs.csv").read_text().splitlines()))
            self.assertEqual(len(rows), 6)
            logs = list(root.rglob("convergence_log_*.csv"))
            self.assertEqual(len(logs), 6)
            refs = {tuple((row["hv_ref_f1"], row["hv_ref_f2"]))
                    for path in logs for row in csv.DictReader(path.read_text().splitlines())}
            self.assertEqual(len(refs), 1)
            self.assertTrue((root / "hv_boxplot.png").stat().st_size > 100)
            self.assertIsNone(cfg["hv_ref_f1"])  # No mutation of caller config.

    def test_raise_policy_stops_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(n_seeds=3, methods=["nsga2_new"], seed=42,
                reference_fronts=None, output_dir=tmp, independent=False)
            cfg = dict(n_seeds=10, auto_verify=True, run_mode="nsga2_new",
                       hv_ref_f1=None, hv_ref_f2=None, output_dir=tmp)
            calls = []
            def fail(*args):
                calls.append(1)
                return 4
            self.assertEqual(ex.run_experiment(args, cfg, fail), 4)
            self.assertEqual(len(calls), 1)

    def test_default_ten_independent_seeds_and_empty_fronts(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(n_seeds=None, methods=["nsga2_old", "nsga2_new"],
                seed=42, reference_fronts=None, output_dir=tmp, independent=True)
            cfg = dict(n_seeds=10, auto_verify=True, run_mode="nsga2_new",
                       hv_ref_f1=None, hv_ref_f2=None, output_dir=tmp)
            seen = {m: [] for m in args.methods}
            def no_solution(args, cfg, captured):
                method = cfg["run_mode"]
                seen[method].append(cfg[method + "_seed"])
                return 3
            self.assertEqual(ex.run_experiment(args, cfg, no_solution), 3)
            self.assertEqual(seen["nsga2_old"], list(range(42, 52)))
            self.assertEqual(seen["nsga2_new"], list(range(52, 62)))
            root = next(Path(tmp).glob("experiment_*"))
            rows = list(csv.DictReader((root / "runs.csv").read_text().splitlines()))
            self.assertEqual(len(rows), 20)
            self.assertTrue(all(r["HV"] == "" for r in rows))


if __name__ == "__main__":
    unittest.main()
