"""Real NSGA/verification/report smoke; no MIP solver calls or fake solutions."""
import sys
from pathlib import Path
import types
import argparse
import tempfile
import json
import csv

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
# run_model imports Gurobi eagerly even for pure-Python heuristic paths.
# Stub ONLY the unused import. Any attempted model construction must fail.
try:
    import gurobipy
except ImportError:
    module = types.ModuleType('gurobipy')
    module.GRB = types.SimpleNamespace()
    def forbidden(*args, **kwargs):
        raise AssertionError('MIP must not be invoked in this smoke test')
    module.Model = module.quicksum = forbidden
    sys.modules['gurobipy'] = module
import run_model as rm
import experiments as ex

cfg = rm.load_config(repo / 'inputs_new/config.xlsx')
cfg.update(num_products=3, nsga2_parallel=False, auto_visualize=False,
           auto_verify=True, verify_on_fail='raise', nsga2_detail_max_solutions=0)
for mode in ('nsga2_old','nsga2_new'):
    cfg.update({mode+'_pop_size':6, mode+'_n_gen':2, mode+'_n_solomon_seeds':1,
                mode+'_results_every':1, mode+'_time_limit_sec':20})
with tempfile.TemporaryDirectory() as tmp:
    args = argparse.Namespace(inputs=repo/'inputs_new', config=None, output_dir=Path(tmp),
        verbose=False, n_seeds=2, methods=['nsga2_old','nsga2_new','heuristic'],
        seed=42, reference_fronts=None, independent=False)
    code = ex.run_experiment(args,cfg,rm._run_once)
    assert code == 0, code
    root = next(Path(tmp).glob('experiment_*'))
    rows = list(csv.DictReader((root/'runs.csv').read_text().splitlines()))
    assert len(rows) == 6
    assert all(r['status']=='ok' for r in rows)
    assert len(list(root.rglob('snapshot_*.csv'))) == 8
    import openpyxl
    for file in root.rglob('result_*.xlsx'):
        wb = openpyxl.load_workbook(file,read_only=True)
        if 'pareto' in wb.sheetnames:
            pts = list(wb['pareto'].values)[1:]
            assert len(pts) == len(set(pts))
        wb.close()
    import logging
    logging.shutdown()  # Windows: close the last run's log file before temp-dir cleanup
    print('REAL_SMOKE_OK', json.dumps(rows))