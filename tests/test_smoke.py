# Smoke test: full pipeline on the compact configuration
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from ssu_lsf import Config, run
from ssu_lsf.data import grid_adjacency
from ssu_lsf.metrics import eval_mask, scs, tci


def test_window_layout():  # Appendix G layout: T_c = 24 and the window ends at the train/test split
    for cfg in (Config(), Config.compact(0)):
        assert cfg.te == cfg.t_split - 1 and cfg.te - cfg.ts + 1 == 24


def test_metric_definitions():  # Appendix G.5: TCI = 1 - sigma_yhat / sigma_y, SCS = TV(Y_hat) / (H*W)
    cfg = Config.compact(0); m = eval_mask(cfg, torch.arange(cfg.n_loc)); y = torch.randn(cfg.n_loc, cfg.T)
    assert abs(tci(y, y, m) - 1.0) < 1e-9               # no model-induced spread  ->  TCI = 1
    assert tci(3 * y, y, m) < 0                         # spread exceeds the signal  ->  TCI < 0
    adj = grid_adjacency(cfg.grid); field = torch.zeros(cfg.n_loc, cfg.T); field[0] = 1.0
    deg = sum(1 for i, _ in adj if i == 0) + sum(1 for _, j in adj if j == 0)  # neighbours of patch 0
    assert abs(scs(field, adj, cfg) - deg / cfg.n_loc) < 1e-9


def test_pipeline_runs():  # CM anchors CRR = 0, OR anchors CRR = 1, others finite
    res = run(Config.compact(0), verbose=False)
    assert abs(res["CM"]["crr"]) < 1e-9 and abs(res["OR"]["crr"] - 1) < 1e-9
    assert all(math.isfinite(res[k]["crr"]) for k in ("GR", "SSU-LSF"))
    assert len(res["_meta"]["footprint"]) > 0
    assert res["_meta"]["confound_gap_pct"] > 20                    # confounding must stay visible,
    assert 0.0 < res["SSU-LSF"]["crr"] <= 1.0                       # else CRR is a ratio of noise
    assert 0.0 < res["GR"]["crr"] <= 1.0
    assert res["SSU-LSF"]["d_rmse_pct"] < res["GR"]["d_rmse_pct"]   # KL + rehearsal protect the clean domain
    assert res["SSU-LSF"]["d_rmse_pct"] < 10.0                      # paper worst case is 4.2%

