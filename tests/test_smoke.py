# Smoke test: full pipeline on the quick config
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssu_lsf import Config, run


def test_window_layout():  # Appendix G layout: T_c = 24 and the window ends at the train/test split
    for cfg in (Config(), Config.quick(0)):
        assert cfg.te == cfg.t_split - 1 and cfg.te - cfg.ts + 1 == 24


def test_pipeline_runs():  # CM anchors CRR = 0, OR anchors CRR = 1, others finite
    res = run(Config.quick(0), verbose=False)
    assert abs(res["CM"]["crr"]) < 1e-9 and abs(res["OR"]["crr"] - 1) < 1e-9
    assert all(math.isfinite(res[k]["crr"]) for k in ("GR", "SSU-LSF"))
    assert len(res["_meta"]["footprint"]) > 0
    gap = res["CM"]["rmse_aff"] / res["OR"]["rmse_aff"] - 1        # confounding must be visible,
    assert gap > 0.20                                              # else CRR is a ratio of noise
    assert 0.0 < res["SSU-LSF"]["crr"] <= 1.0                      # a real fraction of the gap is recovered
    assert 0.0 < res["GR"]["crr"] <= 1.0
    assert res["SSU-LSF"]["d_rmse_pct"] < res["GR"]["d_rmse_pct"]  # KL + rehearsal protect the clean domain
    assert res["SSU-LSF"]["d_rmse_pct"] < 10.0                     # paper worst case is 4.2%
