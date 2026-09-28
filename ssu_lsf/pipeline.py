# End-to-end experiment: CM, OR, GR baseline, SSU-LSF, plus helpers
import json, os, random
from dataclasses import replace
import numpy as np
import torch
from .data import make_archive, affected_mask, inject_labels, train_weights, grid_adjacency
from .model import DiagSSM, fit
from .influence import fisher_diag, step_scores, footprint
from .unlearn import ssu_lsf
from .metrics import test_mask, rmse, crr, scs, tci


def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)  # reproducibility


def save_json(obj, path): os.makedirs(os.path.dirname(path) or ".", exist_ok=True); json.dump(obj, open(path, "w"), indent=2)


def table(res):  # pretty results table
    rows = [f"{'Method':<9}{'CRR':>8}{'RMSE_aff':>10}{'dRMSE%':>9}{'SCS':>8}{'TCI':>8}", "-" * 52]
    for k, v in res.items():
        if not k.startswith("_"):
            rows.append(f"{k:<9}{v['crr']:>8.3f}{v['rmse_aff']:>10.4f}{v['d_rmse_pct']:>9.1f}{v['scs']:>8.4f}{v['tci']:>8.3f}")
    return "\n".join(rows)


def run(cfg, verbose=True):
    log = print if verbose else (lambda *a: None); torch.set_num_threads(max(1, os.cpu_count() // 2))
    set_seed(cfg.seed); X, Y = make_archive(cfg); aff = affected_mask(cfg)
    Yc, adj, w = inject_labels(Y, cfg, aff), grid_adjacency(cfg.grid), train_weights(cfg)
    w_clean = w.clone(); w_clean[aff, cfg.ts:cfg.te + 1] = 0.0         # everything except the forget window
    log("[1/5] training confounded model (CM)"); set_seed(cfg.seed); cm = fit(DiagSSM(cfg), X, Yc, w, cfg)
    log("[2/5] oracle retrain without forget window (OR)"); set_seed(cfg.seed); orc = fit(DiagSSM(cfg), X, Yc, w_clean, cfg)
    log("[3/5] curvature-weighted influence and footprint")
    phi = step_scores(cm, X, Yc, aff, cfg, fisher_diag(cm, X, Yc, w)); rho = cm.rho().max().item()
    steps, ext = footprint(phi, cfg, rho)
    log(f"      rho_max={rho:.3f}  tail ext={ext}  |Phi|={len(steps)}")
    log("[4/5] SSU-LSF unlearning"); ssu, hist = ssu_lsf(cm, X, Yc, aff, steps, w_clean, adj, cfg)
    log("[5/5] gradient reversal baseline (GR)")
    gr, _ = ssu_lsf(cm, X, Yc, aff, np.arange(cfg.ts, cfg.te + 1), w_clean, adj, replace(cfg, beta=0.0), use_kl=False, use_tv=False)
    ta, tu = test_mask(cfg, aff), test_mask(cfg, ~aff); res = {}
    for name, m in [("CM", cm), ("OR", orc), ("GR", gr), ("SSU-LSF", ssu)]:
        with torch.no_grad(): mu, _ = m(X)
        res[name] = dict(rmse_aff=rmse(mu, Y, ta), rmse_clean=rmse(mu, Y, tu), scs=scs(mu, adj, cfg), tci=tci(mu, Y, ta | tu))
    r_cm, r_or, c_or = res["CM"]["rmse_aff"], res["OR"]["rmse_aff"], res["OR"]["rmse_clean"]
    for v in res.values(): v["crr"] = crr(v["rmse_aff"], r_cm, r_or); v["d_rmse_pct"] = 100 * (v["rmse_clean"] / c_or - 1)
    res["_meta"] = dict(rho_max=rho, tail_ext=ext, footprint=[int(s) for s in steps], ul_epochs_run=len(hist),
                        accepted_steps=sum(h is not None for h in hist))
    return res
