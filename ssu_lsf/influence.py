# Curvature-preconditioned influence scores and temporal footprint Phi
import math
import numpy as np
import torch
from .model import nll


def grads(model, loss):  # flattened gradient, graph kept for reuse
    gs = torch.autograd.grad(loss, list(model.parameters()), retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, model.parameters())])


def fisher_diag(model, X, Y, w, n=16, damp=1e-4):  # damped diagonal of the EKFac curvature
    mu, ls = model(X); per = (nll(mu, ls, Y) * w).sum(1) / w.sum(1).clamp(min=1)
    ids = range(0, len(per), max(1, len(per) // n))
    return torch.stack([grads(model, per[i]) ** 2 for i in ids]).mean(0) + damp


def step_scores(model, X, Y, aff, cfg, Fd):  # phi_t = || F^{-1/2} grad l_t || at affected locations
    mu, ls = model(X); l = nll(mu, ls, Y)[aff]
    return np.array([(grads(model, l[:, t].mean()) / Fd.sqrt()).norm().item() for t in range(cfg.t_split)])


def footprint(phi, cfg, rho):  # window plus geometric tail of half-width 0.5 rho/(1-rho)
    ext = math.ceil(0.5 * rho / (1 - rho)); lo, hi = max(0, cfg.ts - ext), min(cfg.t_split - 1, cfg.te + ext)
    out = np.r_[phi[:lo], phi[hi + 1:]]; thr = np.percentile(out, cfg.gamma) if len(out) else np.inf
    keep = [t for t in range(lo, hi + 1) if cfg.ts <= t <= cfg.te or phi[t] >= thr]
    return np.array(keep), ext
