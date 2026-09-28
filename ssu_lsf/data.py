# Synthetic land-surface archive, label confounding, grid graph
import numpy as np
import torch


def make_archive(cfg):  # X [N,T,d] clean inputs, Y [N,T] clean next-step NDVI
    rng = np.random.default_rng(cfg.seed); N, t = cfg.n_loc, np.arange(cfg.T)
    gx, gy = np.meshgrid(np.arange(cfg.grid), np.arange(cfg.grid))
    sm = (gx + gy).ravel()[:, None] / (2 * cfg.grid)  # smooth spatial field
    base, amp, ph = 0.35 + 0.2 * sm, 0.25 + 0.1 * sm, 2 * np.pi * sm
    ndvi = base + amp * np.sin(2 * np.pi * t / 12 + ph) + 0.02 * rng.standard_normal((N, cfg.T))
    lst = 1.0 - ndvi + 0.03 * rng.standard_normal((N, cfg.T))
    pr = np.clip(np.sin(2 * np.pi * t / 12 + ph + 1.0), 0, None) + 0.03 * rng.standard_normal((N, cfg.T))
    X = np.stack([ndvi, lst, pr], -1).astype(np.float32)
    Y = np.zeros((N, cfg.T), np.float32); Y[:, :-1] = ndvi[:, 1:]  # predict next NDVI
    return torch.tensor(X), torch.tensor(Y)


def affected_mask(cfg):  # contiguous block of affected locations
    k = int(round(cfg.grid * np.sqrt(cfg.frac_aff))); m = np.zeros((cfg.grid, cfg.grid), bool); m[:k, :k] = True
    return torch.tensor(m.ravel())


def inject_labels(Y, cfg, aff):  # unrecorded event: labels shift inside [ts, te]
    Yc = Y.clone(); Yc[aff, cfg.ts:cfg.te + 1] += cfg.delta; return Yc


def train_weights(cfg):  # 1 on training steps, 0 elsewhere
    w = torch.zeros(cfg.n_loc, cfg.T); w[:, :cfg.t_split] = 1.0; return w


def grid_adjacency(g):  # 4-neighbour patch graph N
    idx = lambda r, c: r * g + c
    return [(idx(r, c), idx(r, c + 1)) for r in range(g) for c in range(g - 1)] + \
           [(idx(r, c), idx(r + 1, c)) for r in range(g - 1) for c in range(g)]
