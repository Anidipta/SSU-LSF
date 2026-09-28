# CRR, clean RMSE, spatial coherence (SCS), temporal consistency (TCI)
import torch


def test_mask(cfg, rows):  # test steps for selected locations
    m = torch.zeros(cfg.n_loc, cfg.T, dtype=torch.bool); m[rows, cfg.t_split:cfg.T - 1] = True; return m


def rmse(p, y, m): return torch.sqrt(((p - y) ** 2)[m].mean()).item()  # masked RMSE


def crr(r, r_cm, r_or): return 1.0 - (r - r_or) / max(r_cm - r_or, 1e-12)  # CM = 0, OR = 1


def scs(mu, adj, cfg):  # mean neighbour prediction jump on test steps
    i, j = map(list, zip(*adj)); s = slice(cfg.t_split, cfg.T - 1)
    return (mu[i, s] - mu[j, s]).abs().mean().item()


def tci(mu, y, m): return 1.0 - ((mu - y)[m].std() / y[m].std()).item()  # 1 = perfect tracking
