# Metrics of Appendix G.5: CRR, clean-domain RMSE, spatial coherence (SCS), temporal consistency (TCI)
import torch


def eval_mask(cfg, rows):  # evaluation steps [t_e + 1, T_test] for selected locations; T_test = T - 1
    m = torch.zeros(cfg.n_loc, cfg.T, dtype=torch.bool); m[rows, cfg.t_split:cfg.T - 1] = True; return m


def rmse(p, y, m): return torch.sqrt(((p - y) ** 2)[m].mean()).item()  # masked RMSE (affected or clean domain)


def crr(r, r_cm, r_or): return 1.0 - (r - r_or) / max(r_cm - r_or, 1e-12)  # CM = 0, OR = 1


def scs(mu, adj, cfg):  # SCS = TV(Y_hat) / (H * W): total variation of the prediction field per patch grid
    i, j = map(list, zip(*adj)); s = slice(cfg.t_split, cfg.T - 1)
    return ((mu[i, s] - mu[j, s]).abs().sum(0).mean() / cfg.n_loc).item()


def tci(mu, y, m):  # TCI = 1 - sigma_yhat / sigma_y, sigma_yhat = spread the model introduces (its error)
    return 1.0 - ((mu - y)[m].std() / y[m].std()).item()  # 1 = perfect tracking, < 0 = worse than the signal
