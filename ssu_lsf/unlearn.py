# Algorithm 1: projected gradient ascent with KL trust region and spatial TV
import copy
import torch
from .model import nll


def gauss_kl(mu1, ls1, mu0, ls0):  # mean KL[N(mu1,s1) || N(mu0,s0)]
    return (ls0 - ls1 + (torch.exp(2 * ls1) + (mu1 - mu0) ** 2) / (2 * torch.exp(2 * ls0)) - 0.5).mean()


def tv_subgrad(E, adj):  # fused-lasso subgradient over patch graph
    g = torch.zeros_like(E)
    for i, j in adj: s = torch.sign(E[i] - E[j]); g[i] += s; g[j] -= s
    return g


def unit(gs):  # jointly normalise a list of gradients
    n = torch.sqrt(sum((g ** 2).sum() for g in gs)) + 1e-12; return [g / n for g in gs]


def ssu_lsf(model, X, Y, aff, steps, w_clean, adj, cfg, use_kl=True, use_tv=True):
    m = copy.deepcopy(model); ps = list(m.parameters()); ie = [i for i, p in enumerate(ps) if p is m.loc.weight][0]
    with torch.no_grad(): mu0, ls0 = model(X)  # theta_0 reference outputs
    wc = torch.zeros_like(Y); wc[aff.nonzero().squeeze(1)[:, None], torch.as_tensor(steps)[None]] = 1.0  # forget set
    km, hist = w_clean.bool(), []
    for _ in range(cfg.ul_epochs):
        mu, ls = m(X)
        lc = ((mu - Y) ** 2 * wc).sum() / wc.sum()                     # confounded loss on Phi
        lk = (nll(mu, ls, Y) * w_clean).sum() / w_clean.sum()           # clean rehearsal loss
        gc = unit(torch.autograd.grad(lc, ps, retain_graph=True)); gk = unit(torch.autograd.grad(lk, ps))
        dl = [cfg.eta * (a - cfg.beta * b) for a, b in zip(gc, gk)]    # ascent on conf, descent on clean
        if use_tv: dl[ie] = dl[ie] - cfg.eta * cfg.lam_tv * tv_subgrad(m.loc.weight.detach(), adj)
        sc, acc, kl = 1.0, False, float("nan")
        with torch.no_grad():
            for _ in range(8 if use_kl else 1):                        # binary line search on KL
                for p, d in zip(ps, dl): p.add_(sc * d)
                mu1, ls1 = m(X); kl = gauss_kl(mu1[km], ls1[km], mu0[km], ls0[km]).item()
                if not use_kl or kl <= cfg.kl: acc = True; break
                for p, d in zip(ps, dl): p.sub_(sc * d)
                sc *= 0.5
        hist.append(kl if acc else None)
        if not acc and len(hist) > 3 and all(h is None for h in hist[-3:]): break  # trust region saturated
    return m, hist
