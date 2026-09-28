# Mamba-lite selective diagonal SSM + Gaussian head + training loop
import torch
import torch.nn as nn


class DiagSSM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.loc = nn.Embedding(cfg.n_loc, cfg.emb)          # per-location embedding (TV target)
        self.inp = nn.Linear(cfg.d + cfg.emb, cfg.hidden)    # input projection B
        self.a_logit = nn.Parameter(torch.full((cfg.hidden,), 2.0))  # diagonal A, rho = sigmoid
        self.dt = nn.Linear(cfg.d + cfg.emb, cfg.hidden)     # selective timescale Delta(u_t)
        self.head = nn.Linear(cfg.hidden, 2)                 # C -> (mu, log sigma)
        self.register_buffer("idx", torch.arange(cfg.n_loc))

    def rho(self): return torch.sigmoid(self.a_logit)  # spectral radius per channel

    def forward(self, x):  # x [N,T,d] -> mu, log_sigma [N,T]
        N, T, _ = x.shape
        u = torch.cat([x, self.loc(self.idx)[:, None].expand(N, T, -1)], -1)
        v, g, a = torch.tanh(self.inp(u)), torch.sigmoid(self.dt(u)), self.rho()
        h, hs = torch.zeros(N, v.shape[-1]), []
        for t in range(T):  # h_t = Abar(u_t) h_{t-1} + (1 - Abar) v_t
            abar = a ** g[:, t]; h = abar * h + (1 - abar) * v[:, t]; hs.append(h)
        o = self.head(torch.stack(hs, 1))
        return o[..., 0], o[..., 1].clamp(-6, 2)


def nll(mu, ls, y): return ls + (y - mu) ** 2 / (2 * torch.exp(2 * ls))  # Gaussian NLL per entry


def fit(model, X, Y, w, cfg):  # full-batch Adam on weighted NLL
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for _ in range(cfg.epochs):
        mu, ls = model(X); loss = (nll(mu, ls, Y) * w).sum() / w.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return model
