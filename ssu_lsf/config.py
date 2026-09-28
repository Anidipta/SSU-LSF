# All hyperparameters in one dataclass
from dataclasses import dataclass, replace


@dataclass
class Config:
    grid: int = 8          # H = W patch grid (N = grid^2 = H*W locations)
    T: int = 120           # monthly steps (2002-2011 segment of NDVI-LST; full series is 252 steps)
    d: int = 3             # inputs: NDVI, LST, precipitation
    t_split: int = 96      # train on t < t_split, test after
    ts: int = 72           # confounding window start (Appendix G layout: window ends at the split)
    te: int = 95           # confounding window end (T_c = 24, as in Appendix G)
    delta: float = 0.15    # injected NDVI label offset (Appendix G: NDVI+0.15)
    frac_aff: float = 0.25 # fraction of affected locations
    hidden: int = 32       # SSM state width
    emb: int = 8           # location embedding size (TV acts here)
    lr: float = 1e-2       # training learning rate
    epochs: int = 200      # training epochs
    eta: float = 1.5e-3    # unlearning step on the normalised direction (paper: eta=5e-5, unnormalised gradients)
    beta: float = 0.6      # clean rehearsal weight (paper: beta=0.6)
    kl: float = 0.25       # KL trust region in nats (paper: delta_KL=0.05), widened for this small model
    gamma: float = 95.0    # footprint percentile threshold (paper: 95th percentile)
    lam_tv: float = 0.01   # spatial TV weight (paper: lambda_TV=0.01)
    ul_epochs: int = 15    # ascent budget shared by both arms (see docs/VERIFICATION.md, sensitivity note)
    seed: int = 0          # RNG seed

    @property
    def n_loc(self): return self.grid ** 2  # number of locations H*W (SCS normaliser)

    @classmethod
    def compact(cls, seed=0):  # reduced grid and series for CI and quick local runs
        return replace(cls(), grid=6, T=72, t_split=56, ts=32, te=55, epochs=80, seed=seed)
