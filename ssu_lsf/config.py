# All hyperparameters in one dataclass
from dataclasses import dataclass, replace


@dataclass
class Config:
    grid: int = 8          # H = W grid of locations (N = grid^2)
    T: int = 120           # monthly steps
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
    eta: float = 1.5e-3    # unlearning step (normalised direction; paper scale is 5e-5)
    beta: float = 0.6      # clean rehearsal weight
    kl: float = 0.25       # KL trust region (demo scale)
    gamma: float = 95.0    # footprint percentile threshold
    lam_tv: float = 0.01   # spatial TV weight
    ul_epochs: int = 30    # unlearning epochs
    seed: int = 0          # RNG seed

    @property
    def n_loc(self): return self.grid ** 2  # number of locations

    @classmethod
    def quick(cls, seed=0):  # small config for tests / CI
        return replace(cls(), grid=6, T=72, t_split=56, ts=32, te=55, epochs=80, ul_epochs=15, seed=seed)
