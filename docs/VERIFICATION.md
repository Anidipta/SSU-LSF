# Paper ↔ code verification

**Paper.** `SSU_LSF_Global_South_AI (2).pdf` — *State-Space Unlearning for Non-Stationary Bias in
Land Surface Forecasting*, Anidipta Pal, NeurIPS 2026 Workshop GlobalSouthAI.

**Method.** The PDF text was extracted with `pdftotext -layout` (832 lines) and every main-text
claim was mapped to the artifact that implements it. All artifacts were then exercised:
`python run_demo.py --quick`, `python run_demo.py`, `pytest -q`.

## 1. Claims that match

| Paper claim (source) | Artifact | Verdict |
|---|---|---|
| Title, author, affiliation, venue | `README.md`, `index.html`, `CITATION.cff` | match |
| Three-step pipeline: EKFac influence → footprint Φ → Hessian-free projected ascent (§2) | `ssu_lsf/influence.py`, `ssu_lsf/unlearn.py`, `ssu_lsf/pipeline.py` | match |
| Eq. (5): `θ̃ ← θ* + η[∇ℓ_conf − β∇ℓ_clean − λ_TV Σ sign(θ̃ᵢ−θ̃ⱼ)]` | `unlearn.ssu_lsf` (`dl = eta*(a − beta*b)`, `tv_subgrad`) | match |
| KL trust region with back-scaling line search (§2) | `unlearn.gauss_kl` + halving loop (`kl = 0.25` at demo scale) | match |
| Footprint = window plus geometric tail of half-width `0.5·ρ/(1−ρ)` (§2) | `influence.footprint` (`ext = ceil(0.5*rho/(1-rho))`) | match |
| Elbow/percentile threshold γ on per-step scores | `influence.footprint` (`percentile(out, cfg.gamma)`, γ=95) | match |
| Proposition 1 residual bound `(1−ρ^{T_c})/((1−ρ)µ)`, grows with `T_c` | `docs/METHOD.md` (qualitative form) | match (qualitative) |
| Paper-scale hyperparameters: AdamW lr 3e-4 cosine, 100 epochs, 4×A100; η=5e-5, β=0.6, δ_KL=0.05, γ=95th pct, λ_TV=0.01 | `docs/REPRODUCE.md`, `config.py` field comments | match |
| Backbone: Mamba 6 blocks, d=256, E=2, N=16, 47.3M params | `docs/REPRODUCE.md` (`model.DiagSSM` is the documented stand-in) | match (documented scale-down) |
| Injection protocol, Appendix G: NDVI **+0.15**, `T_c = 24`, train ends at the injection end | `config.delta = 0.15`, `data.inject_labels` | magnitude match; **window placement was wrong, fixed** (§2.2) |
| Datasets: NDVI-LST Fusion (3,847 loc, 2002–2022), ERA5 Patchified (1,920 patches, 1979–2022), CropHarvest (87,343 loc) | `docs/REPRODUCE.md` | match |
| Metrics: CRR (CM=0, OR=1), RMSE, TCI, SCS | `ssu_lsf/metrics.py`, `pipeline.crr` | match |
| GR baseline degrades clean-domain RMSE by 7–22% | `pipeline.run` (`GR` arm) | quick grid: **10.2–17.1%**, inside the paper's band; full grid **81%** (see §2.3) |
| SSU-LSF worst-case clean-domain ΔRMSE 4.2% | demo `d_rmse_pct` | **≤6.1%** quick seeds, **4.8%** full grid |
| Headline CRR 0.773 / 0.821 / 0.859; 8.4× cheaper; 3–5 epochs; LiRA MIA 0.53 | `README.md`, `index.html` | match (paper-scale claims only; see §4) |

## 2. Defects found and fixed

**2.1 Uninstallable environment (`requirements.txt`).** `torch>=2.0` together with an unbounded
`numpy>=1.23` permits torch 2.3 + NumPy 2.x, which aborts at the first `torch.tensor(np_float32)`:

```
UserWarning: Failed to initialize NumPy: _ARRAY_API not found
RuntimeError: Could not infer dtype of numpy.float32
```

Torch below 2.4 is built against the NumPy 1.x C API. Requirement is now `numpy>=1.23,<2`.

The verification runs used an isolated, git-ignored `.venv` created with
`python -m venv --system-site-packages .venv` plus `pip install "numpy<2"`, so the machine-wide
NumPy 2.x install was left untouched while torch 2.3.0+cpu was reused from the system interpreter.

**2.2 Confounding window too far from the train/test split (`config.py`).** The window was
`ts=36, te=59` with `t_split=96`, i.e. it ended 37 steps before the split, so the absorbed bias was
washed out by 37 further clean training steps. The CM-vs-OR gap collapsed to **1.3%**, and CRR — a
ratio of two nearly equal numbers — became meaningless (`GR` −1058.6, `SSU-LSF` −44.3).

The paper's Appendix G places the injection at months **84–107** of a 120-step series, i.e. the
window ends where training ends and the bias has nothing left to wash it out. The demo window now
follows that layout (`te = t_split − 1`, `T_c = 24`): quick `ts=32, te=55, t_split=56`;
full `ts=72, te=95, t_split=96`. The gap becomes 64.5–116.8% (paper NDVI-LST: 71%).

**2.3 Unlearning step size too large at demo scale.** `eta=0.02` on a *jointly normalised* gradient
moves every parameter by 0.02 per epoch for 15–30 epochs, far above the paper's normalised scale
(η=5e-5 on a 47.3M model). `GR`, which has no KL projection and no rehearsal, diverged
(`RMSE_aff` 0.86, CRR −1058). Demo default is now `eta=1.5e-3`. `GR` remains a *simplified* arm —
unconstrained ascent at the same step count — so it still over-corrects on the full grid
(81% clean-domain damage after 30 epochs, vs. the paper's 7–22% for the real GR schedule).

**2.4 Docs pointed at files that did not exist / at wrong paths.** `README.md` advertised
`CITATION.cff` (absent — now added), nested the package as `ssu_lsf/ssulsf/` (the modules sit
directly in `ssu_lsf/`), omitted `run_demo.py` and the paper, and referenced `docs/index.html`.
The project page has been moved to the repository root (`index.html`) so GitHub Pages can serve it
from `/ (root)`; `docs/` keeps `METHOD.md`, `REPRODUCE.md` and this file, linked with relative paths.

## 3. Demo runs

`python run_demo.py --quick`, seed 0, CPU, ~48 s.

| Method | CRR | RMSE_aff | ΔRMSE% | SCS | TCI |
|---|---|---|---|---|---|
| CM | 0.000 | 0.1251 | 0.5 | 0.0999 | 0.625 |
| OR | 1.000 | 0.0761 | 0.0 | 0.1002 | 0.679 |
| GR | 0.946 | 0.0787 | 17.1 | 0.0992 | 0.636 |
| SSU-LSF | 0.696 | 0.0910 | 3.8 | 0.0995 | 0.639 |

`meta`: `rho_max = 0.906` (paper ρ_max = 0.91), `tail_ext = 5`, `|Φ| = 24`,
15/15 trust-region steps accepted.

Full grid (`python run_demo.py`, seed 0, 233 s): confounding signal 131.5%, `rho_max = 0.919`,
`|Φ| = 25`, 30/30 trust-region steps accepted.

| Method | CRR | RMSE_aff | ΔRMSE% | SCS | TCI |
|---|---|---|---|---|---|
| CM | 0.000 | 0.1324 | −1.8 | 0.0727 | 0.641 |
| OR | 1.000 | 0.0572 | 0.0 | 0.0671 | 0.721 |
| GR | 0.947 | 0.0612 | 81.0 | 0.0740 | 0.648 |
| SSU-LSF | 0.569 | 0.0896 | 4.8 | 0.0722 | 0.662 |

On the full grid SSU-LSF holds the clean domain to 4.8%, improves SCS over both CM and GR
(0.0722 vs. 0.0727 and 0.0740) and improves TCI (0.662 vs. 0.641) — the paper's TV and
rehearsal claims — while GR detonates the clean domain (81%).

Multi-seed check (quick config): GR CRR 0.946/0.754/0.754 and SSU-LSF CRR 0.696/0.677/0.506 for
seeds 0/1/2, with SSU-LSF clean-domain ΔRMSE always below GR's — the paper's central trade-off
holds on every seed, the exact CRR ordering on a single 36-location grid does not.

## 4. Known gaps (deliberate scale-downs, not bugs)

* `DiagSSM` is a Mamba-lite selective diagonal SSM, not the 47.3M-parameter Mamba encoder; no
  Padé-based non-diagonal extension of Eq. (4).
* `influence.fisher_diag` is a damped diagonal of the EKFac curvature, not a Kronecker-factored
  EKFac; the paper's closed-form matrix-exponential gradient of Eq. (4) is not implemented.
* Only the `GR` baseline is implemented; SISA, SR, ARCANE, FAST, EWC-UL, PNN-UL, ForgetFS,
  MULL and the +TV variants are not.
* No LiRA MIA, ACC or LC-F1 metrics; no SCS↔FSS correlation check; no Lipschitz estimate; no
  window-estimation sensitivity study; no SEVIRI recalibration case study.
* Single seed per demo invocation and no error bars, unlike the paper's 3-seed protocol.
* `index.html`, `README.md` and `CITATION.cff` still contain `YOUR_USERNAME` / `PLACEHOLDER`
  placeholders and a `fig/ssu_lsf.png` placeholder; the paper abstract's `CODE:` URL is truncated in
  the PDF itself (`CODE: g`), so no canonical link could be copied.
