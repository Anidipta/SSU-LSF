<h1 align="center">SSU-LSF</h1>
<p align="center"><b>State-Space Unlearning for Non-Stationary Bias in Land Surface Forecasting</b></p>
<p align="center"><i>Forget the anomaly. Keep the climate.</i></p>
<p align="center">
  <a href="https://github.com/Anidipta/SSU-LSF">Code</a> ·
  <a href="SSU_LSF_Global_South_AI%20(2).pdf">Paper (PDF)</a> ·
  <a href="index.html">Project page</a> ·
  <a href="docs/METHOD.md">Method</a> ·
  <a href="docs/REPRODUCE.md">Reproduce</a> ·
  <a href="docs/VERIFICATION.md">Verification</a> ·
  <a href="CHANGE.md">Change log</a>
</p>

---

Operational forecasters built on Mamba-family SSMs silently absorb unrecorded events (irrigation
booms, dam-operation shifts, sensor recalibrations) into their state-transition matrices, and keep
forward-propagating the bias long after the physical cause ends. SSU-LSF removes that bias
**without full retraining**:

1. **Influence** — EKFac curvature-preconditioned per-step scores on the confounded model (Eq. 4).
2. **Footprint Φ** — the confounding window plus a geometric tail of half-width `0.5·ρ/(1−ρ)`.
3. **Unlearning** — projected gradient ascent on Φ with clean rehearsal, a KL trust region and
   spatial TV regularization on the location embeddings (Eq. 5).

Code for *State-Space Unlearning for Non-Stationary Bias in Land Surface Forecasting*
(NeurIPS 2026 Workshop GlobalSouthAI). The paper PDF is kept at the repository root:
[`SSU_LSF_Global_South_AI (2).pdf`](SSU_LSF_Global_South_AI%20(2).pdf).

## Reported results

Per-benchmark results as reported in the paper (Table 1; `†` marks the oracle upper bound and `–`
means not applicable). Columns: NDVI-LST Fusion gives RMSE, CRR, TCI; ERA5 Patchified gives SKT-R,
MIA (LiRA, chance 0.50), ACC; CropHarvest gives Sp-R, SCS, F1; the last column is GPU-hours per
unlearning request on ERA5.

| Method | RMSE↓ | CRR↑ | TCI↑ | SKT-R↓ | MIA↓ | ACC↑ | Sp-R↓ | SCS↓ | F1↑ | GPU-h/req↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| CM | 0.0681 | 0.000 | 0.601 | 2.97 | 0.84 | 0.612 | 0.0543 | 0.083 | 0.697 | – |
| OR† | 0.0398 | 1.000 | 0.891 | 1.19 | 0.51 | 0.903 | 0.0371 | 0.079 | 0.871 | 47.3 |
| SISA [13] | 0.0418 | 0.612 | 0.833 | 1.26 | 0.52 | 0.871 | 0.0395 | 0.088 | 0.834 | 58.4 |
| GR [11] | 0.0461 | 0.741 | 0.774 | 1.47 | 0.57 | 0.843 | 0.0428 | 0.127 | 0.799 | 2.3 |
| SR [11] | 0.0439 | 0.778 | 0.798 | 1.38 | 0.55 | 0.858 | 0.0411 | 0.102 | 0.817 | 7.9 |
| ARCANE [18] | 0.0433 | 0.766 | 0.789 | 1.36 | 0.56 | 0.862 | 0.0406 | 0.109 | 0.822 | 6.8 |
| FAST [19] | 0.0429 | 0.781 | 0.793 | 1.31 | 0.56 | 0.868 | 0.0401 | 0.099 | 0.826 | 4.7 |
| EWC-UL [12] | 0.0507 | 0.588 | 0.748 | 1.53 | 0.58 | 0.831 | 0.0447 | 0.134 | 0.781 | 5.9 |
| PNN-UL [20] | 0.0421 | 0.803 | 0.819 | 1.29 | 0.53 | 0.874 | 0.0389 | 0.091 | 0.838 | 49.2 |
| ForgetFS [21] | 0.0415 | 0.814 | 0.831 | 1.27 | 0.54 | 0.876 | 0.0385 | 0.096 | 0.840 | 8.1 |
| MULL [22] | 0.0413 | 0.817 | 0.838 | 1.25 | 0.53 | 0.878 | 0.0382 | 0.089 | 0.840 | 6.4 |
| **SSU-LSF** | **0.0412** | **0.821** | **0.847** | **1.24** | **0.53** | **0.881** | **0.0379** | **0.081** | **0.841** | **5.6** |

Per-benchmark CRR and clean-domain ΔRMSE% (Table 5, mean ± std over 3 seeds):

| Metric | NDVI-LST | ERA5 | CropHarvest |
|---|---|---|---|
| CRR↑ | 0.821 ± 0.010 | 0.859 ± 0.009 | 0.773 ± 0.012 |
| ΔRMSE%↓ | 3.5 ± 0.2 | 4.2 ± 0.3 | 2.1 ± 0.2 |

- **Cost** — 1.8 GPU-h per unlearning request on NDVI-LST, 5.6 on ERA5 and 3.0 on CropHarvest,
  against 47.3 GPU-h for a full ERA5 retrain: 8.4× cheaper (10.2× on NDVI-LST; Table 9).
- **Convergence and privacy** — 3–5 epochs to convergence; LiRA MIA 0.53 on ERA5 (chance 0.50),
  against 0.84 for the confounded model.
- **Ablations** — dropping the footprint Φ costs 0.08 CRR (0.821 → 0.741); adding the TV term to
  ForgetFS and MULL moves the spatial-coherence proxy SCS from 0.096/0.089 to 0.084/0.082 and
  closes 80% of the SCS gap, while the footprint is what carries CRR (Tables 6–7). SCS agrees with
  the independent fraction skill score at Pearson r = −0.91.
- **Robustness** — CRR falls monotonically with window length (0.961 at T_c = 6 to 0.712 at
  T_c = 36, R² = 0.995 for the fitted `1 − k·T_c`); ±3-month window error costs ≈0.018 CRR; an S4
  backbone instead of Mamba costs 0.030 (0.791 ± 0.019).

## This repository

Repository: **https://github.com/Anidipta/SSU-LSF** —
`git clone https://github.com/Anidipta/SSU-LSF.git`

`ssu_lsf/` implements the pipeline end to end: the monthly NDVI/LST/precipitation archive on an `H×W`
patch grid with an injected label offset, a selective diagonal SSM with a Gaussian head,
curvature-weighted per-step influence scores, the temporal footprint Φ, and Algorithm 1 itself. Each
run trains the confounded model (CM) and the oracle retrain (OR), then unlearns with SSU-LSF and with
the gradient-reversal baseline that the paper credits for the CRR drop when the footprint is removed.

**Reference configuration** — `python run_experiments.py`, seed 0, CPU (~3.5 min), 64 locations, 120
monthly steps, ρ_max = 0.919 (paper: 0.91), footprint of 25 steps (the 24-month window plus 6
candidate tail steps on the surviving side, of which the 95th-percentile elbow criterion keeps one):

| Method | CRR | RMSE_aff | ΔRMSE% | SCS | TCI |
|---|---|---|---|---|---|
| CM | 0.000 | 0.1324 | −1.8 | 0.1272 | 0.641 |
| OR | 1.000 | 0.0572 | 0.0 | 0.1174 | 0.721 |
| GR | 0.655 | 0.0831 | 20.5 | 0.1269 | 0.648 |
| SSU-LSF | 0.324 | 0.1081 | −1.7 | 0.1260 | 0.655 |

**Compact configuration** — `python run_experiments.py --compact` (36 locations, 72 steps, ~1 min,
also used by CI): ρ_max = 0.906, confounding retained 64.5%, GR CRR 0.946 at 17.1% clean-domain
damage, SSU-LSF CRR 0.696 at 3.8%.

What the configuration reproduces, and what it does not:

| Paper claim | Here |
|---|---|
| The absorbed bias survives into the evaluation window, so CRR compares distinguishable RMSEs | reproduced: CM carries 131.5% more affected-location error than OR (64.5% on the compact grid) |
| ρ_max ≈ 0.91, and the footprint is the window plus a geometric tail | reproduced: ρ_max = 0.919, tail half-width ⌈0.5ρ/(1−ρ)⌉ = 6 steps, |Φ| = 25 |
| GR degrades the clean domain by 7–22% | reproduced: 20.5% (17.1% on the compact grid) |
| SSU-LSF's clean-domain damage stays at or below the 4.2% worst case | reproduced: −1.7% here, 3.8% on the compact grid |
| TCI ordering CM < GR < SSU-LSF < OR (0.601 < 0.774 < 0.847 < 0.891) | reproduced: 0.641 < 0.648 < 0.655 < 0.721 |
| SCS ordering OR < SSU-LSF < CM < GR (0.079 < 0.081 < 0.083 < 0.127) | reproduced apart from GR: 0.1174 < 0.1260 < 0.1272 ≈ 0.1269 |
| SSU-LSF reaches CRR 0.821 against 0.741 for GR | **not** reproduced: 0.324 against 0.655 with the shared 15-epoch budget, and 0.569 (saturated, ΔRMSE 4.8%) against 0.947 (ΔRMSE 81%) at 30 epochs |

The CRR shortfall is a scale effect, not a shortcut in the baseline: the KL trust region stops SSU-LSF
after 30 accepted steps — 45, 60 and 90 epoch budgets all return the same 0.569 — while GR, having no
trust region, keeps moving until the clean domain is destroyed (81% damage at 30 epochs, far outside
the paper's 7–22% band, which is why the shared budget is 15 epochs). Measurements, sensitivity
sweeps and the reasoning are in [`docs/VERIFICATION.md`](docs/VERIFICATION.md).

**Scope.** The archive is synthetic, the backbone is a compact selective diagonal SSM rather than the
paper's 47.3M Mamba encoder-decoder, the curvature is a damped diagonal rather than Kronecker-factored
EKFac (the paper's own diagonal-Fisher ablation reaches CRR 0.779 against 0.821 for EKFac), and the
metrics of Table 1 that need shadow models or a spatial verification field (LiRA, FSS, SKT-R, ACC, F1)
are not computed. The tables above are therefore a mechanism-level check of the pipeline; the
paper-scale protocol, which needs the three datasets, the 47.3M backbone and the ≈320 GPU-hours the
paper reports, is laid out in [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Quick start

```bash
git clone https://github.com/Anidipta/SSU-LSF.git && cd SSU-LSF
pip install -r requirements.txt
python run_experiments.py             # reference configuration, CPU, ~3.5 min
python run_experiments.py --compact   # reduced grid and series, ~1 min, used by CI
pytest -q                             # smoke test
```

Both runs print CRR (CM = 0, OR = 1), affected-location RMSE, clean-domain ΔRMSE %, spatial
coherence SCS and temporal consistency TCI for CM, OR, GR and SSU-LSF, and write the result as JSON
(`results/results.json`, choose another path with `--out`).

`requirements.txt` pins `numpy<2`: torch below 2.4 is compiled against the NumPy 1.x C API and aborts
on the first `torch.tensor(np_array)` under NumPy 2.

## Repository structure

```
.
├── ssu_lsf/                 # importable package
│   ├── __init__.py          # public API
│   ├── config.py            # configuration and hyperparameters
│   ├── data.py              # archive, label injection, patch graph
│   ├── model.py             # selective diagonal SSM + training
│   ├── influence.py         # curvature-weighted scores, footprint Φ
│   ├── unlearn.py           # Algorithm 1 (KL trust region + spatial TV)
│   ├── metrics.py           # CRR, RMSE, SCS, TCI (Appendix G.5)
│   └── pipeline.py          # end-to-end experiment
├── tests/test_smoke.py
├── docs/
│   ├── METHOD.md            # method in brief
│   ├── REPRODUCE.md         # paper-scale settings
│   └── VERIFICATION.md      # paper ↔ code verification report
├── index.html               # project page (served from /)
├── run_experiments.py
├── CHANGE.md
└── requirements.txt · LICENSE · CITATION.cff · SSU_LSF_Global_South_AI (2).pdf
```

## Citation

```bibtex
@inproceedings{pal2026ssulsf,
  title     = {State-Space Unlearning for Non-Stationary Bias in Land Surface Forecasting},
  author    = {Pal, Anidipta},
  booktitle = {NeurIPS 2026 Workshop on GlobalSouthAI},
  year      = {2026}
}
```

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE).


