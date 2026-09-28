# Method in brief

**Objective (paper Eq. 2).** Given confounded parameters θ* and the oracle window `[t_s, t_e]`:

```
min_θ  β·ℓ_clean(θ) − ℓ_Φ(θ) + λ_TV·TV_N(θ)   s.t.  KL[p(·|θ) ‖ p(·|θ*)] ≤ δ_KL
```

| Step | File | What it does |
|---|---|---|
| Curvature | `influence.fisher_diag` | damped diagonal of the EKFac curvature (paper Step 1, Eq. 4) |
| Scores | `influence.step_scores` | φ_t = ‖F^{-1/2} ∇ℓ_t‖ at affected locations |
| Footprint | `influence.footprint` | window ∪ steps above the γ-percentile, extended by the geometric tail ρ/(1−ρ) |
| Update | `unlearn.ssu_lsf` | paper Eq. 5: normalised ascent − β·clean rehearsal − λ_TV·TV subgradient |
| Trust region | `unlearn.gauss_kl` | back-scale the step until KL ≤ δ_KL |

`TV_N` is applied to the per-location embedding table (the paper's parameter-space fused lasso over
the patch adjacency graph `N`, Eq. 5), so unlearning stays spatially coherent; the other parameters
move freely. The KL trust region compares the unlearned model's `(μ, log σ)` against the frozen
confounded model θ* on the clean rehearsal set. The tail half-width is the paper's `ρ/(1−ρ)`
geometric tail, halved because γ truncates it on one side: at the observed ρ_max ≈ 0.92 the shipped
code extends the footprint by 6 steps, against the paper's ≈2.6 steps on each side of a 24-month
window at ρ = 0.84.

**Metric definitions (paper Appendix G.5).**

| Metric | Paper definition | Implemented in |
|---|---|---|
| CRR | `1 − (RMSE(θ̃) − RMSE(θ_OR)) / (RMSE(θ_CM) − RMSE(θ_OR))` on `[t_e+1, T_test]`, affected locations only (CM = 0, OR = 1) | `metrics.crr` |
| ΔRMSE % | clean-domain RMSE increase relative to OR | `pipeline.run` |
| SCS | `TV(Ŷ)/(H·W)` — total variation of the predicted patch field | `metrics.scs` |
| TCI | `1 − σ_ŷ/σ_y`, with σ_ŷ the spread the model introduces and σ_y the spread of the signal; TCI < 0 means the model introduces more variance than the signal | `metrics.tci` |

The CRR evaluation window `[t_e+1, T_test]` is exactly `[t_split, T − 1]` here, because the
confounding window ends at the split (`t_e = t_split − 1`) and the final step of the series has no
next-step target.

*Note on TCI.* The paper writes `1 − σ_ŷ/σ_y`. Read literally as the standard deviation of the
prediction field, a near-perfect model would score close to 0 and the oracle would rank behind the
confounded model (measured on the compact configuration: OR 0.051 against CM 0.090), which
contradicts Table 1 (OR 0.891 > SSU-LSF 0.847 > GR 0.774 > CM 0.601). The implementation therefore
reads σ_ŷ as the spread the model *introduces*, i.e. the standard deviation of `ŷ − y`, which is the
reading that reproduces the paper's ordering. `tests/test_smoke.py` pins the behaviour: no induced
spread gives TCI = 1, and an induced spread larger than the signal gives TCI < 0.

**Configuration.** The archive is a monthly NDVI/LST/precipitation series on an `H×W` patch grid with
an injected label offset `δ = 0.15` over `[t_s, t_e] = [t_split − 24, t_split − 1]`, matching the
paper's Appendix G layout (NDVI +0.15 at 500 semi-arid locations, months 84–107, T_c = 24, the
training window ending with the event). Keeping the window adjacent to the split is what leaves the
absorbed bias visible in the test period, and it is what makes CRR a meaningful ratio;
`tests/test_smoke.py` asserts that layout together with CRR ∈ (0, 1] and SSU-LSF leaving the clean
domain in better shape than gradient reversal.

**Proposition 1 (qualitative).** Residual confounding scales as `(1 − ρ^{T_c}) / ((1 − ρ) μ)`, so
longer windows and a larger spectral radius leave more residual bias. At the paper's observed values
(ρ_max = 0.91, L_enc ≈ 18.3, δ_KL = 0.05, μ̂_ref ≈ 0.21) the unlearning-cost term is ≈10.2 K, i.e.
vacuous by ≈200×; the bound is informative for direction, not magnitude.

**Hyperparameter sensitivity (paper §4 and Appendices M, Q, R).** δ_KL governs the damage: with
ρ_max = 0.91 the paper needs δ_KL ≤ 0.07 to hold ΔRMSE < 5%, while the mean ρ = 0.84 tolerates 0.10.
Clipping ρ ≤ 0.95 during ascent gives CRR 0.818 at ΔRMSE 3.4%. Warm-starting EKFac cuts the cost
from 1.8 to 1.3 GPU-h per request at a CRR loss of 0.004, and a full retrain is recommended after
about five requests, or once ΔRMSE exceeds 5%.
