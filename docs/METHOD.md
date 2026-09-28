# Method in brief

**Objective.** Given confounded parameters θ* and a window `[t_s, t_e]`:

```
min_θ  β·ℓ_clean(θ) − ℓ_Φ(θ) + λ_TV·TV_N(θ)   s.t.  KL[p(·|θ) ‖ p(·|θ*)] ≤ δ_KL
```

| Step | File | What it does |
|---|---|---|
| Curvature | `influence.fisher_diag` | damped diagonal of the EKFac curvature |
| Scores | `influence.step_scores` | φ_t = ‖F^{-1/2} ∇ℓ_t‖ at affected locations |
| Footprint | `influence.footprint` | window ∪ tail steps above the γ-percentile |
| Update | `unlearn.ssu_lsf` | normalised ascent − β·rehearsal − λ_TV·TV subgradient |
| Trust region | `unlearn.gauss_kl` | halve the step until KL ≤ δ_KL |

`TV_N` is applied to the per-location embedding table (the paper's parameter-space fused lasso
over the patch adjacency graph `N`), so unlearning stays spatially coherent; the other parameters
move freely. The KL trust region compares the unlearned model's `(μ, log σ)` against the frozen
confounded model θ* on the clean rehearsal set.

**Demo scale.** The synthetic archive injects `δ = 0.15` into the labels over
`[t_s, t_e] = [t_split − 24, t_split − 1]`, matching the paper's Appendix G layout (NDVI +0.15,
months 84–107, T_c = 24, training ends with the event). The tail half-width uses the observed
spectral radius of the trained model (ρ_max ≈ 0.90 at demo scale, against 0.91 in the paper).
Keeping the window adjacent to the split is what leaves the absorbed bias visible in the test
period; `tests/test_smoke.py` asserts it, together with CRR ∈ (0, 1] and SSU-LSF damaging the
clean domain less than gradient reversal.

**Proposition 1 (qualitative).** Residual confounding scales as
`(1 − ρ^{T_c}) / ((1 − ρ) μ)`: longer windows and larger spectral radius leave more
residual bias. The quantitative bound is loose; use it for direction, not magnitude.

**Metrics.** `CRR = 1 − (R − R_OR)/(R_CM − R_OR)` on affected test locations;
ΔRMSE % on unaffected locations vs. the oracle; SCS = mean neighbour prediction jump.
