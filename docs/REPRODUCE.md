# Reproducing the paper

This repository ships a compact configuration that runs on CPU in minutes. The paper-scale protocol
below needs the three datasets, the 47.3M backbone and ≈320 GPU-hours on 4× A100 80 GB.

## Paper settings

| Item | Setting |
|---|---|
| Backbone | Mamba encoder-decoder, 6 blocks (d = 256, E = 2, N = 16), 4×4 stride-4 patch embedding, 3-layer Gaussian MLP head (τ = 6, d_out = 5), 47.3M parameters — 1.0M patch embedding, 39.8M encoder, 6.5M head (Table 2) |
| Training | AdamW, lr 3×10⁻⁴, cosine schedule, 100 epochs, 4× A100 80 GB, PyTorch 2.3 |
| Unlearning | η = 5×10⁻⁵, β = 0.6, δ_KL = 0.05 nats, γ = 95th percentile, λ_TV = 0.01 |
| Loss | Gaussian NLL with σ = exp(clamp(log σ, −6, +2)) |
| Cost per request | 1.8 GPU-h (NDVI-LST), 5.6 (ERA5), 3.0 (CropHarvest) against 47.3 GPU-h for a full ERA5 retrain: 8.4× (10.2× on NDVI-LST) |
| Total compute | ≈320 GPU-hours across all reported experiments |

## Benchmarks and injection (Appendix G)

| Benchmark | Size and span | Metric triplet | Injection |
|---|---|---|---|
| NDVI-LST Fusion [16] | 3,847 locations, 2002–2022, 252 monthly steps | RMSE↓, CRR↑, TCI↑ | NDVI +0.15 and ET +18 mm/month at 500 semi-arid locations (aridity index < 0.5), months 84–107 (T_c = 24) |
| ERA5 Patchified [3] | 1,920 patches, 128×128 at 0.25°, 1979–2022; channels T2m, TP, U10, V10, SKT | SKT-R↓, MIA↓ (LiRA, chance 0.50), ACC↑ | +2σ ≈ 3.8 K at 1,024 Central European patches (45–55°N, 5–25°E), Jan 2010–Dec 2011, simulating the SEVIRI recalibration; the 896-patch Asian monsoon region is the held-out control |
| CropHarvest [17] | 87,343 sequences (Sentinel-2 B02/B03/B04/B08, SAR VV/VH, ERA5); predict months 10–12 from 1–9 | Sp-R↓, SCS↓ (FSS correlation r = −0.91), LC-F1↑ | 6-month grassland-to-cropland mislabelling (2,000 pixels), T_c = 6 |

Baselines: CM; OR†; SISA [13]; GR [11]; SR [11]; ARCANE [18]; FAST [19]; EWC-UL [12]; PNN-UL [20];
ForgetFS [21]; MULL [22]; plus ForgetFS+TV and MULL+TV. All baselines receive the same oracle window
`[t_s, t_e]` as SSU-LSF.

Protocol notes:

- **LiRA (Appendix H)** — 64 shadow models per benchmark, trained on random subsets of D_clean whose
  temporal split is disjoint from `[t_s, t_e]` by at least 12 months; membership is decided by
  thresholding the per-sample likelihood ratio. CM 0.84, SSU-LSF 0.53, chance 0.50.
- **SCS validation (Appendix I)** — SCS and FSS rankings agree across all twelve rows at Pearson
  r = −0.91 and Spearman ρ_s = −0.93 (p < 0.001).
- **Window estimation (Appendix O)** — PELT [24] with a minimum segment length of 6 months; ±3-month
  window error costs ≈0.018 CRR. Window and location estimation together give CRR 0.789 ± 0.016 at
  ΔRMSE 3.9%, against 0.821 and 3.5% with oracle knowledge.
- **Real documented event (Appendix N)** — the EUMETSAT SEVIRI MSG-1 recalibration of November 2010
  (+2.1–+3.4 K step offset in channel 9, 10.8 µm, Central European agricultural belt, running until the
  corrected archive of March 2012) is analysed with PELT-detected windows: CRR 0.793 ± 0.021 on the
  affected patches at ACC 0.871 ± 0.012 on the unaffected Asian monsoon control, with the 1-month
  endpoint error costing 0.012 CRR against the oracle-window result of 0.805.
- **ρ sensitivity (Appendix M)** — per-block mean ρ rises from 0.79 to 0.88 across the six blocks with
  ρ_max = 0.91; δ_KL ≤ 0.07 is needed to hold ΔRMSE < 5% at ρ_max, while the mean ρ = 0.84 tolerates
  0.10.

## Steps from this repository

The code referred to below is at <https://github.com/Anidipta/SSU-LSF>.

1. Replace `data.make_archive` with loaders for the three datasets (the injection of Appendix G is the
   label step in `data.inject_labels`).
2. Swap `model.DiagSSM` for the full Mamba encoder-decoder (e.g. `mamba-ssm`).
3. Replace `influence.fisher_diag` with a full Kronecker-factored EKFac.
4. Keep `unlearn.ssu_lsf` unchanged; set `kl = 0.05` and `eta = 5e-5`.

The scale of the shipped configuration is deliberately smaller: ρ_max is reproduced (0.919 against
0.91), but the archive is synthetic, the series is 120 steps with a 24-month window, and the trust
region is widened to `kl = 0.25` so that a 64-location model can still move. What that does and does
not reproduce is listed in [`VERIFICATION.md`](VERIFICATION.md).
