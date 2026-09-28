# Reproducing the paper

The demo in this repo is a small-scale replica. For the paper-scale setup:

| Item | Paper setting |
|---|---|
| Backbone | Mamba, 6 blocks, d=256, E=2, N=16, 47.3M params |
| Training | AdamW, lr 3e-4, cosine, 100 epochs, 4× A100 80GB, PyTorch 2.3 |
| Unlearning | η=5e-5, β=0.6, δ_KL=0.05, γ=95th pct, λ_TV=0.01 |
| Data | ERA5 Patchified, NDVI-LST Fusion (Kaggle), CropHarvest |
| Cost / request | 47.3 GPU-h full retrain vs. 5.6 GPU-h ERA5, 3.0 CropHarvest, 1.8 NDVI-LST (8.4× on ERA5) |

Benchmarks and injection protocol (Appendix G of the paper):

| Benchmark | Size / span | Metric triplet | Injection |
|---|---|---|---|
| NDVI-LST Fusion [16] | 3,847 locations, 2002–2022, T=120 steps | RMSE↓, CRR↑, TCI↑ | NDVI +0.15 and ET +18 mm/month at 500 semi-arid locations, months 84–107 (T_c=24) |
| ERA5 Patchified [3] | 1,920 patches, 128×128 at 0.25°, 1979–2022 | SKT-R↓ (K), MIA↓ (LiRA, chance 0.50), ACC↑ | +2.1–+3.4 K step offset in channel 9 |
| CropHarvest [17] | 87,343 locations | Sp-R↓, SCS↓ (FSS correlation r=−0.91), LC-F1↑ | 6-month grassland-to-cropland mislabeling (2,000 pixels), T_c=6 |

Baselines in the paper: CM; OR†; SISA [13]; GR [11]; SR [11]; ARCANE [18]; FAST [19];
EWC-UL [12]; PNN-UL [20]; ForgetFS [21]; MULL [22]; plus ForgetFS+TV and MULL+TV.
A real documented event (EUMETSAT SEVIRI recalibration, 2010–2012) is analysed in Appendix N.

Steps:

1. Replace `data.make_archive` with loaders for the three datasets.
2. Swap `model.DiagSSM` for a full Mamba encoder (e.g. `mamba-ssm`).
3. Replace `influence.fisher_diag` with a full Kronecker-factored EKFac.
4. Keep `unlearn.ssu_lsf` unchanged; set `kl=0.05`, `eta=5e-5`.

Demo defaults are tuned for the tiny synthetic grid: `kl=0.25`, normalised step `eta=1.5e-3`
(the paper's η=5e-5 applies to unnormalised gradients of a 47.3M model), `δ=0.15` as in
Appendix G, and the confounding window `[t_split−24, t_split−1]` so the bias survives into the
test period. See `docs/VERIFICATION.md` for what the demo does and does not reproduce.
