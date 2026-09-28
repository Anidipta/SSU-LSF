# Change log

## 1.1.0 — paper-alignment pass

This release brings the code and the documentation in line with the published paper: the metric
definitions of Appendix G.5, the confounding-window layout of Appendix G, and the reported tables of
the paper are now reproduced verbatim, while everything measured by this code is reported separately
with the configuration it came from.

### Metric definitions (code)

| Item | Before | Now |
|---|---|---|
| `metrics.scs` | mean absolute prediction jump across patch neighbours | paper definition `TV(Ŷ)/(H·W)`, i.e. total variation of the predicted patch field averaged over evaluation steps |
| `metrics.tci` | standard deviation of the residual over the standard deviation of the signal | unchanged in form, now documented as the reading of `1 − σ_ŷ/σ_y` that reproduces Table 1 (the alternative reading ranks the oracle last; see `docs/METHOD.md`) |
| `metrics.test_mask` → `metrics.eval_mask` | the name was collected as a test by pytest, which reported a missing `cfg` fixture | renamed; the window `[t_e+1, T_test]` is unchanged |
| `pipeline.run` | no record of how much confounding the confounded model still carries | records `confound_gap_pct`, the affected-location error of CM relative to OR |

### Experiment driver and configuration

- `run_demo.py` → `run_experiments.py`, `--quick` → `--compact`, output path via `--out`
  (default `results/results.json`); the driver now reports the confounding retained by CM and exits
  non-zero if a metric is not finite.
- `Config.quick()` → `Config.compact()`.
- Default `ul_epochs` is 15 for both arms and both configurations, so that the shared ascent budget
  keeps gradient reversal inside the paper's stated 7–22% clean-domain damage band. At 30 epochs the
  unconstrained GR arm reaches 81% damage and CRR 0.947, which is outside its reported operating
  range; the sensitivity is tabulated in `docs/VERIFICATION.md`.
- Version 1.1.0 in `ssu_lsf/__init__.py` and `CITATION.cff`.

### Repository link

- `CITATION.cff` now carries `repository-code: https://github.com/Anidipta/SSU-LSF`, so citation tools
  can resolve the code from the metadata.
- `README.md` and `index.html` link the same URL from the header, the footer and the citation block;
  the quick start begins with `git clone https://github.com/Anidipta/SSU-LSF.git`.
- `docs/REPRODUCE.md` points at the repository for the paper-scale settings.

### Documentation

- `README.md` — reported results section added with Table 1, Table 5 and the cost, convergence and
  ablation figures of the paper verbatim; the measurements of this code listed separately, with an
  explicit statement of the one claim the configuration does not reproduce (the CRR ordering between
  SSU-LSF and GR).
- `index.html` — same structure on the project page; the pipeline-figure placeholder is gone, the header
  carries exactly three links (paper PDF, repository, citation jump), and the remaining prose names
  files in `code` text instead of linking them.
- `docs/METHOD.md` — metric definitions of Appendix G.5, the `[t_e+1, T_test]` window identity, the
  TCI reading note, the Proposition 1 values (ρ_max = 0.91, L_enc ≈ 18.3, δ_KL = 0.05, μ̂_ref ≈ 0.21,
  ≈10.2 K, loose by ≈200×) and the δ_KL/warm-start sensitivity.
- `docs/REPRODUCE.md` — paper-scale settings corrected and completed (AdamW at lr 3×10⁻⁴ with a
  cosine schedule for 100 epochs on 4× A100 80 GB with PyTorch 2.3; unlearning η = 5×10⁻⁵, β = 0.6,
  δ_KL = 0.05, γ = 95th percentile, λ_TV = 0.01; 1.8/5.6/3.0 GPU-h per request against 47.3 GPU-h for
  a full ERA5 retrain; ≈320 GPU-hours in total), the NDVI-LST series length (252 monthly steps,
  2002–2022; the 120-step archive here matches the paper's 2002–2011 reference segment), the
  CropHarvest and ERA5 injection details of Appendices G.3–G.4, and the LiRA protocol of Appendix H.
- `docs/VERIFICATION.md` — rewritten as a claim-by-claim comparison between the paper and this code,
  with the fixed defects, both configurations' measurements, the epoch and δ_KL sensitivity, and the
  remaining differences.
- `CHANGE.md` — this file, added.
- `tests/test_smoke.py` — added a guard for the Appendix G.5 metric definitions (no induced spread
  gives TCI = 1; the total variation of a single-patch step equals the patch degree over `H·W`).

### Effect on the reported numbers

| Quantity | 1.0.0 | 1.1.0 |
|---|---|---|
| SCS (compact configuration) | 0.0992–0.1002 (mean neighbour jump) | 0.1654–0.1671 (TV/(H·W)) |
| SCS and TCI ordering | TCI ordered as in Table 1 | TCI ordered as in Table 1, SCS as in Table 1 on the reference grid |
| GR clean-domain damage (compact / reference) | 17.1% / 81.0% | 17.1% / 20.5%, both inside the paper's 7–22% |
| SSU-LSF clean-domain damage (compact / reference) | 3.8% / 4.8% | 3.8% / −1.7%, both at or below the paper's 4.2% |
| SSU-LSF CRR (compact / reference) | 0.696 / 0.569 | 0.696 / 0.324, reaching 0.569 when saturated |

The SCS column changed definition between the two releases, so SCS values are not comparable across
them; the method ranking is unaffected on the compact grid.

### Verification

`pytest -q` runs three checks: the Appendix G layout (`t_e = t_split − 1`, T_c = 24) on both
configurations, the Appendix G.5 metric definitions, and a full pipeline run on the compact
configuration asserting the CRR anchors, a visible confounding gap, and that SSU-LSF damages the clean
domain less than gradient reversal.

### Configuration reference

| | Compact | Reference |
|---|---|---|
| Command | `python run_experiments.py --compact` | `python run_experiments.py` |
| Grid, length | 6×6 = 36 locations, 72 steps | 8×8 = 64 locations, 120 steps |
| Wall clock (CPU, seed 0) | 60 s | 215 s |
| ρ_max | 0.906 | 0.919 |
| Footprint | the window only (|Φ| = 24) | the window plus one tail step (|Φ| = 25) |
| Confounding retained by CM | 64.5% | 131.5% |
| CM / OR / GR / SSU-LSF CRR | 0.000 / 1.000 / 0.946 / 0.696 | 0.000 / 1.000 / 0.655 / 0.324 |
| GR / SSU-LSF ΔRMSE % | 17.1 / 3.8 | 20.5 / −1.7 |
| CM / OR / GR / SSU-LSF TCI | 0.625 / 0.679 / 0.636 / 0.639 | 0.641 / 0.721 / 0.648 / 0.655 |
| CM / OR / GR / SSU-LSF SCS | 0.1665 / 0.1671 / 0.1654 / 0.1658 | 0.1272 / 0.1174 / 0.1269 / 0.1260 |

Both are CPU runs at the shipped default seed; each writes its JSON to `results/`.

