<h1 align="center">SSU-LSF</h1>
<p align="center"><b>State-Space Unlearning for Non-Stationary Bias in Land Surface Forecasting</b></p>
<p align="center"><i>Forget the anomaly. Keep the climate.</i></p>
<p align="center">
  <a href="SSU_LSF_Global_South_AI%20(2).pdf">Paper (PDF)</a> ·
  <a href="https://YOUR_USERNAME.github.io/ssu_lsf/">Project page</a> ·
  <a href="docs/VERIFICATION.md">Verification</a> ·
  <a href="#citation">Citation</a>
</p>

---

Operational forecasters built on Mamba-family SSMs silently absorb unrecorded events
(irrigation booms, dam-operation shifts, sensor recalibrations) into their state matrices.
SSU-LSF removes that bias **without full retraining**:

1. **Influence**: curvature-preconditioned per-step scores on the confounded model.
2. **Footprint Φ**: confounding window plus a geometric tail of half-width `0.5·ρ/(1−ρ)`.
3. **Unlearning**: ascent on Φ, clean rehearsal, KL trust region, spatial TV on location embeddings.

Accepted at the **NeurIPS 2026 Workshop GlobalSouthAI**. The paper PDF is kept at the repository
root: [`SSU_LSF_Global_South_AI (2).pdf`](SSU_LSF_Global_South_AI%20(2).pdf).

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/ssu_lsf.git && cd ssu_lsf
pip install -r requirements.txt
python run_demo.py --quick      # ~1 min on CPU
python run_demo.py              # full demo grid
pytest -q                       # smoke test
```

The demo prints a table with **CRR** (CM = 0, oracle retrain = 1), affected-location RMSE,
clean-domain ΔRMSE %, spatial coherence (SCS) and temporal consistency (TCI) for
CM, OR, gradient reversal (GR) and SSU-LSF, and writes `results/demo_results.json`.

`requirements.txt` pins `numpy<2`: torch below 2.4 is compiled against the NumPy 1.x C API and
aborts on the first `torch.tensor(np_array)` under NumPy 2. Verified quick run (seed 0, CPU, ~48 s,
ρ_max = 0.906 vs. the paper's 0.91):

| Method | CRR | RMSE_aff | ΔRMSE% | SCS | TCI |
|---|---|---|---|---|---|
| CM | 0.000 | 0.1251 | 0.5 | 0.0999 | 0.625 |
| OR | 1.000 | 0.0761 | 0.0 | 0.1002 | 0.679 |
| GR | 0.946 | 0.0787 | 17.1 | 0.0992 | 0.636 |
| SSU-LSF | 0.696 | 0.0910 | 3.8 | 0.0995 | 0.639 |

GR's 17.1% clean-domain damage (quick grid) sits in the paper's stated 7–22% range and SSU-LSF
stays near the paper's 4.2% worst case (3.8% quick, 4.8% full grid). The full paper ↔ code
mapping, the defects this run uncovered and the deliberate scale-downs are in
[`docs/VERIFICATION.md`](docs/VERIFICATION.md).

## Repository structure

```
.
├── ssu_lsf/                 # importable package (modules sit directly here)
│   ├── __init__.py          # public API
│   ├── config.py            # hyperparameters
│   ├── data.py              # synthetic archive, injection, patch graph
│   ├── model.py             # selective diagonal SSM + training
│   ├── influence.py         # curvature-weighted scores, footprint Φ
│   ├── unlearn.py           # Algorithm 1 (KL trust region + TV)
│   ├── metrics.py           # CRR, RMSE, SCS, TCI
│   └── pipeline.py          # end-to-end experiment
├── tests/test_smoke.py
├── docs/
│   ├── METHOD.md            # method in brief
│   ├── REPRODUCE.md         # paper-scale reproduction notes
│   └── VERIFICATION.md      # paper ↔ code verification report
├── index.html               # GitHub Pages project page (served from /)
├── run_demo.py
├── requirements.txt · LICENSE · CITATION.cff · SSU_LSF_Global_South_AI (2).pdf
```

## Scope of this code

This repo is a compact, CPU-runnable reference implementation on a synthetic archive
with injected label confounding. It mirrors the paper's pipeline; paper-scale
experiments (ERA5, NDVI-LST, CropHarvest, 47.3M backbone) are described in
[`docs/REPRODUCE.md`](docs/REPRODUCE.md), and the claim-by-claim comparison with the paper,
including the deliberate scale-downs, is in [`docs/VERIFICATION.md`](docs/VERIFICATION.md).

The confounding window follows the paper's Appendix G layout: it ends at the train/test split
(`te = t_split − 1`, `T_c = 24`) so the absorbed bias is still present when the test period starts.
Moving that window back into the middle of the training set washes the bias out and makes CRR —
a ratio of two nearly equal RMSEs — meaningless; `tests/test_smoke.py` guards against that.

## Citation

```bibtex
@inproceedings{pal2026ssulsf,
  title     = {State-Space Unlearning for Non-Stationary Bias in Land Surface Forecasting},
  author    = {Pal, Anidipta},
  booktitle = {NeurIPS 2026 Workshop on GlobalSouthAI},
  year      = {2026},
  note      = {PLACEHOLDER: add URL / pages}
}
```

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE).
