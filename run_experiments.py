# Experiment driver: runs CM, OR, GR and SSU-LSF and writes a results table + JSON
#   python run_experiments.py                 # reference configuration (grid 8x8, T=120)
#   python run_experiments.py --compact       # reduced grid and series (CI / quick local runs)
import argparse, math, sys, time
from ssu_lsf import Config, run
from ssu_lsf.pipeline import save_json, table


def main():
    ap = argparse.ArgumentParser(description="SSU-LSF experiments")
    ap.add_argument("--compact", action="store_true", help="reduced grid and series, for CI and quick runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/results.json", help="JSON output path")
    a = ap.parse_args()
    cfg = Config.compact(a.seed) if a.compact else Config(seed=a.seed)
    t0 = time.time(); res = run(cfg)
    print("\n" + table(res)); print("meta:", res["_meta"])
    gap = res["_meta"]["confound_gap_pct"]  # confounding retained by the confounded model
    print(f"confounding retained by CM vs OR on affected test locations: {gap:.1f}%")
    if gap < 10:
        print("warning: confounding is too weak for CRR to be meaningful; keep te = t_split - 1 so "
              "the injected bias survives into the test period")
    ok = all(math.isfinite(v[k]) for n, v in res.items() if not n.startswith("_") for k in ("crr", "rmse_aff", "scs"))
    save_json(res, a.out)
    print(f"\nsaved {a.out} | {time.time() - t0:.1f}s | all finite: {ok}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
