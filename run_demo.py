# One-command demo: python run_demo.py [--quick]
import argparse, math, sys, time
from ssu_lsf import Config, run
from ssu_lsf.pipeline import save_json, table


def main():
    ap = argparse.ArgumentParser(description="SSU-LSF demo")
    ap.add_argument("--quick", action="store_true", help="small grid for a fast check")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = Config.quick(a.seed) if a.quick else Config(seed=a.seed)
    t0 = time.time(); res = run(cfg)
    print("\n" + table(res)); print("meta:", res["_meta"])
    gap = 100 * (res["CM"]["rmse_aff"] / max(res["OR"]["rmse_aff"], 1e-12) - 1)  # confounding signal
    print(f"confounding signal (CM vs OR on affected test locations): {gap:.1f}%")
    if gap < 10:
        print("warning: confounding is too weak for CRR to be meaningful; keep te = t_split - 1 "
              "so the injected bias survives into the test period")
    ok = all(math.isfinite(v[k]) for n, v in res.items() if not n.startswith("_") for k in ("crr", "rmse_aff", "scs"))
    save_json(res, "results/demo_results.json")
    print(f"\nsaved results/demo_results.json | {time.time() - t0:.1f}s | all finite: {ok}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
