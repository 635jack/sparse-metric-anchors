"""How much of the run-to-run spread comes from the scorer itself.

The F@2 of one prediction is not a fixed number: `align_prediction_to_gt` and
`compute_chamfer_and_fscore` both sample points at random, and the ICP can settle in a
different basin from one call to the next. This script re-scores the saved meshes of
the frozen-frame noise-floor repetitions several times each, then compares:

  - the spread of re-scoring the very same mesh (the scorer alone);
  - the spread across repetitions, as the paper's noise floor reports it (one score
    per repetition, read from the stored results);
  - what is left for generation once the scorer is removed.

    python tools/scoring_noise.py --out_root out --data_dir data/training_ycb --k 4

Scoring goes through eval_fusion only, the same calls `run_guidance_campaign.score`
makes for F@2, so the model code is not needed.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_fusion import load_and_normalize_mesh, align_prediction_to_gt, compute_chamfer_and_fscore  # noqa: E402

# floor directory -> stored results of that floor (one score per repetition)
FLOORS = {
    "plancher_gel": "plancher_gel.json",
    "plancher_gel_006_mustard_bottle": "consolidation/plancher_gel_006_mustard_bottle.json",
    "plancher_gel_025_mug": "consolidation/plancher_gel_025_mug.json",
    "plancher_gel_035_power_drill": "consolidation/plancher_gel_035_power_drill.json",
}
CONDITIONS = {"w0": "temoin", "g": "guide"}      # mesh suffix -> key in the stored results


def f2(gt, path):
    pred, _ = load_and_normalize_mesh(path)
    return compute_chamfer_and_fscore(gt, align_prediction_to_gt(gt, pred))["f_scores"]["2.0%"]["f_score"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out_root", default="out")
    ap.add_argument("--data_dir", default="data/training_ycb")
    ap.add_argument("--results", default="results")
    ap.add_argument("--k", type=int, default=4, help="re-scorings per saved mesh")
    ap.add_argument("--out", default="results/scoring_noise.json")
    args = ap.parse_args()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    report = {}
    for floor, stored_file in FLOORS.items():
        stored = json.load(open(Path(args.results) / stored_file))
        reps = sorted({p.name.rsplit("_", 1)[0] for p in (Path(args.out_root) / floor).iterdir() if p.is_dir()},
                      key=lambda r: int(r[1:]))
        obj = next((Path(args.out_root) / floor / f"{reps[0]}_g").glob("*.obj")).stem
        gt, _ = load_and_normalize_mesh(Path(args.data_dir) / obj / "mesh.obj")
        t0 = time.time()
        scores = {c: np.array([[f2(gt, next((Path(args.out_root) / floor / f"{r}_{c}").glob("*.obj")))
                                for _ in range(args.k)] for r in reps]) for c in CONDITIONS}
        row = {"object": obj, "repetitions": len(reps), "k": args.k, "scores": {c: s.tolist() for c, s in scores.items()}}
        for c, key in CONDITIONS.items():
            s = scores[c]
            within = float(np.sqrt(s.var(axis=1, ddof=1).mean()))              # same mesh, re-scored
            across = float(np.std([e[key] for e in stored], ddof=1))          # the paper's floor
            gen_var = float(s.mean(axis=1).var(ddof=1) - within ** 2 / args.k)  # repetitions, scorer averaged out
            row[key] = {"scorer_sd": within, "floor_sd": across,
                        "scorer_share_of_floor_variance": within ** 2 / across ** 2,
                        "generation_sd": float(np.sqrt(max(gen_var, 0.0)))}
        d_across = float(np.std([e["diff"] for e in stored], ddof=1))
        d_scorer = float(np.sqrt(row["temoin"]["scorer_sd"] ** 2 + row["guide"]["scorer_sd"] ** 2))
        row["diff"] = {"scorer_sd": d_scorer, "floor_sd": d_across, "scorer_share_of_floor_variance": d_scorer ** 2 / d_across ** 2}
        row["seconds"] = time.time() - t0
        report[obj] = row
        print(f"{obj:22s} scorer sd unanchored {row['temoin']['scorer_sd']:.2f} (floor {row['temoin']['floor_sd']:.2f}) | "
              f"anchored {row['guide']['scorer_sd']:.2f} (floor {row['guide']['floor_sd']:.2f}) | "
              f"paired diff {d_scorer:.2f} (floor {d_across:.2f}) | {row['seconds']:.0f} s", flush=True)

    pool = lambda key, field: float(np.sqrt(np.mean([r[key][field] ** 2 for r in report.values()])))
    report["_pooled"] = {key: {"scorer_sd": pool(key, "scorer_sd"), "floor_sd": pool(key, "floor_sd")}
                         for key in ("temoin", "guide", "diff")}
    Path(args.out).write_text(json.dumps(report, indent=1))
    print("pooled:", json.dumps(report["_pooled"]))


if __name__ == "__main__":
    main()
