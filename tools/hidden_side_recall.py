"""Where the gain lands: recall of the side the camera cannot see, and of the side it sees.

The premise of the work was that touch completes what vision misses. If it does,
anchors on the hidden face should improve the hidden side of the reconstruction more
than anchors on the visible face do. This script measures it on the Table I rerun and
the anchor-set arms, with visibility decided from the camera that rendered the images
(the campaigns' own recall columns used a misplaced camera).

For every object and seed, one mesh per arm: the unanchored pass, the anchored pass with
the frame estimated, and the oracle-frame arms of the 2 x 2 anchor-set design (8
palpation sites or uniform coverage, on the hidden or the visible face). Arms whose
meshes are not all present are skipped. Each mesh is aligned onto the reference as
`score` aligns it, with the random generator seeded so that a rerun gives the same
numbers, and the reference surface points are split by hidden-point removal from the
rendering camera. Recall is the share of those points within 2 % of the diagonal of
the prediction.

    python tools/hidden_side_recall.py --root . --out results/hidden_side_recall.json
"""
import argparse
import json
import statistics as st
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_fusion import load_and_normalize_mesh, align_prediction_to_gt  # noqa: E402
from visibility_render_camera import CAM_RENDER                          # noqa: E402

SEEDS = (0, 1, 2)
ARMS = {  # arm -> (output directory, mesh sub-directory pattern)
    "unanchored": ("out/full42_render_combined", "w0_s{s}"),
    "estimated": ("out/full42_render_combined", "palpation8128_w0.05_s{s}"),
    "patches_hidden": ("out/full42_render_oracle", "palpation8128_w0.05_s{s}"),
    "uniform_visible": ("out/full42_render_oracle_visible", "visible128_w0.05_s{s}"),
    "uniform_hidden": ("out/full42_render_oracle_occluded", "occluded128_w0.05_s{s}"),
    "patches_visible": ("out/full42_render_oracle_vispatch", "palpation8visible128_w0.05_s{s}"),
}
CONTRASTS = {  # name -> (arm a, arm b): a minus b, on the same unanchored pass
    "face_within_patches": ("patches_visible", "patches_hidden"),
    "face_within_uniform": ("uniform_visible", "uniform_hidden"),
    "spread_within_hidden": ("uniform_hidden", "patches_hidden"),
    "spread_within_visible": ("uniform_visible", "patches_visible"),
}


def sign_test(d):
    d = [x for x in d if x != 0]
    n, k = len(d), sum(x > 0 for x in d)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n) if n else float("nan")
    return k, n, p


def summary(v):
    v = [float(x) for x in v]          # numpy scalars are not JSON-serialisable
    k, n, p = sign_test(v)
    return {"mean": st.mean(v), "median": st.median(v), "se": st.stdev(v) / len(v) ** 0.5,
            "positive": k, "nonzero": n, "p_two_sided": p}


def mesh_path(R, arm, obj, s):
    root, sub = ARMS[arm]
    return next((R / root / obj / sub.format(s=s)).glob("*.obj"), None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--data_dir", default="data/training_ycb")
    ap.add_argument("--n_points", type=int, default=20000)
    ap.add_argument("--out", default="results/hidden_side_recall.json")
    args = ap.parse_args()
    R = Path(args.root)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    objects = sorted(json.load(open(R / "full42_render_combined.json")))
    arms = [a for a in ARMS if all(mesh_path(R, a, o, s) for o in objects for s in SEEDS)]
    print("arms:", ", ".join(arms), "| skipped:", ", ".join(a for a in ARMS if a not in arms) or "none", flush=True)

    rows, t0 = [], time.time()
    for i, obj in enumerate(objects):
        gt, _ = load_and_normalize_mesh(R / args.data_dir / obj / "mesh.obj")
        o3d.utility.random.seed(0)
        pcd = gt.sample_points_uniformly(number_of_points=args.n_points)
        pts = np.asarray(pcd.points)
        _, idx = pcd.hidden_point_removal(CAM_RENDER, np.linalg.norm(pts.max(0) - pts.min(0)) * 100)
        hidden = np.ones(len(pts), dtype=bool)
        hidden[np.asarray(idx)] = False
        thr = 0.02 * float(np.linalg.norm(gt.get_axis_aligned_bounding_box().get_extent()))
        query = o3d.core.Tensor(pts.astype(np.float32))
        for s in SEEDS:
            row = {"object": obj, "seed": s, "hidden_share_of_surface": float(hidden.mean())}
            for arm in arms:
                pred, _ = load_and_normalize_mesh(mesh_path(R, arm, obj, s))
                o3d.utility.random.seed(0)
                aligned = align_prediction_to_gt(gt, pred)
                scene = o3d.t.geometry.RaycastingScene()
                scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(aligned))
                close = scene.compute_distance(query).numpy() < thr
                row[f"{arm}_hidden"] = 100.0 * float(close[hidden].mean())
                row[f"{arm}_visible"] = 100.0 * float(close[~hidden].mean())
            rows.append(row)
        print(f"[{i + 1}/{len(objects)}] {obj} ({time.time() - t0:.0f} s)", flush=True)

    col = lambda k: np.array([r[k] for r in rows])
    out = {"cases": len(rows), "arms": arms,
           "recall": {f"{arm}_{side}": float(col(f"{arm}_{side}").mean()) for arm in arms for side in ("hidden", "visible")}}
    for side in ("hidden", "visible"):
        for arm in arms[1:]:
            out[f"{arm}_minus_unanchored_{side}"] = summary(col(f"{arm}_{side}") - col(f"unanchored_{side}"))
        for name, (a, b) in CONTRASTS.items():
            if a in arms and b in arms:
                out[f"{name}_{side}"] = summary(col(f"{a}_{side}") - col(f"{b}_{side}"))
    out["rows"] = rows
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))

    f = lambda s: f"{s['mean']:+.2f} ± {s['se']:.2f}, {s['positive']}/{s['nonzero']}, p = {s['p_two_sided']:.2g}"
    print(f"\n{len(rows)} cases; recall, % of the side:")
    for side in ("hidden", "visible"):
        print(f"  {side} side: unanchored {out['recall'][f'unanchored_{side}']:.1f}")
        for arm in arms[1:]:
            print(f"    {arm:16s} minus unanchored  {f(out[f'{arm}_minus_unanchored_{side}'])}")
        for name in CONTRASTS:
            if f"{name}_{side}" in out:
                print(f"    {name:34s} {f(out[f'{name}_{side}'])}")


if __name__ == "__main__":
    main()
