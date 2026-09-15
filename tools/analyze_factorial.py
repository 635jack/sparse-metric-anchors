"""The 2 x 2 design on the anchor set: which face the anchors are on, and how they spread.

Oracle frame, 42 objects, 3 seeds, the same unanchored pass and the same frame
transform in every cell (all four arms reuse the unanchored passes of the Table I rerun
with the rendering camera):

                       hidden face                        visible face
  8 palpation sites    full42_render_oracle.json          full42_render_oracle_vispatch.json
                       (palpation8_128)                   (palpation8visible_128)
  uniform coverage     full42_render_oracle_occluded.json full42_render_oracle_visible.json
                       (occluded_128)                     (visible_128)

Because the unanchored pass is shared, contrasts are taken on the anchored scores and the
unanchored score cancels. Gains are against the unanchored score recorded when that pass
was generated. Sign tests are two-sided with ties discarded, as in the paper.

Each cell is also described by how hidden its anchors are from the rendering camera and
how clustered they are, because the design is not perfectly balanced: the uniform
hidden-face points are less hidden than the hidden-face palpation sites.

    python tools/analyze_factorial.py --root . --out results/anchor_set_factorial.json
"""
import argparse
import json
import re
import statistics as st
import sys
from math import comb
from pathlib import Path

import numpy as np

SEEDS = (0, 1, 2)
CELLS = {  # cell -> (results file, condition key, contacts file pattern)
    "patches_hidden": ("full42_render_oracle.json", "palpation8_128", "data/contacts_render/{o}/palpation8_128.pt"),
    "patches_visible": ("full42_render_oracle_vispatch.json", "palpation8visible_128", "data/contacts_render_vispatch/{o}/palpation8visible_128.pt"),
    "uniform_hidden": ("full42_render_oracle_occluded.json", "occluded_128", "data/contacts_render/{o}/occluded_128.pt"),
    "uniform_visible": ("full42_render_oracle_visible.json", "visible_128", "data/contacts_render/{o}/visible_128.pt"),
}
LOGS = ("table1_render.log", "visible_control.log", "factorial_arms.log")


def sign_test(d):
    d = [x for x in d if x != 0]
    n, k = len(d), sum(x > 0 for x in d)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n) if n else float("nan")
    return k, n, p


def summary(values):
    v = list(values)
    k, n, p = sign_test(v)
    return {"n_cases": len(v), "mean": st.mean(v), "median": st.median(v), "se": st.stdev(v) / len(v) ** 0.5,
            "positive": k, "nonzero": n, "p_two_sided": p}


def out_of_grid(logs_dir):
    """Cases whose placed anchors partly fell outside the [-1, 1] grid, per oracle-frame set."""
    hits = {}
    for name in LOGS:
        p = Path(logs_dir) / name
        if not p.exists():
            continue
        arm = obj = seed = None
        for line in p.read_text().splitlines():
            m = re.search(r"start: .*--oracle_frame .*--sets (\S+)", line)
            if m:
                arm = m.group(1).replace(":", "_")
            elif re.search(r"start: .*--combined_frame", line):
                arm = None
            m = re.match(r"=== (\S+) :", line)
            if m:
                obj = m.group(1)
            m = re.search(r"témoin s(\d)", line)
            if m:
                seed = int(m.group(1))
            m = re.search(r"(\d+) % des contacts hors de la grille", line)
            if m and arm:
                hits.setdefault(arm, {})[(obj, seed)] = int(m.group(1))
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--data_dir", default="data/training_ycb")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out", default="results/anchor_set_factorial.json")
    args = ap.parse_args()
    R = Path(args.root)

    est = json.load(open(R / "full42_render_combined.json"))
    res = {c: json.load(open(R / f)) for c, (f, _, _) in CELLS.items()}
    cases = [(o, s) for o in sorted(est) for s in SEEDS
             if all(f"{CELLS[c][1]}_s{s}" in res[c].get(o, {}) for c in CELLS)]
    base = {k: est[k[0]][f"baseline_s{k[1]}"]["f2"] for k in cases}
    anch = {c: {k: res[c][k[0]][f"{CELLS[c][1]}_s{k[1]}"]["f2"] for k in cases} for c in CELLS}

    def contrasts(keep):
        d = lambda a, b: summary(anch[a][k] - anch[b][k] for k in keep)
        return {
            "gain": {c: summary(anch[c][k] - base[k] for k in keep) for c in CELLS},
            "face_effect_within_patches": d("patches_visible", "patches_hidden"),
            "face_effect_within_uniform": d("uniform_visible", "uniform_hidden"),
            "spread_effect_within_hidden": d("uniform_hidden", "patches_hidden"),
            "spread_effect_within_visible": d("uniform_visible", "patches_visible"),
            "interaction": summary((anch["uniform_visible"][k] - anch["uniform_hidden"][k])
                                   - (anch["patches_visible"][k] - anch["patches_hidden"][k]) for k in keep),
        }

    out = {"cases": len(cases), "all_cases": contrasts(cases)}
    oog = out_of_grid(args.logs)
    flagged = {k for arm in oog.values() for k in arm}
    keep = [k for k in cases if k not in flagged]
    out["out_of_grid"] = {arm: {f"{o}_s{s}": v for (o, s), v in hits.items()} for arm, hits in oog.items()}
    out["without_out_of_grid"] = {"cases_kept": len(keep), **contrasts(keep)}

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import open3d as o3d
    import torch
    from eval_fusion import load_and_normalize_mesh
    from visibility_render_camera import hidden_ray, CAM_RENDER
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    desc = {c: {"hidden": [], "nearest_neighbour": [], "gyration_radius": []} for c in CELLS}
    for obj in sorted({o for o, _ in cases}):
        gt, _ = load_and_normalize_mesh(R / args.data_dir / obj / "mesh.obj")
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(gt))
        for c, (_, _, pattern) in CELLS.items():
            P = torch.load(R / pattern.format(o=obj), weights_only=True).reshape(-1, 3).numpy().astype(np.float64)
            dist = np.linalg.norm(P[:, None] - P[None], axis=2)
            np.fill_diagonal(dist, np.inf)
            desc[c]["hidden"].append(float(hidden_ray(scene, CAM_RENDER, P).mean()))
            desc[c]["nearest_neighbour"].append(float(dist.min(1).mean()))
            desc[c]["gyration_radius"].append(float(np.linalg.norm(P - P.mean(0), axis=1).mean()))
    out["anchor_sets"] = {c: {k: float(np.mean(v)) for k, v in d.items()} for c, d in desc.items()}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))

    f = lambda s: f"{s['mean']:+.2f} ± {s['se']:.2f}, {s['positive']}/{s['nonzero']}, p = {s['p_two_sided']:.2g}"
    a = out["all_cases"]
    print(f"{len(cases)} cases, oracle frame, gain over the shared unanchored pass (F@2 points):")
    print(f"{'':18s}{'hidden face':>34s}{'visible face':>34s}")
    for row in ("patches", "uniform"):
        print(f"{row:18s}" + "".join(f"{f(a['gain'][f'{row}_{face}']):>34s}" for face in ("hidden", "visible")))
    print("anchors hidden from the rendering camera: " + ", ".join(f"{c} {v['hidden']:.2f}" for c, v in out["anchor_sets"].items()))
    for k in ("face_effect_within_patches", "face_effect_within_uniform", "spread_effect_within_hidden", "spread_effect_within_visible", "interaction"):
        print(f"{k:32s} {f(a[k])}")
    w = out["without_out_of_grid"]
    print(f"without the {len(cases) - w['cases_kept']} out-of-grid cases: " + " | ".join(
        f"{k} {w[k]['mean']:+.2f} (p = {w[k]['p_two_sided']:.2g})" for k in ("face_effect_within_patches", "face_effect_within_uniform", "spread_effect_within_hidden", "spread_effect_within_visible")))


if __name__ == "__main__":
    main()
