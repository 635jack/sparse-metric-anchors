"""Table I rerun with the rendering camera, and the visible-face control: the numbers.

Reads, from --root:
  full42_render_combined.json        frame estimated, palpation8_128 drawn from the
                                     camera that rendered the images
  full42_render_oracle.json          frame given, same anchors, reusing the unanchored
                                     passes of the run above
  full42_render_oracle_visible.json  frame given, visible_128 (the face the camera sees),
                                     same unanchored passes
  full42_combined.json, full42_oracle.json
                                     the published Table I, whose palpation sites were
                                     drawn from the misplaced camera
and writes one JSON with every number, then prints a summary.

Gains of the three new arms are all taken against the same unanchored score, the one
recorded when that pass was generated (full42_render_combined.json). Arms that share an
unanchored pass are then compared on their anchored scores, so the unanchored score
cancels exactly. Comparisons with the published campaigns do not share an unanchored
pass and carry frame re-estimation noise on top.

Sign tests are two-sided with ties discarded, as in the paper.

    python tools/analyze_render_rerun.py --root . --data_dir data/training_ycb \
        --logs logs --out results/render_rerun_analysis.json
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


def sign_test(d):
    d = [x for x in d if x != 0]
    n, k = len(d), sum(x > 0 for x in d)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n) if n else float("nan")
    return k, n, p


def summary(values):
    v = list(values)
    k, n, p = sign_test(v)
    return {"n_cases": len(v), "mean": st.mean(v), "median": st.median(v), "sd": st.stdev(v),
            "se": st.stdev(v) / len(v) ** 0.5, "positive": k, "nonzero": n, "p_two_sided": p}


def per_object_positive(gains):
    objs = sorted({o for o, _ in gains})
    return sum(st.mean(gains[(o, s)] for s in SEEDS) > 0 for o in objs), len(objs)


def out_of_grid_cases(log_text, arm_flag):
    """Cases whose placed anchors partly fell outside the [-1, 1] grid, per arm."""
    hits, arm, obj, seed = {}, None, None, None
    for line in log_text.splitlines():
        m = re.search(r"start: .*run_guidance_campaign\.py (--\w+)", line)
        if m:
            arm = m.group(1)
        m = re.match(r"=== (\S+) :", line)
        if m:
            obj = m.group(1)
        m = re.search(r"témoin s(\d)", line)
        if m:
            seed = int(m.group(1))
        m = re.search(r"(\d+) % des contacts hors de la grille", line)
        if m and arm == arm_flag:
            hits[(obj, seed)] = int(m.group(1))
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--data_dir", default="data/training_ycb")
    ap.add_argument("--contacts_new", default="data/contacts_render")
    ap.add_argument("--contacts_old", default="data/contacts")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out", default="results/render_rerun_analysis.json")
    args = ap.parse_args()
    R = Path(args.root)
    load = lambda f: json.load(open(R / f))

    est, ora, vis = load("full42_render_combined.json"), load("full42_render_oracle.json"), load("full42_render_oracle_visible.json")
    pub_est, pub_ora = load("full42_combined.json"), load("full42_oracle.json")
    cases = [(o, s) for o in sorted(est) for s in SEEDS
             if all(f"{k}_s{s}" in d.get(o, {}) for d, k in ((est, "palpation8_128"), (ora, "palpation8_128"), (vis, "visible_128")))]
    base = {c: est[c[0]][f"baseline_s{c[1]}"]["f2"] for c in cases}
    anch = {"estimated": {c: est[c[0]][f"palpation8_128_s{c[1]}"]["f2"] for c in cases},
            "oracle": {c: ora[c[0]][f"palpation8_128_s{c[1]}"]["f2"] for c in cases},
            "oracle_visible": {c: vis[c[0]][f"visible_128_s{c[1]}"]["f2"] for c in cases}}
    gain = {arm: {c: a[c] - base[c] for c in cases} for arm, a in anch.items()}
    pub = {arm: {c: d[c[0]][f"palpation8_128_s{c[1]}"]["f2"] - d[c[0]][f"baseline_s{c[1]}"]["f2"] for c in cases}
           for arm, d in (("estimated", pub_est), ("oracle", pub_ora))}

    out = {"cases": len(cases), "unanchored_mean": st.mean(base.values()),
           "published_unanchored_mean": st.mean(pub_est[o][f"baseline_s{s}"]["f2"] for o, s in cases)}
    for arm in gain:
        out[f"gain_{arm}"] = {**summary(gain[arm].values()),
                              "objects_positive": per_object_positive(gain[arm])}
    for arm in pub:
        out[f"published_gain_{arm}"] = {**summary(pub[arm].values()),
                                        "objects_positive": per_object_positive(pub[arm])}
        out[f"rerun_minus_published_{arm}"] = summary(gain[arm][c] - pub[arm][c] for c in cases)

    # what the frame costs: same anchors, same unanchored pass, placement estimated vs given
    out["frame_cost"] = {**summary(anch["oracle"][c] - anch["estimated"][c] for c in cases),
                         "realised_share_of_ceiling": st.mean(gain["estimated"].values()) / st.mean(gain["oracle"].values())}
    # the control: same frame transform, same unanchored pass, anchors on the visible face instead
    out["visible_minus_hidden_oracle"] = summary(anch["oracle_visible"][c] - anch["oracle"][c] for c in cases)

    # sensitivity: gains against each arm's own re-scoring of the shared unanchored mesh
    own = {"oracle": {c: ora[c[0]][f"palpation8_128_s{c[1]}"]["f2"] - ora[c[0]][f"baseline_s{c[1]}"]["f2"] for c in cases},
           "oracle_visible": {c: vis[c[0]][f"visible_128_s{c[1]}"]["f2"] - vis[c[0]][f"baseline_s{c[1]}"]["f2"] for c in cases}}
    out["sensitivity_own_baseline"] = {arm: summary(g.values()) for arm, g in own.items()}
    out["baseline_rescoring_sd"] = st.stdev([ora[o][f"baseline_s{s}"]["f2"] - base[(o, s)] for o, s in cases]) / 2 ** 0.5

    # sensitivity: drop the cases where some anchors fell outside the grid, in any arm
    logs = (Path(args.logs) / "table1_render.log").read_text() + "\n" + (Path(args.logs) / "visible_control.log").read_text()
    oog = {flag: out_of_grid_cases(logs, flag) for flag in ("--combined_frame", "--oracle_frame")}
    flagged = set(oog["--combined_frame"]) | set(oog["--oracle_frame"])
    keep = [c for c in cases if c not in flagged]
    out["out_of_grid"] = {"estimated": {f"{o}_s{s}": v for (o, s), v in oog["--combined_frame"].items()},
                          "oracle_arms": {f"{o}_s{s}": v for (o, s), v in oog["--oracle_frame"].items()},
                          "cases_kept": len(keep),
                          "gain_estimated": summary(gain["estimated"][c] for c in keep),
                          "gain_oracle": summary(gain["oracle"][c] for c in keep),
                          "visible_minus_hidden_oracle": summary(anch["oracle_visible"][c] - anch["oracle"][c] for c in keep)}

    # how hidden each anchor set is, seen from the rendering camera
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import open3d as o3d
    import torch
    from eval_fusion import load_and_normalize_mesh
    from visibility_render_camera import hidden_ray, CAM_RENDER
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    hid = {"palpation_rerun": [], "palpation_published": [], "visible_control": []}
    for obj in sorted({o for o, _ in cases}):
        gt, _ = load_and_normalize_mesh(R / args.data_dir / obj / "mesh.obj")
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(gt))
        for key, path in (("palpation_rerun", R / args.contacts_new / obj / "palpation8_128.pt"),
                          ("palpation_published", R / args.contacts_old / obj / "palpation8_128.pt"),
                          ("visible_control", R / args.contacts_new / obj / "visible_128.pt")):
            P = torch.load(path, weights_only=True).reshape(-1, 3).numpy().astype(np.float64)
            hid[key].append(float(hidden_ray(scene, CAM_RENDER, P).mean()))
    out["hidden_fraction_render_camera"] = {k: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}
                                            for k, v in hid.items()}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))

    f = lambda s: f"{s['mean']:+.2f} (median {s['median']:+.2f}), {s['positive']}/{s['nonzero']} positive, p = {s['p_two_sided']:.2g}"
    print(f"{len(cases)} cases | unanchored {out['unanchored_mean']:.1f} (published {out['published_unanchored_mean']:.1f})")
    for arm in ("estimated", "oracle", "oracle_visible"):
        g = out[f"gain_{arm}"]
        print(f"gain {arm:15s} {f(g)} | objects positive {g['objects_positive'][0]}/{g['objects_positive'][1]}")
    for arm in ("estimated", "oracle"):
        print(f"published {arm:10s} {f(out[f'published_gain_{arm}'])} | rerun - published {f(out[f'rerun_minus_published_{arm}'])}")
    print(f"frame cost (oracle - estimated, anchored) {f(out['frame_cost'])} | realised share {out['frame_cost']['realised_share_of_ceiling']:.0%}")
    print(f"visible - hidden (oracle, anchored)       {f(out['visible_minus_hidden_oracle'])}")
    print(f"hidden fraction from the rendering camera: " + ", ".join(f"{k} {v['mean']:.2f}" for k, v in out["hidden_fraction_render_camera"].items()))
    o = out["out_of_grid"]
    print(f"without the {len(cases) - o['cases_kept']} out-of-grid cases: estimated {f(o['gain_estimated'])} | oracle {f(o['gain_oracle'])} | visible - hidden {f(o['visible_minus_hidden_oracle'])}")
    print(f"sensitivity, own re-scored baseline: " + " | ".join(f"{k} {f(v)}" for k, v in out["sensitivity_own_baseline"].items())
          + f" | re-scoring sd of one unanchored mesh {out['baseline_rescoring_sd']:.2f}")


if __name__ == "__main__":
    main()
