"""Visibility recomputed from the camera that actually rendered the input images.

tools/render_blender.py places the camera at (2.2, -2.2, 1.8) in Blender's Z-up world,
and Blender's OBJ importer brings each Y-up mesh into that world as (x, y, z) -> (x, -z, y)
before rendering. analyze_occlusion.py, make_contacts.py, make_contacts_strategies.py
and run_strategies.py placed the same camera in the meshes' own frame instead, 68 degrees
away. pose_from_silhouette.py already applied the turn (BLENDER_IMPORT).

The campaigns ran with the misplaced camera, and their result files keep what it gave.
This script measures the error and recomputes what the paper reports about visibility:

  1. silhouette IoU against the input images, with and without the turn (42 objects);
  2. share of the surface whose visible/hidden label changes (hidden-point removal);
  3. share of the palpation anchors (palpation8_128) hidden from the rendering camera,
     and the same for the occluded / visible / all sets of the five test objects;
  4. hidden fraction of each grasp strategy, and its correlation with the gain (Fig. 2b).

    python tools/visibility_render_camera.py --data_dir data/training_ycb

The projection is the one in pose_from_silhouette.py, copied rather than imported:
that module loads the model code, and this measurement needs none of it.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_fusion import load_and_normalize_mesh      # noqa: E402
from analyze_occlusion import N_SAMPLE                # noqa: E402

CAM_LOC = np.array([2.2, -2.2, 1.8])                  # render_blender.py, Blender world
CAM_EULER = (np.radians(60.0), 0.0, np.radians(45.0))
RES, FOCAL_MM, SENSOR_MM = 512, 50.0, 36.0
RENDER_EXTENT, NORM_EXTENT = 1.3, 2.0
BLENDER_IMPORT = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])

CAM_MISPLACED = CAM_LOC.copy()                                          # what the campaigns used
CAM_RENDER = BLENDER_IMPORT.T @ CAM_LOC * (NORM_EXTENT / RENDER_EXTENT)  # dataset frame, scaled
CAM_RENDER_UNSCALED = BLENDER_IMPORT.T @ CAM_LOC                        # turn only
TEST_OBJECTS = ["002_master_chef_can", "006_mustard_bottle", "011_banana", "025_mug", "035_power_drill"]
STRATEGIES = {"fb": "front_back", "lr": "left_right", "rl": "right_left"}


def camera_rotation():
    rx, ry, rz = CAM_EULER
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def silhouette(pts_render):
    """Occupancy mask of points given in the render frame, at the image resolution."""
    cam = (camera_rotation().T @ (pts_render - CAM_LOC).T).T
    depth = -cam[:, 2]
    ok = depth > 1e-6
    f = FOCAL_MM / SENSOR_MM * RES
    u = np.round(RES / 2 + f * cam[ok, 0] / depth[ok]).astype(int)
    v = np.round(RES / 2 - f * cam[ok, 1] / depth[ok]).astype(int)
    m = np.zeros((RES, RES), dtype=bool)
    inside = (u >= 0) & (u < RES) & (v >= 0) & (v < RES)
    m[v[inside], u[inside]] = True
    m[1:, :] |= m[:-1, :]; m[:-1, :] |= m[1:, :]; m[:, 1:] |= m[:, :-1]; m[:, :-1] |= m[:, 1:]
    return m


def image_mask(path):
    im = np.asarray(Image.open(path).convert("RGB")).astype(float)
    bg = np.median(np.concatenate([im[0], im[-1], im[:, 0], im[:, -1]]), axis=0)
    return np.abs(im - bg).sum(2) > 20


def iou(a, b):
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def hidden_hpr(pcd, cam):
    pts = np.asarray(pcd.points)
    _, idx = pcd.hidden_point_removal(cam, np.linalg.norm(pts.max(0) - pts.min(0)) * 100)
    seen = np.zeros(len(pts), dtype=bool)
    seen[np.asarray(idx)] = True
    return ~seen


def hidden_normal(pos, nrm, cam):
    """The campaign's test (run_strategies.py): the normal faces away from the camera."""
    nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
    to_cam = (cam - pos) / np.linalg.norm(cam - pos, axis=1, keepdims=True)
    return (nrm * to_cam).sum(1) < 0


def hidden_ray(scene, cam, P, tol=0.02):
    """A ray from the camera meets the surface before reaching the point."""
    d = P - cam
    L = np.linalg.norm(d, axis=1)
    rays = np.hstack([np.tile(cam, (len(P), 1)), d / L[:, None]]).astype(np.float32)
    return scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy() < L - tol


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data_dir", default="data/training_ycb")
    ap.add_argument("--results", default="results")
    ap.add_argument("--contacts", default="data/contacts")
    ap.add_argument("--out", default="results/visibility_render_camera.json")
    args = ap.parse_args()
    data, res, con = Path(args.data_dir), Path(args.results), Path(args.contacts)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    ang = np.degrees(np.arccos(CAM_MISPLACED @ CAM_RENDER / np.linalg.norm(CAM_MISPLACED) / np.linalg.norm(CAM_RENDER)))

    per_object = {}
    for obj in sorted(json.load(open(res / "full42_combined.json"))):
        gt, _ = load_and_normalize_mesh(data / obj / "mesh.obj")
        o3d.utility.random.seed(0)
        pcd = gt.sample_points_uniformly(number_of_points=N_SAMPLE)
        mask = image_mask(data / obj / "image.png")
        scale = RENDER_EXTENT / NORM_EXTENT
        h_mis, h_ren = hidden_hpr(pcd, CAM_MISPLACED), hidden_hpr(pcd, CAM_RENDER)
        # Silhouettes need a denser sampling than hidden-point removal: at N_SAMPLE points
        # the rasterised mask keeps holes and the IoU drops by about 0.03.
        o3d.utility.random.seed(0)
        dense = np.asarray(gt.sample_points_uniformly(number_of_points=60000).points)
        row = {"iou_with_turn": iou(mask, silhouette(dense @ BLENDER_IMPORT.T * scale)),
               "iou_without_turn": iou(mask, silhouette(dense * scale)),
               "surface_hidden_misplaced": float(h_mis.mean()),
               "surface_hidden_render": float(h_ren.mean()),
               "surface_label_changed": float((h_mis != h_ren).mean())}
        p8 = con / obj / "palpation8_128.pt"
        if p8.exists():
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(gt))
            P = torch.load(p8, weights_only=True).reshape(-1, 3).numpy().astype(np.float64)
            row["palpation8_hidden_misplaced"] = float(hidden_ray(scene, CAM_MISPLACED, P).mean())
            row["palpation8_hidden_render"] = float(hidden_ray(scene, CAM_RENDER, P).mean())
            if obj in TEST_OBJECTS:
                for s in ("occluded_128", "visible_128", "all_128"):
                    f = con / obj / f"{s}.pt"
                    if f.exists():
                        Q = torch.load(f, weights_only=True).reshape(-1, 3).numpy().astype(np.float64)
                        row[f"{s}_hidden_misplaced"] = float(hidden_ray(scene, CAM_MISPLACED, Q).mean())
                        row[f"{s}_hidden_render"] = float(hidden_ray(scene, CAM_RENDER, Q).mean())
        per_object[obj] = row

    S = json.load(open(res / "strategies.json"))
    npz = np.load(con / "strategies.npz", allow_pickle=True)
    cases = []
    for obj in S:
        for c, strat in STRATEGIES.items():
            pos, nrm = npz[f"{obj}|{strat}|pos"], npz[f"{obj}|{strat}|nrm"]
            parts = {k: float(hidden_normal(pos, nrm, cam).mean())
                     for k, cam in [("misplaced", CAM_MISPLACED), ("render", CAM_RENDER), ("render_unscaled", CAM_RENDER_UNSCALED)]}
            for s in (0, 1, 2):
                e = S[obj][f"{c}_s{s}"]
                cases.append({"object": obj, "strategy": c, "seed": s, "stored_part_occultee": e["part_occultee"],
                              **{f"hidden_{k}": v for k, v in parts.items()},
                              "gain": e["f2"] - S[obj][f"baseline_s{s}"]["f2"]})
    g = np.array([x["gain"] for x in cases])
    col = lambda k: np.array([x[k] for x in cases])
    assert np.allclose(col("stored_part_occultee"), col("hidden_misplaced")), "the stored fractions are not reproduced"
    slope, intercept = np.polyfit(col("hidden_render"), g, 1)
    strategies = {c: {"hidden_misplaced": float(np.mean([x["hidden_misplaced"] for x in cases if x["strategy"] == c])),
                      "hidden_render": float(np.mean([x["hidden_render"] for x in cases if x["strategy"] == c])),
                      "gain": float(np.mean([x["gain"] for x in cases if x["strategy"] == c]))} for c in STRATEGIES}

    agg = lambda k: [v[k] for v in per_object.values() if k in v]
    summary = {
        "angle_between_cameras_deg": float(ang),
        "iou_with_turn": {"mean": float(np.mean(agg("iou_with_turn"))), "min": float(np.min(agg("iou_with_turn")))},
        "iou_without_turn": {"mean": float(np.mean(agg("iou_without_turn"))), "max": float(np.max(agg("iou_without_turn")))},
        "objects_where_raw_frame_overlaps_better": int(sum(v["iou_without_turn"] > v["iou_with_turn"] for v in per_object.values())),
        "surface_label_changed": {"mean": float(np.mean(agg("surface_label_changed"))), "min": float(np.min(agg("surface_label_changed"))), "max": float(np.max(agg("surface_label_changed")))},
        "palpation8_hidden_render": {"mean": float(np.mean(agg("palpation8_hidden_render"))), "min": float(np.min(agg("palpation8_hidden_render"))), "max": float(np.max(agg("palpation8_hidden_render"))), "n": len(agg("palpation8_hidden_render"))},
        "palpation8_hidden_misplaced_mean": float(np.mean(agg("palpation8_hidden_misplaced"))),
        **{f"{s}_hidden_render_mean_test_objects": float(np.mean(agg(f"{s}_hidden_render"))) for s in ("occluded_128", "visible_128", "all_128")},
        "strategies": strategies,
        "correlation_hidden_gain": {"misplaced": float(np.corrcoef(col("hidden_misplaced"), g)[0, 1]),
                                    "render": float(np.corrcoef(col("hidden_render"), g)[0, 1]),
                                    "render_unscaled": float(np.corrcoef(col("hidden_render_unscaled"), g)[0, 1])},
        "fit_render": {"slope": float(slope), "intercept": float(intercept)},
        "cases_with_no_hidden_anchor": {"misplaced": int((col("hidden_misplaced") == 0).sum()), "render": int((col("hidden_render") == 0).sum())},
        "cases_changed_by_scale": int((col("hidden_render") != col("hidden_render_unscaled")).sum()),
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "per_object": per_object, "grasp_cases": cases}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
