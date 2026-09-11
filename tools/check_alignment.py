#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/check_alignment.py

Mesure ce que le recalage d'évaluation coûtait quand son ICP partait de l'identité.

Le repère de sortie de WaLa ne coïncide pas avec celui du jeu de données, et l'écart
d'ordre des axes diffère selon l'objet (mesuré sur cinq objets YCB,
`tools/guided_sampling.py:register_contacts`). Un ICP point-à-point initialisé à
l'identité ne peut pas rattraper un échange d'axes à 90° : il converge vers un
minimum local, et la prédiction est alors notée à travers un recalage raté. Comme
l'échec dépend de l'objet, il peut inverser le classement de deux méthodes.

Ce script rejoue les maillages déjà produits avec les deux recalages et compare.

Usage :
    python tools/check_alignment.py --eval_root eval_output_ycb_pc_v2
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import open3d as o3d

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_fusion import (align_prediction_to_gt, compute_chamfer_and_fscore,
                         load_and_normalize_mesh)


def align_from_identity(gt_mesh, pred_mesh, num_points=10000):
    """L'ancien recalage : Sim3 grossier puis ICP parti de l'identité."""
    pred = copy.deepcopy(pred_mesh)
    gt_pcd = gt_mesh.sample_points_uniformly(number_of_points=num_points)
    pr_pcd = pred.sample_points_uniformly(number_of_points=num_points)
    gc, pc = gt_pcd.get_center(), pr_pcd.get_center()
    gs = np.linalg.norm(np.asarray(gt_pcd.points) - gc, axis=1).mean()
    ps = np.linalg.norm(np.asarray(pr_pcd.points) - pc, axis=1).mean()
    pred.translate(-pc)
    if ps > 1e-9:
        pred.scale(gs / ps, center=[0, 0, 0])
    pred.translate(gc)
    diag = np.linalg.norm(gt_mesh.get_axis_aligned_bounding_box().get_extent())
    reg = o3d.pipelines.registration.registration_icp(
        pred.sample_points_uniformly(number_of_points=num_points), gt_pcd,
        max_correspondence_distance=0.3 * diag,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80),
    )
    pred.transform(reg.transformation)
    return pred


def f2(gt, pred):
    return compute_chamfer_and_fscore(gt, pred)["f_scores"]["2.0%"]["f_score"]


def main():
    ap = argparse.ArgumentParser(description="Ancien recalage contre nouveau")
    ap.add_argument("--data_dir", default="data/training_ycb")
    ap.add_argument("--eval_root", default="eval_output_ycb_pc_v2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.eval_root)
    rows, results = [], {}
    for cfg_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for obj_dir in sorted(p for p in cfg_dir.iterdir() if p.is_dir()):
            preds = sorted(obj_dir.rglob("*.obj"))
            gt = Path(args.data_dir) / obj_dir.name / "mesh.obj"
            if not preds or not gt.exists():
                continue
            gt_mesh, _ = load_and_normalize_mesh(gt)
            pred_mesh, _ = load_and_normalize_mesh(preds[0])
            old = f2(gt_mesh, align_from_identity(gt_mesh, pred_mesh))
            new_mesh, info = align_prediction_to_gt(gt_mesh, pred_mesh, return_info=True)
            new = f2(gt_mesh, new_mesh)
            rows.append((cfg_dir.name, obj_dir.name, old, new, info["from_identity"]))
            results.setdefault(cfg_dir.name, {})[obj_dir.name] = {
                "f2_identity": old, "f2_rot_init": new,
                **{k: v for k, v in info.items() if k != "transform"}}
            print(f"{cfg_dir.name:14s} {obj_dir.name:22s} "
                  f"{old:6.2f} -> {new:6.2f}  ({new - old:+6.2f})"
                  f"{'' if info['from_identity'] else '   [rotation]'}", flush=True)

    if rows:
        d = np.array([r[3] - r[2] for r in rows])
        print(f"\nÉcart moyen {d.mean():+.2f}, médian {np.median(d):+.2f}, "
              f"max {d.max():+.2f}, min {d.min():+.2f}")
        print(f"{sum(1 for r in rows if not r[4])}/{len(rows)} recalages retenus "
              f"partent d'une rotation, pas de l'identité")
        print(f"{sum(1 for r in rows if r[3] - r[2] > 5)}/{len(rows)} gagnent plus "
              f"de 5 points de F@2 %")

    out = Path(args.out) if args.out else Path(f"alignment_{root.name}.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
