#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/analyze_occlusion.py

Décompose la qualité de reconstruction selon que la surface est VISIBLE ou
OCCULTÉE depuis la caméra qui a produit l'image d'entrée.

Pourquoi cette décomposition. La grille voxel et le nuage de points décrivent
l'objet entier, face visible comprise. Or sur la face visible la géométrie est
redondante avec l'image : elle ne peut qu'entrer en concurrence avec une
information que la vision possède déjà. La seule information réellement nouvelle
qu'elle apporte est celle de la face cachée.

C'est aussi la formulation la plus fidèle au capteur tactile, qui renseigne
précisément ce que la caméra ne voit pas.

Une moyenne globale mélange les deux régimes et masque l'effet. La mesure globale
donnait une corrélation r = -0.841 entre qualité de la baseline et gain de la
fusion, sans dire OÙ le gain se produit sur l'objet.

Méthode : la visibilité est déterminée par hidden_point_removal d'Open3D depuis la
position de caméra du rendu Blender (2.2, -2.2, 1.8), appliquée au mesh de
référence normalisé comme au rendu. Le F-Score est ensuite calculé séparément sur
le sous-ensemble visible et sur le sous-ensemble occulté.

Usage :
    python tools/analyze_occlusion.py --eval_root eval_output_ycb_vox_v2
    python tools/analyze_occlusion.py --eval_root eval_output_ycb_pc_v2 --out occl_pc.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

# Position de la caméra dans tools/render_blender.py (setup_scene)
CAMERA = np.array([2.2, -2.2, 1.8], dtype=float)
N_SAMPLE = 20000


def normalize(mesh):
    """Centre à l'origine, plus grande étendue ramenée à 2.0 — comme au rendu."""
    bb = mesh.get_axis_aligned_bounding_box()
    mesh.translate(-bb.get_center())
    ext = float(np.max(bb.get_extent()))
    if ext > 0:
        mesh.scale(2.0 / ext, center=[0, 0, 0])
    return mesh


def split_visible_occluded(gt_mesh, n=N_SAMPLE):
    """Partitionne la surface du GT en points visibles et occultés depuis CAMERA."""
    pcd = gt_mesh.sample_points_uniformly(number_of_points=n)
    pts = np.asarray(pcd.points)
    # Le rayon doit englober largement la scène pour que la sphère de projection
    # de hidden_point_removal soit valide.
    diameter = np.linalg.norm(pts.max(0) - pts.min(0))
    _, idx = pcd.hidden_point_removal(CAMERA, diameter * 100)
    vis = np.zeros(len(pts), dtype=bool)
    vis[np.asarray(idx)] = True
    return pts, vis


def f_score(gt_pts, pred_pcd, threshold):
    """Rappel sur un sous-ensemble du GT : fraction couverte par la prédiction.

    On mesure le RAPPEL et non le F-Score complet : la précision se définit depuis
    les points prédits, qu'on ne peut pas attribuer sans ambiguïté à la région
    visible ou occultée du GT. Le rappel répond directement à la question posée —
    quelle proportion de cette région est correctement reconstruite.
    """
    if len(gt_pts) == 0:
        return float("nan")
    g = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_pts))
    d = np.asarray(g.compute_point_cloud_distance(pred_pcd))
    return float((d < threshold).mean() * 100.0)


def main():
    ap = argparse.ArgumentParser(description="Qualité de reconstruction visible vs occulté")
    ap.add_argument("--data_dir", default="data/training_ycb")
    ap.add_argument("--eval_root", default="eval_output_ycb_vox_v2")
    ap.add_argument("--configs", nargs="+", default=["baseline", "fused_ft"])
    ap.add_argument("--threshold_pct", type=float, default=2.0,
                    help="Seuil en %% de la diagonale de la bbox du GT (défaut 2, comme eval_fusion).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    split = json.loads((data_dir / "split.json").read_text())
    objs = split["test"]

    results = {}
    print(f"{'objet':22s} {'région':>9s} " + " ".join(f"{c:>12s}" for c in args.configs) + f" {'écart':>8s}")
    print("-" * 78)

    for o in objs:
        gt_path = data_dir / o / "mesh.obj"
        if not gt_path.exists():
            continue
        gt = normalize(o3d.io.read_triangle_mesh(str(gt_path)))
        pts, vis = split_visible_occluded(gt)
        bb = gt.get_axis_aligned_bounding_box()
        thr = (args.threshold_pct / 100.0) * float(np.linalg.norm(bb.get_max_bound() - bb.get_min_bound()))

        per_cfg = {}
        for cfg in args.configs:
            hits = sorted((Path(args.eval_root) / cfg / o).rglob("*.obj"))
            if not hits:
                per_cfg[cfg] = None
                continue
            pred = normalize(o3d.io.read_triangle_mesh(str(hits[0])))
            ppcd = pred.sample_points_uniformly(number_of_points=N_SAMPLE)
            per_cfg[cfg] = {
                "visible": f_score(pts[vis], ppcd, thr),
                "occlude": f_score(pts[~vis], ppcd, thr),
            }

        if any(v is None for v in per_cfg.values()):
            print(f"{o:22s}   (prédiction manquante)")
            continue

        results[o] = {"n_visible": int(vis.sum()), "n_occlude": int((~vis).sum()), **per_cfg}
        for region in ("visible", "occlude"):
            vals = [per_cfg[c][region] for c in args.configs]
            ecart = vals[-1] - vals[0]
            print(f"{o:22s} {region:>9s} " + " ".join(f"{v:11.2f}%" for v in vals) + f" {ecart:+7.2f}")
        print()

    if results:
        print("=" * 78)
        for region in ("visible", "occlude"):
            moys = [np.mean([r[c][region] for r in results.values()]) for c in args.configs]
            label = "VISIBLE (redondant avec l'image)" if region == "visible" else "OCCULTÉ (info nouvelle)"
            print(f"{label:38s} " + " ".join(f"{m:11.2f}%" for m in moys)
                  + f" {moys[-1]-moys[0]:+7.2f}")
        print("=" * 78)
        print("Rappel : fraction de la surface de référence correctement reconstruite.")

    out = Path(args.out) if args.out else Path(f"occlusion_{Path(args.eval_root).name}.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\n✓ {out}")


if __name__ == "__main__":
    main()
