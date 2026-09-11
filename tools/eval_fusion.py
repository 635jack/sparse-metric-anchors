#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/eval_fusion.py

Script d'évaluation quantitative pour comparer les maillages (meshes) générés
par rapport à un mesh de ground truth (GT).
Calcule :
  1. La distance de Chamfer (symétrique, moyenne et RMS)
  2. Le F-Score à différents seuils (ex: 1% et 2% de la diagonale de la Bounding Box)
  3. L'IoU Voxel en convertissant les meshes en grilles voxels

Dépendances : open3d, numpy
  pip install open3d numpy

Usage :
    python tools/eval_fusion.py \
        --gt_mesh examples/ring/ring.obj \
        --pred_mesh examples/fused_output/0_fused/0.obj \
        --num_points 10000 \
        --voxel_resolution 32
"""

import argparse
import json
import sys
import copy
from pathlib import Path
import numpy as np

try:
    import open3d as o3d
except ImportError:
    raise ImportError("open3d est requis : pip install open3d")


def load_and_normalize_mesh(mesh_path: Path):
    """Charge un mesh et le centre/normalise dans une Bounding Box de taille 2.0 ([-1, 1])."""
    if not mesh_path.exists():
        raise FileNotFoundError(f"Fichier mesh introuvable : {mesh_path}")
    
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()
    
    # Bounding box
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = np.max(bbox.get_extent())
    
    # Translation au centre
    mesh.translate(-center)
    # Scale pour que l'étendue max soit 2.0
    if extent > 0:
        mesh.scale(2.0 / extent, center=[0, 0, 0])
        
    return mesh, extent


def _proper_rotations():
    """Les 24 rotations qui envoient les axes sur les axes (permutations + signes)."""
    import itertools
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for i, p in enumerate(perm):
                R[i, p] = signs[i]
            if abs(np.linalg.det(R) - 1.0) < 1e-9:
                yield R


def _pca_frame(P):
    """Centre, base principale et rayon quadratique moyen d'un nuage.

    La base est forcée à être une **rotation**. Les vecteurs singuliers de la SVD sont
    orthogonaux mais pas nécessairement directs : sans ce redressement, `Vtᵀ R₂₄ Vs`
    peut avoir un déterminant négatif, et l'ICP part alors d'un miroir. Mesuré avant
    correction : 5 recalages d'évaluation sur 15 étaient des symétries, dont les trois
    graines de la banane. Une reconstruction inversée est un autre objet — mais elle
    obtient le même F-Score, la métrique ne comparant que des distances.
    """
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    if np.linalg.det(Vt) < 0:
        Vt[-1] *= -1
    return c, Vt, np.sqrt(((P - c) ** 2).sum(1).mean())


def align_prediction_to_gt(gt_mesh, pred_mesh, num_points=10000, return_info=False):
    """Recale la prédiction sur la référence par une similitude (Sim3) puis ICP.

    Sans ce recalage, la métrique mesure autant l'erreur de POSE que l'erreur de
    forme. Mesuré sur la campagne voxel : le F-Score @2% moyen passe de 22.61% à
    31.23% pour la baseline, la reconstruction étant globalement juste mais dans une
    orientation différente. L'étude matériaux appliquait déjà ce recalage ; sans lui
    les deux campagnes n'étaient pas comparables.

    L'ICP est initialisé par les 24 rotations axe-sur-axe entre les bases
    principales des deux nuages, et non depuis l'identité. Raison mesurée
    (`tools/guided_sampling.py:register_contacts`) : WaLa génère dans un repère
    canonique dont l'ordre des axes diffère de celui du jeu de données, et il en
    diffère **différemment selon l'objet**. Un ICP point-à-point parti de l'identité
    est hors de son bassin de convergence pour un échange d'axes à 90°, il converge
    alors vers un minimum local et la prédiction est notée à travers un recalage
    raté. Comme l'échec dépend de l'objet, il ne se contente pas d'abaisser les
    scores : il peut en inverser le classement.
    """
    pred = copy.deepcopy(pred_mesh)
    gt_pcd = gt_mesh.sample_points_uniformly(number_of_points=num_points)
    pr_pcd = pred.sample_points_uniformly(number_of_points=num_points)

    # Alignement grossier : centres et échelles
    gc, pc = gt_pcd.get_center(), pr_pcd.get_center()
    gs = np.linalg.norm(np.asarray(gt_pcd.points) - gc, axis=1).mean()
    ps = np.linalg.norm(np.asarray(pr_pcd.points) - pc, axis=1).mean()
    s = gs / ps if ps > 1e-9 else 1.0
    pred.translate(-pc)
    pred.scale(s, center=[0, 0, 0])
    pred.translate(gc)
    coarse_T = np.eye(4)
    coarse_T[:3, :3] = s * np.eye(3)
    coarse_T[:3, 3] = gc - s * pc

    diag = np.linalg.norm(gt_mesh.get_axis_aligned_bounding_box().get_extent())
    src = pred.sample_points_uniformly(number_of_points=num_points)
    src_np, tgt_np = np.asarray(src.points), np.asarray(gt_pcd.points)
    cs, Vs, _ = _pca_frame(src_np)
    ct, Vt, _ = _pca_frame(tgt_np)

    # Recherche grossière sur nuages réduits : 24 ICP courts, on ne garde que la
    # rotation initiale gagnante, puis on l'affine à pleine résolution.
    coarse = max(2000, num_points // 5)
    src_c = src.random_down_sample(min(1.0, coarse / num_points))
    tgt_c = gt_pcd.random_down_sample(min(1.0, coarse / num_points))

    inits = [np.eye(4)]                       # identité : cas des nuages dégénérés
    for R24 in _proper_rotations():
        M = Vt.T @ R24 @ Vs
        init = np.eye(4)
        init[:3, :3] = M
        init[:3, 3] = ct - M @ cs
        inits.append(init)

    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    best_init, best_rmse = np.eye(4), np.inf
    for init in inits:
        reg = o3d.pipelines.registration.registration_icp(
            src_c, tgt_c, max_correspondence_distance=0.3 * diag, init=init,
            estimation_method=p2p,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
        )
        # `inlier_rmse` seul favorise les recalages qui n'apparient presque rien ;
        # on le pondère donc par le recouvrement (`fitness`).
        score = reg.inlier_rmse / max(reg.fitness, 1e-6)
        if reg.fitness > 0 and score < best_rmse:
            best_init, best_rmse = init, score

    reg = o3d.pipelines.registration.registration_icp(
        src, gt_pcd, max_correspondence_distance=0.3 * diag, init=best_init,
        estimation_method=p2p,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80),
    )
    pred.transform(reg.transformation)
    if return_info:
        return pred, {"from_identity": bool(np.allclose(best_init, np.eye(4))),
                      "inlier_rmse": reg.inlier_rmse, "fitness": reg.fitness,
                      # Transformation complète, grossière comprise : permet de
                      # rejouer plusieurs prédictions dans un repère commun.
                      "transform": np.asarray(reg.transformation) @ coarse_T}
    return pred


def compute_chamfer_and_fscore(gt_mesh, pred_mesh, num_points=10000, thresholds_pct=[1.0, 2.0]):
    """Calcule la distance de Chamfer symétrique et le F-Score."""
    # Échantillonnage de nuages de points uniformes sur la surface des meshes
    gt_pcd = gt_mesh.sample_points_uniformly(number_of_points=num_points)
    pred_pcd = pred_mesh.sample_points_uniformly(number_of_points=num_points)
    
    # 1. Distances de GT vers Pred
    dists_gt_to_pred = np.asarray(gt_pcd.compute_point_cloud_distance(pred_pcd))
    # 2. Distances de Pred vers GT
    dists_pred_to_gt = np.asarray(pred_pcd.compute_point_cloud_distance(gt_pcd))
    
    # Distance de Chamfer moyenne
    chamfer_gt_to_pred = np.mean(dists_gt_to_pred)
    chamfer_pred_to_gt = np.mean(dists_pred_to_gt)
    chamfer_symmetric = 0.5 * (chamfer_gt_to_pred + chamfer_pred_to_gt)
    
    # Chamfer RMS (Root Mean Square)
    chamfer_rms = np.sqrt(0.5 * (np.mean(dists_gt_to_pred**2) + np.mean(dists_pred_to_gt**2)))
    
    # Diagonale de la bounding box du GT (pour normaliser les seuils)
    # Dans notre cas, normalisé dans [-1, 1]³, la diagonale max théorique est sqrt(3)*2 = 3.464
    bbox = gt_mesh.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    
    f_scores = {}
    for pct in thresholds_pct:
        threshold = (pct / 100.0) * diag
        
        # Précision (Precision) : proportion de points prédits proches du GT
        precision = np.mean(dists_pred_to_gt < threshold) * 100.0
        # Rappel (Recall) : proportion de points GT proches de la prédiction
        recall = np.mean(dists_gt_to_pred < threshold) * 100.0
        
        # F-score (Moyenne harmonique)
        if precision + recall > 0:
            f_score = 2 * (precision * recall) / (precision + recall)
        else:
            f_score = 0.0
            
        f_scores[f"{pct}%"] = {
            "threshold_val": threshold,
            "precision": precision,
            "recall": recall,
            "f_score": f_score
        }
        
    return {
        "chamfer_gt_to_pred": chamfer_gt_to_pred,
        "chamfer_pred_to_gt": chamfer_pred_to_gt,
        "chamfer_symmetric": chamfer_symmetric,
        "chamfer_rms": chamfer_rms,
        "gt_diag": diag,
        "f_scores": f_scores,
        **compute_connectivity(pred_mesh),
    }


def compute_connectivity(mesh):
    """Mesure si la prédiction est une surface ou un amas de fragments.

    Chamfer et F-Score échantillonnent des points sur la surface et ne comparent
    que des distances : ils sont aveugles à la topologie. Un semis d'éclats
    disjoints posés près de la vraie surface obtient donc un excellent score sans
    être une reconstruction.

    Mesuré : le modèle nuage à 128 points produit 43 à 101 composantes dont la plus
    grosse porte 4 à 11 % des faces, et marque jusqu'à 48,7 % de F@2 % — au-dessus
    du modèle image (24,6 %) qui, lui, produit une surface unique. Cet artefact a
    été pris pour de la complémentarité entre modalités et a motivé un balayage
    entier avant d'être vu sur une planche de rendus.

    `largest_component_ratio` est le garde-fou : au-dessus de 0,9 la prédiction est
    une surface, en dessous de ~0,5 les métriques de distance ne veulent plus rien
    dire et doivent être lues comme non valides.
    """
    _, counts, _ = mesh.cluster_connected_triangles()
    counts = np.asarray(counts)
    total = int(counts.sum())
    return {
        "num_components": int(len(counts)),
        "largest_component_ratio": float(counts.max() / total) if total else 0.0,
        "is_surface": bool(total and counts.max() / total >= 0.9),
    }


def compute_voxel_iou(gt_mesh, pred_mesh, resolution=32):
    """Calcule l'Intersection over Union (IoU) voxel après voxelisation des deux meshes."""
    voxel_size = 2.0 / resolution
    
    # Voxelisation des deux meshes
    gt_voxels = o3d.geometry.VoxelGrid.create_from_triangle_mesh(gt_mesh, voxel_size=voxel_size)
    pred_voxels = o3d.geometry.VoxelGrid.create_from_triangle_mesh(pred_mesh, voxel_size=voxel_size)
    
    # Extraire les indices de coordonnées grid de chaque voxel
    gt_indices = set(tuple(v.grid_index) for v in gt_voxels.get_voxels())
    pred_indices = set(tuple(v.grid_index) for v in pred_voxels.get_voxels())
    
    if not gt_indices and not pred_indices:
        return 0.0
        
    intersection = len(gt_indices.intersection(pred_indices))
    union = len(gt_indices.union(pred_indices))
    
    iou = (intersection / union) * 100.0 if union > 0 else 0.0
    return {
        "gt_count": len(gt_indices),
        "pred_count": len(pred_indices),
        "intersection": intersection,
        "union": union,
        "iou": iou
    }


def main():
    parser = argparse.ArgumentParser(description="Évaluation quantitative de la reconstruction 3D")
    parser.add_argument("--gt_mesh", type=str, required=True, help="Chemin vers le mesh Ground Truth (.obj/.ply)")
    parser.add_argument("--pred_mesh", type=str, required=True, help="Chemin vers le mesh Prédit (.obj/.ply)")
    parser.add_argument("--num_points", type=int, default=10000, help="Nombre de points pour Chamfer/F-score")
    parser.add_argument("--voxel_resolution", type=int, default=32, help="Résolution de la grille voxel pour IoU")
    parser.add_argument("--no_align", action="store_true", default=False,
                        help="Désactive le recalage Sim3/ICP. Sans recalage la métrique "
                             "mesure aussi l'erreur de pose : F-Score @2%% moyen 22.61%% "
                             "contre 31.23%% avec, sur la campagne voxel.")
    parser.add_argument("--json_out", type=str, default=None,
                        help="Si fourni, écrit les métriques dans ce fichier JSON "
                             "(plus fiable que de parser la sortie texte).")
    args = parser.parse_args()
    
    gt_path = Path(args.gt_mesh)
    pred_path = Path(args.pred_mesh)
    
    print("=" * 60)
    print("ÉVALUATION RECONSTRUCTION 3D")
    print("=" * 60)
    print(f"Ground Truth : {gt_path}")
    print(f"Prédiction   : {pred_path}")
    
    try:
        # 1. Charger et normaliser
        print("\n[1/3] Chargement et normalisation des maillages...")
        gt_mesh, gt_scale = load_and_normalize_mesh(gt_path)
        pred_mesh, _ = load_and_normalize_mesh(pred_path)
        print("  ✓ Maillages chargés et normalisés au centre dans [-1, 1]")

        if not args.no_align:
            pred_mesh = align_prediction_to_gt(gt_mesh, pred_mesh)
            print("  ✓ Prédiction recalée sur la référence (Sim3 + ICP)")
        
        # 2. Chamfer & F-Score
        print("\n[2/3] Calcul de la distance de Chamfer & F-Score...")
        dist_metrics = compute_chamfer_and_fscore(
            gt_mesh, pred_mesh, num_points=args.num_points, thresholds_pct=[1.0, 2.0]
        )
        
        print(f"  Chamfer GT -> Pred : {dist_metrics['chamfer_gt_to_pred']:.6f}")
        print(f"  Chamfer Pred -> GT : {dist_metrics['chamfer_pred_to_gt']:.6f}")
        print(f"  Chamfer Symétrique : {dist_metrics['chamfer_symmetric']:.6f}")
        print(f"  Chamfer RMS        : {dist_metrics['chamfer_rms']:.6f}")
        print(f"  Diagonale GT BBox  : {dist_metrics['gt_diag']:.4f}")
        
        for k, v in dist_metrics["f_scores"].items():
            print(f"  F-Score @ {k} (seuil={v['threshold_val']:.4f}) :")
            print(f"    - Précision : {v['precision']:.2f}%")
            print(f"    - Rappel    : {v['recall']:.2f}%")
            print(f"    - F-Score   : {v['f_score']:.2f}%")
            
        # 3. IoU Voxel
        print(f"\n[3/3] Calcul de l'IoU Voxel à la résolution {args.voxel_resolution}...")
        voxel_metrics = compute_voxel_iou(gt_mesh, pred_mesh, resolution=args.voxel_resolution)
        print(f"  Voxels GT occupés  : {voxel_metrics['gt_count']}")
        print(f"  Voxels Pred occupés: {voxel_metrics['pred_count']}")
        print(f"  Intersection       : {voxel_metrics['intersection']}")
        print(f"  Union              : {voxel_metrics['union']}")
        print(f"  IoU Voxel          : {voxel_metrics['iou']:.2f}%")
        
        print("\n" + "=" * 60)
        print("RÉSUMÉ DES RÉSULTATS")
        print("=" * 60)
        print(f"Chamfer Symétrique : {dist_metrics['chamfer_symmetric']:.6f}")
        print(f"F-Score @ 1%       : {dist_metrics['f_scores']['1.0%']['f_score']:.2f}%")
        print(f"F-Score @ 2%       : {dist_metrics['f_scores']['2.0%']['f_score']:.2f}%")
        print(f"IoU Voxel {args.voxel_resolution}       : {voxel_metrics['iou']:.2f}%")
        print("=" * 60)

        if args.json_out:
            payload = {
                "gt_mesh": str(gt_path),
                "pred_mesh": str(pred_path),
                "chamfer": float(dist_metrics["chamfer_symmetric"]),
                "chamfer_rms": float(dist_metrics["chamfer_rms"]),
                "f1": float(dist_metrics["f_scores"]["1.0%"]["f_score"]),
                "f2": float(dist_metrics["f_scores"]["2.0%"]["f_score"]),
                "iou": float(voxel_metrics["iou"]),
                "voxel_resolution": args.voxel_resolution,
            }
            out_path = Path(args.json_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2))
            print(f"✓ Métriques écrites dans {out_path}")

    except Exception as e:
        import traceback
        print(f"\n✗ ERREUR : {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
