#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/pose_determinism.py

La pose de sortie du modèle mono-vue est-elle une fonction de l'image ?

L'enjeu. Si elle l'est, un petit régresseur peut l'apprendre — et le verrou du projet
(4,8 points de F-Score perdus dans l'estimation de repère) se lève sans réentraîner le
générateur. Si elle dépend du bruit initial, aucun régresseur n'est possible, parce
que la cible n'est pas déterminée par l'entrée.

Le piège à éviter. Mesurer directement l'écart de rotation entre deux graines donne
139,5° en médiane sur 42 objets, ce qui semble trancher. Mais la mesure est
**confondue par les symétries** : sur un solide de révolution, deux sorties identiques
se recalent avec des rotations très différentes, toutes également valides. Le chiffre
brut mélange donc l'instabilité du générateur et l'indétermination du recalage.

D'où ce script : on note d'abord chaque objet sur son degré d'asymétrie, en recalant
sa référence sur elle-même tournée. Un objet asymétrique ne peut se superposer à
aucune de ses rotations non triviales ; un objet de révolution le peut parfaitement.
On ne mesure ensuite la stabilité que sur les objets dont le recalage a un sens.

Usage :
    python tools/pose_determinism.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_fusion import load_and_normalize_mesh, align_prediction_to_gt, _proper_rotations


def asymetrie(mesh, n=6000):
    """Distance minimale entre l'objet et ses rotations non triviales.

    Grande valeur = objet asymétrique, son orientation est identifiable.
    Valeur faible = l'objet se superpose à lui-même tourné, et toute mesure de
    rotation sur lui est arbitraire.
    """
    pts = np.asarray(mesh.sample_points_uniformly(number_of_points=n).points)
    ref = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    diag = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    pire = np.inf
    for R in _proper_rotations():
        if np.allclose(R, np.eye(3)):
            continue
        rot = o3d.geometry.PointCloud(o3d.utility.Vector3dVector((R @ pts.T).T))
        d = float(np.mean(np.asarray(ref.compute_point_cloud_distance(rot))))
        pire = min(pire, d)
    return pire / diag          # sans dimension, comparable entre objets


def rotations_sorties(gt, obj, mesh_dir, seeds):
    """Rotation entre chaque sortie non guidée et la référence."""
    from scipy.spatial.transform import Rotation
    out = []
    for s in seeds:
        f = Path(mesh_dir) / obj / f"w0_s{s}" / f"{obj}.obj"
        if not f.exists():
            continue
        pred, _ = load_and_normalize_mesh(f)
        _, info = align_prediction_to_gt(gt, pred, return_info=True)
        M = info["transform"][:3, :3]
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        if np.linalg.det(R) > 0:
            out.append(R)
    return out


def main():
    ap = argparse.ArgumentParser(description="Déterminisme de la pose de sortie")
    ap.add_argument("--data_dir", default="data/training_ycb")
    ap.add_argument("--mesh_dir", default="out/full42_combined")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seuil", type=float, default=0.06,
                    help="Asymétrie au-dessus de laquelle l'orientation est identifiable.")
    ap.add_argument("--out", default="pose_determinism.json",
                    help="Le papier cite 140,5° / 129,8° / 1 sur 16 / 0 sur 26 : ce fichier "
                         "est ce qui les porte. Sans lui, le script n'imprimait que.")
    args = ap.parse_args()

    from scipy.spatial.transform import Rotation
    objs = sorted(p.name for p in Path(args.mesh_dir).iterdir() if p.is_dir())
    lignes = []
    detail = {}
    for obj in objs:
        gt, _ = load_and_normalize_mesh(Path(args.data_dir) / obj / "mesh.obj")
        a = asymetrie(gt)
        Rs = rotations_sorties(gt, obj, args.mesh_dir, args.seeds)
        if len(Rs) < 2:
            continue
        ec = [np.degrees(Rotation.from_matrix(Rs[i] @ Rs[j].T).magnitude())
              for i in range(len(Rs)) for j in range(i + 1, len(Rs))]
        lignes.append((obj, a, float(np.median(ec))))
        detail[obj] = {"asymetrie": float(a), "ecart_median_deg": float(np.median(ec)),
                       "ecarts_deg": [float(x) for x in ec], "n_graines": len(Rs)}
        print(f"  {obj:26s} asymétrie {a:.3f}   écart entre graines {np.median(ec):6.1f}°",
              flush=True)

    A = np.array([l[1] for l in lignes])
    E = np.array([l[2] for l in lignes])
    asym = A >= args.seuil
    import json
    A_ = np.array([l[1] for l in lignes]); E_ = np.array([l[2] for l in lignes])
    asym_ = A_ > args.seuil
    Path(args.out).write_text(json.dumps({
        "seuil_asymetrie": args.seuil, "seeds": args.seeds, "mesh_dir": args.mesh_dir,
        "objets": detail,
        "agrege": {
            "asymetriques": {"n": int(asym_.sum()), "ecart_median_deg": float(np.median(E_[asym_])),
                             "stables_sous_30": int((E_[asym_] < 30).sum())},
            "symetriques": {"n": int((~asym_).sum()), "ecart_median_deg": float(np.median(E_[~asym_])),
                            "stables_sous_30": int((E_[~asym_] < 30).sum())},
            "correlation_asymetrie_stabilite": float(np.corrcoef(A_, E_)[0, 1]),
        }}, indent=2))
    print(f"→ {args.out}")
    print(f"\n{'':4s}{'objets':>28s} {'écart médian':>14s} {'stables (<30°)':>16s}")
    print(f"    {'asymétriques (identifiables)':>28s} {np.median(E[asym]):13.1f}° "
          f"{f'{(E[asym] < 30).sum()}/{asym.sum()}':>16s}")
    print(f"    {'symétriques (mesure aveugle)':>28s} {np.median(E[~asym]):13.1f}° "
          f"{f'{(E[~asym] < 30).sum()}/{(~asym).sum()}':>16s}")
    print(f"\nCorrélation asymétrie / stabilité : {np.corrcoef(A, E)[0, 1]:+.3f}")
    print("\nUn régresseur de pose n'est envisageable que si les objets asymétriques "
          "sont stables.\nS'ils ne le sont pas, la pose dépend du bruit initial et "
          "n'est pas une fonction de l'image.")


if __name__ == "__main__":
    main()
