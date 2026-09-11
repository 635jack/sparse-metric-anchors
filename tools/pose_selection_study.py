#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/pose_selection_study.py

Cherche une meilleure règle de sélection de pose, sans lancer une seule génération.

Le constat à traiter. Sur 15 objets des 42, l'information tactile est disponible
(plafond +6,71 en moyenne) mais le placement la perd (gain réel +0,25). Résoudre ce
point ferait passer le résultat global de +3,21 à +8,05 points, sans rien changer au
guidage lui-même.

L'hypothèse testée. La règle actuelle est **séquentielle** : la silhouette retient les
six meilleures poses, puis le résidu des contacts tranche entre elles. Si la vraie
pose n'est pas dans ces six-là, elle est perdue avant que les contacts aient voix au
chapitre. Une règle **conjointe**, qui note chaque candidat sur les deux critères à la
fois et sur un vivier plus large, ne peut pas se faire piéger de cette façon.

Le protocole ne coûte aucun calcul GPU : les maillages non guidés des 42 objets
existent déjà (`out/full42_combined/*/w0_s*`), et on compare chaque règle au placement
oracle, qui est connu. On mesure donc directement ce qu'une règle récupérerait, sans
avoir à régénérer quoi que ce soit.

Usage :
    python tools/pose_selection_study.py --out pose_selection.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_fusion import load_and_normalize_mesh, _proper_rotations
from pose_from_silhouette import (image_mask, silhouette, to_render_frame, iou,
                                  _centered, refine_pose)
from run_guidance_campaign import oracle_frame, _normalization

# Les 15 objets où l'information tactile existe mais où le placement la perd,
# établis par la comparaison plafond / réalisé sur la campagne à 42 objets.
ECHECS = ["002_master_chef_can", "003_cracker_box", "006_mustard_bottle",
          "008_pudding_box", "016_pear", "021_bleach_cleanser", "022_windex_bottle",
          "024_bowl", "025_mug", "037_scissors", "038_padlock", "048_hammer",
          "050_medium_clamp", "052_extra_large_clamp", "065-a_cups"]


def candidates(mesh, mask, n_rot=2000, keep=12, maxiter=250, seed=0):
    """Vivier de poses plausibles : tirage dans SO(3), classement, raffinement."""
    from scipy.spatial.transform import Rotation

    from PIL import Image
    m128 = np.array(Image.fromarray(mask).resize((128, 128), Image.NEAREST))
    pts = np.asarray(mesh.sample_points_uniformly(number_of_points=20000).points)
    pool = list(Rotation.random(n_rot, random_state=seed).as_matrix()) + \
        [np.asarray(R) for R in _proper_rotations()]
    scored = sorted(
        ((iou(silhouette(to_render_frame(_centered((R @ pts.T).T)), res=128), m128), R)
         for R in pool), key=lambda x: -x[0])
    return refine_pose(mesh, mask, top_k=keep, maxiter=maxiter,
                       candidates=[R for _, R in scored[:keep]], return_all=True)


def evaluate(obj, seed, mesh_dir, data_dir, contacts_dir):
    """Pour un (objet, graine) : chaque candidat, ses deux scores, et sa vraie erreur."""
    ref = Path(mesh_dir) / obj / f"w0_s{seed}" / f"{obj}.obj"
    if not ref.exists():
        return None
    gt, _ = load_and_normalize_mesh(Path(data_dir) / obj / "mesh.obj")
    mask = image_mask(Path(data_dir) / obj / "image.png")
    pred, _ = load_and_normalize_mesh(ref)

    c = torch.load(Path(contacts_dir) / obj / "palpation8_128.pt", weights_only=True)
    if c.dim() == 3:
        c = c[0]
    c = c.numpy()

    O = oracle_frame(gt, ref)
    exact = (O[:3, :3] @ c.T).T + O[:3, 3]          # placement de référence
    surf = pred.sample_points_uniformly(number_of_points=20000)
    N = _normalization(ref)

    out = []
    for iou_score, A in candidates(pred, mask):
        Ainv = np.linalg.inv(A @ N)
        pts = (Ainv[:3, :3] @ c.T).T + Ainv[:3, 3]
        # Résidu : les contacts tombent-ils sur la surface de la première passe ?
        An = np.linalg.inv(A)
        pn = (An[:3, :3] @ c.T).T + An[:3, 3]
        d = np.asarray(o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(pn)).compute_point_cloud_distance(surf))
        out.append({"iou": float(iou_score),
                    "residu": float(np.median(d)),
                    "erreur": float(np.median(np.linalg.norm(pts - exact, axis=1)))})
    return out


def regles(cands):
    """Erreur obtenue par chaque règle de sélection, sur un même vivier."""
    if not cands:
        return {}
    ious = np.array([c["iou"] for c in cands])
    res = np.array([c["residu"] for c in cands])
    err = np.array([c["erreur"] for c in cands])
    # Scores normalisés : l'IoU se maximise, le résidu se minimise.
    zi = (ious - ious.mean()) / (ious.std() + 1e-9)
    zr = (res.mean() - res) / (res.std() + 1e-9)
    return {
        "silhouette seule": float(err[int(np.argmax(ious))]),
        "residu seul": float(err[int(np.argmin(res))]),
        "sequentielle (actuelle)": float(err[int(np.argmin(res[:6]))]),
        "conjointe egale": float(err[int(np.argmax(zi + zr))]),
        "conjointe 2:1 silhouette": float(err[int(np.argmax(2 * zi + zr))]),
        "conjointe 1:2 residu": float(err[int(np.argmax(zi + 2 * zr))]),
        "meilleur du vivier": float(err.min()),
    }


def main():
    ap = argparse.ArgumentParser(description="Règles de sélection de pose")
    ap.add_argument("--data_dir", default="data/training_ycb")
    ap.add_argument("--contacts_dir", default="data/contacts")
    ap.add_argument("--mesh_dir", default="out/full42_combined")
    ap.add_argument("--objects", nargs="*", default=ECHECS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="pose_selection.json")
    args = ap.parse_args()

    results, table = {}, {}
    for obj in args.objects:
        results[obj] = {}
        for s in args.seeds:
            c = evaluate(obj, s, args.mesh_dir, args.data_dir, args.contacts_dir)
            if not c:
                continue
            r = regles(c)
            results[obj][f"s{s}"] = {"candidats": c, "regles": r}
            for k, v in r.items():
                table.setdefault(k, []).append(v)
            print(f"{obj:24s} s{s}  " +
                  "  ".join(f"{k.split()[0][:6]}={v:.2f}" for k, v in r.items()),
                  flush=True)
        Path(args.out).write_text(json.dumps(results, indent=2))

    print(f"\n{'règle':28s} {'erreur médiane':>15s} {'moyenne':>9s}")
    print("-" * 56)
    for k, v in sorted(table.items(), key=lambda x: np.median(x[1])):
        print(f"{k:28s} {np.median(v):15.3f} {np.mean(v):9.3f}")
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
