#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/poisson_temoin.py

Témoin sans modèle génératif : que valent quelques contacts orientés, seuls ?

Le papier affirme que la modalité tactile est une bonne contrainte et un mauvais
générateur. Ce script transforme l'argument en mesure, par une méthode qui n'a rien à
voir avec la diffusion : les contacts orientés du simulateur de préhension sont
reconstruits par Poisson écranté — qui consomme précisément des points orientés — et
notés avec **exactement** la même machinerie que les prédictions de WaLa
(`compute_chamfer_and_fscore` après `align_prediction_to_gt`).

Deux réserves, toutes deux optimistes, à rapporter avec le chiffre : la profondeur
d'octree est choisie après coup contre la vérité terrain (aucun système ne peut faire
cela), et la notation recale la prédiction sur la référence.

Ce script a d'abord vécu dans un répertoire temporaire, qui a été effacé ; le papier
citait 19,81 sans qu'aucun fichier du dépôt ne le porte. Il est ici pour que le
chiffre soit reproductible.

Usage :
    python tools/poisson_temoin.py --contacts data/contacts/strategies.npz \
        --out poisson_temoin.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_fusion import (load_and_normalize_mesh, align_prediction_to_gt,
                         compute_chamfer_and_fscore)

OBJECTS = ["002_master_chef_can", "006_mustard_bottle", "011_banana",
           "025_mug", "035_power_drill"]
STRATEGIES = ["front_back", "left_right", "right_left"]


def poisson(pos, nrm, depth):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pos.astype(np.float64)))
    pc.normals = o3d.utility.Vector3dVector(nrm.astype(np.float64))
    try:
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pc, depth=depth, linear_fit=False)
    except Exception:
        return None
    if len(mesh.triangles) == 0:
        return None
    # Même normalisation que toute prédiction évaluée dans le dépôt.
    bb = mesh.get_axis_aligned_bounding_box()
    mesh.translate(-bb.get_center())
    ext = float(np.max(bb.get_extent()))
    if ext > 0:
        mesh.scale(2.0 / ext, center=[0, 0, 0])
    return mesh


def main():
    ap = argparse.ArgumentParser(description="Poisson sur contacts orientés, noté comme WaLa")
    ap.add_argument("--data_dir", default="data/training_ycb")
    ap.add_argument("--contacts", default="data/contacts/strategies.npz")
    ap.add_argument("--depths", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--out", default="poisson_temoin.json")
    args = ap.parse_args()

    npz = np.load(args.contacts, allow_pickle=True)
    res = {}
    for obj in OBJECTS:
        gt, _ = load_and_normalize_mesh(Path(args.data_dir) / obj / "mesh.obj")
        res[obj] = {}
        for label, strats in [("6_contacts", ["front_back"]), ("18_contacts", STRATEGIES)]:
            pos = np.concatenate([npz[f"{obj}|{s}|pos"] for s in strats])
            nrm = np.concatenate([npz[f"{obj}|{s}|nrm"] for s in strats])
            best = None
            for depth in args.depths:
                m = poisson(pos, nrm, depth)
                if m is None:
                    continue
                try:
                    al = align_prediction_to_gt(gt, m)
                    mt = compute_chamfer_and_fscore(gt, al)
                except Exception:
                    continue
                f2 = mt["f_scores"]["2.0%"]["f_score"]
                if best is None or f2 > best["f2"]:
                    best = {"depth": depth, "f2": f2, "chamfer": mt["chamfer_symmetric"],
                            "num_components": mt["num_components"],
                            "largest_component_ratio": mt["largest_component_ratio"],
                            "is_surface": mt["is_surface"]}
            res[obj][label] = best
            if best:
                print(f"{obj:22s} {label:12s} prof.{best['depth']}  F@2 {best['f2']:6.2f}  "
                      f"{best['num_components']} comp. (max {best['largest_component_ratio']:.2f})  "
                      f"surface={best['is_surface']}", flush=True)

    Path(args.out).write_text(json.dumps(res, indent=2))
    print("\n--- moyennes (meilleure profondeur par cas, donc optimiste) ---")
    for label in ["6_contacts", "18_contacts"]:
        v = [res[o][label]["f2"] for o in res if res[o].get(label)]
        closed = all(res[o][label]["is_surface"] for o in res if res[o].get(label))
        print(f"{label:12s} F@2 moyen {np.mean(v):6.2f}   surfaces closes partout : {closed}")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
