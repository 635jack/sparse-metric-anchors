#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/make_pointclouds.py

Échantillonne un nuage de points sur la surface des meshes d'un jeu de données,
et l'écrit sous <sample>/points.pt au format (1, N, 3).

Pourquoi cette modalité, en plus du voxel :

- La grille 16^3 décrit très inégalement les objets selon leur forme. Mesuré sur
  les formes synthétiques : une sphère occupe 1250 voxels, une règle plate
  seulement 51. La campagne matériaux a montré que c'est ce facteur — et non le
  matériau — qui pilote le gain de la fusion.
- Un nuage échantillonné sur la surface décrit correctement une forme fine comme
  une forme épaisse, à nombre de points égal.
- `PointNet_Simple` produit un nombre de tokens fixe (`num_inds`) quel que soit le
  nombre de points en entrée. La densité devient donc un paramètre libre, qu'on
  peut faire varier sans toucher à l'architecture.
- Enfin, un nuage épars est plus fidèle à un capteur tactile réel, qui renvoie des
  points de contact et non une occupation volumique complète.

Les coordonnées sont centrées et normalisées dans [-1, 1], comme les meshes le
sont ailleurs dans le pipeline d'évaluation.

Usage :
    python tools/make_pointclouds.py --data_dir data/training_ycb --num_points 256
    python tools/make_pointclouds.py --data_dir data/training_ycb --num_points 64 --suffix _64
"""

import argparse
from pathlib import Path

import numpy as np
import torch

try:
    import open3d as o3d
except ImportError:
    raise ImportError("open3d est requis : pip install open3d")


def sample_pointcloud(mesh_path: Path, num_points: int, seed: int = 0) -> np.ndarray:
    """Échantillonne uniformément `num_points` sur la surface, centré dans [-1, 1]."""
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.triangles) == 0:
        raise ValueError(f"Mesh sans triangles : {mesh_path}")
    mesh.compute_vertex_normals()

    # Normalisation identique à celle de tools/eval_fusion.py : centre à l'origine,
    # plus grande étendue ramenée à 2.0.
    bbox = mesh.get_axis_aligned_bounding_box()
    mesh.translate(-bbox.get_center())
    extent = float(np.max(bbox.get_extent()))
    if extent > 0:
        mesh.scale(2.0 / extent, center=[0, 0, 0])

    o3d.utility.random.seed(seed)
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return np.asarray(pcd.points, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Génère les nuages de points d'un jeu de données")
    parser.add_argument("--data_dir", default="data/training_ycb")
    parser.add_argument("--num_points", type=int, default=256,
                        help="Nombre de points échantillonnés par objet (défaut 256).")
    parser.add_argument("--suffix", default="",
                        help="Suffixe du fichier de sortie, pour comparer plusieurs densités "
                             "(ex. --suffix _64 -> points_64.pt).")
    parser.add_argument("--mesh_name", default="mesh.obj")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_name = f"points{args.suffix}.pt"

    samples = sorted(d for d in data_dir.iterdir() if d.is_dir())
    n_ok, n_skip, n_err = 0, 0, 0

    for d in samples:
        mesh_path = d / args.mesh_name
        out_path = d / out_name

        if not mesh_path.exists():
            print(f"  [ignoré] {d.name} : pas de {args.mesh_name}")
            n_skip += 1
            continue
        if out_path.exists() and not args.force:
            print(f"  [existe] {d.name} : {out_name}")
            n_skip += 1
            continue

        try:
            pts = sample_pointcloud(mesh_path, args.num_points, seed=args.seed)
        except Exception as e:
            print(f"  [ERREUR] {d.name} : {e}")
            n_err += 1
            continue

        torch.save(torch.from_numpy(pts).unsqueeze(0), out_path)  # (1, N, 3)
        rng = f"[{pts.min():+.2f}, {pts.max():+.2f}]"
        print(f"  {d.name:24s} -> {pts.shape[0]:5d} points, étendue {rng}")
        n_ok += 1

    print(f"\n{n_ok} nuage(s) écrit(s), {n_skip} ignoré(s), {n_err} en erreur.")
    if n_err:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
