#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/make_contacts.py

Fabrique les jeux de points de contact pour la campagne de guidage.

Pourquoi ne pas réutiliser `points_128.pt`. Ces nuages sont échantillonnés
uniformément sur toute la surface de référence, **face visible comprise**. Guider
avec eux ne peut donc pas répondre à la question du stage — « le toucher apporte-t-il
ce que la vision ne voit pas ? » — puisque la contrainte porte autant sur la face que
la caméra voit déjà que sur celle qu'elle ne voit pas. Un gain uniforme serait le
résultat attendu, et ne dirait rien.

On produit donc, pour chaque objet, trois conditions à cardinalité égale :

    occluded : contacts pris uniquement là où la caméra ne voit pas
    visible  : contacts pris uniquement là où elle voit — témoin, l'information y est
               redondante avec l'image
    all      : contacts pris partout — la condition des campagnes précédentes

La visibilité est celle de `tools/analyze_occlusion.py` (suppression des points
cachés depuis la caméra de rendu), et la normalisation celle de
`tools/make_pointclouds.py` (centre à l'origine, plus grande étendue ramenée à 2,0) —
les contacts vivent donc dans le même repère que les nuages existants.

Usage :
    python tools/make_contacts.py --objects 002_master_chef_can 011_banana
    python tools/make_contacts.py            # les cinq objets de test
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

# Position de la caméra dans tools/render_blender.py (setup_scene), reprise telle
# quelle de tools/analyze_occlusion.py.
CAMERA = np.array([2.2, -2.2, 1.8], dtype=float)
# ATTENTION : (2.2, -2.2, 1.8) est la position dans le monde Z-up de Blender, où l'import
# OBJ a d'abord tourné le maillage Y-up : (x, y, z) -> (x, -z, y). Placée telle quelle dans
# le repère du maillage, cette caméra est à 68° de celle qui a rendu les images.
# Les sites de palpation « côté caché » ne sont cachés de la vraie caméra qu'à 61 %.
# Les campagnes publiées ont tourné avec cette valeur ; elle est laissée pour qu'elles
# restent reproductibles. tools/visibility_render_camera.py mesure l'écart et recalcule
# la visibilité depuis la caméra du rendu.
POOL = 20000
TEST_OBJECTS = ["002_master_chef_can", "006_mustard_bottle", "011_banana",
                "025_mug", "035_power_drill"]


def normalize(mesh):
    """Centre à l'origine, plus grande étendue ramenée à 2.0 — comme au rendu."""
    bb = mesh.get_axis_aligned_bounding_box()
    mesh.translate(-bb.get_center())
    ext = float(np.max(bb.get_extent()))
    if ext > 0:
        mesh.scale(2.0 / ext, center=[0, 0, 0])
    return mesh


def visibility_split(mesh, n=POOL):
    """Partitionne la surface en points visibles et occultés depuis CAMERA."""
    pcd = mesh.sample_points_uniformly(number_of_points=n)
    pts = np.asarray(pcd.points)
    diameter = np.linalg.norm(pts.max(0) - pts.min(0))
    _, idx = pcd.hidden_point_removal(CAMERA, diameter * 100)
    vis = np.zeros(len(pts), dtype=bool)
    vis[np.asarray(idx)] = True
    return pts, vis


def farthest_point_sample(pts, k, seed=0):
    """Sous-échantillonne k points bien répartis.

    Un tirage uniforme laisserait des paquets et des trous ; à 32 points la
    couverture décide de ce que la contrainte peut dire, donc on la contrôle.
    """
    if len(pts) <= k:
        return pts
    rng = np.random.default_rng(seed)
    sel = [int(rng.integers(len(pts)))]
    d = np.linalg.norm(pts - pts[sel[0]], axis=1)
    for _ in range(k - 1):
        i = int(np.argmax(d))
        sel.append(i)
        d = np.minimum(d, np.linalg.norm(pts - pts[i], axis=1))
    return pts[sel]


def palpation_contacts(mesh, n, n_sites=8, finger_radius=0.06, seed=0, n_rays=20000,
                       return_normals=False, face="hidden"):
    """Simule une palpation au lieu d'échantillonner la surface de référence.

    Les contacts utilisés jusqu'ici sont des points tirés uniformément sur le maillage
    de référence : ils viennent de la réponse. C'est l'objection la plus lourde contre
    tout résultat de guidage, et elle ne se lève pas par une réserve écrite — il faut
    des contacts qu'un capteur pourrait produire.

    Trois contraintes physiques, dans l'ordre :

    1. **Accessibilité en ligne droite.** Un doigt arrive de l'extérieur selon une
       direction et touche la première surface rencontrée. On lance donc des rayons
       depuis l'extérieur et on ne garde que les premiers impacts — ce qui exclut
       d'office les poches internes qu'un échantillonnage uniforme retiendrait.
    2. **Côté occulté.** On ne conserve que les impacts dont la normale tourne le dos
       à la caméra : c'est le régime où le toucher apporte une information que
       l'image n'a pas.
    3. **Encombrement du doigt.** Une pulpe de rayon fini n'entre pas dans une
       concavité étroite. On rejette les points où une sphère de ce rayon, posée sur
       la normale, traverserait la surface.

    Enfin les contacts sont **groupés en quelques sites** plutôt que dispersés : une
    main touche par plaques, et la couverture change ce que la contrainte peut dire.

    `return_normals` renvoie en plus la normale sortante en chaque contact. Elle est
    déjà calculée ici pour les contraintes 2 et 3 (côté occulté, encombrement) et
    n'était que jetée ; un capteur qui connaît la pose de sa pulpe au contact peut
    l'estimer, et c'est une information par contact que le guidage n'exploite pas
    encore.
    """
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n_rays, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    origins = -d * 3.0
    rays = o3d.core.Tensor(np.hstack([origins, d]).astype(np.float32))
    hit = scene.cast_rays(rays)

    ok = np.isfinite(hit["t_hit"].numpy())
    t = hit["t_hit"].numpy()[ok]
    pts = origins[ok] + d[ok] * t[:, None]
    nrm = hit["primitive_normals"].numpy()[ok]
    # Normales orientées vers l'extérieur : elles doivent s'opposer au rayon.
    flip = (nrm * d[ok]).sum(1) > 0
    nrm[flip] *= -1

    to_cam = CAMERA - pts
    to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True)
    # face="visible" keeps the impacts that face the camera instead: the same palpation
    # model on the other side, to separate where the anchors are from how they spread.
    facing = (nrm * to_cam).sum(1)
    keep = facing < 0 if face == "hidden" else facing > 0
    pts, nrm = pts[keep], nrm[keep]
    if len(pts) == 0:
        return None

    # Encombrement : la sphère du doigt ne doit pas traverser la surface.
    centers = o3d.core.Tensor((pts + finger_radius * nrm).astype(np.float32))
    clear = scene.compute_distance(centers).numpy() >= 0.9 * finger_radius
    pts, nrm = pts[clear], nrm[clear]
    if len(pts) < n:
        return None

    # Regroupement en sites : quelques plaques, pas un semis uniforme.
    sites = farthest_point_sample(pts, n_sites, seed=seed)
    per = max(1, n // n_sites)
    out, out_n = [], []
    for site in sites:
        d2 = np.linalg.norm(pts - site, axis=1)
        take = np.argsort(d2)[:per]
        out.append(pts[take])
        out_n.append(nrm[take])
    out = np.concatenate(out, 0)
    out_n = np.concatenate(out_n, 0)
    if len(out) < n:
        return None
    out, out_n = out[:n], out_n[:n]
    out_n = out_n / np.linalg.norm(out_n, axis=1, keepdims=True)
    return (out, out_n) if return_normals else out


def main():
    ap = argparse.ArgumentParser(description="Jeux de contacts par visibilité")
    ap.add_argument("--data_dir", default="data/training_ycb")
    ap.add_argument("--objects", nargs="*", default=TEST_OBJECTS)
    ap.add_argument("--densities", type=int, nargs="+", default=[32, 128, 512])
    ap.add_argument("--out_dir", default="data/contacts")
    ap.add_argument("--sites", type=int, default=8,
                    help="Nombre de plaques de contact — une main touche par zones.")
    ap.add_argument("--finger_radius", type=float, default=0.06,
                    help="Rayon de la pulpe, en unités du repère normalisé "
                         "(0,06 ~ 6 mm sur un objet de 20 cm).")
    ap.add_argument("--face", choices=["hidden", "visible"], default="hidden",
                    help="Face où tombent les sites de palpation. visible écrit "
                         "palpation<N>visible_<n>.pt.")
    ap.add_argument("--camera", choices=["misplaced", "render"], default="misplaced",
                    help="misplaced : la caméra des campagnes publiées, placée dans le "
                         "repère du maillage, à 68° de celle du rendu. render : la caméra "
                         "qui a rendu les images, ramenée dans ce repère.")
    args = ap.parse_args()
    global CAMERA
    if args.camera == "render":
        # Import OBJ de Blender (x, y, z) -> (x, -z, y), puis normalisation 1,3 -> 2,0.
        blender_import = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        CAMERA = blender_import.T @ np.array([2.2, -2.2, 1.8]) * (2.0 / 1.3)
    print(f"[contacts] caméra {args.camera} : {np.round(CAMERA, 3)}")

    out_root = Path(args.out_dir)
    for obj in args.objects:
        mesh_path = Path(args.data_dir) / obj / "mesh.obj"
        if not mesh_path.exists():
            print(f"[contacts] {obj} : mesh.obj introuvable, ignoré")
            continue
        mesh = normalize(o3d.io.read_triangle_mesh(str(mesh_path)))
        pts, vis = visibility_split(mesh)
        pools = {"occluded": pts[~vis], "visible": pts[vis], "all": pts}
        print(f"[contacts] {obj} : {(~vis).sum()} occultés, {vis.sum()} visibles "
              f"sur {len(pts)}")

        for n in args.densities:
            pal = palpation_contacts(mesh, n, n_sites=args.sites,
                                     finger_radius=args.finger_radius, face=args.face)
            if pal is None:
                print(f"  ! palpation n={n} : pas assez de points accessibles")
                continue
            d = out_root / obj
            d.mkdir(parents=True, exist_ok=True)
            # Le nombre de sites est dans le nom : c'est la variable qu'un système
            # réel contrôle (combien de fois toucher), et celle que le balayage
            # explore. Le nombre de points, lui, n'a montré aucun effet.
            torch.save(torch.from_numpy(pal).float().unsqueeze(0),
                       d / f"palpation{args.sites}{'' if args.face == 'hidden' else args.face}_{n}.pt")
            print(f"  palpation n={n} : {args.sites} sites, "
                  f"étendue {np.ptp(pal, axis=0).round(2)}")

        for cond, pool in pools.items():
            for n in args.densities:
                if len(pool) < n:
                    print(f"  ! {cond} n={n} : seulement {len(pool)} points "
                          f"disponibles, ignoré")
                    continue
                sub = farthest_point_sample(pool, n)
                d = out_root / obj
                d.mkdir(parents=True, exist_ok=True)
                torch.save(torch.from_numpy(sub).float().unsqueeze(0),
                           d / f"{cond}_{n}.pt")
        print(f"  → {out_root / obj}")


if __name__ == "__main__":
    main()
