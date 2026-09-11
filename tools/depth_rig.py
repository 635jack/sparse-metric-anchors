#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/depth_rig.py

Cartes de profondeur aux vues du rig WaLa, et **reprojection d'une vue vers une autre**.

L'idée qu'on veut tester. Les modèles multi-vues sont les seuls dont la pose de sortie
soit ancrée (mesuré : rotation reproductible à 0,8° près), mais ils exigent quatre
entrées, et quatre vraies vues reconstruisent si bien qu'il ne reste rien à apporter
au toucher. Les remplir d'images noires échoue : mesuré, une seule vue réelle sur
quatre donne 43,07 de F-Score contre 95,40 à quatre — les vues vides *retirent* de
l'information.

Une carte de profondeur, elle, est de la géométrie 3D dans le repère de la caméra. On
peut donc la rétroprojeter en points, puis la reprojeter depuis une autre position du
rig. La vue synthétique est **géométriquement juste** là où elle a de la donnée, et
trouée ailleurs — bien plus proche de la distribution d'entraînement qu'une image
noire. Et surtout elle n'ajoute **aucune information** : c'est le même unique regard
réencodé, donc la reconstruction doit rester au niveau mono-vue, avec sa marge pour le
toucher, pendant que les indices de caméra ancrent la pose.

Tout est fait par lancer de rayons plutôt que sous Blender : la profondeur obtenue est
métrique et exacte, ce qui est nécessaire pour reprojeter, alors qu'une passe Z
normalisée à l'image perdrait l'échelle.

Usage :
    python tools/depth_rig.py --objects 035_power_drill --views 3 6 10 26
"""

import argparse
import math
import re
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

RADIUS = 3.594          # même distance que la caméra du projet
FOCAL_MM, SENSOR_MM = 50.0, 36.0
RENDER_EXTENT = 1.3     # normalisation des rendus amont


def load_rig(readme="README.md"):
    """Table indice -> (azimut, élévation), extraite du README amont."""
    rig = {}
    for line in open(readme, encoding="utf-8"):
        m = re.match(r"^\s*\|\s*(\d+)\s*\|\s*(-?[\d.]+)\s*\|\s*(-?[\d.]+)\s*\|", line)
        if m:
            rig[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    if len(rig) < 50:
        raise SystemExit(f"[rig] {len(rig)} vues trouvées, attendu 55")
    return rig


def camera_pose(rotation_deg, elevation_deg, radius=RADIUS):
    """Position et matrice monde->caméra, convention Open3D (+Z devant, Y vers le bas)."""
    az, el = math.radians(rotation_deg), math.radians(elevation_deg)
    eye = np.array([radius * math.cos(el) * math.cos(az),
                    radius * math.cos(el) * math.sin(az),
                    radius * math.sin(el)])
    fwd = -eye / np.linalg.norm(eye)              # regard vers l'origine
    up_w = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up_w)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])              # lignes = axes caméra
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ eye
    return eye, T


def intrinsics(res):
    f = FOCAL_MM / SENSOR_MM * res
    return o3d.core.Tensor([[f, 0, res / 2], [0, f, res / 2], [0, 0, 1]],
                           dtype=o3d.core.Dtype.Float64)


def normalize_mesh(mesh):
    """Centre et met à l'échelle comme les rendus du rig (étendue max 1,3)."""
    bb = mesh.get_axis_aligned_bounding_box()
    mesh.translate(-bb.get_center())
    ext = float(np.max(bb.get_extent()))
    if ext > 0:
        mesh.scale(RENDER_EXTENT / ext, center=[0, 0, 0])
    return mesh


def depth_at_view(mesh, rot, elev, res=256):
    """Profondeur métrique exacte, par lancer de rayons. inf = fond."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    _, T = camera_pose(rot, elev)
    rays = scene.create_rays_pinhole(intrinsics(res),
                                     o3d.core.Tensor(T, dtype=o3d.core.Dtype.Float64),
                                     res, res)
    return scene.cast_rays(rays)["t_hit"].numpy()


def backproject(depth, rot, elev, res=None):
    """Carte de profondeur -> nuage de points 3D dans le repère monde."""
    res = res or depth.shape[0]
    eye, T = camera_pose(rot, elev)
    f = FOCAL_MM / SENSOR_MM * res
    v, u = np.nonzero(np.isfinite(depth))
    d = depth[v, u]
    # Direction du rayon en repère caméra, puis retour au monde.
    dirs_cam = np.stack([(u - res / 2) / f, (v - res / 2) / f, np.ones_like(d)], 1)
    dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True)
    Rw = T[:3, :3].T
    dirs_w = dirs_cam @ Rw.T
    return eye + dirs_w * d[:, None]


def project_points(pts, rot, elev, res=256):
    """Nuage 3D -> carte de profondeur, avec tampon de profondeur. inf = trou."""
    _, T = camera_pose(rot, elev)
    f = FOCAL_MM / SENSOR_MM * res
    cam = (T[:3, :3] @ pts.T).T + T[:3, 3]
    z = cam[:, 2]
    ok = z > 1e-6
    u = np.round(f * cam[ok, 0] / z[ok] + res / 2).astype(int)
    v = np.round(f * cam[ok, 1] / z[ok] + res / 2).astype(int)
    zz = z[ok]
    m = (u >= 0) & (u < res) & (v >= 0) & (v < res)
    out = np.full((res, res), np.inf)
    # Tampon de profondeur : on garde le point le plus proche par pixel.
    order = np.argsort(-zz[m])
    out[v[m][order], u[m][order]] = zz[m][order]
    return out


def depth_to_png(depth, near=2.4, far=4.8):
    """Image 8 bits au format des exemples amont : fond noir, blanc = proche.

    L'intervalle est **fixe** et non normalisé par image : deux vues d'un même objet
    doivent partager la même échelle, sinon la reprojection introduirait un facteur
    d'échelle invisible entre elles.
    """
    d = np.clip((far - depth) / (far - near), 0, 1)
    d[~np.isfinite(depth)] = 0.0
    return Image.fromarray((d * 255).astype(np.uint8)).convert("RGB")


def main():
    ap = argparse.ArgumentParser(description="Profondeur au rig et reprojection")
    ap.add_argument("--data_dir", default="data/training_ycb")
    ap.add_argument("--out_dir", default="data/depth_rig")
    ap.add_argument("--objects", nargs="+", required=True)
    ap.add_argument("--views", type=int, nargs="+", default=[3, 6, 10, 26])
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--upsample", type=int, default=4,
                    help="Facteur de suréchantillonnage de la vue source.")
    args = ap.parse_args()

    rig = load_rig()
    for obj in args.objects:
        mesh = normalize_mesh(o3d.io.read_triangle_mesh(
            str(Path(args.data_dir) / obj / "mesh.obj")))
        d = Path(args.out_dir) / obj
        (d / "reelle").mkdir(parents=True, exist_ok=True)
        (d / "reprojetee").mkdir(parents=True, exist_ok=True)

        # Vues réelles, pour référence et pour comparaison.
        depths = {}
        for v in args.views:
            rot, el = rig[v]
            depths[v] = depth_at_view(mesh, rot, el, args.res)
            depth_to_png(depths[v]).save(d / "reelle" / f"{v:03d}.png")

        # Reprojections depuis la PREMIÈRE vue seulement : c'est le régime visé,
        # une seule observation réelle qui alimente les quatre entrées.
        #
        # La source est échantillonnée plus finement que la destination : un pixel
        # source ne donne qu'un pixel destination, et la surface s'inclinant, les
        # points s'écartent — une reprojection à résolution égale sort mouchetée là où
        # les vues d'entraînement sont pleines. Suréchantillonner la source d'un
        # facteur k densifie d'un facteur k² sans rien inventer : ce sont les mêmes
        # rayons, tirés plus serré sur la même surface visible.
        src = args.views[0]
        depth_hi = depth_at_view(mesh, *rig[src], args.res * args.upsample)
        pts = backproject(depth_hi, *rig[src], args.res * args.upsample)
        for v in args.views:
            rep = depths[src] if v == src else project_points(pts, *rig[v], args.res)
            depth_to_png(rep).save(d / "reprojetee" / f"{v:03d}.png")
            trous = float(np.mean(~np.isfinite(rep)))
            plein = float(np.mean(~np.isfinite(depths[v])))
            print(f"  {obj} vue {v:03d} : {(1-trous)*100:5.1f} % de pixels remplis "
                  f"(vue réelle : {(1-plein)*100:5.1f} %)")
        print(f"  → {d}")


if __name__ == "__main__":
    main()
