#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/make_contacts_strategies.py

Contacts du simulateur de préhension `grasp-dataset-gen`, sur les maillages YCB.

Pourquoi passer par le simulateur plutôt que par `make_contacts.palpation_contacts`.
Ce dernier place ses plaques par échantillonnage du plus lointain : aussi écartées que
possible. Une main fait l'inverse — les quatre longs doigts couvrent 80° d'arc, tous du
même côté. Mesuré sur ces cinq objets, l'écart minimal entre contacts vaut 0,058 à
0,344 pour le simulateur contre 0,446 à 1,281 pour les plaques. Or l'étalement est
précisément ce que le balayage de sites désigne comme décisif.

En plus des positions et des normales, on exporte **l'origine du rayon** de chaque
doigt. Le segment origine→contact est de l'espace libre mesuré : le doigt y est passé.
Le guidage n'impose que « la surface passe ici » et jamais « la surface ne passe pas
ici » ; ces origines sont ce qui manque pour lever l'asymétrie.

Deux environnements : le simulateur veut trimesh + rtree (venv du projet), la campagne
veut open3d et le code de WL-VisioTouch. D'où le NPZ comme point de passage, et
l'exécution de ce script avec le python du venv de `grasp-dataset-gen`.

Usage :
    ../grasp-dataset-gen/venv/bin/python tools/make_contacts_strategies.py \
        --out data/contacts/strategies.npz
"""

import argparse
import sys
import types
from pathlib import Path

import numpy as np


class _Stub(types.ModuleType):
    """pyrender n'est importé que pour `build_camera_pose`, qui est du numpy pur."""
    def __getattr__(self, name):
        return lambda *a, **k: None


sys.modules.setdefault("pyrender", _Stub("pyrender"))
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "grasp-dataset-gen"))

import trimesh
from grasp_dataset_gen.grasp_sampler import (
    GraspSampler, _mesh_principal_axis, _build_finger_dirs, FINGER_LABELS)
from grasp_dataset_gen.config import GraspConfig, CameraConfig, GraspStrategy

# La caméra de WL-VisioTouch/tools/make_contacts.py, pas celle du simulateur : c'est
# elle qui définit ce qui est occulté dans toute la campagne.
CAMERA = (2.2, -2.2, 1.8)
# ATTENTION : (2.2, -2.2, 1.8) est la position dans le monde Z-up de Blender, où l'import
# OBJ a d'abord tourné le maillage Y-up : (x, y, z) -> (x, -z, y). Placée telle quelle dans
# le repère du maillage, cette caméra est à 68° de celle qui a rendu les images.
# Les trois stratégies restent trois prises distinctes ; leur part cachée est recalculée.
# Les campagnes publiées ont tourné avec cette valeur ; elle est laissée pour qu'elles
# restent reproductibles. tools/visibility_render_camera.py mesure l'écart et recalcule
# la visibilité depuis la caméra du rendu.
TEST_OBJECTS = ["002_master_chef_can", "006_mustard_bottle", "011_banana",
                "025_mug", "035_power_drill"]
STRATEGIES = ["front_back", "left_right", "right_left"]


def normalize(mesh):
    """Centre à l'origine, plus grande étendue à 2,0 — la normalisation du dépôt."""
    m = mesh.copy()
    m.apply_translation(-m.bounds.mean(0))
    m.apply_scale(2.0 / max(m.extents))
    return m


def ray_origins(mesh, strategy, cam_config, config, contacts):
    """Reconstruit l'origine du rayon de chaque doigt, comme `_cast_ray_to_surface`.

    Le simulateur les calcule puis les jette. On refait le même calcul plutôt que de
    modifier le simulateur, pour que les deux restent indépendants.
    """
    center = mesh.centroid
    bsphere = float(np.max(np.linalg.norm(mesh.vertices - center, axis=1)))
    radius = bsphere * config.grasp_radius_factor
    axis = _mesh_principal_axis(mesh)
    dirs, _ = _build_finger_dirs(strategy, cam_config, axis)
    half = float(np.max(np.abs(np.dot(mesh.vertices - center, axis))))

    out = []
    for c in contacts:
        offset = -axis * half * 0.30 if c.finger == "palm" else np.zeros(3)
        out.append(center + offset - dirs[c.finger] * radius)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description="Contacts du simulateur de préhension")
    ap.add_argument("--data_dir", default="data/training_ycb")
    ap.add_argument("--objects", nargs="*", default=TEST_OBJECTS)
    ap.add_argument("--strategies", nargs="*", default=STRATEGIES)
    ap.add_argument("--out", default="data/contacts/strategies.npz")
    args = ap.parse_args()

    cam = CameraConfig(position=CAMERA, target=(0., 0., 0.), up=(0., 0., 1.))
    cfg = GraspConfig()
    sampler = GraspSampler(cfg, cam)
    cam_np = np.array(CAMERA)

    out = {}
    for obj in args.objects:
        p = Path(args.data_dir) / obj / "mesh.obj"
        if not p.exists():
            print(f"[strat] {obj} : mesh.obj introuvable, ignoré")
            continue
        mesh = normalize(trimesh.load(str(p), force="mesh", process=False))

        for name in args.strategies:
            strat = GraspStrategy(name)
            cs = sampler.sample(mesh, strat)
            if not cs:
                print(f"  ! {obj} {name} : aucun contact")
                continue
            pos = np.array([c.position for c in cs], np.float32)
            nrm = np.array([c.normal for c in cs], np.float32)
            org = ray_origins(mesh, strat, cam, cfg, cs).astype(np.float32)

            to_cam = cam_np - pos
            to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True)
            occl = int(((nrm * to_cam).sum(1) < 0).sum())

            k = f"{obj}|{name}"
            out[f"{k}|pos"], out[f"{k}|nrm"], out[f"{k}|org"] = pos, nrm, org
            out[f"{k}|fng"] = np.array([c.finger for c in cs])

            d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
            d[np.diag_indices(len(pos))] = np.inf
            print(f"  {obj:22s} {name:11s} {len(cs)} contacts  "
                  f"occultés {occl}/{len(cs)}  écart min {d.min():.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"\n→ {args.out}  ({len(out)//4} prises)")


if __name__ == "__main__":
    main()
