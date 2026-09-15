#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/run_guidance_campaign.py

Campagne de guidage par contact : où le toucher aide, et combien il en faut.

Elle répond à deux questions d'un seul run, en mutualisant le chargement du modèle
et la passe non guidée (témoin apparié, un par objet et par graine).

  Axe A — visibilité. À cardinalité égale (128 points), on guide avec des contacts
  pris uniquement sur la face occultée, uniquement sur la face visible, ou partout.
  C'est le test de la thèse du stage : si le gain se concentre sur la face cachée
  quand les contacts y sont, le toucher apporte bien ce que la vision ne voit pas.
  Les campagnes précédentes utilisaient `points_128.pt`, échantillonné sur toute la
  surface — elles ne pouvaient donc pas trancher, la contrainte portant autant sur
  ce que la caméra voit déjà que sur le reste.

  Axe C — densité. Combien de points de contact faut-il ? 32 / 128 / 512 sur la
  condition occultée, la seule opérationnelle pour un vrai capteur tactile.

Chaque prédiction est notée après le recalage à 24 initialisations de rotation
(`eval_fusion.align_prediction_to_gt`) : un ICP parti de l'identité échoue sur les
objets dont le repère de sortie diffère par un échange d'axes, et sous-note alors
la prédiction de plus de 20 points de F@2 % (mesuré, `tools/check_alignment.py`).

Usage :
    python tools/make_contacts.py
    python tools/run_guidance_campaign.py --out guidance_campaign.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyze_occlusion import split_visible_occluded, f_score, N_SAMPLE
from eval_fusion import (align_prediction_to_gt, compute_chamfer_and_fscore,
                         load_and_normalize_mesh)
from guided_sampling import (load_model, image_condition, make_sampler,
                             register_contacts)

# (condition, cardinalité). Le croisement complet des deux axes serait du gâchis :
# la visibilité se tranche à cardinalité fixée, la densité sur une seule condition.
CONTACT_SETS = [("occluded", 128), ("visible", 128), ("all", 128),
                ("occluded", 32), ("occluded", 512)]


def oracle_frame(gt_mesh, ref_path):
    """Transformation qui envoie le repère du jeu de données vers celui du modèle.

    Elle est obtenue en recalant la première passe sur le maillage de **référence**,
    puis en inversant — donc en utilisant une information qu'un système réel n'a pas.
    C'est délibéré : elle sert à mesurer le plafond du guidage une fois le problème
    de repère supposé résolu, pas à faire tourner un système.

    Le recalage habituel (`register_contacts`) place les contacts à une tolérance de
    F-Score entière de la vraie surface (mesuré : ratio 0,91 à 1,43 selon l'objet),
    parce qu'il les aligne sur une prédiction qui porte déjà l'erreur de forme qu'on
    cherche à corriger. Tant que ce biais n'est pas retiré, un guidage même parfait
    tire la surface vers de mauvais points, et on ne peut pas savoir si la technique
    vaut quelque chose.
    """
    N = _normalization(ref_path)
    pred_mesh, _ = load_and_normalize_mesh(ref_path)
    _, info = align_prediction_to_gt(gt_mesh, pred_mesh, return_info=True)
    return np.linalg.inv(info["transform"] @ N)      # référence -> repère brut


def _normalization(ref_path):
    """N : repère brut du modèle -> repère normalisé, celui que voit l'évaluation."""
    raw = o3d.io.read_triangle_mesh(str(ref_path))
    bb = raw.get_axis_aligned_bounding_box()
    c, ext = bb.get_center(), float(np.max(bb.get_extent()))
    N = np.eye(4)
    N[:3, :3] = (2.0 / ext) * np.eye(3)
    N[:3, 3] = -(2.0 / ext) * c
    return N


def combined_placement(ref_path, image_path, contacts, n_rot=2000):
    """Comme `silhouette_placement`, mais les contacts départagent les poses retenues.

    Le placement dépend alors du jeu de contacts, ce qui est le comportement d'un
    système réel : on tranche avec les mesures dont on dispose.
    """
    from pose_from_silhouette import combined_pose, image_mask

    pred, _ = load_and_normalize_mesh(ref_path)
    A, iou_best, resid = combined_pose(pred, image_mask(image_path), contacts,
                                       n_rot=n_rot)
    return np.linalg.inv(A @ _normalization(ref_path)), iou_best, resid


def silhouette_placement(ref_path, image_path, n_rot=2000):
    """Transformation jeu de données -> repère du modèle, estimée par l'image.

    Contrairement à `oracle_frame`, elle n'utilise **aucune vérité terrain** : la pose
    est celle dont la silhouette projetée recouvre le mieux l'image de
    conditionnement, que le modèle a déjà reçue. C'est donc un placement qu'un
    système réel peut produire.

    Mesuré contre le placement oracle : écart médian 0,183 sur les objets sans
    symétrie de révolution, contre 0,689 pour le recalage ICP en deux passes. Sur les
    quasi-solides de révolution l'écart reste grand, mais il porte sur l'orbite de
    symétrie, où il laisse la contrainte valide.
    """
    from pose_from_silhouette import global_pose, image_mask

    pred, _ = load_and_normalize_mesh(ref_path)
    A, iou_best, iou_second = global_pose(pred, image_mask(image_path), n_rot=n_rot)
    return np.linalg.inv(A @ _normalization(ref_path)), iou_best, iou_second


def score(gt_mesh, gt_pts, vis, thr, pred_path):
    """Métriques globales et par région de visibilité, après recalage."""
    pred_mesh, _ = load_and_normalize_mesh(pred_path)
    aligned = align_prediction_to_gt(gt_mesh, pred_mesh)
    m = compute_chamfer_and_fscore(gt_mesh, aligned)
    ppcd = aligned.sample_points_uniformly(number_of_points=N_SAMPLE)
    return {
        "f2": m["f_scores"]["2.0%"]["f_score"],
        "f1": m["f_scores"]["1.0%"]["f_score"],
        "chamfer": m["chamfer_symmetric"],
        "num_components": m["num_components"],
        "largest_component_ratio": m["largest_component_ratio"],
        "is_surface": m["is_surface"],
        "recall_visible": f_score(gt_pts[vis], ppcd, thr),
        "recall_occlude": f_score(gt_pts[~vis], ppcd, thr),
    }


def _save(path, results):
    """Écriture atomique : une campagne arrêtée en pleine écriture ne doit pas laisser un
    JSON tronqué, que la reprise ne saurait plus relire."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(results, indent=2))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description="Campagne guidage : visibilité × densité")
    ap.add_argument("--data_dir", default="data/training_ycb")
    ap.add_argument("--contacts_dir", default="data/contacts")
    ap.add_argument("--out_dir", default="out/guidance_campaign")
    ap.add_argument("--out", default="guidance_campaign.json")
    ap.add_argument("--objects", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--weight", type=float, default=0.05)
    ap.add_argument("--timesteps", type=int, default=20)
    ap.add_argument("--scale", type=float, default=1.5)
    ap.add_argument("--sets", nargs="*", default=None,
                    help="Sous-ensemble de CONTACT_SETS, au format condition:n.")
    ap.add_argument("--oracle_frame", action="store_true",
                    help="Place les contacts par le maillage de référence au lieu de "
                         "la première passe. Mesure de plafond, pas un système.")
    ap.add_argument("--combined_frame", action="store_true",
                    help="Silhouette pour restreindre les poses, résidu des contacts "
                         "pour trancher. Sans vérité terrain.")
    ap.add_argument("--silhouette_frame", action="store_true",
                    help="Place les contacts par la pose dont la silhouette recouvre "
                         "l'image de conditionnement. Sans vérité terrain.")
    ap.add_argument("--keep_meshes", action="store_true", default=True,
                    help="Les maillages sont conservés : la campagne précédente ne "
                         "les avait pas gardés, ce qui a imposé de tout régénérer "
                         "pour poser une question nouvelle sur les mêmes tirages.")
    ap.add_argument("--baseline_from", default=None,
                    help="Dossier de sortie d'une campagne dont on réutilise les passes "
                         "non guidées (même objet, même graine) au lieu de les régénérer : "
                         "deux bras lancés l'un après l'autre partagent alors le même témoin.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    objects = args.objects or json.loads((data_dir / "split.json").read_text())["test"]
    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    sets = ([(c.split(":")[0], int(c.split(":")[1])) for c in args.sets]
            if args.sets else CONTACT_SETS)
    model, net, device = load_model(scale=args.scale, timesteps=args.timesteps)
    t_start = time.time()

    for obj in objects:
        gt_path = data_dir / obj / "mesh.obj"
        img_path = data_dir / obj / "image.png"
        if not gt_path.exists() or not img_path.exists():
            print(f"[campagne] {obj} : entrées manquantes, ignoré")
            continue

        gt_mesh, _ = load_and_normalize_mesh(gt_path)
        gt_pts, vis = split_visible_occluded(gt_mesh)
        bb = gt_mesh.get_axis_aligned_bounding_box()
        thr = 0.02 * float(np.linalg.norm(bb.get_max_bound() - bb.get_min_bound()))
        print(f"\n=== {obj} : {(~vis).sum()} points occultés, {vis.sum()} visibles")

        # Reprise : on saute ce qui est déjà noté. Sans ça, relancer après une
        # interruption recalcule tout depuis le début — une campagne de six heures
        # n'est pas relançable à l'identique, et celle du 9 août a été coupée en
        # cours de route.
        attendu = {f"baseline_s{s}" for s in args.seeds}
        attendu |= {f"{c}_{n}_s{s}" for c, n in sets for s in args.seeds}
        if attendu <= set(results.get(obj, {})):
            print(f"\n=== {obj} : déjà complet, sauté")
            continue

        cond = image_condition(model, net, img_path, device)
        sample = make_sampler(model, net, cond, obj, args.scale, device)
        dummy = torch.zeros(1, 3, device=device)
        results.setdefault(obj, {})

        for seed in args.seeds:
          # Une campagne à 42 objets dure plusieurs heures : un maillage dégénéré ou
          # un recalage qui échoue ne doit pas emporter tout le reste. On note
          # l'objet comme raté et on continue — les résultats sont écrits au fur et
          # à mesure, donc rien de ce qui précède n'est perdu.
          try:
            # Passe 1 sans guidage : témoin apparié, et repère canonique dans lequel
            # les contacts doivent être exprimés. Mutualisée entre les 5 jeux.
            t0 = time.time()
            reuse = (Path(args.baseline_from) / obj / f"w0_s{seed}" / f"{obj}.obj"
                     if args.baseline_from else None)
            if reuse is not None and reuse.exists():
                ref_obj = reuse                    # le témoin de l'autre bras
            else:
                if reuse is not None:
                    print(f"  ! témoin {reuse} absent : régénéré", flush=True)
                ref_obj, _ = sample(0.0, seed, dummy,
                                    Path(args.out_dir) / obj / f"w0_s{seed}")
            base = score(gt_mesh, gt_pts, vis, thr, ref_obj)
            results[obj][f"baseline_s{seed}"] = base
            print(f"  témoin s{seed} : F@2 {base['f2']:.2f} "
                  f"(vu {base['recall_visible']:.1f} / caché "
                  f"{base['recall_occlude']:.1f})  [{time.time()-t0:.0f}s]", flush=True)
            ref_mesh = o3d.io.read_triangle_mesh(str(ref_obj))
            if args.oracle_frame:
                O = oracle_frame(gt_mesh, ref_obj)
            elif args.silhouette_frame:
                O, i1, i2 = silhouette_placement(ref_obj, img_path)
                print(f"  pose par silhouette : IoU {i1:.3f} (2e {i2:.3f})", flush=True)
            else:
                O = None

            for condition, n in sets:
                cpath = Path(args.contacts_dir) / obj / f"{condition}_{n}.pt"
                if not cpath.exists():
                    print(f"  ! {condition}_{n} absent, ignoré")
                    continue
                contacts = torch.load(cpath, map_location="cpu", weights_only=True)
                if contacts.dim() == 3:
                    contacts = contacts[0]

                if args.combined_frame:
                    # Le placement dépend du jeu de contacts : il est réestimé pour
                    # chacun, contrairement aux poses oracle et silhouette.
                    O, i1, resid = combined_placement(ref_obj, img_path,
                                                      contacts.numpy())
                    print(f"  pose combinée {condition}_{n} : IoU {i1:.3f}, "
                          f"résidu contacts {resid:.4f}", flush=True)

                if O is not None:
                    c_np = contacts.numpy()
                    pts_reg = (O[:3, :3] @ c_np.T).T + O[:3, 3]
                    rmse = float(np.abs(pts_reg).max())   # doit rester sous 1 :
                    if rmse > 1.0:                        # au-delà, la grille SDF
                        print(f"  ! {condition}_{n} : {(np.abs(pts_reg) > 1).any(1).mean()*100:.0f} %"
                              f" des contacts hors de la grille [-1,1], guidage tronqué")
                else:
                    # Le recalage dépend du jeu de contacts : il est refait pour
                    # chacun, mais il ne coûte que de l'ICP sur CPU.
                    pts_reg, rmse = register_contacts(contacts.numpy(), ref_mesh)
                pts_t = torch.from_numpy(pts_reg).float().to(device)

                t0 = time.time()
                pred_obj, trace = sample(args.weight, seed, pts_t,
                                         Path(args.out_dir) / obj /
                                         f"{condition}{n}_w{args.weight:g}_s{seed}")
                s = score(gt_mesh, gt_pts, vis, thr, pred_obj)
                s.update(register_rmse=rmse,
                         sdf_start=trace[0][1] if trace else None,
                         sdf_end=trace[-1][1] if trace else None)
                results[obj][f"{condition}_{n}_s{seed}"] = s
                print(f"  {condition:8s} n={n:3d} s{seed} : "
                      f"F@2 {s['f2']:6.2f} ({s['f2']-base['f2']:+6.2f})  "
                      f"vu {s['recall_visible']:5.1f} "
                      f"({s['recall_visible']-base['recall_visible']:+5.1f})  "
                      f"caché {s['recall_occlude']:5.1f} "
                      f"({s['recall_occlude']-base['recall_occlude']:+5.1f})  "
                      f"|SDF| {s['sdf_start']:.4f}->{s['sdf_end']:.4f}  "
                      f"[{time.time()-t0:.0f}s]", flush=True)

                _save(out_path, results)

          except Exception as e:
            print(f"  !! {obj} graine {seed} abandonné : {type(e).__name__} {e}",
                  flush=True)
            results[obj][f"echec_s{seed}"] = f"{type(e).__name__}: {e}"
            _save(out_path, results)

    _save(out_path, results)
    print(f"\nTerminé en {(time.time()-t_start)/60:.0f} min → {out_path}")


if __name__ == "__main__":
    main()
