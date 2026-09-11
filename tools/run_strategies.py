#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/run_strategies.py

Deux questions d'une seule campagne, sur les contacts du simulateur de préhension.

**Axe 1 — la thèse du stage, testée avec une pose de main.** Les campagnes précédentes
séparaient les contacts par visibilité en partitionnant la surface : une construction,
pas une prise. Le simulateur (`grasp-dataset-gen`) produit au contraire des prises
réalisables, et la visibilité des contacts en découle au lieu d'être imposée. Mesuré
sur les cinq objets : `front_back` met 5 contacts sur 6 du côté occulté pour tous,
`right_left` tombe à 0 ou 1 sur la perceuse et la banane. Si le gain suit l'occultation
à cardinalité et objet identiques, la thèse tient sur un geste, pas sur une partition.

**Axe 2 — l'espace libre.** Le guidage n'impose que « la surface passe ici », jamais
« la surface ne passe pas ici », alors que le trajet parcouru par un doigt avant
l'impact est mesuré au même titre que le contact. `fb_libre` ajoute ce terme à
`fb`, tout le reste égal : c'est une comparaison appariée d'un terme de perte.

Placement **oracle** : mesure de plafond, pas un système. Voir la discipline du dépôt.

Usage :
    python tools/run_strategies.py --contacts <strategies.npz> --out strategies.json
"""

import argparse
import json
import subprocess
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyze_occlusion import split_visible_occluded
from eval_fusion import load_and_normalize_mesh
from guided_sampling import load_model, image_condition, make_sampler
from run_guidance_campaign import oracle_frame, score
from run_dh116_regime import FIELD_SIGN
from make_contacts import TEST_OBJECTS

# (nom, stratégie, espace libre)
CONDITIONS = [
    ("fb",       "front_back", False),
    ("lr",       "left_right", False),
    ("rl",       "right_left", False),
    ("fb_libre", "front_back", True),
]


def batterie():
    """(pourcentage, sur_secteur). Renvoie (None, True) si l'état est illisible."""
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True,
                             text=True, timeout=5).stdout
    except Exception:
        return None, True
    ac = "AC Power" in out
    pct = None
    for tok in out.replace(";", " ").split():
        if tok.endswith("%"):
            try:
                pct = int(tok[:-1]); break
            except ValueError:
                pass
    return pct, ac


def points_libres(origines, contacts, O, n=24, marge=0.06, bord=0.98):
    """Points d'espace libre le long des segments origine→contact, dans le repère modèle.

    Deux filtrages, tous deux nécessaires :

    - **marge** près du contact. La surface est là ; y imposer « pas de matière »
      combattrait la contrainte de position, qui est la seule qu'on sait exacte.
    - **bord** de la grille. `trilinear` borne ses coordonnées : un point hors de
      [-1,1] serait rabattu sur la face du cube et imposerait une contrainte à un
      endroit que le doigt n'a jamais traversé. Les origines de rayon sont à 1,2 fois
      le rayon de la sphère englobante, donc systématiquement dehors — sans ce filtre,
      la majorité des points seraient faux.
    """
    R, t3 = O[:3, :3], O[:3, 3]
    q = []
    for org, ct in zip(origines, contacts):
        seg = ct - org
        L = np.linalg.norm(seg)
        if L < 1e-6:
            continue
        ts = np.linspace(0.0, 1.0 - marge / L, n)
        q.append(org[None] + ts[:, None] * seg[None])
    if not q:
        return None
    q = np.concatenate(q, 0)
    q = (R @ q.T).T + t3
    return q[(np.abs(q) < bord).all(1)]


def main():
    ap = argparse.ArgumentParser(description="Stratégies de préhension × espace libre")
    ap.add_argument("--data_dir", default="data/training_ycb")
    ap.add_argument("--contacts", required=True, help="NPZ produit par gen_strategies")
    ap.add_argument("--out_dir", default="out/strategies")
    ap.add_argument("--out", default="strategies.json")
    ap.add_argument("--objects", nargs="*", default=TEST_OBJECTS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--weight", type=float, default=0.05)
    ap.add_argument("--timesteps", type=int, default=20)
    ap.add_argument("--scale", type=float, default=1.5)
    ap.add_argument("--lambda_free", type=float, default=1.0)
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="Sous-ensemble de CONDITIONS. `fb_libre` est mesurée inerte "
                         "(perte d'espace libre nulle du premier au dernier pas, "
                         "133 points) : la retirer economise un quart du calcul sans "
                         "rien perdre.")
    ap.add_argument("--min_batterie", type=int, default=35,
                    help="Sur batterie, s'arrête proprement sous ce pourcentage. "
                         "Les résultats sont écrits au fil de l'eau et la reprise "
                         "saute ce qui est déjà noté : on rebranche et on relance.")
    args = ap.parse_args()

    conds = ([c for c in CONDITIONS if c[0] in args.conditions]
             if args.conditions else CONDITIONS)

    npz = np.load(args.contacts, allow_pickle=True)
    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    pct, ac = batterie()
    print(f"[batterie] {pct}% {'sur secteur' if ac else 'sur batterie'} — "
          f"arrêt sous {args.min_batterie}%", flush=True)

    model, net, device = load_model(scale=args.scale, timesteps=args.timesteps)
    t_start = time.time()
    arrete = False

    for obj in args.objects:
        if arrete:
            break
        gt_path, img_path = data_dir / obj / "mesh.obj", data_dir / obj / "image.png"
        if not gt_path.exists() or not img_path.exists():
            continue

        gt_mesh, _ = load_and_normalize_mesh(gt_path)
        gt_pts, vis = split_visible_occluded(gt_mesh)
        bb = gt_mesh.get_axis_aligned_bounding_box()
        thr = 0.02 * float(np.linalg.norm(bb.get_max_bound() - bb.get_min_bound()))

        attendu = {f"baseline_s{s}" for s in args.seeds}
        attendu |= {f"{c[0]}_s{s}" for c in conds for s in args.seeds}
        if attendu <= set(results.get(obj, {})):
            print(f"\n=== {obj} : déjà complet, sauté", flush=True)
            continue
        print(f"\n=== {obj}", flush=True)

        cond = image_condition(model, net, img_path, device)
        sample = make_sampler(model, net, cond, obj, args.scale, device)
        dummy = torch.zeros(1, 3, device=device)
        results.setdefault(obj, {})

        for seed in args.seeds:
          pct, ac = batterie()
          if not ac and pct is not None and pct < args.min_batterie:
              print(f"\n[batterie] {pct}% sur batterie — arrêt propre. "
                    f"Rebrancher et relancer la même commande pour reprendre.",
                    flush=True)
              arrete = True
              break
          try:
            t0 = time.time()
            ref_obj, _ = sample(0.0, seed, dummy,
                                Path(args.out_dir) / obj / f"w0_s{seed}")
            base = score(gt_mesh, gt_pts, vis, thr, ref_obj)
            results[obj][f"baseline_s{seed}"] = base
            print(f"  témoin s{seed} : F@2 {base['f2']:.2f}  "
                  f"[{time.time()-t0:.0f}s, batt {pct}%]", flush=True)
            O = oracle_frame(gt_mesh, ref_obj)
            R = O[:3, :3]

            for name, strat, libre in conds:
                k = f"{obj}|{strat}"
                if f"{k}|pos" not in npz:
                    continue
                pos0, nrm0 = npz[f"{k}|pos"].astype(float), npz[f"{k}|nrm"].astype(float)
                org0 = npz[f"{k}|org"].astype(float)

                pts = (R @ pos0.T).T + O[:3, 3]
                nrm = (R @ nrm0.T).T
                nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)

                free_t = None
                n_free = 0
                if libre:
                    q = points_libres(org0, pos0, O)
                    if q is not None and len(q):
                        free_t = torch.from_numpy(q).float().to(device)
                        n_free = len(q)

                hors = float((np.abs(pts) > 1).any(1).mean())
                if hors > 0:
                    print(f"  ! {name} : {hors*100:.0f} % des contacts hors grille",
                          flush=True)

                t0 = time.time()
                pred_obj, trace = sample(
                    args.weight, seed, torch.from_numpy(pts).float().to(device),
                    Path(args.out_dir) / obj / f"{name}_s{seed}",
                    normals=torch.from_numpy(nrm).float().to(device),
                    field_sign=FIELD_SIGN, free_points=free_t,
                    lambda_free=args.lambda_free)
                s = score(gt_mesh, gt_pts, vis, thr, pred_obj)
                # Part des contacts sur la face occultee : la variable de l'axe 1.
                cam = np.array([2.2, -2.2, 1.8])
                tc = cam - pos0
                tc /= np.linalg.norm(tc, axis=1, keepdims=True)
                s.update(strategie=strat, espace_libre=bool(libre),
                         n_contacts=int(len(pts)), n_points_libres=n_free,
                         part_occultee=float(((nrm0 * tc).sum(1) < 0).mean()),
                         sdf_start=trace[0][1] if trace else None,
                         sdf_end=trace[-1][1] if trace else None,
                         libre_start=trace[0][3] if trace else None,
                         libre_end=trace[-1][3] if trace else None)
                results[obj][f"{name}_s{seed}"] = s
                extra = (f"  libre {s['libre_start']:.4f}->{s['libre_end']:.4f} "
                         f"({n_free} pts)" if s.get("libre_start") is not None else "")
                print(f"  {name:9s} s{seed} : F@2 {s['f2']:6.2f} "
                      f"({s['f2']-base['f2']:+6.2f})  occ {s['part_occultee']:.2f}  "
                      f"|SDF| {s['sdf_start']:.4f}->{s['sdf_end']:.4f}{extra}  "
                      f"[{time.time()-t0:.0f}s]", flush=True)
                out_path.write_text(json.dumps(results, indent=2))
          except Exception as e:
            print(f"  !! {obj} graine {seed} : {type(e).__name__} {e}", flush=True)
            results[obj][f"echec_s{seed}"] = f"{type(e).__name__}: {e}"
            out_path.write_text(json.dumps(results, indent=2))

    out_path.write_text(json.dumps(results, indent=2))
    pct, ac = batterie()
    print(f"\nTerminé en {(time.time()-t_start)/60:.0f} min → {out_path} "
          f"[batt {pct}%]", flush=True)

    def st(g):
        n = len(g); p = sum(1 for x in g if x > 0)
        return p, n, sum(comb(n, i) for i in range(p, n + 1)) / 2 ** n

    print(f"\n{'condition':10s} {'occ.':>5s} {'moy':>7s} {'med':>7s} {'positifs':>9s} {'p':>7s}")
    gains = {}
    for name, strat, libre in conds:
        g, occ = [], []
        for o in results:
            for s_ in args.seeds:
                if f"{name}_s{s_}" in results[o] and f"baseline_s{s_}" in results[o]:
                    g.append(results[o][f"{name}_s{s_}"]["f2"]
                             - results[o][f"baseline_s{s_}"]["f2"])
                    occ.append(results[o][f"{name}_s{s_}"]["part_occultee"])
        if not g:
            continue
        gains[name] = g
        p, n, pv = st(g)
        print(f"{name:10s} {np.mean(occ):5.2f} {np.mean(g):+7.2f} "
              f"{np.median(g):+7.2f} {p:4d}/{n:<4d} {pv:7.3f}")

    if "fb_libre" in gains and "fb" in gains:
        d = [a - b for a, b in zip(gains["fb_libre"], gains["fb"])]
        p, n, pv = st(d)
        print(f"\nespace libre (fb_libre - fb, apparié) : moy {np.mean(d):+.2f}  "
              f"med {np.median(d):+.2f}  {p}/{n} positifs  p={pv:.3f}")


if __name__ == "__main__":
    main()
