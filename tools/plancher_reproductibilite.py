#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/plancher_reproductibilite.py

Combien vaut un écart de F-Score entre deux exécutions **strictement identiques** ?

Pourquoi cette mesure. Tout le dépôt compare une génération guidée à un témoin de
**même graine**, en supposant que la seule différence entre les deux est le guidage.
Observé le 25 août : relancer deux fois la même commande, même objet, même graine,
mêmes contacts, donne 49,10 puis 48,87 sur le témoin et 45,18 puis 48,83 sur le guidé.
Les traces divergent dès le premier pas. `torch.manual_seed` n'y peut rien : les noyaux
Metal ne garantissent pas un ordre de réduction constant, et la diffusion amplifie
l'écart sur vingt pas.

Tant que ce plancher n'est pas chiffré, aucune taille d'effet du projet n'est
interprétable — ni les +0,93 des normales, ni le +3,21 de la campagne à 42 objets, dont
le « bruit apparié σ 2,3 à 10,7 » mélange variation de graine et non-déterminisme.

Ce qui est mesuré : N répétitions du couple (témoin, guidé), tout fixé. On publie
l'écart-type du témoin seul, du guidé seul, et surtout **de leur différence appariée**,
qui est la grandeur que les campagnes rapportent.

Usage :
    python tools/plancher_reproductibilite.py --contacts <strategies.npz> --repeats 8
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyze_occlusion import split_visible_occluded
from eval_fusion import load_and_normalize_mesh
from guided_sampling import load_model, image_condition, make_sampler
from run_guidance_campaign import oracle_frame, score
from run_dh116_regime import FIELD_SIGN
from run_strategies import batterie


def main():
    ap = argparse.ArgumentParser(description="Plancher de reproductibilité MPS")
    ap.add_argument("--data_dir", default="data/training_ycb")
    ap.add_argument("--contacts", required=True)
    ap.add_argument("--object", default="011_banana")
    ap.add_argument("--strategie", default="front_back")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--weight", type=float, default=0.05)
    ap.add_argument("--timesteps", type=int, default=20)
    ap.add_argument("--scale", type=float, default=1.5)
    ap.add_argument("--out", default="plancher.json")
    ap.add_argument("--out_dir", default="out/plancher")
    ap.add_argument("--min_batterie", type=int, default=30)
    ap.add_argument("--gel_pose", action="store_true",
                    help="Calcule le repère oracle UNE FOIS, sur le témoin de la "
                         "première répétition, puis le fige. Sans ce drapeau, il est "
                         "réestimé à chaque répétition depuis le témoin de cette "
                         "répétition-là — donc les contacts fournis au guidage bougent, "
                         "et l'écart mesuré mélange le non-déterminisme d'exécution "
                         "avec l'amplification de la variation du témoin. Le drapeau "
                         "sépare les deux.")
    args = ap.parse_args()

    npz = np.load(args.contacts, allow_pickle=True)
    d = Path(args.data_dir) / args.object
    gt_mesh, _ = load_and_normalize_mesh(d / "mesh.obj")
    gt_pts, vis = split_visible_occluded(gt_mesh)
    bb = gt_mesh.get_axis_aligned_bounding_box()
    thr = 0.02 * float(np.linalg.norm(bb.get_max_bound() - bb.get_min_bound()))

    k = f"{args.object}|{args.strategie}"
    pos0 = npz[f"{k}|pos"].astype(float)
    nrm0 = npz[f"{k}|nrm"].astype(float)

    pct, ac = batterie()
    print(f"[batterie] {pct}% {'secteur' if ac else 'batterie'}", flush=True)

    model, net, device = load_model(scale=args.scale, timesteps=args.timesteps)
    cond = image_condition(model, net, d / "image.png", device)
    sample = make_sampler(model, net, cond, args.object, args.scale, device)
    dummy = torch.zeros(1, 3, device=device)

    out_path = Path(args.out)
    res = json.loads(out_path.read_text()) if out_path.exists() else []
    O_fige = None
    t0 = time.time()

    for r in range(len(res), args.repeats):
        pct, ac = batterie()
        if not ac and pct is not None and pct < args.min_batterie:
            print(f"\n[batterie] {pct}% — arrêt propre à {r} répétitions", flush=True)
            break

        # Même graine et même appel. ATTENTION : cela ne suffit pas à rendre les
        # répétitions identiques. Sans --gel_pose, `oracle_frame` est réestimé sur le
        # témoin de CETTE répétition, qui varie lui-même ; les contacts fournis au
        # guidage bougent donc d'une répétition à l'autre.
        ref, _ = sample(0.0, args.seed, dummy, Path(args.out_dir) / f"r{r}_w0")
        b = score(gt_mesh, gt_pts, vis, thr, ref)

        if args.gel_pose and O_fige is not None:
            O = O_fige
        else:
            O = oracle_frame(gt_mesh, ref)
            if args.gel_pose:
                O_fige = O
        R = O[:3, :3]
        pts = (R @ pos0.T).T + O[:3, 3]
        nrm = (R @ nrm0.T).T
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)

        pred, _ = sample(args.weight, args.seed,
                         torch.from_numpy(pts).float().to(device),
                         Path(args.out_dir) / f"r{r}_g",
                         normals=torch.from_numpy(nrm).float().to(device),
                         field_sign=FIELD_SIGN)
        g = score(gt_mesh, gt_pts, vis, thr, pred)

        res.append({"temoin": b["f2"], "guide": g["f2"],
                    "diff": g["f2"] - b["f2"], "batt": pct})
        out_path.write_text(json.dumps(res, indent=2))
        print(f"  r{r} : témoin {b['f2']:6.2f}  guidé {g['f2']:6.2f}  "
              f"écart {g['f2']-b['f2']:+6.2f}  [batt {pct}%]", flush=True)

    if len(res) < 2:
        print("pas assez de répétitions")
        return

    t = np.array([x["temoin"] for x in res])
    g = np.array([x["guide"] for x in res])
    dd = np.array([x["diff"] for x in res])
    print(f"\n=== {len(res)} exécutions strictement identiques, "
          f"{(time.time()-t0)/60:.0f} min ===")
    print(f"{'':10s} {'moyenne':>9s} {'écart-type':>11s} {'étendue':>9s}")
    for nom, v in [("témoin", t), ("guidé", g), ("différence", dd)]:
        print(f"{nom:10s} {v.mean():9.2f} {v.std(ddof=1):11.2f} "
              f"{v.max()-v.min():9.2f}")
    print(f"\nPlancher : un effet apparié doit dépasser ~{2*dd.std(ddof=1):.2f} points "
          f"sur un cas unique pour être distinguable du bruit d'exécution.")
    print(f"Sur une moyenne de 15 cas, l'erreur type vaut "
          f"{dd.std(ddof=1)/np.sqrt(15):.2f}.")


if __name__ == "__main__":
    main()
