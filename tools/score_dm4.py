#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/score_dm4.py

Note la campagne DM4 avec **exactement** le protocole de la campagne mono-vue.

Pourquoi un script séparé. `guided_sampling_dm4.py` ne fait que générer ; la campagne
mono-vue, elle, notait au fil de l'eau dans `run_guidance_campaign.py`. Si on notait
autrement ici, on comparerait le +3,21 obtenu d'une façon à un nombre obtenu d'une
autre, et la comparaison ne vaudrait rien. Les fonctions de mesure sont donc importées
telles quelles, pas réécrites.

Ce qui est mesuré, pour chaque paire (témoin, guidé) de même graine :
  - F-Score @2 % et @1 % — le second parce que le résultat mono-vue n'y survit pas
    (+1,57, p = 0,42), et qu'il faut savoir si celui-ci y survit ;
  - distance de Chamfer, sur laquelle le résultat mono-vue est le plus solide ;
  - connectivité, pour écarter les gains obtenus par fragmentation ;
  - rappel séparé sur face visible et face occultée.

Usage :
    python tools/score_dm4.py --mesh_dir out/dm4_campagne --out dm4_resultats.json
"""

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyze_occlusion import split_visible_occluded, f_score, N_SAMPLE
from eval_fusion import (align_prediction_to_gt, compute_chamfer_and_fscore,
                         load_and_normalize_mesh)


def noter(gt_mesh, gt_pts, vis, thr, chemin):
    """Mêmes mesures que `run_guidance_campaign.score`, sur le même recalage."""
    pred, _ = load_and_normalize_mesh(chemin)
    aligne = align_prediction_to_gt(gt_mesh, pred)
    m = compute_chamfer_and_fscore(gt_mesh, aligne)
    pcd = aligne.sample_points_uniformly(number_of_points=N_SAMPLE)
    return {
        "f2": m["f_scores"]["2.0%"]["f_score"],
        "f1": m["f_scores"]["1.0%"]["f_score"],
        "chamfer": m["chamfer_symmetric"],
        "num_components": m["num_components"],
        "largest_component_ratio": m["largest_component_ratio"],
        "recall_visible": f_score(gt_pts[vis], pcd, thr),
        "recall_occlude": f_score(gt_pts[~vis], pcd, thr),
    }


def test_des_signes(v):
    v = np.asarray(v)
    n, k = len(v), int((v > 0).sum())
    if n == 0:
        return float("nan"), 0, 0
    tail = min(k, n - k)
    return min(2 * sum(comb(n, i) for i in range(tail + 1)) / 2 ** n, 1.0), k, n


def main():
    ap = argparse.ArgumentParser(description="Notation de la campagne DM4")
    ap.add_argument("--data_dir", default="data/training_ycb")
    ap.add_argument("--mesh_dir", default="out/dm4_campagne")
    ap.add_argument("--weight", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="dm4_resultats.json")
    args = ap.parse_args()

    racine = Path(args.mesh_dir)
    objets = sorted(p.name for p in racine.iterdir()
                    if p.is_dir() and (p / f"w0_s{args.seeds[0]}").exists())
    resultats = {}

    for obj in objets:
        gt, _ = load_and_normalize_mesh(Path(args.data_dir) / obj / "mesh.obj")
        gt_pts, vis = split_visible_occluded(gt)
        bb = gt.get_axis_aligned_bounding_box()
        thr = 0.02 * float(np.linalg.norm(bb.get_max_bound() - bb.get_min_bound()))
        resultats[obj] = {}
        for s in args.seeds:
            for tag, w in (("baseline", 0.0), ("guide", args.weight)):
                f = racine / obj / f"w{w:g}_s{s}" / f"{obj}.obj"
                if f.exists():
                    resultats[obj][f"{tag}_s{s}"] = noter(gt, gt_pts, vis, thr, f)
        g = [resultats[obj][f"guide_s{s}"]["f2"] - resultats[obj][f"baseline_s{s}"]["f2"]
             for s in args.seeds
             if f"guide_s{s}" in resultats[obj] and f"baseline_s{s}" in resultats[obj]]
        if g:
            print(f"  {obj:26s} {np.mean(g):+6.2f}", flush=True)
        Path(args.out).write_text(json.dumps(resultats, indent=2))

    def ecarts(champ, signe=1):
        return np.array([signe * (resultats[o][f"guide_s{s}"][champ]
                                  - resultats[o][f"baseline_s{s}"][champ])
                         for o in resultats for s in args.seeds
                         if f"guide_s{s}" in resultats[o]
                         and f"baseline_s{s}" in resultats[o]])

    print(f"\nCAMPAGNE DM4 — {len(resultats)} objets\n")
    print(f"{'mesure':26s} {'moyenne':>9s} {'médiane':>9s} {'positifs':>10s} {'p':>10s}")
    for lbl, champ, signe in (("F@2 %", "f2", 1), ("F@1 %", "f1", 1),
                              ("Chamfer (baisse)", "chamfer", -1),
                              ("rappel face visible", "recall_visible", 1),
                              ("rappel face cachée", "recall_occlude", 1)):
        v = ecarts(champ, signe)
        p, k, n = test_des_signes(v)
        print(f"{lbl:26s} {v.mean():+9.3f} {np.median(v):+9.3f} "
              f"{f'{k}/{n}':>10s} {p:10.2e}")

    par_objet = {o: np.mean([resultats[o][f'guide_s{s}']['f2']
                             - resultats[o][f'baseline_s{s}']['f2']
                             for s in args.seeds if f"guide_s{s}" in resultats[o]])
                 for o in resultats}
    pos = sum(1 for v in par_objet.values() if v > 0)
    print(f"\nobjets à gain moyen positif : {pos}/{len(par_objet)}")
    tri = sorted(par_objet.items(), key=lambda x: -x[1])
    print("  meilleurs : " + ", ".join(f"{o.split('_', 1)[1]} {v:+.1f}" for o, v in tri[:4]))
    print("  pires     : " + ", ".join(f"{o.split('_', 1)[1]} {v:+.1f}" for o, v in tri[-4:]))
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
