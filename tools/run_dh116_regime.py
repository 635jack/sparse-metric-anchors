#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/run_dh116_regime.py

Le guidage par contact tient-il au régime de cardinalité que la DH116 peut produire ?

La question. Toutes les campagnes du dépôt tournent à 128 points de contact, la plus
pauvre à 32, et le balayage de densité (32 / 128 / 512 → +0,23 / +1,21 / +0,07) dit que
le **nombre** de points n'est pas l'axe qui décide. Mais il ne descend jamais au régime
de la main réelle : neuf zones tactiles exploitables, dont une saisie n'en excite que
trois à cinq. Un point par plaque, pas seize. Ce régime n'a jamais été mesuré, et c'est
exactement là que se pose la question « la fusion avec la main réelle mène-t-elle
quelque part ».

Ce qui est comparé, à plaques identiques (mêmes sites que `palpation8_128`, dont le
plafond oracle vaut **+6,77** sur ces cinq objets) :

    dh8       8 contacts, position seule          — le régime nominal de la main saine
    dh8n      8 contacts, position + normale      — le levier : plus d'information par
                                                     contact au lieu de plus de contacts
    dh8n_bruit 8 contacts, normale, placement bruité — ce que la cinématique donnera
    dh4       4 contacts, position seule          — ce qu'une saisie excite vraiment

Placement **oracle** dans les trois premières conditions, bruit compris : c'est une
mesure de plafond, pas un système. La discipline du dépôt s'applique — un chiffre
oracle se rapporte comme borne supérieure, avec mention de ce qui a été donné
gratuitement. Ici : la pose exacte des contacts, que `pose_from_silhouette` ne sait
retrouver qu'à 0,111-0,501 près.

La notation est importée de `run_guidance_campaign`, pas réécrite : comparer +6,77
obtenu d'une façon à un nombre obtenu d'une autre ne vaudrait rien.

Usage :
    python tools/make_contacts_dh116.py
    python tools/run_dh116_regime.py --out dh116_regime.json
"""

import argparse
import json
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyze_occlusion import split_visible_occluded, N_SAMPLE
from eval_fusion import load_and_normalize_mesh
from guided_sampling import load_model, image_condition, make_sampler
from run_guidance_campaign import oracle_frame, score
from make_contacts import TEST_OBJECTS

# Mesuré par `tools/probe_field_sign.py` : 8/8 des contacts donnent
# champ(p+eps*n) > champ(p-eps*n). L'extérieur est positif.
FIELD_SIGN = +1.0

# (nom, k sites, normales, sigma position, sigma angulaire en degrés)
CONDITIONS = [
    ("dh8",        8, False, 0.0,  0.0),
    ("dh8n",       8, True,  0.0,  0.0),
    ("dh8n_bruit", 8, True,  0.03, 15.0),
    ("dh4",        4, False, 0.0,  0.0),
]


def perturb(pts, nrm, sigma, sigma_deg, rng):
    """Bruit de placement : un capteur ne sait pas où son doigt a touché.

    Le repère est normalisé — plus grande étendue de l'objet ramenée à 2,0 — donc
    sigma 0,03 vaut environ 3 mm sur un objet de 20 cm. La normale est perturbée en
    même temps que la position : l'erreur vient de la cinématique du doigt, qui se
    trompe sur les deux à la fois. Le roulement du contact sur la pulpe donne
    facilement une quinzaine de degrés.
    """
    if sigma <= 0 and sigma_deg <= 0:
        return pts, nrm, 0.0, 0.0
    p = pts + rng.normal(scale=sigma, size=pts.shape) if sigma > 0 else pts.copy()
    n = nrm.copy()
    if sigma_deg > 0:
        for i in range(len(n)):
            axis = rng.normal(size=3)
            axis -= axis.dot(n[i]) * n[i]           # rotation autour d'un axe
            na = np.linalg.norm(axis)                # perpendiculaire à la normale
            if na < 1e-9:
                continue
            axis /= na
            a = np.deg2rad(rng.normal(scale=sigma_deg))
            n[i] = n[i] * np.cos(a) + np.cross(axis, n[i]) * np.sin(a)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
    d_pos = float(np.linalg.norm(p - pts, axis=1).mean())
    d_ang = float(np.degrees(np.arccos(np.clip((n * nrm).sum(1), -1, 1))).mean())
    return p, n, d_pos, d_ang


def main():
    ap = argparse.ArgumentParser(description="Guidage au régime de cardinalité DH116")
    ap.add_argument("--data_dir", default="data/training_ycb")
    ap.add_argument("--contacts_dir", default="data/contacts")
    ap.add_argument("--out_dir", default="out/dh116_regime")
    ap.add_argument("--out", default="dh116_regime.json")
    ap.add_argument("--objects", nargs="*", default=TEST_OBJECTS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--weight", type=float, default=0.05)
    ap.add_argument("--timesteps", type=int, default=20)
    ap.add_argument("--scale", type=float, default=1.5)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--lambda_rel", type=float, default=1.0)
    ap.add_argument("--conditions", nargs="*", default=None)
    args = ap.parse_args()

    conds = ([c for c in CONDITIONS if c[0] in args.conditions]
             if args.conditions else CONDITIONS)
    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    model, net, device = load_model(scale=args.scale, timesteps=args.timesteps)
    t_start = time.time()

    for obj in args.objects:
        gt_path, img_path = data_dir / obj / "mesh.obj", data_dir / obj / "image.png"
        if not gt_path.exists() or not img_path.exists():
            print(f"[dh116] {obj} : entrées manquantes, ignoré", flush=True)
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

        sets = {}
        for k in sorted({c[1] for c in conds}):
            f = Path(args.contacts_dir) / obj / f"dh116_{k}.npz"
            if f.exists():
                z = np.load(f)
                sets[k] = (z["points"].astype(float), z["normals"].astype(float))
            else:
                print(f"  ! dh116_{k}.npz absent", flush=True)

        cond = image_condition(model, net, img_path, device)
        sample = make_sampler(model, net, cond, obj, args.scale, device)
        dummy = torch.zeros(1, 3, device=device)
        results.setdefault(obj, {})

        for seed in args.seeds:
          try:
            t0 = time.time()
            ref_obj, _ = sample(0.0, seed, dummy,
                                Path(args.out_dir) / obj / f"w0_s{seed}")
            base = score(gt_mesh, gt_pts, vis, thr, ref_obj)
            results[obj][f"baseline_s{seed}"] = base
            print(f"  témoin s{seed} : F@2 {base['f2']:.2f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            O = oracle_frame(gt_mesh, ref_obj)
            R = O[:3, :3]

            for name, k, use_n, sig, sig_d in conds:
                if k not in sets:
                    continue
                pts0, nrm0 = sets[k]
                rng = np.random.default_rng(1000 + seed)
                pts0, nrm0, d_pos, d_ang = perturb(pts0, nrm0, sig, sig_d, rng)

                pts = (R @ pts0.T).T + O[:3, 3]
                # Normale : partie linéaire puis renormalisation. Le recalage est une
                # similitude (ICP avec échelle uniforme), donc l'inverse-transposée
                # est proportionnelle à la rotation — renormaliser suffit.
                nrm = (R @ nrm0.T).T
                nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)

                hors = float((np.abs(pts) > 1).any(1).mean())
                if hors > 0:
                    print(f"  ! {name} : {hors*100:.0f} % des contacts hors grille",
                          flush=True)

                pt_t = torch.from_numpy(pts).float().to(device)
                nr_t = (torch.from_numpy(nrm).float().to(device) if use_n else None)

                t0 = time.time()
                pred_obj, trace = sample(
                    args.weight, seed, pt_t,
                    Path(args.out_dir) / obj / f"{name}_s{seed}",
                    normals=nr_t, eps=args.eps, lambda_rel=args.lambda_rel,
                    field_sign=FIELD_SIGN)
                s = score(gt_mesh, gt_pts, vis, thr, pred_obj)
                s.update(n_contacts=int(len(pts)), sites=k, normals=bool(use_n),
                         sigma=sig, sigma_deg=sig_d,
                         bruit_pos=d_pos, bruit_ang=d_ang,
                         sdf_start=trace[0][1] if trace else None,
                         sdf_end=trace[-1][1] if trace else None,
                         cos_start=trace[0][2] if trace else None,
                         cos_end=trace[-1][2] if trace else None)
                results[obj][f"{name}_s{seed}"] = s
                extra = (f"  cos {s['cos_start']:.3f}->{s['cos_end']:.3f}"
                         if s.get("cos_start") is not None else "")
                print(f"  {name:11s} s{seed} : F@2 {s['f2']:6.2f} "
                      f"({s['f2']-base['f2']:+6.2f})  "
                      f"|SDF| {s['sdf_start']:.4f}->{s['sdf_end']:.4f}{extra}  "
                      f"[{time.time()-t0:.0f}s]", flush=True)
                out_path.write_text(json.dumps(results, indent=2))

          except Exception as e:
            print(f"  !! {obj} graine {seed} abandonné : {type(e).__name__} {e}",
                  flush=True)
            results[obj][f"echec_s{seed}"] = f"{type(e).__name__}: {e}"
            out_path.write_text(json.dumps(results, indent=2))

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nTerminé en {(time.time()-t_start)/60:.0f} min → {out_path}", flush=True)

    # Synthèse immédiate : test des signes, comme la campagne à 42 objets.
    print(f"\n{'condition':12s} {'n':>3s} {'moy':>7s} {'med':>7s} {'positifs':>9s} "
          f"{'p(signe)':>9s}")
    for name, k, use_n, sig, sig_d in conds:
        g = [results[o][f"{name}_s{s}"]["f2"] - results[o][f"baseline_s{s}"]["f2"]
             for o in results for s in args.seeds
             if f"{name}_s{s}" in results[o] and f"baseline_s{s}" in results[o]]
        if not g:
            continue
        n, pos = len(g), sum(1 for x in g if x > 0)
        p = sum(comb(n, i) for i in range(pos, n + 1)) / 2 ** n
        print(f"{name:12s} {n:3d} {np.mean(g):+7.2f} {np.median(g):+7.2f} "
              f"{pos:4d}/{n:<4d} {p:9.3f}")


if __name__ == "__main__":
    main()
