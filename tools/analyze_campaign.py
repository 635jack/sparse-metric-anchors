#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/analyze_campaign.py

Analyse finale de la campagne à 42 objets : tableaux et classification des objets.

Fonctionne sur le bras combiné seul (résultat principal, déjà acquis) et enrichit
l'analyse si le bras oracle est présent, même partiel.

Ce que le plafond apporte. Six objets perdent avec le placement réalisable, et sans
plafond on ne peut pas dire pourquoi. Avec, la lecture est mécanique :

    plafond ~ 0  -> le tactile n'apporte rien sur cet objet ; limite du problème
    plafond > 0  -> l'information était là, le placement a raté ; limite de méthode

C'est ce qui remplace la taxonomie par symétrie, réfutée par la campagne : l'orange
(+13,8) et la balle de baseball (−0,4) sont toutes deux quasi sphériques.

Usage :
    python tools/analyze_campaign.py
    python tools/analyze_campaign.py --markdown > paper/resultats_42.md
"""

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np


def sign_test(v):
    """p bilatéral du test des signes — sans hypothèse sur la loi des écarts.

    Choisi plutôt qu'un test de Student parce que la distribution des gains est
    fortement asymétrique (médiane +1,69 pour une moyenne +3,21) et que quelques
    objets dominent : une moyenne y est fragile, un compte de signes ne l'est pas.
    """
    v = np.asarray(v)
    n, k = len(v), int((v > 0).sum())
    if n == 0:
        return float("nan"), 0, 0
    tail = min(k, n - k)
    p = 2 * sum(comb(n, i) for i in range(0, tail + 1)) / 2 ** n
    return min(p, 1.0), k, n


def load_arm(path, tag="palpation8_128"):
    """Gains appariés par objet : guidé − témoin, même graine."""
    if not Path(path).exists():
        return {}
    d = json.load(open(path))
    out = {}
    for o in d:
        g = [d[o][f"{tag}_s{i}"]["f2"] - d[o][f"baseline_s{i}"]["f2"]
             for i in range(3)
             if f"{tag}_s{i}" in d[o] and f"baseline_s{i}" in d[o]]
        b = [d[o][f"baseline_s{i}"]["f2"] for i in range(3)
             if f"baseline_s{i}" in d[o]]
        if g:
            out[o] = {"gain": float(np.mean(g)), "gains": g,
                      "baseline": float(np.mean(b))}
    return out


def classify(gain, ceiling, noise=2.0):
    """Range un objet selon ce que le plafond révèle de la cause de son échec."""
    if ceiling is None:
        return "plafond inconnu"
    if ceiling < noise:
        return "tactile sans apport"         # rien à gagner, quel que soit le placement
    if gain > 0.5 * ceiling:
        return "placement réussi"
    if gain > noise:
        return "placement partiel"
    return "placement en échec"              # l'information était là, elle est perdue


def main():
    ap = argparse.ArgumentParser(description="Analyse de la campagne à 42 objets")
    ap.add_argument("--combined", default="full42_combined.json")
    ap.add_argument("--oracle", default="full42_oracle.json")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    comb_arm = load_arm(args.combined)
    orac_arm = load_arm(args.oracle)
    if not comb_arm:
        raise SystemExit(f"{args.combined} introuvable ou vide.")

    allg = [x for v in comb_arm.values() for x in v["gains"]]
    p, k, n = sign_test(allg)
    a = np.array(allg)

    print("## Résultat principal — placement combiné, contacts de palpation\n")
    print(f"- objets : **{len(comb_arm)}**, cas appariés : **{n}**")
    print(f"- gain moyen de F@2 % : **{a.mean():+.2f}**, médiane **{np.median(a):+.2f}**, "
          f"écart-type {a.std(ddof=1):.2f}")
    print(f"- cas positifs : **{k}/{n}**, test des signes **p = {p:.2e}**")
    print(f"- objets à gain moyen positif : "
          f"**{sum(1 for v in comb_arm.values() if v['gain'] > 0)}/{len(comb_arm)}**")
    print("\nAucune vérité terrain n'intervient : contacts issus d'un modèle de "
          "capteur, pose estimée par cohérence de silhouette avec l'image d'entrée "
          "et résidu des contacts.\n")

    b = np.array([v["baseline"] for v in comb_arm.values()])
    g = np.array([v["gain"] for v in comb_arm.values()])
    print(f"Corrélation entre niveau du témoin et gain : **{np.corrcoef(b, g)[0, 1]:+.3f}** "
          f"— témoins sous 40 % {g[b < 40].mean():+.2f}, au-dessus {g[b >= 40].mean():+.2f}.\n")

    if orac_arm:
        print(f"## Plafond ({len(orac_arm)}/{len(comb_arm)} objets mesurés)\n")
        common = [o for o in comb_arm if o in orac_arm]
        cg = np.array([comb_arm[o]["gain"] for o in common])
        og = np.array([orac_arm[o]["gain"] for o in common])
        if og.mean() > 0:
            print(f"- plafond moyen **{og.mean():+.2f}**, réalisable {cg.mean():+.2f} "
                  f"— **{cg.mean() / og.mean() * 100:.0f} %** récupérés\n")

        groups = {}
        for o in common:
            groups.setdefault(
                classify(comb_arm[o]["gain"], orac_arm[o]["gain"]), []).append(o)
        for lbl in ("placement réussi", "placement partiel", "placement en échec",
                    "tactile sans apport"):
            if lbl in groups:
                print(f"**{lbl}** ({len(groups[lbl])}) : "
                      f"{', '.join(sorted(groups[lbl]))}\n")
    else:
        print("_Bras oracle absent — lancer `tools/resume_oracle.sh` pour obtenir "
              "le plafond par objet et la classification des échecs._\n")

    print("## Détail par objet\n")
    print("| Objet | Témoin | Gain réalisable | Plafond | Régime |")
    print("|---|---|---|---|---|")
    for o, v in sorted(comb_arm.items(), key=lambda x: -x[1]["gain"]):
        c = orac_arm.get(o, {}).get("gain")
        print(f"| {o} | {v['baseline']:.1f} | {v['gain']:+.2f} | "
              f"{'—' if c is None else f'{c:+.2f}'} | "
              f"{classify(v['gain'], c)} |")


if __name__ == "__main__":
    main()
