#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/make_contacts_dh116.py

Jeux de contacts au régime que la DH116 peut réellement produire, avec normales.

Pourquoi ce fichier. Toutes les campagnes du dépôt tournent à 128 points, et la plus
pauvre à 32. Or la main du banc n'offre pas cela : neuf zones tactiles exploitables
(`ring.pad` mort), dont une saisie n'en excite que trois à cinq — l'index lit 0,000 en
poussant à 620 ‰ parce qu'il appuie à côté de sa zone instrumentée. Le régime réel est
donc **un point par plaque**, pas seize, et il n'a jamais été mesuré.

Le jeu produit ici partage **exactement les mêmes plaques** que `palpation8_128`, dont
le plafond oracle vaut +6,77 sur ces cinq objets : `palpation_contacts` tire ses sites
par échantillonnage du plus lointain à graine fixée, donc les huit centres sont les
mêmes qu'on en demande 8 points ou 128. Seule la cardinalité change, ce qui est
précisément la variable à isoler.

Chaque contact porte en plus sa **normale sortante**, déjà calculée par le modèle de
palpation pour ses contraintes d'occultation et d'encombrement, et jusqu'ici jetée.

Usage :
    python tools/make_contacts_dh116.py
    python tools/make_contacts_dh116.py --sites 4 8 --objects 011_banana
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from make_contacts import normalize, palpation_contacts, TEST_OBJECTS


def main():
    ap = argparse.ArgumentParser(description="Contacts au régime DH116, avec normales")
    ap.add_argument("--data_dir", default="data/training_ycb")
    ap.add_argument("--out_dir", default="data/contacts")
    ap.add_argument("--objects", nargs="*", default=TEST_OBJECTS)
    ap.add_argument("--sites", type=int, nargs="+", default=[4, 8],
                    help="Nombre de zones excitées. 8 = la main complète et saine, "
                         "4 = ce qu'une saisie réelle excite (régime déjà mesuré "
                         "négatif à 128 points : -0,42).")
    ap.add_argument("--finger_radius", type=float, default=0.06)
    args = ap.parse_args()

    for obj in args.objects:
        mesh_path = Path(args.data_dir) / obj / "mesh.obj"
        if not mesh_path.exists():
            print(f"[dh116] {obj} : mesh.obj introuvable, ignoré")
            continue
        mesh = normalize(o3d.io.read_triangle_mesh(str(mesh_path)))
        d = Path(args.out_dir) / obj
        d.mkdir(parents=True, exist_ok=True)

        for k in args.sites:
            res = palpation_contacts(mesh, k, n_sites=k,
                                     finger_radius=args.finger_radius,
                                     return_normals=True)
            if res is None:
                print(f"  ! {obj} k={k} : pas assez de points accessibles")
                continue
            pts, nrm = res
            np.savez(d / f"dh116_{k}.npz", points=pts.astype(np.float32),
                     normals=nrm.astype(np.float32))
            # Écart minimal entre plaques : si deux contacts se touchent, la
            # contrainte est redondante et le régime n'est pas celui qu'on croit.
            dm = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
            dm[np.diag_indices(len(pts))] = np.inf
            print(f"  {obj} k={k} : {len(pts)} contacts, "
                  f"écart min {dm.min():.3f}, étendue {np.ptp(pts, 0).round(2)}")


if __name__ == "__main__":
    main()
