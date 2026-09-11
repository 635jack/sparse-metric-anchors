#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/download_ycb.py

Télécharge un ensemble d'objets YCB (meshes) de taille adaptée à la manipulation
depuis le dataset Hugging Face 'll4ma-lab/ycb-fixed-meshes' via requêtes HTTP directes.
"""

import os
import sys
import urllib.request
from pathlib import Path

REPO_ID = "ll4ma-lab/ycb-fixed-meshes"
# Utiliser des chemins absolus par rapport au répertoire de travail
BASE_DIR = Path(__file__).resolve().parent.parent   # racine du dépôt, où que soit le clone
OUTPUT_DIR = BASE_DIR / "data" / "ycb_source_meshes"

DEXTER_OBJECTS = [
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "024_bowl",
    "025_mug",
    "035_power_drill",
    "036_wood_block",
    "037_scissors",
    "040_large_marker",
    "051_large_clamp",
    "052_extra_large_clamp",
    "061_foam_brick",
]

# Extension du 7 août. Le guidage par contact ne s'entraîne pas : la séparation
# train/test ne servait qu'au module appris (technique n°6, abandonnée). Tous les
# objets sont donc des objets de test, et la taille de l'échantillon n'est plus
# limitée que par le temps de rendu.
#
# Le dépôt en contient 101, mais les prendre tous gonflerait N sans gagner en
# puissance : des familles entières y sont quasi identiques (11 avions jouets,
# 13 lego duplo, 10 gobelets, 6 sacs de billes, 6 balles). On en garde un ou deux
# représentants. Sont aussi écartés les objets sous ~3 cm (clé, écrous, billes,
# petits marqueurs), que la palpation à pulpe de 6 mm ne peut pas échantillonner
# proprement, et dont la reconstruction à 256³ n'a pas de sens.
# 023_wine_glass a été retiré après contrôle : le dépôt n'en propose que la version
# `tsdf`, qui compte 84 triangles. Ce n'est pas un verre, c'est un placeholder — même
# une comparaison appariée n'a pas de sens contre une référence pareille.
# 001_chips_can et 076_timer ont été retirés au contrôle des rendus : le dépôt n'en
# propose que la version `poisson`, dont les normales sont inversées (0 % sortantes
# sur la boîte de chips) et la surface trouée — les rendus montrent des zones noires
# traversantes. 049_small_clamp est retiré aussi : 0,33 % de l'image, trop peu de
# pixels pour estimer une pose et trop peu de faces pour une référence.
#
# Note conservée : la référence est grossière,
# ce qui abaisse les scores absolus, mais témoin et guidé sont notés contre la même
# référence et l'écart apparié reste valide.
YCB_EXTENSION = [
    "005_tomato_soup_can-1",
    "013_apple",
    "016_pear",
    "017_orange",
    "022_windex_bottle",
    "026_sponge",
    "028_skillet_lid",
    "029_plate",
    "031_spoon",
    "033_spatula",
    "038_padlock",
    "042_adjustable_wrench",
    "043_phillips_screwdriver",
    "048_hammer",
    "050_medium_clamp",
    "053_mini_soccer_ball",
    "055_baseball",
    "062_dice",
    "065-a_cups",
    "070-a_colored_wood_blocks",
    "071_nine_hole_peg_test",
    "077_rubiks_cube",
]

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"[YCB] Démarrage du téléchargement direct depuis Hugging Face...")
    
    downloaded_count = 0
    
    for obj in DEXTER_OBJECTS + YCB_EXTENSION:
        # Ordre de préférence des fichiers sur le dépôt Hugging Face
        candidates = [
            f"{obj}/google_16k/textured.obj",
            f"{obj}/poisson/textured.obj",
            f"{obj}/poisson/nontextured.ply",
            f"{obj}/google_16k/nontextured_proc.stl",
            f"{obj}/tsdf/textured.obj",
            f"{obj}/poisson/nontextured.stl",
        ]
        
        success = False
        for target_file in candidates:
            # Encoder les espaces ou caractères spéciaux si nécessaire (pas de problème ici)
            url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{target_file}"
            suffix = Path(target_file).suffix
            dest_path = OUTPUT_DIR / f"{obj}{suffix}"
            
            try:
                # Tentative de téléchargement
                # Utiliser un User-Agent pour éviter le blocage de certains serveurs
                req = urllib.request.Request(
                    url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                with urllib.request.urlopen(req) as response:
                    with open(dest_path, 'wb') as out_file:
                        out_file.write(response.read())
                
                print(f"  [+] {obj} -> téléchargé avec succès depuis {target_file}")
                success = True
                downloaded_count += 1
                break # On a trouvé un candidat valide, on passe à l'objet suivant
            except Exception as e:
                # Si l'erreur est un 404, on tente le candidat suivant
                continue
                
        if not success:
            print(f"  [-] Aucun mesh valide téléchargeable trouvé pour {obj}")
            
    print(f"\n[YCB] Téléchargement terminé : {downloaded_count}/{len(DEXTER_OBJECTS + YCB_EXTENSION)} objets récupérés.")

if __name__ == "__main__":
    main()
