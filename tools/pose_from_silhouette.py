#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/pose_from_silhouette.py

Estime le repère de sortie de WaLa en confrontant la prédiction à l'image d'entrée.

Le problème. Pour utiliser des points de contact comme contrainte, il faut les
exprimer dans le repère où le modèle génère. Le recalage sur la première passe
(`register_contacts`) échoue : mesuré point à point contre un placement oracle, les
contacts sont déplacés d'une médiane de 0,66 dans un repère où l'objet occupe
[−1, 1] — un tiers de l'objet. L'ICP superpose le nuage à la forme sans respecter les
correspondances, et les quasi-symétries lui laissent toute latitude pour le faire.
Le consensus sur plusieurs graines n'y change rien : l'erreur est induite par l'image,
donc commune aux tirages.

L'idée. La pose de sortie est déterminée par l'image de conditionnement, et la caméra
de rendu est connue — position (2,2, −2,2, 1,8), euler XYZ (60°, 0, 45°), 50 mm sur
capteur 36 mm, 512². On peut donc, pour chaque pose candidate, projeter la prédiction
et comparer sa silhouette à celle de l'image d'entrée. Le critère n'utilise **aucune
vérité terrain** : seulement l'image que le modèle a déjà reçue. Contrairement à
l'ICP, il ne peut pas être satisfait par un glissement le long de la surface, puisque
glisser change la silhouette.

Usage :
    python tools/pose_from_silhouette.py --out pose_silhouette.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_fusion import load_and_normalize_mesh, align_prediction_to_gt, _proper_rotations
from guided_sampling import register_contacts
from run_guidance_campaign import oracle_frame

CAM_LOC = np.array([2.2, -2.2, 1.8])
CAM_EULER = (math.radians(60), 0.0, math.radians(45))
RES = 512
FOCAL_MM, SENSOR_MM = 50.0, 36.0
RENDER_EXTENT = 1.3        # tools/render_blender.py normalise à 1,3 ; nous à 2,0
NORM_EXTENT = 2.0


def camera_matrix():
    """Rotation monde -> caméra de Blender (euler XYZ, regard sur −Z local)."""
    rx, ry, rz = CAM_EULER
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def project(pts, res=RES):
    """Projette des points du repère de rendu vers les pixels d'une image res × res.

    La résolution est un paramètre : la câbler à 512 alors que `silhouette` construit
    un masque plus petit ne garde que le quart supérieur gauche de l'image, et
    l'IoU calculée à 128² devient sans rapport avec celle à pleine résolution.
    """
    R = camera_matrix()
    cam = (R.T @ (pts - CAM_LOC).T).T
    depth = -cam[:, 2]
    ok = depth > 1e-6
    f = FOCAL_MM / SENSOR_MM * res
    u = res / 2 + f * cam[ok, 0] / depth[ok]
    v = res / 2 - f * cam[ok, 1] / depth[ok]
    return u, v


def silhouette(pts, res=RES, dilate=1):
    """Masque d'occupation des points projetés, à la résolution de l'image."""
    u, v = project(pts, res)
    m = np.zeros((res, res), dtype=bool)
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    ok = (ui >= 0) & (ui < res) & (vi >= 0) & (vi < res)
    m[vi[ok], ui[ok]] = True
    # Les points échantillonnés laissent des trous ; on les bouche par dilatation.
    for _ in range(dilate):
        m[1:, :] |= m[:-1, :]
        m[:-1, :] |= m[1:, :]
        m[:, 1:] |= m[:, :-1]
        m[:, :-1] |= m[:, 1:]
    return m


def image_mask(path):
    """Silhouette de l'objet : tout ce qui s'écarte du fond uniforme."""
    im = np.asarray(Image.open(path).convert("RGB")).astype(float)
    bg = np.median(np.concatenate([im[0], im[-1], im[:, 0], im[:, -1]]), axis=0)
    return np.abs(im - bg).sum(2) > 20


def iou(a, b):
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter / union) if union else 0.0


# L'importateur OBJ de Blender convertit Y-up en Z-up, donc applique Rx(+90°) à tout
# maillage chargé. Sans elle, la référence projetée à l'identité ne recouvre l'image
# qu'à 0,09 (banane) ; avec elle, 0,93. C'est une convention de la chaîne de rendu,
# pas une pose à estimer.
BLENDER_IMPORT = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])


def to_render_frame(pts):
    """Repère normalisé (étendue 2,0) -> repère du rendu (étendue 1,3, Z-up)."""
    return (BLENDER_IMPORT @ pts.T).T * (RENDER_EXTENT / NORM_EXTENT)


def _centered(pts):
    return pts - (pts.max(0) + pts.min(0)) / 2          # recentrage comme au rendu


def best_pose(mesh, mask, num_points=60000):
    """Choisit parmi les 24 rotations axe-sur-axe celle dont la silhouette colle."""
    pts = np.asarray(mesh.sample_points_uniformly(number_of_points=num_points).points)
    scores = []
    for R in _proper_rotations():
        rot = _centered((R @ pts.T).T)
        scores.append((iou(silhouette(to_render_frame(rot)), mask), R))
    scores.sort(key=lambda x: -x[0])
    return scores[0][1], scores[0][0], scores[1][0]


def global_pose(mesh, mask, n_rot=2000, top_k=3, maxiter=400, seed=0, return_all=False):
    """Cherche la pose dans SO(3) entier, puis raffine les meilleurs candidats.

    Le vote sur les 24 permutations d'axes est insuffisant : mesuré contre la pose
    oracle, celle-ci s'écarte de 46 à 53° de la plus proche permutation sur la
    conserve, et l'IoU y plafonne à 0,66 quand la pose oracle atteint 0,97. La pose
    de sortie de WaLa est une rotation quelconque déterminée par l'image, pas une
    convention d'axes.

    Deux étages : tirage uniforme dans SO(3) classé en 128², puis optimisation
    continue de la similitude complète sur les meilleurs candidats.

    Un premier étage en 64² sur 8 000 points a été essayé et retiré : à cette
    résolution le nuage projeté est trop clairsemé pour que l'IoU classe
    correctement, et les candidats retenus étaient moins bons que les 24 rotations
    d'axes (banane : 0,148 d'IoU contre 0,736). Le classement grossier se fait donc
    directement en 128².

    Les 24 permutations d'axes sont ajoutées au vivier : la recherche ne peut ainsi
    jamais faire moins bien que le vote qu'elle remplace.
    """
    from scipy.spatial.transform import Rotation

    m128 = np.array(Image.fromarray(mask).resize((128, 128), Image.NEAREST))
    pts_m = np.asarray(mesh.sample_points_uniformly(number_of_points=20000).points)

    pool = list(Rotation.random(n_rot, random_state=seed).as_matrix()) + \
        [np.asarray(R) for R in _proper_rotations()]
    scored = [(iou(silhouette(to_render_frame(_centered((R @ pts_m.T).T)), res=128), m128), R)
              for R in pool]
    scored.sort(key=lambda x: -x[0])
    return refine_pose(mesh, mask, top_k=top_k, maxiter=maxiter,
                       candidates=[R for _, R in scored[:top_k]],
                       return_all=return_all)


def refine_pose(mesh, mask, top_k=3, num_points=20000, res=128, maxiter=400,
                candidates=None, return_all=False):
    """Optimise la pose en continu à partir des meilleures rotations discrètes.

    Le vote sur 24 rotations est trop grossier : la vraie pose n'est pas exactement
    axe-sur-axe, et la marge entre la meilleure et la deuxième reste alors sous 0,06
    sur 10 cas sur 15 — c'est-à-dire indécidable. On raffine donc chaque candidat sur
    une similitude complète (rotation, translation, échelle).

    L'IoU n'est pas différentiable à travers la rastérisation, d'où Nelder-Mead sur
    sept paramètres. L'optimisation travaille en 128² pour lisser le critère, et le
    score final est réévalué à pleine résolution.
    """
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation

    pts_hi = np.asarray(mesh.sample_points_uniformly(number_of_points=60000).points)
    pts_lo = np.asarray(mesh.sample_points_uniformly(number_of_points=num_points).points)
    small = np.array(Image.fromarray(mask).resize((res, res), Image.NEAREST))

    if candidates is None:
        candidates = [R for _, R in
                      sorted(((iou(silhouette(to_render_frame(_centered((R @ pts_hi.T).T))), mask), R)
                              for R in _proper_rotations()), key=lambda x: -x[0])[:top_k]]

    def apply(p, R0, pts):
        M = Rotation.from_rotvec(p[:3]).as_matrix() @ R0
        return _centered((M @ pts.T).T) * np.exp(p[6]) + p[3:6]

    results = []
    for R0 in candidates:
        def neg_iou(p):
            m = silhouette(to_render_frame(apply(p, R0, pts_lo)), res=res)
            return -iou(m, small)
        # Simplexe initial explicite : avec x0 = 0, Nelder-Mead construit le sien avec
        # des pas de 2,5·10⁻⁴, microscopiques devant l'échelle du problème — il
        # converge alors immédiatement sur place et rend l'IoU de départ inchangée.
        step = np.array([0.25, 0.25, 0.25, 0.06, 0.06, 0.06, 0.05])
        simplex = np.vstack([np.zeros(7), np.diag(step)])
        r = minimize(neg_iou, np.zeros(7), method="Nelder-Mead",
                     options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-4,
                              "initial_simplex": simplex})
        M = Rotation.from_rotvec(r.x[:3]).as_matrix() @ R0
        score = iou(silhouette(to_render_frame(apply(r.x, R0, pts_hi))), mask)
        # Transformation complète prédiction -> jeu de données, recentrage compris.
        rot = (M @ pts_hi.T).T
        c = (rot.max(0) + rot.min(0)) / 2
        s = np.exp(r.x[6])
        A = np.eye(4)
        A[:3, :3] = s * M
        A[:3, 3] = r.x[3:6] - s * c
        results.append((score, A))

    results.sort(key=lambda x: -x[0])
    if return_all:
        return results
    second = results[1][0] if len(results) > 1 else 0.0
    return results[0][1], results[0][0], second


def combined_pose(mesh, mask, contacts, n_rot=2000, top_k=6, maxiter=400):
    """Silhouette pour restreindre les poses, contacts pour trancher entre elles.

    Aucun des deux critères ne suffit seul. L'ICP sur les contacts explore tout SE(3)
    et tombe dans un minimum où le nuage se superpose à la forme sans respecter les
    correspondances (écart médian 0,76 à la position oracle). La silhouette restreint
    correctement, mais reste aveugle aux retournements de 180° qui laissent la
    projection presque identique en changeant la surface — l'anse du mug, la courbure
    avant/arrière de la moutarde : ces deux objets ne récupèrent rien du plafond.

    En combinant, on ne demande aux contacts que de départager quelques poses déjà
    plausibles, ce qui est un problème beaucoup plus facile que de les trouver. Et le
    critère reste sans vérité terrain : résidu des contacts à la surface de la
    première passe.

    **Une notation conjointe a été essayée et écartée.** L'idée : au lieu de retenir
    les six meilleures poses par IoU puis de les trier par résidu, noter chaque
    candidat sur les deux critères centrés-réduits et sommés. Elle est meilleure là
    où elle a été mise au point — sur les 15 objets dont le placement échouait, +0,25
    devient +2,12 — et **franchement moins bonne sur les 27 objets tenus à l'écart**,
    où +4,86 tombe à +3,19 (10 objets améliorés sur 27 ; la cuillère passe de +2,93 à
    −18,48). Sur les 42 objets, le bilan est de −0,40.

    C'était du sur-mesure : la règle avait été choisie sur les cas difficiles et
    dégrade les cas où la silhouette suffisait déjà. Le code garde donc la règle
    séquentielle, et cette note pour qu'on ne la re-essaie pas.

    Limite connue, indépendante de la règle : dans 29 cas sur 45, **aucune** pose du
    vivier n'est bonne (`tools/pose_selection_study.py`). Le classement par silhouette
    ne fait pas remonter la vraie pose, et aucune règle de sélection ne peut y
    remédier — il y faudrait un critère que la silhouette ne porte pas, l'ombrage par
    exemple.
    """
    cands = global_pose(mesh, mask, n_rot=n_rot, top_k=top_k, maxiter=maxiter,
                        return_all=True)
    surf = mesh.sample_points_uniformly(number_of_points=20000)

    ious, residus, mats = [], [], []
    for iou_score, A in cands:
        Ainv = np.linalg.inv(A)
        pts = (Ainv[:3, :3] @ contacts.T).T + Ainv[:3, 3]
        d = np.asarray(o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(pts)).compute_point_cloud_distance(surf))
        ious.append(float(iou_score))
        residus.append(float(np.median(d)))
        mats.append(A)

    # Règle séquentielle : la silhouette a déjà classé le vivier, les contacts
    # tranchent entre les candidats retenus. Voir la note ci-dessus sur la variante
    # conjointe, essayée puis écartée sur données tenues à l'écart.
    k = int(np.argmin(residus))
    return mats[k], ious[k], residus[k]


def main():
    ap = argparse.ArgumentParser(description="Pose par cohérence de silhouette")
    ap.add_argument("--data_dir", default="data/training_ycb")
    ap.add_argument("--contacts_dir", default="data/contacts")
    ap.add_argument("--mesh_dir", default="out/guidance_campaign")
    ap.add_argument("--condition", default="occluded")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="pose_silhouette.json")
    ap.add_argument("--refine", action="store_true",
                    help="Optimise la pose en continu au lieu de voter sur 24 rotations.")
    ap.add_argument("--so3", action="store_true",
                    help="Cherche dans SO(3) entier puis raffine. La pose de sortie "
                         "n'est pas une permutation d'axes : elle s'en écarte de 46 "
                         "à 53° sur la conserve.")
    ap.add_argument("--n_rot", type=int, default=2000)
    ap.add_argument("--combine", action="store_true",
                    help="Silhouette pour restreindre les poses, résidu des contacts "
                         "pour trancher entre elles. Sans vérité terrain.")
    args = ap.parse_args()

    data_dir, mesh_dir = Path(args.data_dir), Path(args.mesh_dir)
    objects = sorted(p.name for p in mesh_dir.iterdir() if p.is_dir())
    results = {}

    # Contrôle de la chaîne de projection : la référence, placée à l'identité, doit
    # reproduire la silhouette de l'image. Si ce chiffre est bas, tout le reste est
    # sans valeur — c'est la caméra qui est mal reconstituée, pas la pose.
    print("contrôle : IoU de la référence projetée contre l'image d'entrée")
    for obj in objects:
        gt, _ = load_and_normalize_mesh(data_dir / obj / "mesh.obj")
        mask = image_mask(data_dir / obj / "image.png")
        pts = np.asarray(gt.sample_points_uniformly(number_of_points=60000).points)
        print(f"  {obj:22s} {iou(silhouette(to_render_frame(pts)), mask):.3f}")

    print(f"\n{'objet':22s} {'graine':>6s} {'IoU':>6s} {'2e':>6s} "
          f"{'écart oracle : ICP':>19s} {'silhouette':>11s}")
    print("-" * 78)
    for obj in objects:
        gt, _ = load_and_normalize_mesh(data_dir / obj / "mesh.obj")
        mask = image_mask(data_dir / obj / "image.png")
        contacts = torch.load(Path(args.contacts_dir) / obj /
                              f"{args.condition}_{args.n}.pt", weights_only=True)
        if contacts.dim() == 3:
            contacts = contacts[0]
        contacts = contacts.numpy()
        results[obj] = {}

        for s in args.seeds:
            ref = mesh_dir / obj / f"w0_s{s}" / f"{obj}.obj"
            if not ref.exists():
                continue
            pred, _ = load_and_normalize_mesh(ref)
            if args.combine:
                A, best, second = combined_pose(pred, mask, contacts, n_rot=args.n_rot)
            elif args.so3:
                A, best, second = global_pose(pred, mask, n_rot=args.n_rot)
            elif args.refine:
                A, best, second = refine_pose(pred, mask)
            else:
                R, best, second = best_pose(pred, mask)
                A = np.eye(4)
                A[:3, :3] = R

            # A envoie la prédiction **normalisée** sur le repère du jeu de données.
            # Les placements ICP et oracle vivent, eux, dans le repère **brut** du
            # modèle : il faut donc composer par la normalisation avant de comparer.
            # Sans ça on soustrait des points exprimés dans deux repères différents,
            # et l'écart mesuré contient l'écart de normalisation.
            from run_guidance_campaign import _normalization
            Ainv = np.linalg.inv(A @ _normalization(ref))
            pts_sil = (Ainv[:3, :3] @ contacts.T).T + Ainv[:3, 3]

            O = oracle_frame(gt, ref)
            pts_or = (O[:3, :3] @ contacts.T).T + O[:3, 3]
            pts_icp, _ = register_contacts(contacts, o3d.io.read_triangle_mesh(str(ref)))
            d_icp = float(np.median(np.linalg.norm(pts_icp - pts_or, axis=1)))
            d_sil = float(np.median(np.linalg.norm(pts_sil - pts_or, axis=1)))

            results[obj][f"s{s}"] = {"iou": best, "iou_second": second,
                                     "disp_icp": d_icp, "disp_silhouette": d_sil}
            print(f"{obj:22s} {s:>6d} {best:6.3f} {second:6.3f} "
                  f"{d_icp:19.3f} {d_sil:11.3f}")
        Path(args.out).write_text(json.dumps(results, indent=2))

    if results:
        a = np.array([v["disp_icp"] for o in results for v in results[o].values()])
        b = np.array([v["disp_silhouette"] for o in results for v in results[o].values()])
        print(f"\nécart médian à la position oracle : ICP {np.median(a):.3f} → "
              f"silhouette {np.median(b):.3f}   ({(b < a).sum()}/{len(b)} améliorés)")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
