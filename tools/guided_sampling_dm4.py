#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/guided_sampling_dm4.py

Guidage par contact sur `WaLa-DM4-1B`, avec placement **analytique** des contacts.

Ce que ça change par rapport à `guided_sampling.py`. Sur le modèle mono-vue, le
repère de sortie est inconnu et instable — changer la seule graine de bruit fait
tourner la sortie de plus de 130°. Tout l'appareillage d'estimation (recherche dans
SO(3), cohérence de silhouette, résidu des contacts, règles de sélection) sert à le
retrouver, et il en perd 4,8 points de F-Score sur 42 objets.

Sur DM4 alimenté par quatre vues de profondeur reprojetées depuis **une seule
observation**, la pose est ancrée : mesuré sur 4 objets et 2 graines, l'angle reste
entre 118° et 122° autour d'un axe constant. Le placement des contacts devient donc
une transformation fixe, calibrée une fois, et il n'y a plus rien à estimer.

Le guidage lui-même est inchangé : à chaque pas, on décode ẑ₀ en champ de distance
signée, on l'évalue aux contacts, et on corrige ẑ₀ par le gradient de l'écart à zéro.
C'est le placement qui change, pas le mécanisme.

Usage :
    python tools/depth_rig.py --objects 011_banana
    python tools/guided_sampling_dm4.py --object 011_banana --guidance_weight 0.05
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_utils import Model
from src.dataset_utils import get_mv_dm_data, get_image_transform_latent_model
from src.experiments.utils.wavelet_utils import WaveletData
from guided_sampling import trilinear, ContactGuidedUNet
from depth_rig import load_rig, depth_at_view, backproject, camera_pose, RENDER_EXTENT
from eval_fusion import load_and_normalize_mesh, align_prediction_to_gt

VUES = [3, 6, 10, 26]


def calibrer_pose(model, tf, device, objets, data_dir, depth_dir, seed=0,
                  timesteps=5):
    """Estime une fois pour toutes la transformation repère jeu de données -> modèle.

    Elle est mesurée plutôt que codée en dur : l'angle observé (118° à 122°) dépend du
    point de contrôle et du rig, et une constante recopiée deviendrait fausse en
    silence si l'un des deux changeait. On la réestime donc sur quelques objets, et on
    garde la médiane des rotations obtenues.

    Les objets à quasi-symétrie sont écartés : leur recalage choisit librement parmi
    des solutions équivalentes, et leur inclure fausserait la médiane.

    Ce qui est calibré est **la rotation seule**, exprimée dans le repère normalisé de
    la prédiction. L'échelle et le centre ne sont pas des constantes du modèle : ils
    dépendent de la boîte englobante de chaque maillage produit, et une matrice globale
    unique les fait donc nécessairement fausses. Mesuré en les incluant : le nuage de
    contacts ressortait 24 % trop grand et à 0,209 de sa position exacte, contre 0,05
    de distance à la surface pour un placement correct.
    """
    from scipy.spatial.transform import Rotation
    rots, facteurs = [], []
    for obj in objets:
        files = [Path(depth_dir) / obj / "reprojetee" / f"{v:03d}.png" for v in VUES]
        if not all(f.exists() for f in files):
            continue
        gt, _ = load_and_normalize_mesh(Path(data_dir) / obj / "mesh.obj")
        data = get_mv_dm_data([str(f) for f in files], VUES, tf, device)
        out = Path("/tmp/calib") / obj
        out.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(seed)
        with torch.no_grad():
            p = model.test_inference(data, data_idx=0, image_name="c",
                                     save_dir=str(out), output_format="obj")
        pred, _ = load_and_normalize_mesh(Path(p))
        _, info = align_prediction_to_gt(gt, pred, return_info=True)
        M = info["transform"]
        U, S, Vt = np.linalg.svd(M[:3, :3])
        R = U @ Vt
        if np.linalg.det(R) < 0:
            continue
        rots.append(Rotation.from_matrix(R))
        # Le recalage applique une échelle en plus de la normalisation par boîte
        # englobante : celle-ci ne fait pas coïncider les tailles, la prédiction
        # n'ayant pas exactement les proportions de la référence. Mesuré sans ce
        # facteur, le nuage de contacts ressortait 18 à 21 % trop grand.
        facteurs.append(float(np.mean(S)))
        print(f"  calibrage {obj:24s} {np.degrees(rots[-1].magnitude()):6.1f}°  "
              f"facteur d'échelle {facteurs[-1]:.3f}", flush=True)

    if not rots:
        raise SystemExit("[calibrage] aucun objet exploitable")
    # Moyenne de rotations : `Rotation.mean()` et non une moyenne de quaternions.
    # q et -q désignent la même rotation, et une moyenne naïve part n'importe où si
    # les signes diffèrent — mesuré : 119,5° puis 83,9° sur deux exécutions
    # identiques, pour les mêmes rotations individuelles.
    R = Rotation.concatenate(rots).mean().as_matrix()
    f = float(np.median(facteurs))
    print(f"[calibrage] rotation {np.degrees(Rotation.from_matrix(R).magnitude()):.1f}°, "
          f"facteur d'échelle {f:.3f}, sur {len(rots)} objets "
          f"(étendue des facteurs {min(facteurs):.3f}-{max(facteurs):.3f})")
    return R, f


def placer_contacts(contacts, R, facteur, ref_path, nuage_observe=None):
    """Place les contacts dans le repère du modèle.

    Deux variantes, selon qu'on dispose ou non du nuage observé.

    **Sans nuage** : rotation calibrée, puis normalisation lue sur la boîte englobante
    du maillage non guidé, corrigée d'un facteur d'échelle calibré. Mesuré : écart de
    0,107 à 0,141 à la position exacte — mieux que l'ICP en deux passes du pipeline
    mono-vue (0,66), comparable à son meilleur estimateur (0,111), mais trois à quatre
    fois moins précis que le placement exact.

    **Avec nuage** : on recale la prédiction sur le nuage de points issu de la carte de
    profondeur d'entrée. C'est légitime — la profondeur est une **entrée** du système,
    pas une vérité terrain — et bien mieux posé que le recalage contacts vers
    prédiction : le nuage est dense et couvre une face entière, là où les contacts sont
    épars et d'un seul côté. La rotation calibrée sert d'initialisation, ce qui évite le
    minimum local dans lequel tombait l'ICP du pipeline mono-vue.
    """
    raw = o3d.io.read_triangle_mesh(str(ref_path))
    bb = raw.get_axis_aligned_bounding_box()
    c, ext = bb.get_center(), float(np.max(bb.get_extent()))

    if nuage_observe is None:
        pts = (R.T @ contacts.T).T / facteur
        return pts * (ext / 2.0) + c

    # Init : rotation calibrée + normalisation grossière, dans le repère du rendu.
    src = raw.sample_points_uniformly(number_of_points=20000)
    P = np.asarray(src.points)
    P = (P - c) * (2.0 / ext)                 # brut -> normalisé
    P = (R @ P.T).T * (RENDER_EXTENT / 2.0)   # normalisé -> repère du rendu
    s_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    t_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(nuage_observe))

    reg = o3d.pipelines.registration.registration_icp(
        s_pcd, t_pcd, max_correspondence_distance=0.35,
        estimation_method=o3d.pipelines.registration.
        TransformationEstimationPointToPoint(with_scaling=True),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=120))
    T = np.asarray(reg.transformation)

    # Contacts (repère jeu de données, étendue 2) -> repère du rendu -> prédiction.
    q = contacts * (RENDER_EXTENT / 2.0)
    Tinv = np.linalg.inv(T)
    q = (Tinv[:3, :3] @ q.T).T + Tinv[:3, 3]      # repère du rendu -> init
    q = (R.T @ (q * (2.0 / RENDER_EXTENT)).T).T   # -> normalisé
    return q * (ext / 2.0) + c


def nuage_depuis_profondeur(obj, data_dir, vue, res=256, upsample=4):
    """Nuage de points de la vue réellement observée, dans le repère du rendu.

    C'est la même rétroprojection que celle qui fabrique les vues reprojetées : la
    donnée est déjà là, on ne la calcule qu'une fois de plus.
    """
    from depth_rig import normalize_mesh
    rig = load_rig()
    mesh = normalize_mesh(o3d.io.read_triangle_mesh(
        str(Path(data_dir) / obj / "mesh.obj")))
    d = depth_at_view(mesh, *rig[vue], res * upsample)
    return backproject(d, *rig[vue], res * upsample)


def main():
    ap = argparse.ArgumentParser(description="Guidage par contact sur DM4")
    ap.add_argument("--model", default="ADSKAILab/WaLa-DM4-1B")
    ap.add_argument("--data_dir", default="data/training_ycb")
    ap.add_argument("--depth_dir", default="data/depth_rig")
    ap.add_argument("--contacts_dir", default="data/contacts")
    ap.add_argument("--out_dir", default="out/dm4_guidage")
    ap.add_argument("--objects", nargs="+", required=True)
    ap.add_argument("--calib_objects", nargs="+",
                    default=["035_power_drill", "048_hammer", "011_banana"],
                    help="Objets asymétriques servant à calibrer la pose.")
    ap.add_argument("--contacts", default="palpation8_128")
    ap.add_argument("--guidance_weight", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--timesteps", type=int, default=20)
    ap.add_argument("--scale", type=float, default=1.3)
    ap.add_argument("--recaler_sur_profondeur", action="store_true", default=True,
                    help="Recale la prédiction sur le nuage observé plutôt que de se "
                         "fier à la seule boîte englobante.")
    ap.add_argument("--sans_recalage", dest="recaler_sur_profondeur",
                    action="store_false")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = Model.from_pretrained(args.model).to(device).eval()
    model.set_inference_fusion_params(args.scale, args.timesteps)
    tf = get_image_transform_latent_model()

    print("[calibrage] estimation de la transformation de pose")
    R_calib, f_calib = calibrer_pose(model, tf, device, args.calib_objects,
                                     args.data_dir, args.depth_dir)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    np.save(Path(args.out_dir) / "rotation_calibree.npy", R_calib)
    np.save(Path(args.out_dir) / "facteur_calibre.npy", np.array([f_calib]))

    net = getattr(model.network, "_orig_mod", model.network)
    a = model.args
    shape = (1, a.e_dim, a.grid_size, a.grid_size, a.grid_size)

    for obj in args.objects:
        files = [Path(args.depth_dir) / obj / "reprojetee" / f"{v:03d}.png" for v in VUES]
        cpath = Path(args.contacts_dir) / obj / f"{args.contacts}.pt"
        if not all(f.exists() for f in files) or not cpath.exists():
            print(f"{obj} : entrées manquantes, ignoré")
            continue

        contacts = torch.load(cpath, map_location="cpu", weights_only=True)
        if contacts.dim() == 3:
            contacts = contacts[0]
        c = contacts.numpy()
        data = get_mv_dm_data([str(f) for f in files], VUES, tf, device)
        with torch.no_grad():
            feats = model.extract_input_features(data, data_type="depth",
                                                 is_train=False, to_cuda=True)
            idx = model.extract_img_idx(data, data_idx=0)
            cond = net.process_condition(feats, image_index=idx)

        for seed in args.seeds:
            pts_t = None
            for w in (0.0, args.guidance_weight):
                torch.manual_seed(seed)
                trace = []
                guided = ContactGuidedUNet(
                    net.unet, model,
                    pts_t if pts_t is not None else torch.zeros(1, 3, device=device),
                    w, trace)
                t0 = time.time()
                latent, _ = net.inference_diffusion_module.p_sample_loop(
                    model=guided, shape=shape, device=device, clip_denoised=False,
                    progress=False,
                    model_kwargs={"latent_codes": cond, "condition_zero": None,
                                  "guidance_scale": args.scale, "dp_cond": None})
                with torch.no_grad():
                    pred = model.autoencoder.decode_from_pre_quant(latent[0:1])
                wd = WaveletData(shape_list=model.dwt_sparse_composer.shape_list,
                                 output_stage=a.max_training_level,
                                 max_depth=a.max_depth, wavelet_volume=pred)
                low, highs = wd.convert_low_highs()
                d = Path(args.out_dir) / obj / f"w{w:g}_s{seed}"
                d.mkdir(parents=True, exist_ok=True)
                model.save_visualization_obj(None, obj, obj_path=str(d / f"{obj}.obj"),
                                             samples=(low, highs))
                if w == 0.0:
                    # La passe non guidée sert de témoin ET fournit la normalisation
                    # propre à cette sortie, seule grandeur non constante du placement.
                    nuage = nuage_depuis_profondeur(obj, args.data_dir, VUES[0]) \
                        if args.recaler_sur_profondeur else None
                    pts = placer_contacts(c, R_calib, f_calib, d / f"{obj}.obj",
                                          nuage_observe=nuage)
                    hors = float(np.mean(np.abs(pts).max(1) > 1.0))
                    if hors > 0.05:
                        print(f"  ! {obj} : {hors*100:.0f} % des contacts hors grille")
                    pts_t = torch.from_numpy(pts).float().to(device)
                msg = (f"{trace[0][1]:.4f} -> {trace[-1][1]:.4f}" if trace else "témoin")
                print(f"  {obj:24s} poids {w:g} graine {seed} : |SDF| {msg}  "
                      f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
