#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/guided_sampling.py

Technique n°5 : guidage de la diffusion par une contrainte de contact.

À chaque pas de débruitage, on décode ẑ₀ en champ de distance signée, on évalue ce
champ aux points de contact, et on corrige ẑ₀ par le gradient de l'écart à zéro.
La contrainte imposée est « la surface passe par ces points ».

Pourquoi cette technique et pas une des cinq autres.

Les techniques 1, 2, 3, 4 et 6 exigent toutes que la modalité géométrique soit un
**générateur** : qu'elle produise une forme complète, que l'on mélange ensuite à
celle issue de l'image. Or c'est exactement ce qu'elle ne sait pas faire ici —
mesuré : le modèle nuage sous 512 points ne produit pas une surface mais 43 à 101
fragments dont le plus gros porte 4 à 11 % des faces, et le rembourrage à la
cardinalité d'entraînement (2500) n'y change rien. C'est la densité d'information qui
manque, pas le format.

Le guidage n'exige de la géométrie qu'elle soit une **contrainte**. Quelques dizaines
de points de contact sont inutilisables pour générer une forme, mais parfaitement
valides pour dire où passe la surface — c'est précisément ce qu'un capteur tactile
fournit. Et comme on ne mélange jamais deux prédictions, l'interférence destructive
qui condamne les techniques 2 et 3 (effondrement à poids égal, mesuré à 13,94 % de
F@2 % contre 44,88 % pour l'image seule) ne peut pas se produire.

Faisabilité mesurée avant écriture : 3,44 s par pas de guidage, gradient fini sur ẑ₀,
8,98 Go de crête MPS. Soit 17 s pour un échantillonnage à 5 pas.

Usage :
    python tools/guided_sampling.py --image data/training_ycb/011_banana/image.png \\
        --contacts data/training_ycb/011_banana/points_64.pt \\
        --out_dir out/ --name 011_banana --guidance_weight 0.05 --seeds 0 1 2
    # témoin : --guidance_weight 0 reproduit exactement l'image seule
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_utils import Model
from src.dataset_utils import get_singleview_data, get_image_transform_latent_model
from src.experiments.utils.wavelet_utils import WaveletData


def trilinear(vol, pts):
    """Échantillonne vol (1,1,R,R,R) aux points pts (N,3), différentiablement.

    Écrit à la main plutôt qu'avec `grid_sample` : le rétropropagé 3D de celui-ci
    (`aten::grid_sampler_3d_backward`) n'est pas implémenté sur MPS, et le repli CPU
    imposerait un aller-retour de 67 Mo par pas. Ici tout est indexation et produit.

    La convention de repère suit celle de l'extraction de surface
    (`latent_module.save_visualization_obj`), qui pose
    `sommet = indice / résolution * 2 - 1`. On inverse donc par
    `indice = (coord + 1) * R / 2`, et non la convention `align_corners` de
    grid_sample — sinon les points de contact seraient décalés d'un demi-voxel par
    rapport au maillage effectivement produit.
    """
    R = vol.shape[-1]
    v = vol[0, 0]
    idx = ((pts.clamp(-1, 1) + 1) * R / 2).clamp(0, R - 1.001)
    lo = idx.floor().long().clamp(0, R - 2)
    frac = idx - lo.float()
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (((1 - frac[:, 0]) if dx == 0 else frac[:, 0])
                     * ((1 - frac[:, 1]) if dy == 0 else frac[:, 1])
                     * ((1 - frac[:, 2]) if dz == 0 else frac[:, 2]))
                out = out + w * v[lo[:, 0] + dx, lo[:, 1] + dy, lo[:, 2] + dz]
    return out


def _proper_rotations():
    """Les 24 rotations qui envoient les axes sur les axes (permutations + signes)."""
    import itertools
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for i, p in enumerate(perm):
                R[i, p] = signs[i]
            if abs(np.linalg.det(R) - 1.0) < 1e-9:
                yield R


def register_contacts(contacts, mesh, num_points=20000):
    """Recale les points de contact sur le repère de sortie du modèle.

    WaLa génère dans son repère canonique, qui ne coïncide pas avec celui du jeu de
    données : mesuré sur cinq objets YCB, l'ordre des axes par étendue diffère entre
    la référence et la prédiction, et il diffère **différemment selon l'objet** — ce
    n'est donc pas une convention fixe que l'on pourrait corriger une fois pour
    toutes, mais une pose déterminée par l'image.

    On estime donc le repère par objet, à partir de la seule première passe non
    guidée et des points de contact — jamais du maillage de référence. C'est ce que
    ferait un système réel où caméra et capteur tactile sont calibrés entre eux mais
    où le repère de sortie du générateur reste inconnu.

    ICP seul échouerait : les deux repères diffèrent d'une rotation quelconque, hors
    de son bassin de convergence. On initialise donc par alignement des axes
    principaux, en essayant les 24 rotations axe-sur-axe, et on garde celle dont
    l'ICP converge au meilleur résidu.

    Limite à garder en tête : le recalage se fait sur la forme de la première passe.
    Il en retire la pose, pas l'erreur de forme — le guidage garde donc de quoi
    corriger la géométrie locale, mais ne peut pas rattraper une orientation
    globalement fausse choisie à la première passe.
    """
    # `mesh` accepte aussi un nuage (N,3) déjà constitué : c'est ce que fournit le
    # consensus de pose, qui fusionne plusieurs passes non guidées pour moyenner
    # l'erreur de forme au lieu de la subir sur une seule.
    tgt_np = (np.asarray(mesh) if isinstance(mesh, np.ndarray) else
              np.asarray(mesh.sample_points_uniformly(number_of_points=num_points).points))
    src_np = contacts

    def frame(P):
        c = P.mean(0)
        _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
        if np.linalg.det(Vt) < 0:
            Vt[-1] *= -1          # base directe : sinon l'init peut être un miroir
        return c, Vt, np.sqrt(((P - c) ** 2).sum(1).mean())

    cs, Vs, rs = frame(src_np)
    ct, Vt, rt = frame(tgt_np)
    scale = rt / max(rs, 1e-9)

    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_np))
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt_np))
    diag = np.linalg.norm(tgt.get_axis_aligned_bounding_box().get_extent())

    best, best_rmse = np.eye(4), np.inf
    for R24 in _proper_rotations():
        M = Vt.T @ R24 @ Vs * scale
        init = np.eye(4)
        init[:3, :3] = M
        init[:3, 3] = ct - M @ cs
        reg = o3d.pipelines.registration.registration_icp(
            src, tgt, max_correspondence_distance=0.3 * diag, init=init,
            estimation_method=o3d.pipelines.registration.
            TransformationEstimationPointToPoint(with_scaling=True),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
        if reg.inlier_rmse > 0 and reg.inlier_rmse < best_rmse and reg.fitness > 0.5:
            best, best_rmse = np.asarray(reg.transformation), reg.inlier_rmse

    out = (best[:3, :3] @ src_np.T).T + best[:3, 3]
    return out, best_rmse


class ContactGuidedUNet(nn.Module):
    """Corrige ẑ₀ à chaque pas pour que la surface passe par les points de contact.

    WaLa prédit directement ẑ₀ (`diffusion_model_mean_type = START_X`, vérifié dans
    `args.json`), donc la sortie du réseau *est* l'estimation de forme courante et on
    peut la corriger sans toucher au calendrier de bruit ni au sampler.

    Le pas est normalisé — on avance de `weight × ‖ẑ₀‖` dans la direction du gradient
    — pour que le réglage ne dépende ni de l'échelle du champ de distance ni de celle
    du latent, qui varient toutes deux au cours de l'échantillonnage.
    """

    def __init__(self, unet, model, contacts, weight, log=None,
                 normals=None, eps=0.03, lambda_rel=1.0, field_sign=None,
                 free_points=None, lambda_free=1.0):
        super().__init__()
        self.unet, self.model = unet, model
        self.contacts, self.weight, self.log = contacts, float(weight), log
        # Contrainte de normale, optionnelle. `normals=None` reproduit exactement le
        # comportement de la campagne à 42 objets : aucune des mesures publiées ne
        # change.
        self.normals = normals
        self.eps = float(eps)
        self.lambda_rel = float(lambda_rel)
        self.field_sign = field_sign
        # Contrainte d'espace libre, optionnelle. Le guidage n'imposait que « la
        # surface passe ici » et jamais « la surface ne passe pas ici », alors que le
        # trajet parcouru par un doigt avant l'impact est de l'espace libre **mesuré**
        # — au même titre que le contact, et gratuit. C'est même la seule information
        # que produit un doigt qui ne touche rien.
        self.free_points = free_points
        self.lambda_free = float(lambda_free)
        self.lam = None          # calibrés au premier pas guidé, puis figés
        self.lam_free = None
        self._calibre = False

    def _fd_gradient(self, vol, pts):
        """Gradient du champ aux contacts, par différences finies centrées.

        Pourquoi pas `torch.autograd.grad` sur les points. Il faudrait alors
        rétropropager une seconde fois à travers le décodeur pour obtenir le gradient
        en ẑ₀ (`create_graph=True`), et le double rétropropagé du décodeur d'ondelettes
        n'est ni bon marché ni acquis sur MPS. Six évaluations trilinéaires de plus
        par contact coûtent, elles, un temps négligeable devant le décodage — qui est
        le seul poste réel (3,44 s par pas).

        Le pas `eps` est exprimé dans le repère normalisé [-1,1] : 0,03 vaut environ
        3 mm sur un objet de 20 cm, soit quatre voxels de la grille 256³ — assez grand
        pour que la différence ne mesure pas l'interpolation trilinéaire elle-même.
        """
        g = []
        for k in range(3):
            off = torch.zeros_like(pts)
            off[:, k] = self.eps
            g.append((trilinear(vol, pts + off) - trilinear(vol, pts - off))
                     / (2 * self.eps))
        return torch.stack(g, dim=1)

    def _losses(self, z):
        """(perte de position, perte de normale ou None) pour un latent donné."""
        vol = self.decode_sdf(z)
        v = trilinear(vol, self.contacts)
        loss_pos = (v ** 2).mean()
        if self.normals is None:
            return vol, v, loss_pos, None
        # Contrainte scale-free : seule la **direction** du gradient est imposée.
        # Une cible absolue serait fausse — le champ de WaLa n'est pas une distance
        # métrique de gradient unitaire, et rien ne garantit sa pente.
        g = self._fd_gradient(vol, self.contacts)
        gn = g.norm(dim=1, keepdim=True).clamp_min(1e-8)
        cos = (g / gn * self.normals).sum(1)
        loss_nrm = (1.0 - self.field_sign * cos).mean()
        return vol, v, loss_pos, loss_nrm

    def _loss_free(self, vol):
        """Pénalise la matière là où un doigt est passé.

        Charnière **unilatérale** : l'espace libre dit « pas de matière ici », il ne dit
        pas à quelle distance se trouve la surface. Pénaliser un écart signé imposerait
        une distance qu'aucune mesure ne fournit ; on ne pénalise donc que le côté
        intérieur du champ, et on laisse le côté extérieur libre.

        Les points sont filtrés hors de cette classe : `trilinear` **borne** ses
        coordonnées, donc un point hors de la grille [-1,1] y serait rabattu sur le
        bord et imposerait une contrainte à un endroit qui n'a pas été mesuré.
        """
        q = trilinear(vol, self.free_points)
        return torch.relu(-self.field_sign * q).pow(2).mean()

    def decode_sdf(self, z):
        a = self.model.args
        pred = self.model.autoencoder.decode_from_pre_quant(z)
        wd = WaveletData(shape_list=self.model.dwt_sparse_composer.shape_list,
                         output_stage=a.max_training_level, max_depth=a.max_depth,
                         wavelet_volume=pred)
        low, highs = wd.convert_low_highs()
        return self.model.dwt_inverse_3d((low, highs))

    def _grad_of(self, z0, which):
        """Gradient en ẑ₀ d'un seul terme, graphe libéré aussitôt.

        La calibration a besoin de la norme de chaque gradient séparément. Les obtenir
        par trois `retain_graph=True` successifs garde trois graphes du décodeur en
        mémoire en même temps — le pic mesuré du guidage est déjà de 9 Go, et le
        système a tué le processus. On recalcule donc chaque terme, ce qui coûte un
        décodage de plus mais ramène le pic à celui d'un seul graphe. Le surcoût ne
        porte que sur le premier pas guidé de chaque échantillon.
        """
        # `enable_grad` explicite : la calibration est appelée depuis `forward`, que le
        # sampler exécute sous `torch.no_grad()`. Sans ça les pertes n'ont pas de
        # graphe et `autograd.grad` échoue.
        with torch.enable_grad():
            z = z0.detach().requires_grad_(True)
            vol, _, loss_pos, loss_nrm = self._losses(z)
            if which == "pos":
                L = loss_pos
            elif which == "nrm":
                L = loss_nrm
            else:
                L = self._loss_free(vol)
            g, = torch.autograd.grad(L, z)
        return g.detach()

    def _calibrer(self, z0):
        """Fixe une fois pour toutes le poids des termes auxiliaires.

        Les pertes n'ont pas la même unité — l'écart au zéro est en unités de champ au
        carré, le cosinus est sans dimension, la charnière d'espace libre est encore
        autre chose. Les additionner à poids brut laisserait l'une écraser les autres,
        et le résultat mesurerait un choix d'échelle plutôt que l'apport de chaque
        contrainte. On impose donc un rapport de norme entre **gradients**, mesuré au
        premier pas guidé, puis figé pour tout l'échantillonnage.
        """
        npos = float(self._grad_of(z0, "pos").norm())
        if self.normals is not None:
            n = float(self._grad_of(z0, "nrm").norm())
            self.lam = (self.lambda_rel * npos / n) if n > 0 else 0.0
        if self.free_points is not None:
            n = float(self._grad_of(z0, "free").norm())
            self.lam_free = (self.lambda_free * npos / n) if n > 0 else 0.0

    def forward(self, x, t, latent_codes=None, **kw):
        z0 = self.unet(x, t, latent_codes=latent_codes)
        if self.weight <= 0.0:
            return z0

        if not self._calibre and (self.normals is not None
                                  or self.free_points is not None):
            self._calibrer(z0)
            self._calibre = True

        # Le sampler enveloppe ses appels dans torch.no_grad() ; il faut donc
        # réactiver explicitement le graphe pour obtenir le gradient sur ẑ₀.
        with torch.enable_grad():
            z = z0.detach().requires_grad_(True)
            vol, v, loss_pos, loss_nrm = self._losses(z)
            loss_free = (self._loss_free(vol)
                         if self.free_points is not None else None)

            loss = loss_pos
            total = loss_pos
            if loss_nrm is not None:
                total = total + self.lam * loss_nrm
            if loss_free is not None:
                total = total + self.lam_free * loss_free
            grad, = torch.autograd.grad(total, z)

        gn = grad.norm()
        if not torch.isfinite(gn) or gn == 0:
            return z0
        step = self.weight * z0.norm() / gn
        if self.log is not None:
            self.log.append((float(loss_pos), float(v.abs().mean()),
                             float(loss_nrm) if loss_nrm is not None else None,
                             float(loss_free) if loss_free is not None else None))
        return (z0.detach() - step * grad).to(z0.dtype)


def load_model(model_name="ADSKAILab/WaLa-SV-1B", scale=1.5, timesteps=5, device=None):
    """Charge WaLa une fois pour toutes — le chargement domine le coût d'un objet."""
    device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = Model.from_pretrained(model_name).to(device).eval()
    model.set_inference_fusion_params(scale, timesteps)
    net = getattr(model.network, "_orig_mod", model.network)
    return model, net, device


def image_condition(model, net, image_path, device):
    """Encode l'image une fois : elle ne change pas d'un poids ou d'une graine à l'autre."""
    tf = get_image_transform_latent_model()
    data = get_singleview_data(image_file=Path(image_path), image_transform=tf,
                               device=device, image_over_white=False)
    with torch.no_grad():
        feats = model.extract_input_features(data, data_type="image",
                                             is_train=False, to_cuda=True)
        return net.process_condition(feats)


def make_sampler(model, net, cond, name, scale, device):
    """Renvoie `sample(weight, seed, pts, out_dir)` pour un objet donné."""
    a = model.args
    shape = (1, a.e_dim, a.grid_size, a.grid_size, a.grid_size)

    def sample(weight, seed, pts, out_dir, normals=None, **gkw):
        torch.manual_seed(seed)          # même bruit initial à chaque poids :
        trace = []                       # la comparaison entre poids est appariée
        guided = ContactGuidedUNet(net.unet, model, pts, weight, trace,
                                   normals=normals, **gkw)
        latent, _ = net.inference_diffusion_module.p_sample_loop(
            model=guided, shape=shape, device=device, clip_denoised=False,
            progress=False,
            model_kwargs={"latent_codes": cond, "condition_zero": None,
                          "guidance_scale": scale, "dp_cond": None})
        with torch.no_grad():
            pred = model.autoencoder.decode_from_pre_quant(latent[0:1])
        wd = WaveletData(shape_list=model.dwt_sparse_composer.shape_list,
                         output_stage=a.max_training_level, max_depth=a.max_depth,
                         wavelet_volume=pred)
        low, highs = wd.convert_low_highs()
        out_dir.mkdir(parents=True, exist_ok=True)
        obj = out_dir / f"{name}.obj"
        model.save_visualization_obj(None, name, obj_path=str(obj),
                                     samples=(low, highs))
        return obj, trace

    return sample


def main():
    ap = argparse.ArgumentParser(description="Diffusion guidée par contrainte de contact")
    ap.add_argument("--image", required=True)
    ap.add_argument("--contacts", required=True,
                    help="Fichier .pt (N,3) ou (1,N,3) de points de contact, dans le "
                         "repère normalisé [-1,1] du maillage.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="ADSKAILab/WaLa-SV-1B")
    ap.add_argument("--guidance_weight", type=float, nargs="+", default=[0.05],
                    help="Un ou plusieurs poids, balayés en rechargeant le modèle une "
                         "seule fois. 0 désactive le guidage et reproduit exactement "
                         "l'image seule, ce qui donne le témoin apparié.")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--timesteps", type=int, default=5)
    ap.add_argument("--scale", type=float, default=1.5)
    args = ap.parse_args()

    model, net, device = load_model(args.model, args.scale, args.timesteps)

    contacts = torch.load(args.contacts, map_location="cpu", weights_only=True)
    if contacts.dim() == 3:
        contacts = contacts[0]
    contacts = contacts.to(device).float()
    print(f"[Guidage] {contacts.shape[0]} points de contact, poids {args.guidance_weight}")

    cond = image_condition(model, net, args.image, device)
    sample = make_sampler(model, net, cond, args.name, args.scale, device)

    for seed in args.seeds:
        # Passe 1, sans guidage : sert de témoin apparié ET établit le repère
        # canonique dans lequel les points de contact doivent être exprimés.
        t0 = time.time()
        ref_obj, _ = sample(0.0, seed, contacts, Path(args.out_dir) / f"g0_s{seed}")
        print(f"[Guidage] poids 0 graine {seed} : témoin ({time.time()-t0:.0f}s)",
              flush=True)

        pts, rmse = register_contacts(contacts.cpu().numpy(),
                                      o3d.io.read_triangle_mesh(str(ref_obj)))
        print(f"[Guidage] recalage des contacts sur la passe 1 : "
              f"résidu {rmse:.4f}", flush=True)
        pts_t = torch.from_numpy(pts).float().to(device)

        for weight in args.guidance_weight:
            if weight <= 0:
                continue                 # déjà produit comme témoin ci-dessus
            t0 = time.time()
            _, trace = sample(weight, seed, pts_t,
                              Path(args.out_dir) / f"g{weight:g}_s{seed}")
            # |SDF| moyen aux contacts : dit si la contrainte a effectivement été
            # rapprochée de zéro, indépendamment de toute métrique de forme.
            msg = (f"{trace[0][1]:.4f} -> {trace[-1][1]:.4f}" if trace else "n/a")
            print(f"[Guidage] poids {weight:g} graine {seed} : |SDF| aux contacts "
                  f"{msg}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
