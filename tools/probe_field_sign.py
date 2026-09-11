#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/probe_field_sign.py

Mesure de quel côté le champ décodé par WaLa est positif.

La contrainte de normale impose une **direction** au gradient du champ aux contacts.
Se tromper de signe reviendrait à demander à la surface de retourner sa normale : le
guidage pousserait exactement à l'envers, et l'expérience mesurerait une erreur de
convention. `save_visualization_obj` appelle `mcubes.marching_cubes(sdf, 0.0)` puis
inverse l'ordre des sommets (`triangles[:, ::-1]`), ce qui rend le raisonnement sur la
documentation peu sûr. On mesure donc.

Protocole : une passe non guidée, le champ décodé depuis le latent final, et la
différence `champ(p + eps*n) - champ(p - eps*n)` aux contacts placés par le repère
oracle, avec `n` la normale **sortante**. Si la moyenne est positive, l'extérieur est
positif.

Usage :
    python tools/probe_field_sign.py --object 002_master_chef_can
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.utils.wavelet_utils import WaveletData
from guided_sampling import load_model, image_condition, make_sampler, trilinear
from run_guidance_campaign import oracle_frame
from eval_fusion import load_and_normalize_mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/training_ycb")
    ap.add_argument("--object", default="002_master_chef_can")
    ap.add_argument("--timesteps", type=int, default=5)
    ap.add_argument("--scale", type=float, default=1.5)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--out_dir", default="out/probe_sign")
    args = ap.parse_args()

    obj = args.object
    d = Path(args.data_dir) / obj
    model, net, device = load_model(scale=args.scale, timesteps=args.timesteps)
    cond = image_condition(model, net, d / "image.png", device)
    sample = make_sampler(model, net, cond, obj, args.scale, device)

    ref_obj, _ = sample(0.0, 0, torch.zeros(1, 3, device=device),
                        Path(args.out_dir) / obj)
    gt_mesh, _ = load_and_normalize_mesh(d / "mesh.obj")
    O = oracle_frame(gt_mesh, ref_obj)

    z = np.load(f"data/contacts/{obj}/dh116_8.npz")
    pts = (O[:3, :3] @ z["points"].T).T + O[:3, 3]
    R = O[:3, :3]
    nrm = (R @ z["normals"].T).T
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)

    # Le champ décodé depuis le latent final : on relance une passe et on garde le
    # volume plutôt que le maillage.
    a = model.args
    torch.manual_seed(0)
    latent, _ = net.inference_diffusion_module.p_sample_loop(
        model=net.unet, shape=(1, a.e_dim, a.grid_size, a.grid_size, a.grid_size),
        device=device, clip_denoised=False, progress=False,
        model_kwargs={"latent_codes": cond, "condition_zero": None,
                      "guidance_scale": args.scale, "dp_cond": None})
    with torch.no_grad():
        pred = model.autoencoder.decode_from_pre_quant(latent[0:1])
        wd = WaveletData(shape_list=model.dwt_sparse_composer.shape_list,
                         output_stage=a.max_training_level, max_depth=a.max_depth,
                         wavelet_volume=pred)
        vol = model.dwt_inverse_3d(wd.convert_low_highs())

        P = torch.from_numpy(pts).float().to(device)
        N = torch.from_numpy(nrm).float().to(device)
        v0 = trilinear(vol, P)
        vout = trilinear(vol, P + args.eps * N)
        vin = trilinear(vol, P - args.eps * N)

    diff = (vout - vin).cpu().numpy()
    print(f"\n=== {obj} — grille {vol.shape[-1]}³, eps {args.eps}")
    print(f"champ aux contacts        : moyenne {v0.mean():+.4f}  "
          f"|.| {v0.abs().mean():.4f}")
    print(f"champ(p+eps*n) (extérieur): moyenne {vout.mean().item():+.4f}")
    print(f"champ(p-eps*n) (intérieur): moyenne {vin.mean().item():+.4f}")
    print(f"différence sortant-entrant: moyenne {diff.mean():+.4f}, "
          f"{(diff > 0).sum()}/{len(diff)} positives")
    s = 1.0 if diff.mean() > 0 else -1.0
    print(f"\n>>> field_sign = {s:+.0f}  "
          f"({'extérieur positif' if s > 0 else 'intérieur positif'})")
    if (diff > 0).sum() not in (0, len(diff)):
        print("!!! signe non unanime : convention non fiable, ne pas guider sur "
              "les normales avant d'avoir compris pourquoi.")


if __name__ == "__main__":
    main()
