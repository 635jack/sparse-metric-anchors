### Standard Library Imports
import os
import logging
import argparse
import subprocess
import json
import random
import math
import datetime
import os.path as osp
import inspect
from pathlib import Path

### Third-Party Library Imports
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import tqdm
import mcubes
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger

### Network and Model Imports
from src.latent_model import continous_diffusion_interface
from src.diffusion_modules.dwt import DWTInverse3d
from src.diffusion_modules.sparse_network import SparseComposer
from src.experiments.utils.wavelet_utils import WaveletData
from src.experiments.utils.wavelet_utils import (
    extract_wavelet_coefficients,
    extract_full_indices,
    extract_highs_from_values,
    pad_with_batch_idx,
)
from src.latent_model.points_network import (
    PointNet_Large,
    PointNet_Simple,
)
from src.latent_model.voxels_network import (
    Encoder_Down_Interpolate,
    Encoder_Down_2,
)
from src.fusion_module import CondFeatureFusion, PointTrunkEncoder

### External Libraries (e.g., CLIP and Ray)
from src.clip_mod import get_clip_model, tokenize
from src.latent_model import wavelet_vq_model

import time

def load_ema_state_dict(checkpoint_path, args):
    print(f"Loading model from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict")
    print("Not using EMA for autoencoder")

    if state_dict is None:
        raise ValueError("Autoencoder path is required to load weights.")

    return state_dict


# def setup_autoencoder(args):
#     logging.info("Setting up Autoencoder")
#     if args.use_ray:
#         checkpoint_path = save_checkpoint_locally(args.exp_name, args.checkpoint_type, args.debug_base_folder)
#     else:
#         checkpoint_path = f"{args.checkpoint_dir_base}/{args.checkpoint_type}.ckpt"

#     ema_state_dict = load_ema_state_dict(checkpoint_path, args)

#     if ema_state_dict:
#         autoencoder_module = Trainer_Geo_Network(args)
#         autoencoder_module.load_state_dict(ema_state_dict)
#         logging.info("Autoencoder model loaded successfully.")
#     else:
#         logging.error("Failed to load EMA state dict. Autoencoder setup incomplete.")

#     return autoencoder_module.network



def setup_autoencoder(args):
    logging.info("Setting up Autoencoder")
    autoencoder_module = Trainer_Geo_Network(args)
    return autoencoder_module.network


class Trainer_Geo_Network(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.network = wavelet_vq_model.get_model(args)
        self.args = args
        if args.use_compile == True:
            self.network = torch.compile(
                self.network
            )  # compiles the model and *step (training/validation/prediction)
        # print(self.network)


class Trainer_Condition_Network(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset_path = args.dataset_path
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.use_autoencoder = args.use_autoencoder
        self.image_transform = args.image_transform
        self.encoder = None

        # helper.create_dir(args.experiment_dir)
        # Clear GPU cache to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        # Determine return representations based on autoencoder usage
        latent_rep = "latent" if args.use_autoencoder_ema else "latent_code"
        self.return_reps = ["Wavelet"] if self.use_autoencoder else [latent_rep]

        # print(os.listdir("/home/ray/ray_results/"))
        self.autoencoder = setup_autoencoder(args)

        # Setup conditions
        self.setup_conditions()

        # Initialize the main network and autoencoder
        self.network = continous_diffusion_interface.get_model(args)

        # Freeze autoencoder parameters
        # self.freeze_autoencoder()

        # Initialize sparse models
        self.dwt_sparse_composer = SparseComposer(
            input_shape=[args.resolution, args.resolution, args.resolution],
            J=args.max_depth,
            wave=args.wavelet,
            mode=args.padding_mode,
            inverse_dwt_module=None,
        )
        self.dwt_inverse_3d = DWTInverse3d(
            J=args.max_depth, wave=args.wavelet, mode=args.padding_mode
        )

        # Optionally compile models
        self.compile_models_if_needed()

        # Print network structure for debugging
        # print(self.network)
        # print(self.encoder)

    def setup_conditions(self):
        """Setup various condition modules based on the arguments."""
        # ---- Mode fusion multimodale (image + voxel simultanément) ----
        if (
            hasattr(self.args, "use_fusion_conditions")
            and self.args.use_fusion_conditions
        ):
            self.setup_fusion_conditions()
            return  # les encodeurs image et voxel sont initialisés dans setup_fusion_conditions

        if (
            hasattr(self.args, "use_image_conditions")
            and self.args.use_image_conditions
        ):
            self.setup_image_conditions()

        if (
            hasattr(self.args, "use_depth_conditions")
            and self.args.use_depth_conditions
        ):
            self.setup_depth_conditions()

        if (
            hasattr(self.args, "use_wavelet_conditions")
            and self.args.use_wavelet_conditions
        ):
            self.setup_wavelet_conditions()

        if (
            hasattr(self.args, "use_pointcloud_conditions")
            and self.args.use_pointcloud_conditions
        ):
            self.setup_pointcloud_conditions()

        if (
            hasattr(self.args, "use_voxel_conditions")
            and self.args.use_voxel_conditions
        ):
            self.setup_voxel_conditions()

    def setup_depth_conditions(self):
        """Setup image conditions using CLIP if applicable."""
        args, clip_model, clip_preprocess = get_clip_model(self.args)
        self.args.condition_dim = self.args.cond_grid_emb_size
        self.args.num_cond_vectors = self.args.cond_grid_size
        self.clip_model = clip_model
        # Freeze CLIP model parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.return_reps.append("depth")

    def setup_image_conditions(self):
        """Setup image conditions using CLIP if applicable."""
        args, clip_model, clip_preprocess = get_clip_model(self.args)
        self.args.condition_dim = self.args.cond_grid_emb_size
        self.args.num_cond_vectors = self.args.cond_grid_size
        self.clip_model = clip_model
        # Freeze CLIP model parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.return_reps.append("image")

    def setup_wavelet_conditions(self):
        """Setup wavelet conditions if applicable."""
        high_size = (
            511
            if not hasattr(self.args, "max_training_level")
            or self.args.max_training_level == self.args.max_depth
            else (2**3) ** self.args.max_training_level - 1
        )
        self.encoder = WaveletEncoder(ae_input_channel=1 + high_size, args=self.args)

    def setup_pointcloud_conditions(self):
        """Setup point cloud conditions if applicable."""
        if self.args.pc_encoder_type == "PointNet_Simple":
            self.encoder = PointNet_Simple(
                output_dim=self.args.pc_output_dim,
                pc_dims=self.args.pc_dims,
                num_inds=self.args.num_inds,
                num_heads=16,
            )
        else:
            self.encoder = PointNet_Large(
                output_dim=self.args.pc_output_dim,
                pc_dims=self.args.pc_dims,
                num_inds=self.args.num_inds,
                num_heads=16,
            )

        self.return_reps.append("Pointcloud")

    def setup_voxel_conditions(self):
        """Setup voxel conditions if applicable."""
        self.encoder = Encoder_Down_2(
            channel_in=1, channel_out=self.args.voxel_context_dim
        )
        self.return_reps.append(f"Voxel_{self.args.voxel_resolution}")

    def setup_fusion_conditions(self):
        """Setup fusion multimodale image + voxel.

        Initialise :
          - self.clip_model  : encodeur image CLIP (gelé)
          - self.voxel_encoder : Encoder_Down_2 (entraînable)
          - self.fusion_module : CondFeatureFusion (new, entraînable)

        La dimension de sortie du module de fusion doit être égale à
        cond_grid_emb_size (dim du contexte attendu par le DiT).
        La longueur de séquence doit correspondre à cond_grid_size.
        """
        # ---- Encodeur image (CLIP/DINOv2, gelé) ----
        # Réutiliser l'encodeur déjà chargé s'il existe. Quand la fusion est activée
        # après coup (cas de tools/train_fusion.py, qui appelle from_pretrained puis
        # setup_fusion_conditions), un second DINOv2 était chargé et venait remplacer
        # le premier : ~1,2 Go de mémoire unifiée gaspillés, ce qui compte sur un Mac
        # 32 Go où l'entraînement en consomme déjà ~19.
        self.args.condition_dim = self.args.cond_grid_emb_size
        self.args.num_cond_vectors = self.args.cond_grid_size
        if getattr(self, "clip_model", None) is None:
            args, clip_model, clip_preprocess = get_clip_model(self.args)
            self.clip_model = clip_model.to(self.device)
        else:
            print("[Fusion] Encodeur image déjà chargé — réutilisé au lieu d'être rechargé.")
            self.clip_model = self.clip_model.to(self.device)
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # ---- Encodeur géométrique (entraînable) ----
        # Attribut distinct de self.encoder, pour ne pas écraser l'encodeur du
        # chemin mono-modal.
        self.fusion_geom = self.args.get("fusion_geom", None) or "voxel"
        self.voxel_encoder = None
        self.point_encoder = None

        if self.fusion_geom == "pointcloud":
            pc_output_dim = self.args.get("pc_output_dim", None) or self.args.cond_grid_emb_size
            pc_pooling = self.args.get("pc_pooling", None) or "none"
            if pc_pooling == "none":
                # Un token par point, sans agrégation : la cross-attention de
                # CondFeatureFusion agrège déjà, pilotée par les tokens image.
                # Le pooling IAB de PointNet_Simple divisait le pouvoir discriminant
                # par 273 (voir PointTrunkEncoder).
                self.point_encoder = PointTrunkEncoder(
                    output_dim=pc_output_dim,
                    pc_dims=self.args.get("pc_dims", None) or 1024,
                ).to(self.device)
            else:
                self.point_encoder = PointNet_Simple(
                    output_dim=pc_output_dim,
                    pc_dims=self.args.get("pc_dims", None) or 1024,
                    num_inds=self.args.get("num_inds", None) or 256,
                    num_heads=16,
                ).to(self.device)
            print(f"[Fusion] Encodeur de points : pooling '{pc_pooling}'")
            geom_dim = pc_output_dim
        elif self.fusion_geom == "voxel":
            voxel_context_dim = self.args.get("voxel_context_dim", None) or self.args.cond_grid_emb_size
            self.voxel_encoder = Encoder_Down_2(
                channel_in=1, channel_out=voxel_context_dim
            ).to(self.device)
            geom_dim = voxel_context_dim
        else:
            raise ValueError(
                f"fusion_geom inconnu : {self.fusion_geom!r} (attendu 'voxel' ou 'pointcloud')"
            )

        # ---- Module de fusion ----
        # output_dim = cond_grid_emb_size (dim contexte du DiT)
        # output_seq_len = cond_grid_size (longueur de séquence du DiT)
        fused_dim = self.args.get("fusion_fused_dim", None) or self.args.cond_grid_emb_size
        n_heads = self.args.get("fusion_n_heads", None) or 8
        dropout = self.args.get("fusion_dropout", None) or 0.0
        fusion_mode = self.args.get("fusion_mode", None) or "cross"
        self.fusion_module = CondFeatureFusion(
            fused_dim=fused_dim,
            n_heads=n_heads,
            dropout=dropout,
            output_dim=self.args.cond_grid_emb_size,
            output_seq_len=self.args.cond_grid_size,
            mode=fusion_mode,
            gate_mode=self.args.get("gate_mode", None) or "channel",
        ).to(self.device)
        # Instancier tout de suite les projections d'entrée : sinon elles naissent
        # au premier forward, donc après la construction de l'optimiseur, et leurs
        # 2 099 200 paramètres — dont toute la projection des voxels — ne seraient
        # jamais mis à jour.
        self.fusion_module.materialize(
            in_dim_a=self.args.cond_grid_emb_size,
            in_dim_b=geom_dim,
            device=self.device,
        )
        print(f"[Fusion] Mode de fusion : {fusion_mode} | modalité géométrique : {self.fusion_geom}")

        self.return_reps.append("image")
        if self.fusion_geom == "pointcloud":
            self.return_reps.append("Pointcloud")
        else:
            self.return_reps.append(f"Voxel_{self.args.get('voxel_resolution', None) or 16}")

    def freeze_autoencoder(self):
        """Freeze the autoencoder parameters."""
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def compile_models_if_needed(self):
        """Optionally compile models for faster execution."""
        if self.args.use_compile:
            self.network = torch.compile(self.network)
            if self.encoder is not None:
                self.encoder = torch.compile(self.encoder)

    def extract_input_image_features(self, image, is_train=False):
        """Extract image features from the input."""
        with torch.no_grad():
            if (
                hasattr(self.args, "use_multiple_views_grids")
                and self.args.use_multiple_views_grids
            ):
                batch_size = image.size(0)
                image_reshaped = image.view(
                    batch_size * image.size(1),
                    image.size(2),
                    image.size(3),
                    image.size(4),
                )
                image_features, image_features_grids = (
                    self.clip_model.get_image_features(image_reshaped)
                )
                input_features = image_features_grids[-1].view(
                    batch_size, image.size(1), -1, image_features_grids[-1].size(-1)
                )
                input_features = input_features.detach()
            else:
                image_features, image_features_grids = (
                    self.clip_model.get_image_features(image)
                )
                input_features = image_features_grids[-1].detach()
            return input_features.float()

    def extract_input_features(self, data, data_type, is_train=False, to_cuda=False):
        """Extract input features from the data."""
        if to_cuda:
            for key, item in data.items():
                if torch.is_tensor(item):
                    data[key] = item.to(self.device)

        assert data_type in [
            "Wavelet",
            "Pointcloud",
            "Voxel",
            "image",
            "texts",
            "depth",
            "fusion",
        ]
        if data_type == "Wavelet":
            low = data["low"]
            high_indices = data["high_indices"]
            high_values = data["high_values"]
            wavelet_data = WaveletData(
                shape_list=self.network.dwt_sparse_composer.shape_list,
                output_stage=self.args.max_training_level,
                max_depth=self.args.max_depth,
                low=low,
                highs_indices=high_indices,
                highs_values=high_values,
            )
            wavelet_inputs = wavelet_data.convert_wavelet_volume()
            input_features = self.encoder(wavelet_inputs)
        elif data_type == "Pointcloud":
            points_inputs = data["Pointcloud"]
            input_features = self.encoder(points_inputs.type(torch.FloatTensor).to(self.device))
        elif data_type == "Voxel":
            voxels_inputs = data["voxels"]
            input_features = self.encoder(voxels_inputs)
            input_features = torch.permute(input_features, (0, 2, 3, 4, 1))
            input_features = input_features.view(
                (
                    input_features.size(0),
                    input_features.size(1)
                    * input_features.size(2)
                    * input_features.size(3),
                    input_features.size(4),
                )
            )
        elif data_type == "image":
            image = data["images"].type(torch.FloatTensor).to(self.device)
            input_features = self.extract_input_image_features(image, is_train=is_train)
        elif data_type == "depth":
            image = data["depth"].type(torch.FloatTensor).to(self.device)
            input_features = self.extract_input_image_features(image, is_train=is_train)
        elif data_type == "fusion":
            # ---- Mode fusion multimodale : image + géométrie ----
            # 1. Features image via CLIP
            image = data["images"].type(torch.FloatTensor).to(self.device)
            feat_img = self.extract_input_image_features(image, is_train=is_train)
            # feat_img : (B, cond_grid_size, cond_grid_emb_size) ou (B, N_grid, D)

            # 2. Features géométriques, selon la modalité configurée
            feat_geom = self.extract_geometry_features(data)

            # 3. Fusion → (B, cond_grid_size, cond_grid_emb_size)
            input_features = self.fusion_module(feat_img, feat_geom)
        else:
            raise Exception("Unknown features....")

        return input_features

    @property
    def geometry_encoder(self):
        """L'encodeur géométrique actif de la fusion (voxel ou nuage de points).

        Permet aux scripts d'entraînement et d'inférence de geler, collecter et
        sauvegarder les paramètres sans connaître la modalité configurée.
        """
        if getattr(self, "fusion_geom", None) == "pointcloud":
            return self.point_encoder
        return getattr(self, "voxel_encoder", None)

    def extract_geometry_features(self, data):
        """Encode la modalité géométrique de la fusion en tokens (B, N, D).

        Deux modalités, choisies par `args.fusion_geom` :

        - "voxel"      : grille d'occupation 16^3 -> Encoder_Down_2 -> (B, 512, D).
                         Complète et régulière, mais sa capacité à décrire un objet
                         dépend fortement de sa forme : une sphère occupe ~1250
                         voxels, une règle plate seulement 51.

        - "pointcloud" : nuage de points -> PointNet_Simple -> (B, num_inds, D).
                         Le pooling par attention induite rend le nombre de tokens
                         de sortie indépendant du nombre de points en entrée, donc
                         la densité du nuage est un paramètre libre. C'est aussi
                         plus fidèle à un capteur tactile, qui produit des points de
                         contact épars et non une occupation volumique.
        """
        if self.fusion_geom == "pointcloud":
            points = data["Pointcloud"].type(torch.FloatTensor).to(self.device)
            return self.point_encoder(points)  # (B, num_inds, pc_output_dim)

        voxels = data["voxels"].type(torch.FloatTensor).to(self.device)
        feat_vox = self.voxel_encoder(voxels)  # (B, voxel_context_dim, 8, 8, 8)
        feat_vox = torch.permute(feat_vox, (0, 2, 3, 4, 1))
        return feat_vox.view(
            feat_vox.size(0),
            feat_vox.size(1) * feat_vox.size(2) * feat_vox.size(3),
            feat_vox.size(4),
        )

    def concat_images_spatially(self, images):
        """Concatenate images spatially in a square-like manner."""
        batch_size, num_images, channels, height, width = images.size()

        # Calculate the square grid dimensions
        grid_size = math.ceil(math.sqrt(num_images))

        # Pad images to fit into the grid
        padded_images = torch.cat(
            [
                images,
                torch.zeros(
                    batch_size, grid_size**2 - num_images, channels, height, width
                ).to(images.device),
            ],
            dim=1,
        )

        # Reshape and concatenate into a grid for each batch
        padded_images = padded_images.view(
            batch_size, grid_size, grid_size, channels, height, width
        )
        concat_images = padded_images.permute(
            0, 3, 1, 4, 2, 5
        ).contiguous()  # Rearrange dimensions for concatenation
        concat_images = concat_images.view(
            batch_size, channels, grid_size * height, grid_size * width
        )

        return concat_images

    def extract_images(self, data, img_idx=0, data_type="images"):
        image = data[data_type].type(torch.FloatTensor).to(self.device)
        if len(image.size()) > 4:
            return self.concat_images_spatially(image)
        else:
            return image

    def extract_img_idx(self, data, data_idx=None, data_type="img"):
        """Extract image index if using camera index."""
        if hasattr(self.args, "use_camera_index") and self.args.use_camera_index:
            img_idx = data[f"{data_type}_idx"].type(torch.LongTensor).to(self.device)
            return img_idx[data_idx : data_idx + 1] if data_idx is not None else img_idx
        return None

    def encode_to_z(self, data):
        """Encode data into latent space."""

        if not self.use_autoencoder:
            # Return pre_quant or its converted form based on the flag
            return (
                data["pre_quant"]
                if self.args.pre_quant
                else self.autoencoder.convert_to_post_quant(data["pre_quant"])
            )

        # Extract relevant data from the input dictionary
        low = data["low"]
        high_indices = data["high_indices"]
        high_values = data["high_values"]
        high_values_mask = data["high_values_mask"]
        high_indices_empty = data["high_indices_empty"]

        # Perform encoding in no-gradient mode
        with torch.no_grad():
            pre_quant, _, post_quant = self.autoencoder.encode_to_z(
                low,
                high_indices,
                high_values,
                high_indices_empty=high_indices_empty,
                high_values_mask=high_values_mask,
            )

        # Return pre_quant or post_quant based on the pre_quant flag
        return pre_quant if self.args.pre_quant else post_quant

    def save_visualization_obj(self,experiments,image_name, obj_path, samples):
        """Save a visualization object."""
        low, highs = samples
        t7 = time.time()
        sdf_recon = self.dwt_inverse_3d((low, highs))
        t8 = time.time()
        print('Inverse DWT time elapsed', t8- t7,'s')
        vertices, triangles = mcubes.marching_cubes(
            sdf_recon.cpu().detach().numpy()[0, 0], 0.0
        )
        t9 = time.time()
        print('mcubes.marching_cubes time', t9-t8, 's')
        vertices = (vertices / self.args.resolution) * 2.0 - 1.0
        triangles = triangles[:, ::-1]
        mcubes.export_obj(vertices, triangles, obj_path)
        t10 = time.time()
        print('export obj time', time.time() - t9,'s')

        times_7_9 = {
            'Inverse DWT time elapsed': t8 - t7,
            'mcubes.marching_cubes time': t9 - t8,
            'export obj time': t10 - t9
        }
        try:

            for metric_name, value in times_7_9.items():
                experiments.log_metric(metric_name, value)
        except:
            
            pass



    def save_visualization_sdf(self, obj_path, sdf):
        """Save a visualization SDF."""
        sdf = sdf.cpu().detach().numpy()
        if not np.isfinite(sdf).all():
            raise ValueError("SDF contains NaN or infinite values.")
        vertices, triangles = mcubes.marching_cubes(sdf, 0.0)
        vertices = (vertices / self.args.resolution) * 2.0 - 1.0
        mcubes.export_obj(vertices, triangles, obj_path)

    def set_inference_fusion_params(self, scale, diffusion_rescale_timestep):
        self.args.scale = scale
        self.args.diffusion_rescale_timestep = diffusion_rescale_timestep
        self.network.reset_diffusion_module()

    def inference_sample(
        self, data, data_idx, experiments, return_wavelet_volume=False, progress=True
    ):
        # Generate prediction and save visualization

        low_data = data["low"].type(torch.FloatTensor).to(self.device)

        if (
            hasattr(self.args, "use_fusion_conditions")
            and self.args.use_fusion_conditions
        ):
            t0 = time.time()
            condition_features = self.extract_input_features(
                data, data_type="fusion", is_train=False, to_cuda=True
            )
            img_idx = self.extract_img_idx(data, data_idx=data_idx)
            t1 = time.time()
            print(f"[Fusion] Feature extraction: {t1 - t0:.3f}s")
            print(f"[Fusion] feat shape: {condition_features.shape}")

            latent = self.network.inference(
                condition_features.size(0),
                condition_features,
                scale=self.args.scale,
                image_index=img_idx,
            )
            t2 = time.time()
            print(f"[Fusion] Latent diffusion: {t2 - t1:.3f}s")

        elif self.args.use_image_conditions:
            t0 = time.time()

            condition_features = self.extract_input_features(
                data, data_type="image", is_train=False, to_cuda=True
            )
           
            img_idx = self.extract_img_idx(data, data_idx=data_idx)

            t1 = time.time()
            print('Extract Image', t1 - t0, 's')

            latent = self.network.inference(
                condition_features.size(0),
                condition_features,
                scale=self.args.scale,
                image_index=img_idx,
            )
            t2 = time.time()
            print("Latent Diffusion Time", t2-t1,'s')

        elif (
            hasattr(self.args, "use_pointcloud_conditions")
            and self.args.use_pointcloud_conditions
        ):
            t0 = time.time()
            condition_features = self.extract_input_features(
                data, data_type="Pointcloud", is_train=False, to_cuda=True
            )
            t1 = time.time()
            print('Extract Image', t1 - t0, 's')

            latent = self.network.inference(
                condition_features.size(0),
                condition_features,
                None,
            )
            t2 = time.time()
            print("Latent Diffusion Time", t2-t1,'s')

        elif (
            hasattr(self.args, "use_voxel_conditions")
            and self.args.use_voxel_conditions
        ):
            t0 = time.time()
            condition_features = self.extract_input_features(
                data, data_type="Voxel", is_train=False, to_cuda=True
            )
            t1 = time.time()
            print('Extract Voxel', t1 - t0, 's')
            # `inference(batch_size, condition, scale, image_index)` : le troisième
            # argument est l'échelle de guidage. L'amont y passait None, et le
            # sampler substitue alors 3 (`gaussian_diffusion.py:380`). L'échelle
            # retenue par les auteurs pour le voxel est 1,5 — la valeur imposée était
            # donc le double, et un guidage sans classifieur deux fois trop fort
            # sature la sortie : 895 508 faces de bruit sur l'exemple officiel du
            # cheval, au lieu d'un cheval.
            latent = self.network.inference(
                condition_features.size(0), condition_features,
                scale=self.args.scale
            )
            t2 = time.time()
            print("Latent Diffusion Time", t2-t1,'s')

        elif (
            hasattr(self.args, "use_depth_conditions")
            and self.args.use_depth_conditions
        ):
            t0 = time.time()
            condition_features = self.extract_input_features(
                data, data_type="depth", is_train=False, to_cuda=True
            )
            t1 = time.time()
            print('Extract Image', t1 - t0, 's')
            img_idx = self.extract_img_idx(data, data_idx=data_idx)
            print(img_idx)
            latent = self.network.inference(
                condition_features.size(0),
                condition_features,
                scale=self.args.scale,
                image_index=img_idx,
            )
            t2 = time.time()
            print("Latent Diffusion Time", t2-t1,'s')
        else:
            latent = self.network.inference(
                low_data[data_idx : data_idx + 1],
                None,
                None,
                local_rank=0,
                current_stage=self.current_stage,
                return_wavelet_volume=return_wavelet_volume,
                progress=progress,
            )
            t2 = time.time()
            print("Latent Diffusion Time", t2-t1,'s')
        
        pred = self.autoencoder.decode_from_pre_quant(latent[data_idx : data_idx + 1])

        t3 = time.time()
        print('Latent Decoding Time', t3 - t2, 's')

        wavelet_data_pred = WaveletData(
            shape_list=self.dwt_sparse_composer.shape_list,
            output_stage=self.args.max_training_level,
            max_depth=self.args.max_depth,
            wavelet_volume=pred,
        )

        t4 = time.time()

        print('Wavelet Preparation Time', t4-t3,'s')
        low_pred, highs_pred = wavelet_data_pred.convert_low_highs()
        t5 = time.time()
        print('Low to Highs conversion', t5-t4,'s')


        times_1_5 = {'Extract Image': t1-t0,
         'Latent Diffusion Time': t2-t1,
         'Latent Decoding Time': t3-t2, 
         'Wavelet Preparation Time': t4-t3, 
         'Low to Highs conversion': t5-t4}
        
        try:
            for metric_name, value in times_1_5.items():
                experiments.log_metric(metric_name, value)        
        except:
            pass

        return low_pred, highs_pred




    def test_inference(self, data, data_idx,image_name, save_dir,experiments= None, output_format="obj"):
        file_name = data["id"][data_idx]
        with torch.no_grad():
            low_pred, highs_pred = self.inference_sample(
                data, data_idx,experiments, return_wavelet_volume=False
            )

        if output_format == "sdf":
            sdf_path = os.path.join(save_dir, f"{file_name}.npz")
            self.save_sdf(sdf_path, (low_pred, highs_pred))
            return sdf_path
        else:
            t6 = time.time()
            obj_path = os.path.join(save_dir, f"{image_name}.obj")
            self.save_visualization_obj(experiments,image_name,
                obj_path=obj_path, samples=(low_pred, highs_pred)
            )
            print('Time to save visualization from high lows to obj', time.time() - t6,'s')
            time6 = {'Time to save visualization from high lows to obj':time.time() - t6}

            try:

                for metric_name, value in time6.items():
                    experiments.log_metric(metric_name,value)

            except:

                pass
            return obj_path




    def forward(
        self, data, data_idx,experiments = None, output_format="obj"
    ):
        # Generate prediction and save visualization

        low_data = data["low"].type(torch.FloatTensor).to(self.device)

        if self.args.use_image_conditions:
            t0 = time.time()

            condition_features = self.extract_input_features(
                data, data_type="image", is_train=False, to_cuda=True
            )
           
            img_idx = self.extract_img_idx(data, data_idx=data_idx)

            t1 = time.time()
            print('Extract Image', t1 - t0, 's')

            latent = self.network.inference(
                condition_features.size(0),
                condition_features,
                scale=self.args.scale,
                image_index=img_idx,
            )
            t2 = time.time()
            print("Latent Diffusion Time", t2-t1,'s')

        elif (

            hasattr(self.args, "use_pointcloud_conditions")
            and self.args.use_pointcloud_conditions
        ):
            t0 = time.time()
            condition_features = self.extract_input_features(
                data, data_type="Pointcloud", is_train=False, to_cuda=True
            )
            
            t1 = time.time()
            print('Extract Image', t1 - t0, 's')

            latent = self.network.inference(
                condition_features.size(0),
                condition_features,
                None,
            )
            t2 = time.time()
            print("Latent Diffusion Time", t2-t1,'s')
        elif (

            hasattr(self.args, "use_voxel_conditions")
            and self.args.use_voxel_conditions
        ):
            t0 = time.time()
            condition_features = self.extract_input_features(
                data, data_type="Voxel", is_train=False, to_cuda=True
            )

            t1 = time.time()
            print('Extract Image', t1 - t0, 's')

            latent = self.network.inference(
                condition_features.size(0), condition_features, None
            )
            t2 = time.time()
            print("Latent Diffusion Time", t2-t1,'s')
        elif (

            hasattr(self.args, "use_depth_conditions")
            and self.args.use_depth_conditions
        ):
            t0 = time.time()
            condition_features = self.extract_input_features(
                data, data_type="depth", is_train=False, to_cuda=True
            )
            img_idx = self.extract_img_idx(data, data_idx=data_idx)
            print(img_idx)

            t1 = time.time()
            print('Extract Image', t1 - t0, 's')

            latent = self.network.inference(
                condition_features.size(0),
                condition_features,
                scale=self.args.scale,
                image_index=img_idx,
            )
            t2 = time.time()
            print("Latent Diffusion Time", t2-t1,'s')
        else:
                        
            t0 = time.time()
            t1 = time.time()
            print('Extract Image', t1 - t0, 's')
            latent = self.network.inference(
                low_data[data_idx : data_idx + 1],
                None,
                None,
                local_rank=0,
                current_stage=self.current_stage,
                return_wavelet_volume=return_wavelet_volume,
                progress=progress,
            )
            t2 = time.time()
            print("Latent Diffusion Time", t2-t1,'s')
        
        pred = self.autoencoder.decode_from_pre_quant(latent[data_idx : data_idx + 1])

        t3 = time.time()
        print('Latent Decoding Time', t3 - t2, 's')

        wavelet_data_pred = WaveletData(
            shape_list=self.dwt_sparse_composer.shape_list,
            output_stage=self.args.max_training_level,
            max_depth=self.args.max_depth,
            wavelet_volume=pred,
        )

        t4 = time.time()

        print('Wavelet Preparation Time', t4-t3,'s')
        low_pred, highs_pred = wavelet_data_pred.convert_low_highs_trt()
        t5 = time.time()
        print('Low to Highs conversion', t5-t4,'s')


        times_1_5 = {'Extract Image': t1-t0,
         'Latent Diffusion Time': t2-t1,
         'Latent Decoding Time': t3-t2, 
         'Wavelet Preparation Time': t4-t3, 
         'Low to Highs conversion': t5-t4}
        
        try:
            for metric_name, value in times_1_5.items():
                experiments.log_metric(metric_name, value)        
        except:
            pass

        return low_pred, highs_pred
