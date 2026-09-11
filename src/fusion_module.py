#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/fusion_module.py

Module de fusion multimodale pour WaLa.
Fusionne les features extraites par deux encodeurs différents (ex: CLIP image
et Encoder_Down_2 voxel) pour produire un vecteur de conditionnement unique
compatible avec le réseau de diffusion DiT/UVIT.

Design :
  1. Aplatir chaque feature vers (B, N, D_in)
     → accepte (B, D), (B, N, D), (B, C, H, W), (B, C, X, Y, Z)
  2. Projection linéaire vers une dim commune D_fused
  3. Modality embedding (learned) pour différencier les modalités
  4. CLS token + concatenation des deux séquences
  5. 1 bloc transformer léger (self-attention + MLP)
  6. Projection vers la dim de sortie cible (context_emb_dim du DiT)
  7. Troncature/padding vers une longueur fixe pour compatibilité cond_pos_emb

Différences par rapport à fusion_module.py (racine) :
  - Pas de main() / argparse (module pur, pas de script standalone)
  - Ajout de output_seq_len pour forcer une longueur de séquence fixe
  - Ajout de output_dim pour projeter vers la dim du DiT
  - Fix MPS : pas de .half() sur les paramètres LazyLinear (MPS fp32)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _to_tokens(x: torch.Tensor) -> torch.Tensor:
    """Ramène n'importe quel tenseur dense vers (B, N, D).

    Cas gérés :
      (B, D)           → (B, 1, D)        single token
      (B, N, D)        → identité
      (B, C, H, W)     → (B, H*W, C)
      (B, C, X, Y, Z)  → (B, X*Y*Z, C)
    """
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() == 3:
        return x
    if x.dim() == 4:
        B, C, H, W = x.shape
        return x.permute(0, 2, 3, 1).reshape(B, H * W, C)
    if x.dim() == 5:
        B, C, X, Y, Z = x.shape
        return x.permute(0, 2, 3, 4, 1).reshape(B, X * Y * Z, C)
    raise ValueError(f"Forme non supportée pour la fusion : {tuple(x.shape)}")


class _LazyLinear(nn.Module):
    """Projection linéaire avec init paresseuse.
    Évite de hardcoder D_in qui dépend du clip_model_type choisi.
    Compatible MPS (float32 uniquement, pas de .half()).

    `init_identity` : si les dimensions d'entrée et de sortie coïncident, initialise
    la projection à l'identité plutôt qu'aléatoirement. Utilisé pour que la fusion
    parte d'un passthrough exact des features image (voir CondFeatureFusion).
    """

    def __init__(self, out_dim: int, bias: bool = True, init_identity: bool = False):
        super().__init__()
        self.out_dim = out_dim
        self.use_bias = bias
        self.init_identity = init_identity
        self.proj: nn.Linear | None = None  # construit au premier forward

    def _build(self, in_dim: int, device=None, dtype=None) -> nn.Linear:
        proj = nn.Linear(in_dim, self.out_dim, bias=self.use_bias)
        if self.init_identity and in_dim == self.out_dim:
            with torch.no_grad():
                proj.weight.copy_(torch.eye(self.out_dim))
                if self.use_bias:
                    proj.bias.zero_()
        return proj.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.proj is None:
            self.proj = self._build(x.shape[-1], device=x.device, dtype=x.dtype)
        return self.proj(x)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        weight_key = prefix + 'proj.weight'
        if weight_key in state_dict:
            self.proj = self._build(
                state_dict[weight_key].shape[1],
                device=state_dict[weight_key].device,
            )
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


# ---------------------------------------------------------------------------
# Module principal
# ---------------------------------------------------------------------------

class PointTrunkEncoder(nn.Module):
    """Encodeur de nuage de points **sans pooling** : un token par point.

    Reprend le tronc MLP par point de `PointNet_Simple` (4 couches Linear + LayerNorm
    + ReLU) et s'arrête là, au lieu d'agréger via le bloc d'attention induite (IAB).

    Pourquoi retirer le pooling. Mesuré sur 5 objets YCB très différents, distance
    cosinus moyenne entre objets :
        - features par point (sortie du tronc)  : 0.11342
        - après pooling IAB + têtes finales     : 0.00042
        - encodeur voxel Encoder_Down_2         : 0.00194
    Le pooling détruit donc 273x du pouvoir discriminant, et le résultat devient
    4.6x moins discriminant que le voxel. La cause : dans `IAB_Simple`, les requêtes
    sont des points inducteurs appris, identiques quelle que soit l'entrée, et le
    bloc n'a pas de connexion résiduelle. Après LayerNorm, les clés issues de points
    d'une surface normalisée se ressemblent, l'attention devient quasi uniforme, et
    la sortie tend vers une moyenne des valeurs — une statistique proche du
    centroïde, similaire pour tous les objets.

    Ce pooling fait de toute façon doublon avec la cross-attention de
    `CondFeatureFusion`, qui agrège déjà les tokens géométriques — et le fait mieux,
    puisqu'elle est pilotée par les tokens image comme requêtes plutôt que par des
    vecteurs fixes.

    Sortie : (B, N, output_dim), soit un token par point d'entrée.
    """

    def __init__(self, output_dim: int = 1024, pc_dims: int = 1024):
        super().__init__()
        self.linear1 = nn.Linear(3, 64)
        self.linear2 = nn.Linear(64, 256)
        self.linear3 = nn.Linear(256, 1024)
        self.linear4 = nn.Linear(1024, pc_dims)
        self.ln1 = nn.LayerNorm(64)
        self.ln2 = nn.LayerNorm(256)
        self.ln3 = nn.LayerNorm(1024)
        self.ln4 = nn.LayerNorm(pc_dims)
        self.out = nn.Linear(pc_dims, output_dim) if output_dim != pc_dims else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.linear1(x)))
        x = F.relu(self.ln2(self.linear2(x)))
        x = F.relu(self.ln3(self.linear3(x)))
        x = F.relu(self.ln4(self.linear4(x)))
        return self.out(x)


class CondFeatureFusion(nn.Module):
    """Fusion de deux features de condition (modalités quelconques).

    Deux modes, sélectionnables par `mode` :

    - ``"cross"`` (défaut) — **cross-attention** : les tokens image servent de
      *queries*, les tokens voxel de *keys/values*. La longueur de sortie vaut donc
      exactement le nombre de tokens image (= ``cond_grid_size``), sans troncature.

      La contribution voxel passe par une porte scalaire apprise ``gate``,
      initialisée à 0, et ``proj_a`` est initialisée à l'identité. À l'initialisation
      la sortie est donc **exactement égale aux features image** : le modèle fusionné
      démarre au niveau de la baseline, et l'entraînement ne peut qu'apprendre
      combien de voxel injecter. C'est aussi le point d'accroche naturel pour la
      pondération par incertitude visée à terme (les poids d'attention *sont* la
      pondération par token).

    - ``"concat"`` — comportement historique : concaténation des deux séquences,
      un bloc de self-attention, puis troncature à ``output_seq_len``.

      Attention : avec la configuration WaLa-SV-1B (256 tokens image, 512 tokens
      voxel, ``output_seq_len=256``), cette troncature garde précisément la tranche
      image et **jette tous les tokens voxel** ; l'information voxel ne survit
      qu'indirectement, via l'unique couche de self-attention. Ce mode n'est
      conservé que pour pouvoir comparer.

    Args:
        fused_dim     : dimension interne commune après projection
        n_heads       : nb de têtes attention
        dropout       : dropout interne
        output_dim    : dim de sortie (= cond_grid_emb_size du DiT/UVIT).
                        Si None, identique à fused_dim.
        output_seq_len: longueur de séquence de sortie fixe (= cond_grid_size).
                        Utilisé pour la troncature/padding en mode "concat" ;
                        en mode "cross" la longueur est déjà celle des queries.
        mode          : "cross" ou "concat".
    """

    def __init__(
        self,
        fused_dim: int = 512,
        n_heads: int = 8,
        dropout: float = 0.0,
        output_dim: int | None = None,
        output_seq_len: int | None = None,
        mode: str = "cross",
        gate_mode: str = "channel",
    ):
        super().__init__()
        assert fused_dim % n_heads == 0, "fused_dim doit être divisible par n_heads"
        if mode not in ("cross", "concat"):
            raise ValueError(f"mode inconnu : {mode!r} (attendu 'cross' ou 'concat')")

        self.fused_dim = fused_dim
        self.output_dim = output_dim if output_dim is not None else fused_dim
        self.output_seq_len = output_seq_len
        self.mode = mode

        # Projections vers la dim commune (init paresseuse).
        # En mode cross, proj_a démarre à l'identité pour le passthrough.
        self.proj_a = _LazyLinear(fused_dim, init_identity=(mode == "cross"))
        self.proj_b = _LazyLinear(fused_dim)

        # Embedding par modalité (apprend à différencier image vs voxel).
        # Zéro en mode cross pour ne pas perturber le passthrough initial.
        init_mod = torch.zeros(2, fused_dim) if mode == "cross" else torch.randn(2, fused_dim) * 0.02
        self.mod_emb = nn.Parameter(init_mod)

        self.ln1 = nn.LayerNorm(fused_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=fused_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(fused_dim)
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, fused_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim * 2, fused_dim),
            nn.Dropout(dropout),
        )

        if mode == "cross":
            # LayerNorm dédié aux keys/values, et porte initialisée à 0.
            #
            # `gate_mode="channel"` donne une porte par canal (fused_dim valeurs) au
            # lieu d'un scalaire. Une porte scalaire impose un mélange uniforme sur
            # les 1024 canaux : le modèle ne peut pas doser sélectivement, et
            # l'optimisation converge vers une valeur minuscule (3e-03 mesuré) qui
            # revient à ne rien injecter. Une porte par canal permet d'ouvrir
            # certains canaux et d'en fermer d'autres. C'est aussi le point
            # d'accroche naturel pour la pondération par incertitude visée à terme.
            self.ln_kv = nn.LayerNorm(fused_dim)
            self.gate_mode = gate_mode
            n_gate = fused_dim if gate_mode == "channel" else 1
            self.gate = nn.Parameter(torch.zeros(n_gate))
            self.cls_token = None
        else:
            self.ln_kv = None
            self.gate = None
            self.gate_mode = None
            self.cls_token = nn.Parameter(torch.randn(1, 1, fused_dim) * 0.02)

        # Projection finale vers output_dim (si différent de fused_dim)
        if self.output_dim != fused_dim:
            self.output_proj = nn.Linear(fused_dim, self.output_dim)
        else:
            self.output_proj = nn.Identity()

    def materialize(self, in_dim_a: int, in_dim_b: int, device=None, dtype=None) -> None:
        """Construit tout de suite les projections d'entrée.

        `proj_a` et `proj_b` sont paresseuses : elles ne sont instanciées qu'au
        premier forward. Or un optimiseur construit avant ce premier forward ne
        voit pas leurs paramètres et ne les met donc jamais à jour — soit 2 099 200
        paramètres gelés par accident sur les 10,5 M du module, dont l'intégralité
        de la projection des voxels.

        Appeler cette méthode juste après la construction du module, avant de
        collecter les paramètres à optimiser.
        """
        if self.proj_a.proj is None:
            self.proj_a.proj = self.proj_a._build(in_dim_a, device=device, dtype=dtype)
        if self.proj_b.proj is None:
            self.proj_b.proj = self.proj_b._build(in_dim_b, device=device, dtype=dtype)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_a : features modalité A (ex: CLIP image) — forme quelconque
            feat_b : features modalité B (ex: voxel encoder) — forme quelconque

        Returns:
            Tensor (B, output_seq_len, output_dim) si output_seq_len spécifié
            sinon (B, 1 + Na + Nb, output_dim)
        """
        # 1. Ramener tout en (B, N, D_in)
        a = _to_tokens(feat_a)
        b = _to_tokens(feat_b)

        if a.size(0) != b.size(0):
            raise ValueError(
                f"Batch size différent entre les 2 modalités : {a.size(0)} vs {b.size(0)}"
            )
        B = a.size(0)

        # 2. Projection vers fused_dim
        a = self.proj_a(a)  # (B, Na, fused_dim)
        b = self.proj_b(b)  # (B, Nb, fused_dim)

        # 3. Modality embedding
        a = a + self.mod_emb[0].view(1, 1, -1)
        b = b + self.mod_emb[1].view(1, 1, -1)

        if self.mode == "cross":
            # Cross-attention : queries = tokens image, keys/values = tokens voxel.
            # La longueur de sortie reste Na — aucun token voxel n'est jeté, leur
            # contribution est agrégée dans chaque token image.
            q = self.ln1(a)
            kv = self.ln_kv(b)
            attn_out, _ = self.attn(q, kv, kv, need_weights=False)
            x = a + self.gate * attn_out
            x = x + self.gate * self.mlp(self.ln2(x))
            x = self.output_proj(x)  # (B, Na, output_dim)
            return x

        # ---- mode "concat" (historique) ----
        # 4. CLS + concat
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, fused_dim)
        x = torch.cat([cls, a, b], dim=1)       # (B, 1+Na+Nb, fused_dim)

        # 5. Self-attention bloc
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))

        # 6. Suppression du CLS (on garde les tokens des deux modalités)
        x = x[:, 1:]  # (B, Na+Nb, fused_dim)

        # 7. Projection vers output_dim
        x = self.output_proj(x)  # (B, Na+Nb, output_dim)

        # 8. Ajustement de la longueur de séquence si demandé
        if self.output_seq_len is not None:
            current_len = x.size(1)
            if current_len > self.output_seq_len:
                # Troncature
                x = x[:, : self.output_seq_len, :]
            elif current_len < self.output_seq_len:
                # Padding avec des zéros
                pad = torch.zeros(
                    B,
                    self.output_seq_len - current_len,
                    self.output_dim,
                    device=x.device,
                    dtype=x.dtype,
                )
                x = torch.cat([x, pad], dim=1)
            # sinon exactement la bonne taille, rien à faire

        return x  # (B, output_seq_len ou Na+Nb, output_dim)


# ---------------------------------------------------------------------------
# Validation rapide (sans argparse, compatible import)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_fusion(
    fusion: CondFeatureFusion,
    feat_a: torch.Tensor,
    feat_b: torch.Tensor,
    verbose: bool = True,
) -> dict:
    """Passe des features synthétiques dans le module et retourne des métriques.

    En mode "cross", la porte `gate` vaut 0 à l'initialisation : la modalité B n'a
    alors, par construction, aucune influence. On vérifie donc deux choses
    distinctes — le passthrough exact de A à l'init, puis la sensibilité aux deux
    modalités une fois la porte ouverte.
    """
    fusion.eval()
    out = fusion(feat_a, feat_b)

    metrics = {
        "output_shape": tuple(out.shape),
        "has_nan": bool(torch.isnan(out).any().item()),
        "has_inf": bool(torch.isinf(out).any().item()),
        "mean_abs": float(out.abs().mean().item()),
        "std": float(out.std().item()),
    }

    is_cross = fusion.mode == "cross"
    if is_cross:
        # Passthrough : à porte fermée, la sortie doit reproduire A à l'identique
        # (possible seulement si les dimensions coïncident).
        if out.shape == feat_a.shape:
            metrics["passthrough_err"] = float((out - feat_a).abs().max().item())
        else:
            metrics["passthrough_err"] = float("nan")

    # Sensibilité à chaque modalité, porte ouverte en mode cross.
    saved_gate = None
    if is_cross:
        saved_gate = fusion.gate.detach().clone()
        fusion.gate.fill_(1.0)
    try:
        base = fusion(feat_a, feat_b)
        noise_a = feat_a + torch.randn_like(feat_a) * feat_a.abs().mean().clamp(min=1e-6)
        noise_b = feat_b + torch.randn_like(feat_b) * feat_b.abs().mean().clamp(min=1e-6)
        metrics["sensitivity_to_a"] = float((base - fusion(noise_a, feat_b)).abs().mean().item())
        metrics["sensitivity_to_b"] = float((base - fusion(feat_a, noise_b)).abs().mean().item())
    finally:
        if saved_gate is not None:
            fusion.gate.copy_(saved_gate)

    ok = (
        not metrics["has_nan"]
        and not metrics["has_inf"]
        and 1e-4 < metrics["mean_abs"] < 1e3
        and metrics["std"] > 1e-4
        and metrics["sensitivity_to_a"] > 1e-5
        and metrics["sensitivity_to_b"] > 1e-5
    )
    if is_cross and metrics["output_shape"] == tuple(feat_a.shape):
        # Le passthrough initial doit être exact, sinon le modèle fusionné ne
        # démarre pas au niveau de la baseline.
        ok = ok and metrics["passthrough_err"] < 1e-5
    metrics["fusion_ok"] = bool(ok)

    if verbose:
        sep = "=" * 52
        print(f"\n{sep}")
        print("   VALIDATION DU MODULE DE FUSION (src/)")
        print(sep)
        for k, v in metrics.items():
            val_str = f"{v: .6f}" if isinstance(v, float) else str(v)
            print(f"  {k:35s}: {val_str}")
        print("-" * 52)
        print(f"  verdict: {'OK :)' if ok else 'PROBLÈME :('}")
        print(sep)

    return metrics
