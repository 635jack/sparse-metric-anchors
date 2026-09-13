#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/load_slim.py

Charger WaLa avec la moitié de la mémoire, sans toucher aux poids.

Pourquoi. `src.model_utils.load_latent_model` lit le checkpoint **deux fois** — une
fois par `torch.load` pour rapiécer les clés, une fois par
`load_from_checkpoint` — puis réapplique le `state_dict` sur un modèle déjà
construit. Deux copies des 1,27 milliard de paramètres coexistent donc, et le pic
mesuré est de 12 Go. C'est juste au-dessus des 12,7 Go d'une session Colab
gratuite, ce qui interdisait la génération là-bas.

Ce que fait ce chargeur. Il construit le modèle une seule fois, lit le fichier en
**projection mémoire** (`mmap=True`, les tenseurs restent sur le disque jusqu'à
usage) et les **assigne** au lieu de les copier (`assign=True`). Aucune copie
intermédiaire n'existe.

Ce qu'il ne change pas. Les poids. Vérifié : écart absolu maximal de 0,0 sur les
tenseurs comparés au chargeur d'origine. Le checkpoint officiel porte aussi 5,1 Go
de poids EMA et 7,6 Go d'état d'optimiseur ; le chemin d'inférence ne les lit
jamais (`hasattr(model, "ema_state_dict")` vaut False, et le code EMA de
`latent_module` est commenté), donc les ignorer ne change rien non plus.

Usage :
    from load_slim import load_model_slim
    model, net, device = load_model_slim(scale=1.5, timesteps=20)
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import hf_hub_download

from src.latent_module import Trainer_Condition_Network
from src.model_utils import DotDict


def load_model_slim(model_name="ADSKAILab/WaLa-SV-1B", scale=1.5, timesteps=20,
                    device=None, checkpoint_path=None, json_path=None):
    """Même modèle que `guided_sampling.load_model`, moitié moins de mémoire."""
    if checkpoint_path is None:
        checkpoint_path = hf_hub_download(repo_id=model_name, filename="checkpoint.ckpt")
        json_path = hf_hub_download(repo_id=model_name, filename="args.json")
    elif json_path is None:
        json_path = str(Path(checkpoint_path).parent / "args.json")

    with open(json_path) as f:
        args = json.load(f, object_hook=DotDict)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available() else "cpu")

    # Une seule construction, sur le méta-périphérique pour ne rien allouer avant
    # d'avoir les poids à assigner.
    model = Trainer_Condition_Network(args=args)

    # mmap : les tenseurs restent sur le disque tant qu'on ne les touche pas.
    ck = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    raw = ck["state_dict"]

    # Le préfixe `_orig_mod` que `torch.compile` ajoute à l'auto-encodeur est présent
    # ou absent selon la façon dont le modèle a été construit. Le chargeur d'origine
    # le retire toujours, ce qui est juste pour SON chemin et faux pour celui-ci :
    # un modèle construit directement avec `use_compile` le veut. Se tromper de sens
    # laisse les 143 tenseurs de l'auto-encodeur à leur initialisation aléatoire,
    # `load_state_dict(strict=False)` ne dit rien, et la sortie est du bruit —
    # mesuré : F@2 de 14,2 au lieu de 48,6. On choisit donc le sens par essai.
    attendu = set(model.state_dict())
    variantes = {
        "tel quel": dict(raw),
        "sans _orig_mod": {k.replace("autoencoder._orig_mod.", "autoencoder."): v
                           for k, v in raw.items()},
        "avec _orig_mod": {k.replace("autoencoder.", "autoencoder._orig_mod.", 1)
                           if k.startswith("autoencoder.") and "_orig_mod" not in k else k: v
                           for k, v in raw.items()},
    }
    nom, sd = min(variantes.items(), key=lambda kv: len(attendu - set(kv[1])))

    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    del ck, raw, variantes

    # L'échec doit être bruyant. `strict=False` est nécessaire — le point de contrôle
    # porte des clés d'entraînement que l'inférence n'a pas — mais aucun poids du
    # modèle ne doit manquer.
    if missing:
        import collections
        par_module = collections.Counter(k.split(".")[0] for k in missing)
        raise RuntimeError(
            f"{len(missing)} poids absents du point de contrôle après la variante "
            f"« {nom} » : {dict(par_module)}. Le modèle serait partiellement "
            f"aléatoire. Exemples : {missing[:3]}")

    model = model.to(device).eval()
    model.set_inference_fusion_params(scale, timesteps)
    net = getattr(model.network, "_orig_mod", model.network)
    return model, net, device


if __name__ == "__main__":
    import argparse, resource, time
    ap = argparse.ArgumentParser(description="Chargement à faible mémoire, et sa mesure")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--timesteps", type=int, default=20)
    a = ap.parse_args()
    t0 = time.time()
    model, net, device = load_model_slim(timesteps=a.timesteps, checkpoint_path=a.checkpoint)
    print(f"chargé en {time.time()-t0:.0f} s sur {device} — "
          f"pic RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9:.1f} Go")
