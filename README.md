# Sparse Metric Anchors for a Single-View 3D Generative Prior

Companion code and results for the paper *Sparse Metric Anchors for a Single-View 3D
Generative Prior: The Output Frame Is the Bottleneck* (`paper/icra_en.pdf`; French
version `paper/icra_fr.pdf`).

The question the paper asks is what limits the injection of a few metric measurements
— a handful of tactile contacts, one depth map — into a pretrained single-view 3D
generative model. The answer is the generator's **output frame**: it is not a function
of the input image but is resampled with the noise, and estimating it consumes most of
the achievable gain and most of the run-to-run variance.

Everything here is training-free. The backbone is [WaLa](https://github.com/AutodeskAILab/WaLa)
(Sanghi et al., 2024), used through its published weights; nothing is retrained.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/635jack/sparse-metric-anchors/blob/main/colab/reproduce.ipynb)
**Verify the numbers without a GPU:** `colab/reproduce.ipynb` downloads the benchmark
and recomputes every table and figure of the paper from the published campaign
results, with an independent implementation of the statistics, then re-runs the
Poisson witness live. About two minutes on a CPU runtime.

## What is in this repository

```
src/        WaLa's model code, with the modifications listed in NOTICE
tools/      the method, the contact models, the campaigns and the scoring
results/    every campaign result cited in the paper, as JSON
data/       contact sets and reprojected depth views (small); images and meshes are on HF
paper/      LaTeX sources, figures (TikZ), built PDF and ePub, and the ePub build script
notes/      two internal syntheses (French) that document how the results were reached
```

The 42 YCB objects — one rendered image and one reference mesh each — are **not** in
this repository. They are published as a dataset:
[`jack635/sparse-metric-anchors-ycb`](https://huggingface.co/datasets/jack635/sparse-metric-anchors-ycb) on Hugging Face. Download it into
`data/training_ycb/` before running any campaign.

## The method, in one file

`tools/guided_sampling.py` is the constraint-anchoring scheme. At each denoising step
the predicted latent is decoded to a $256^3$ signed distance field, evaluated at the
anchor points by a hand-written differentiable trilinear interpolation, and corrected
along the gradient of the squared field value. Optional terms add a surface-normal
constraint and a free-space constraint; both are off by default, and the paper
measures the first as ineffective and the second as inert.

`tools/depth_rig.py` and `tools/guided_sampling_dm4.py` implement the other route:
reprojecting one depth map into the four canonical views of WaLa's multi-view variant.

## Reproducing the paper

Every number in the paper traces to one of the files below. Campaign scripts resume
from their JSON if interrupted, and write results incrementally.

| Paper | Script | Result file |
|---|---|---|
| Sec. III — output pose resampled with noise (~135°) | `tools/pose_determinism.py` | `results/pose_determinism.json` |
| Table I — constraint anchoring, 42 objects, frame estimated | `tools/run_guidance_campaign.py --combined_frame` | `results/full42_combined.json` |
| Table I — oracle ceiling (+7.82) | `tools/run_guidance_campaign.py --oracle_frame` | `results/full42_oracle.json` |
| Table II — depth reprojection into 4 views | `tools/depth_rig.py`, `tools/guided_sampling_dm4.py`, `tools/score_dm4.py` | `results/dm4_resultats.json` |
| Sec. IV-A — ICP / silhouette / combined placement | `tools/pose_from_silhouette.py`, `tools/pose_selection_study.py` | `results/pose_so3.json`, `pose_silhouette.json`, `pose_combined.json`, `pose_selection.json` |
| Fig. 2a — 4 / 8 / 16 / 32 sites at 128 anchors | `tools/make_contacts.py --sites N` then the campaign | `results/sites_4.json`, `palpation_combined.json`, `sites_16.json`, `sites_32.json` |
| Fig. 2a — 32 / 128 / 512 anchors | `tools/run_guidance_campaign.py --sets occluded:32 occluded:512` | `results/guidance_campaign.json` |
| Fig. 2a, Table IV — hardware cardinality (8 / 4 anchors, normals, noise) | `tools/make_contacts_dh116.py`, `tools/run_dh116_regime.py` | `results/dh116_regime.json` |
| Fig. 2b — grasp strategies and visibility | `tools/make_contacts_strategies.py`, `tools/run_strategies.py`, then `tools/visibility_render_camera.py` | `results/strategies.json`, `results/visibility_render_camera.json` |
| Table III — Poisson witness without a generative model | `tools/poisson_temoin.py` | `results/poisson_temoin.json` |
| Sec. VI-G — run-to-run variance, frame re-estimated / frozen | `tools/plancher_reproductibilite.py [--gel_pose]` | `results/plancher.json`, `plancher_gel.json` |
| Sec. IV-A — sign convention of the decoded field | `tools/probe_field_sign.py` | printed |

The scoring is shared by every campaign: `tools/eval_fusion.py` (alignment with 24
axis-permutation initialisations, right-handed bases enforced, F-Score and Chamfer,
connectivity) and `tools/analyze_occlusion.py` (visible / occluded split). Anything
that changes there changes every number at once, which is deliberate.

`tools/make_contacts_strategies.py` calls the grasp simulator
[`grasp-dataset-gen`](https://github.com/635jack/grasp-dataset-gen) as a sibling
checkout; its output for the five test objects is already in
`data/contacts/strategies.npz`, so the simulator is only needed to regenerate it.

## Environment

Python 3.10–3.13, PyTorch with MPS or CUDA, and `requirements.txt`. The sparse
convolutions need `spconv`, built for the CUDA version of your torch (`spconv-cu118`,
`cu121`, `cu124` or `cu126`; avoid `spconv-cu120`, whose wheels stop at Python 3.11), or a
native build on Apple Silicon. A guided sample takes about 55 s at 20 steps on an M2 Max.

Three traps on Python 3.13, all handled by the first cell of the notebook:
`open3d` has no PyPI wheel, so install Open3D's development build from the
[`main-devel` release](https://github.com/isl-org/Open3D/releases/tag/main-devel);
`pymcubes` has no wheel and builds from source; and `setuptools` 82 removed
`pkg_resources`, which `pytorch_wavelets` imports — pin `setuptools<82`.

The published `WaLa-SV-1B` checkpoint is 18 GB, of which 5.1 GB are weights; the rest is
optimiser state and an EMA copy the inference path never reads. The stock loader reads
it twice and peaks at about 13 GB of RAM; `tools/load_slim.py` memory-maps it and peaks
at about 6 GB, with bit-identical weights (checked tensor by tensor, 1399 of 1399).

Two hardware notes from the paper. `torch.nn.functional.grid_sample` has no 3D
backward kernel on Metal, which is why the interpolation is written by hand. And
Metal kernels do not guarantee a constant reduction order, so identical runs differ
slightly; the paper measures this and shows it is not what dominates the variance.

## Evaluation caveats the paper reports

Three defects of the standard pipeline each silently reversed a conclusion during
this work, and are corrected in `tools/eval_fusion.py`: F-Score is blind to
fragmentation (a scatter of shards scores well — connectivity is now reported), blind
to mirror symmetry (SVD bases are not necessarily right-handed — the determinant is
forced to +1), and an identity-initialised ICP fails silently (24 initialisations are
tried). One defect was found late and is declared in the paper's limitations: every
occlusion computation placed the render camera, `(2.2, −2.2, 1.8)`, in the meshes' Y-up
frame, without the `(x, y, z) → (x, −z, y)` turn that Blender's OBJ importer applied
before rendering — 68° from where it rendered. `tools/visibility_render_camera.py`
measures it (silhouette IoU against the input images: 0.95 with the turn, 0.51 without)
and recomputes the visibility results from the right camera
(`results/visibility_render_camera.json`): Fig. 2b's correlation is −0.001, not −0.145.
The campaign scripts keep the misplaced camera so that the published files stay
reproducible. The palpation sites they drew are 61 % hidden from the rendering camera,
and the visible / occluded recall columns are unreliable.

## Licence

The model code in `src/` derives from WaLa and is governed by the **Autodesk
Non-Commercial License (3D Generative) v1.0** (`LICENSE.md`); `NOTICE` states how it
was modified, as that licence requires. Everything else in this repository — the
tools, the results, the paper — is released under the same non-commercial terms for
simplicity. The YCB meshes and renders on Hugging Face are CC BY 4.0 (see the dataset
card).

## Citation

```bibtex
@unpublished{anchors2026,
  title  = {Sparse Metric Anchors for a Single-View 3D Generative Prior:
            The Output Frame Is the Bottleneck},
  author = {},
  note   = {ISIR, Sorbonne Universit\'e. Manuscript.},
  year   = {2026}
}
```
