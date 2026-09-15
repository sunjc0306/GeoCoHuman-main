# GeoCoHuman

## Single-View 3D Clothed Human Reconstruction via Complementary Geometry Priors

PyTorch implementation and reproduction of **GeoCoHuman**, a framework for reconstructing a detailed 3D clothed human from a single RGB image by combining parametric-body and point-cloud geometry priors.

> **Paper:** [GeoCoHuman: Single-View 3D Clothed Human Reconstruction via Complementary Geometry Priors](https://doi.org/10.1109/TVCG.2026.3717809)  
> **Authors:** Jianchi Sun, Fei Luo, Xiangqian Shen, and Chunxia Xiao  
> **Journal:** IEEE Transactions on Visualization and Computer Graphics, 2026

This repository contains the three principal components described in the paper:

- **PMDM — Parametric Model-guided Diffusion Model:** predicts front- and back-view clothed-human depth maps from an RGB image and rendered SMPL depth maps.
- **PFA — Proximity Feature Aggregation:** extracts domain-specific body and clothing features from an SMPL mesh and an oriented point cloud.
- **CPIF — Collaboration Perception Implicit Function:** combines visual, body-prior, and clothing-prior information through three collaborative prediction heads to reconstruct a signed distance field.




## Repository structure

```text
.
├── configs/
│   ├── default.json              # Paper-scale PMDM configuration
│   ├── geocohuman.json           # Paper-scale PFA/CPIF configuration
├── scripts/
│   ├── build_oriented_pointcloud.py
│   ├── prepare_implicit_sample.py
│   ├── generate_dummy_dataset.py
│   └── generate_dummy_implicit_dataset.py
├── src/
│   ├── human_depth_diffusion/    # PMDM, DDPM/DDIM, data, losses, and metrics
│   └── geocohuman/               # Geometry, PFA, Hourglass, CPIF, and ICON adapter
├── train.py                      # PMDM training
├── infer.py                      # Front/back depth inference
├── evaluate.py                   # PMDM depth evaluation
├── train_cpif.py                 # PFA/CPIF training
├── reconstruct_mesh.py           # Dense SDF query and Marching Cubes
└── tests/                        # Unit and integration tests
```

## Installation

### Requirements

- Python 3.10 or later
- PyTorch 2.1 or later
- A CUDA-capable GPU is strongly recommended for paper-scale training and reconstruction

Install a PyTorch build compatible with your CUDA environment, then install this project:

## Preparing

### 1. PMDM data


### 2. Oriented point-cloud prior


### 3. PFA/CPIF data

## Training

### Stage 1: PMDM



### Stage 2: PFA and CPIF



## Inference

### Predict front/back human depth


### Evaluate PMDM


### Reconstruct a mesh




## Integration with ICON

This implementation follows the pixel-aligned query pattern and implicit-reconstruction interface used by [ICON](https://github.com/YuliangXiu/ICON), while implementing PMDM, PFA, CPIF, the oriented point-cloud conversion, and the prediction MLPs independently.



## Reproduction scope

The following stages are external to this repository and require their original implementations, models, or licenses:

- PyMAF/ECON-based SMPL estimation and refinement;
- foreground matting or background removal;
- joint registration and hand replacement;
- the final depth-refinement stage;
- proprietary or separately licensed datasets and pretrained weights.

See [`docs/REPRODUCTION_NOTES.md`](docs/REPRODUCTION_NOTES.md) for assumptions made where the paper does not fully specify implementation details.

## Citation

If this project or the GeoCoHuman method is useful in your research, please cite the original paper:

```bibtex
@article{sun2026geocohuman,
  title   = {GeoCoHuman: Single-View 3D Clothed Human Reconstruction via Complementary Geometry Priors},
  author  = {Sun, Jianchi and Luo, Fei and Shen, Xiangqian and Xiao, Chunxia},
  journal = {IEEE Transactions on Visualization and Computer Graphics},
  volume  = {32},
  number  = {10},
  pages   = {8364--8378},
  year    = {2026},
  doi     = {10.1109/TVCG.2026.3717809}
}
```


## License

This repository is intended for academic research and open-source release. Third-party code, pretrained models, and datasets remain subject to their original licenses. In particular, ICON has separate non-commercial research terms.

A repository-level `LICENSE` file is not included in this snapshot. Add the intended license before public distribution so that reuse and contribution terms are unambiguous.

## Acknowledgements

This reproduction builds on ideas and interfaces from DDPM and ICON. We thank the authors of both projects and the broader open-source 3D human reconstruction community.

