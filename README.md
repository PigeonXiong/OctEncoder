# OctFormer: Octree-based Transformers for 3D Tetmeshes (TetMesh Edition)

This repository contains my customized implementation of **OctFormer** for tetrahedral mesh (tetmesh) data. 

---

## Features
- Octree-based transformer architecture for efficient 3D point cloud and tetmesh processing
- Masked Autoencoder (MAE) pre-training for self-supervised learning
- Support for tetrahedral mesh (tetmesh) data
- Custom training and evaluation scripts for tetmesh classification

---

## 1. Installation

Tested on Ubuntu 20.04 with Nvidia GPUs.

1. Install [Conda](https://www.anaconda.com/) and create a Conda environment:
    ```bash
    conda create --name octformer python=3.8
    conda activate octformer
    ```
2. Install PyTorch (version matching your CUDA):
    ```bash
    conda install pytorch==1.12.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch
    ```
3. Clone this repository and install requirements:
    ```bash
    git clone https://github.com/mfarazi1991/tetoctformer.git
    cd tetoctformer
    pip install -r requirements.txt
    ```
4. Install the octree-based depthwise convolution library:
    ```bash
    git clone https://github.com/octree-nn/dwconv.git
    pip install ./dwconv
    ```

---

## 2. Tetrahedral Mesh (TetMesh) Classification & MAE Pre-training

### Data Preparation
- Place your tetmesh data in the `data/adni` directory.
- Update the file lists in `data/adni/file_list/train.txt` and `data/adni/file_list/test.txt` as needed.

### Pre-training (MAE)
Run masked autoencoder pre-training on tetmesh data:
```bash
python mae_pretraining.py --config configs/mae_tet.yaml SOLVER.gpu 0,
```

### Classification
Train a classifier on tetmesh data:
```bash
python classification.py --config configs/cls_tet.yaml SOLVER.gpu 0,
```

---

```

---

## Notes

