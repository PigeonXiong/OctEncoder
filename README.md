# OctEncoder: Octree-based Transformers MAE for 3D Tetmeshes and TriMesh

This repository contains my customized implementation of **OctEncoder** for tetmesh and trimesh data. 

---

## Acknowledgements  
Parts of this code are adapted from OctFormer by Peng-Shuai Wang. The original author's information and license are kept.
The modifications for this submission were made by the anonymous author. All information related to us is endacted.

---

## Features
- Octree-based transformer architecture for efficient 3D point cloud and tetmesh/trimesh processing
- Masked Autoencoder (MAE) pre-training for self-supervised learning
- Support for tetrahedral mesh (tetmesh) data
- Custom training and evaluation scripts for tetmesh classification and trimesh segmentation

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
    git clone https://github.com/unknownuser/thisrepo.git
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
- The pre-processed data is not provided in this stage, but will be provided upon acceptance

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
