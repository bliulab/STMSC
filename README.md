# STMSC
STMSC: A Novel Multi-Slice Framework for Precision 3D Spatial Domain Reconstruction and Disease Pathology Analysis
![图片1](https://github.com/user-attachments/assets/8c2d3ae8-8c54-4054-82d1-24c5aa6af756)
# Installation
The STMSC package is developed based on Python and supports GPU acceleration (recommended) and CPU execution.
## Step 1: Clone the Repository
```
git clone https://github.com/bliulab/STMSC.git
cd STMSC
```
## Step 2: Create a Conda Environment
We recommend creating a separate environment for running STMSC:
```
# Create a conda environment named env_STMSC with Python 3.10
conda create -n env_STMSC python=3.10
# Activate the environment
conda activate env_STMSC
```
## Step 3: Install Required Packages
For Linux:
```
pip install -r requirements.txt
```
## Step 4: Install STMSC
```
python setup.py build
python setup.py install
```
## Tutorials and reproducibility
We provided codes for reproducing the experiments of the paper "A novel multi-slice framework for precision 3D spatial domain reconstruction and disease pathology analysis", and comprehensive tutorials for using STMSC. Please check [the tutorial website](https://stmsc-tutorial.readthedocs.io/en/latest/) for more details.
## Parameter Settings

In STMSC, several hyperparameters are used to control the loss weighting across different components of the framework. Below we explain the meaning of each parameter and provide the settings used for various datasets.

### Parameter Definitions

- **`lam`**: Weight of the loss term in the **cell-type deconvolution** module.
- **`bl`**: Weight for incorporating deconvolution-informed features during the **spatial graph correction** step.
- **`bll`**: Weight for enforcing **alignment across adjacent slices** during spatial graph correction. 

> ℹ️ **Tip**: These parameters may need to be adjusted depending on tissue type, slice resolution, or biological heterogeneity.

### Dataset-Specific Settings

```text
# General training configuration
Train_model: epoch=5000, lr=0.01

# Dataset-specific hyperparameter settings
LIBD-151507-151510:       lam=5, bl=0.5, bll=0.1
LIBD-151669-151672:       lam=1, bl=0.6, bll=0.1
LIBD-151673-151676:       lam=7, bl=0.1, bll=0.1
Human breast cancer:      lam=7, bl=0.1
Mouse brain:              lam=3, bl=0.6
Human HER2 breast cancer: lam=9, bl=0.2, bll=0.1
```text
## Hardware specifications
1. Intel(R) Xeon(R) w5-3435X, NVIDIA RTX A6000
2. 13th Gen Intel(R) Core(TM) i9-13900KF, NVIDIA GeForce RTX 4090
