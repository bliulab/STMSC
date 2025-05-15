# STMSC
STMSC: A Novel Multi-Slice Framework for Precision 3D Spatial Domain Reconstruction and Disease Pathology Analysis
![图片1](https://github.com/user-attachments/assets/8c2d3ae8-8c54-4054-82d1-24c5aa6af756)
# Installation
The STMSC package is developed based on Python and supports GPU acceleration (recommended) and CPU execution.
## Step 1: Clone the Repository
`git clone https://github.com/bliulab/STMSC.git
cd STMSC`
## Step 2: Create a Conda Environment
We recommend creating a separate environment for running STMSC:
`# Create a conda environment named env_STMSC with Python 3.8
conda create -n env_STMSC python=3.10`
`# Activate the environment
conda activate env_STMSC`
## Step 3: Install Required Packages
For Linux:
`pip install -r requirements.txt`
## Step 4: Install STMSC
`python setup.py build
python setup.py install`
# Tutorials and reproducibility
We provided codes for reproducing the experiments of the paper "A novel multi-slice framework for precision 3D spatial domain reconstruction and disease pathology analysis", and comprehensive tutorials for using STMSC. Please check the tutorial website for more details.
# Parameter settings
Train_model:epoch=5000,lr=0.01
LIBD-151507-151510:lam=5,bl=0.5,bll=0.1
LIBD-151669-151672:lam=1,bl=0.6,bll=0.1
LIBD-151673-151676:lam=7,bl=0.1,bll=0.1
Human breast cancer:lam=7,bl=0.1
Mouse brain:lam=3,bl=0.6
Human HER2 breast cancer:lam = 9,bl = 0.2,bll = 0.1
# Hardware specifications
1. Intel(R) Xeon(R) w5-3435X, NVIDIA RTX A6000
2. 13th Gen Intel(R) Core(TM) i9-13900KF, NVIDIA GeForce RTX 4090
