# STMSC
STMSC: A Novel Multi-Slice Framework for Precision 3D Spatial Domain Reconstruction and Disease Pathology Analysis
![图片1](https://github.com/user-attachments/assets/8c2d3ae8-8c54-4054-82d1-24c5aa6af756)
# Environment
python==3.8.0
anndata=0.9.2 
torch=2.4.0 
scikit-learn=1.3.2 
scanpy=1.9.8 
numpy=1.21.6  
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
# Run
- **Model/** → Contains the models for reproduction.  
- **STMSC_main/** → Includes the training code for the STMSC model.  
- **LIBD.py, human_breast.py, mouse_brain.py** → Execution scripts for running different experiments.  
