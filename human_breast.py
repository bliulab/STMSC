import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import scipy.io
import matplotlib.pyplot as plt
import sys
from matplotlib.image import imread
# import STitch3D
import torch
import warnings
import scanpy
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
import torch.optim as optim
import os
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from scipy.stats import ranksums, ttest_ind, mannwhitneyu 
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph

import sys
new_path = '/home/Code/New_model/STMSC'
sys.path.append(new_path)

import networks
import utils
import preprocess
import Model
import random

warnings.filterwarnings("ignore")
device = torch.device('cuda:0')
# %%
lam = 7
bl = 0.1

print(f"lam = {lam}")
print(f"bl = {bl}")

dataset = "human_breast_cancer"
# read scRNA data
file_path = f'/disk1/home/Data/{dataset}/scRNA.h5ad' 
adata_ref = sc.read(file_path)
adata_ref.var_names_make_unique()
adata_ref.obs['celltype'] = adata_ref.obs['cell_type']

# st data
file_fold = '/disk1/home/Data/' + str(dataset) 
adata_st = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
adata_st.var_names_make_unique()
meta = pd.read_csv(f"/disk1/home/Data/{dataset}/metadata.tsv", sep="\t", index_col=0)
adata_st.obs['ground_truth'] = meta['ground_truth']
adata_st.obs['annot_type'] = meta['annot_type']

sample_col=None#"sample"
n_hvg_group=500
celltype_ref = None
celltype_ref_col="celltype"
utils.set_seed()
adata_st, adata_basis, adata_ref = preprocess.pre(adata_st,adata_ref,celltype_ref_col=celltype_ref_col,sample_col=sample_col)

utils.set_seed()
X = torch.from_numpy(adata_st.X.toarray()).float().to(device)
A = torch.from_numpy(np.array(adata_st.obsm["graph"])).float().to(device)
Y = torch.from_numpy(np.array(adata_st.obsm["count"])).float().to(device)
lY = torch.from_numpy(np.array(adata_st.obs["library_size"].values.reshape(-1, 1))).float().to(device)

basis = torch.from_numpy(np.array(adata_basis.X)).float().to(device)

model_1 = Model.Model_1(device, basis.shape[0],X.shape[0],learning_rate=0.01,lam=lam)
model_save_dir = '/disk1/home/Result/Result_human_breast/'
model_save_path = os.path.join(model_save_dir, 'model_1_human_breast_map.pt')
if not os.path.exists(model_save_path):
    loss_dev = model_1.train(X, basis, adata_st)
adata_st = model_1.evaluate(adata_st, model_save_path)

output2 = adata_st.obsm['map_matrix'].cpu().detach().numpy()
adata_st.obsm['map_matrix']=output2

adata_st.obsm["graph"] = utils.setGraph(adata_st, output2, bl=bl)
utils.set_seed()
model_2 = Model.Model_2(adata_st, adata_basis, device,lr=0.002)
model_save_dir = '/disk1/home/Result/Result_human_breast/'
model_save_path = os.path.join(model_save_dir, 'model_2_human_breast_latent.pt')
if not os.path.exists(model_save_path):
    model_2.train()

attention_scores = model_2.evaluate_model(model_save_path=model_save_path)

output1 = adata_st.obsm['latent']

utils.set_seed()
gm = GaussianMixture(n_components=20, covariance_type='tied', init_params='kmeans')
y = gm.fit_predict(output1, y=None)

order = [1,2,12 ,6,4,10,7,18,16,11,14,13,8,17,9,19,3,5,0,15] 
pre_label = [order[label] for label in y]
adata_st.obs['pred'] = pre_label
adata_st.obs['pred'] = adata_st.obs['pred'].astype("category")
Result_ARI = ari_score(adata_st.obs['ground_truth'], y)
Result_NMI = nmi_score(adata_st.obs['ground_truth'], y)

print('Gaussian, ARI = %01.4f' % Result_ARI)
print('Gaussian, NMI = %01.4f' % Result_NMI)