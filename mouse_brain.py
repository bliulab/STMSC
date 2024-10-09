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

lam = 3
bl = 0.6

print(f"lam = {lam}")
print(f"bl = {bl}")

# sc data
adata_ref = ad.read_h5ad("/disk1/home/Data/mouse_brain/adult_mouse_brain_sc_STitch3D_cellocation/all_cells_20200625.h5ad")
adata_ref.var.index = adata_ref.var["SYMBOL"].astype(str)
adata_ref.var_names_make_unique()
labels = pd.read_csv("/disk1/home/Data/mouse_brain/adult_mouse_brain_sc_STitch3D_cellocation/snRNA_annotation_astro_subtypes_refined59_20200823.csv", index_col=0)

labels = labels.reindex(index=adata_ref.obs_names)
adata_ref.obs[labels.columns] = labels
adata_ref = adata_ref[~adata_ref.obs['annotation_1'].isna(), :]
adata_ref.obs['celltype'] = adata_ref.obs['annotation_1']

#st data
adata_st = sc.read_h5ad("/disk1/home/Data/mouse_brain/V1_Adult_Mouse_Brain_st.h5ad")

sample_col="sample"
celltype_ref = None
celltype_ref_col="celltype"
adata_st, adata_basis, adata_ref = preprocess.pre(adata_st,adata_ref,celltype_ref_col=celltype_ref_col,sample_col=sample_col)

utils.set_seed()
X = torch.from_numpy(adata_st.X.toarray()).float().to(device)
A = torch.from_numpy(np.array(adata_st.obsm["graph"])).float().to(device)
Y = torch.from_numpy(np.array(adata_st.obsm["count"])).float().to(device)
lY = torch.from_numpy(np.array(adata_st.obs["library_size"].values.reshape(-1, 1))).float().to(device)
# slice = torch.from_numpy(np.array(adata_st.obs["slice"].values)).long().to(device)
basis = torch.from_numpy(np.array(adata_basis.X)).float().to(device)

model_1 = Model.Model_1(device, basis.shape[0],X.shape[0],learning_rate=0.01,lam=lam)
model_save_dir = '/disk1/home/Result/Result_mouse_brain/model/'
model_save_path = os.path.join(model_save_dir, 'model_1_mouse_brain_map_lam_3_bl_0.6.pt')
if not os.path.exists(model_save_path):
    loss_dev = model_1.train(X, basis, adata_st)
adata_st = model_1.evaluate(adata_st, model_save_path)

output2 = adata_st.obsm['map_matrix'].cpu().detach().numpy()
adata_st.obsm['map_matrix']=output2

adata_st.obsm["graph"] = utils.setGraph(adata_st, output2, bl=bl)
utils.set_seed()
model_2 = Model.Model_2(adata_st, adata_basis, device,lr=0.002)
model_save_dir = '/disk1/home/Result/Result_mouse_brain/model/'
model_save_path = os.path.join(model_save_dir, 'model_2_mouse_brain_latent_lam_3_bl_0.6.pt')
if not os.path.exists(model_save_path):
    model_2.train()
attention_scores = model_2.evaluate_model(model_save_path=model_save_path)

output1 = adata_st.obsm['latent']

np.random.seed(2024)
import random
seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
gm = GaussianMixture(n_components=15, covariance_type='tied', init_params='kmeans')
y = gm.fit_predict(output1, y=None)

order = [7,13,0,8,4,12,1,11,2,6,9,14,10,3,5] 
pre_label = [order[label] for label in y]
adata_st.obs['pred'] = pre_label
adata_st.obs['pred'] = adata_st.obs['pred'].astype("category")

Result_ARI = ari_score(adata_st.obs['cluster'], y)
Result_NMI = nmi_score(adata_st.obs['cluster'], y)

print('Gaussian, ARI = %01.4f' % Result_ARI)
print('Gaussian, NMI = %01.4f' % Result_NMI)