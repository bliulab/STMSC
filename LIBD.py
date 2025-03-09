# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# %%
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import scipy.io
import matplotlib.pyplot as plt
import sys
from matplotlib.image import imread
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

import sys
new_path = '/home/Code/STMSC_main/STMSC'
sys.path.append(new_path)

import networks
import utils
import random


warnings.filterwarnings("ignore")
device = torch.device('cuda:0')

# %%
mat = scipy.io.mmread("/disk1/home/Data/LIBD_sc/GSE144136_GeneBarcodeMatrix_Annotated.mtx")
meta = pd.read_csv("/disk1/home/Data/LIBD_sc/GSE144136_CellNames.csv", index_col=0)
meta.index = meta.x.values
group = [i.split('.')[1].split('_')[0] for i in list(meta.x.values)]
condition = [i.split('.')[1].split('_')[1] for i in list(meta.x.values)]
celltype = [i.split('.')[0] for i in list(meta.x.values)]
meta["group"] = group
meta["condition"] = condition
meta["celltype"] = celltype
genename = pd.read_csv("/disk1/home/Data/LIBD_sc/GSE144136_GeneNames.csv", index_col=0)
genename.index = genename.x.values
adata_ref = ad.AnnData(X=mat.tocsr().T)
adata_ref.obs = meta
adata_ref.var = genename
adata_ref = adata_ref[adata_ref.obs.condition.values.astype(str)=="Control", :]

adata_sc = sc.read_h5ad('/disk1/home/Data/LIBD/human_MTG_snrna_norm_by_exon.h5ad')

adata_sc.obs['celltype'] = adata_sc.obs['cluster_label']


adata_ref = adata_sc


#spatial data

slice_idx = [151507, 151508, 151509, 151510]
# slice_idx = [151669, 151670, 151671, 151672]
# slice_idx = [151673, 151674, 151675, 151676]

anno_df = pd.read_csv('/disk1/home/Data/LIBD/barcode_level_layer_map.tsv', sep='\t', header=None)

adata_st1 = sc.read_visium(path="/disk1/home/Data/LIBD/%d" % slice_idx[0],
                        count_file="%d_filtered_feature_bc_matrix.h5" % slice_idx[0])
spatial1=pd.read_csv(os.path.join('/disk1/home/Data/LIBD', str(slice_idx[0]), "spatial/tissue_positions_list.txt"),sep=",",header=None,na_filter=False,index_col=0) 
anno_df1 = anno_df.iloc[anno_df[1].values.astype(str) == str(slice_idx[0])]
anno_df1.columns = ["barcode", "slice_id", "layer"]
anno_df1.index = anno_df1['barcode']
adata_st1.obs = adata_st1.obs.join(anno_df1, how="left")
adata_st1.obs["x_pixel"]=spatial1[4]
adata_st1.obs["y_pixel"]=spatial1[5]
adata_st1 = adata_st1[adata_st1.obs['layer'].notna()]


adata_st2 = sc.read_visium(path="/disk1/home/Data/LIBD/%d" % slice_idx[1],
                        count_file="%d_filtered_feature_bc_matrix.h5" % slice_idx[1])
spatial2=pd.read_csv(os.path.join('/disk1/home/Data/LIBD', str(slice_idx[1]), "spatial/tissue_positions_list.txt"),sep=",",header=None,na_filter=False,index_col=0) 
anno_df2 = anno_df.iloc[anno_df[1].values.astype(str) == str(slice_idx[1])]
anno_df2.columns = ["barcode", "slice_id", "layer"]
anno_df2.index = anno_df2['barcode']
adata_st2.obs = adata_st2.obs.join(anno_df2, how="left")
adata_st2.obs["x_pixel"]=spatial2[4]
adata_st2.obs["y_pixel"]=spatial2[5]
adata_st2 = adata_st2[adata_st2.obs['layer'].notna()]

adata_st3 = sc.read_visium(path="/disk1/home/Data/LIBD/%d" % slice_idx[2],
                        count_file="%d_filtered_feature_bc_matrix.h5" % slice_idx[2])
spatial3=pd.read_csv(os.path.join('/disk1/home/Data/LIBD', str(slice_idx[2]), "spatial/tissue_positions_list.txt"),sep=",",header=None,na_filter=False,index_col=0) 
anno_df3 = anno_df.iloc[anno_df[1].values.astype(str) == str(slice_idx[2])]
anno_df3.columns = ["barcode", "slice_id", "layer"]
anno_df3.index = anno_df3['barcode']
adata_st3.obs = adata_st3.obs.join(anno_df3, how="left")
adata_st3.obs["x_pixel"]=spatial3[4]
adata_st3.obs["y_pixel"]=spatial3[5]
adata_st3 = adata_st3[adata_st3.obs['layer'].notna()]

adata_st4 = sc.read_visium(path="/disk1/home/Data/LIBD/%d" % slice_idx[3],
                        count_file="%d_filtered_feature_bc_matrix.h5" % slice_idx[3])
spatial4=pd.read_csv(os.path.join('/disk1/home/Data/LIBD', str(slice_idx[3]), "spatial/tissue_positions_list.txt"),sep=",",header=None,na_filter=False,index_col=0) 
anno_df4 = anno_df.iloc[anno_df[1].values.astype(str) == str(slice_idx[3])]
anno_df4.columns = ["barcode", "slice_id", "layer"]
anno_df4.index = anno_df4['barcode']
adata_st4.obs = adata_st4.obs.join(anno_df4, how="left")
adata_st4.obs["x_pixel"]=spatial3[4]
adata_st4.obs["y_pixel"]=spatial3[5]
adata_st4 = adata_st4[adata_st4.obs['layer'].notna()]

adata_st_list_raw = [adata_st1, adata_st2, adata_st3, adata_st4]

# %%
seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
adata_st_list = utils.align_spots(adata_st_list_raw, plot=False)

adata_st, adata_basis, adara_ref = utils.preprocess(adata_st_list,
                                                adata_ref,
                                                #   celltype_ref=celltype_list_use,
                                                sample_col=None,#"group",
                                                slice_dist_micron=[10., 300., 10.],
                                                n_hvg_group=500)

adata_basis.obs['cluster_label'] = adata_sc.obs['cluster_label']


loc = adata_st.obsm["3D_coor"]

# %%
from sklearn.metrics import pairwise_distances
pair_dist = pairwise_distances(loc, metric="euclidean")

# %%

output2 = np.load('/disk1/home/Result/DLPFC/151507_bl_0.5_bll_0.1/map_507.npy')

pair_dist_map = pairwise_distances(output2, metric="euclidean")

# %%
loc_3D = np.load('/disk1/home/Result/DLPFC/151507_bl_0.5_bll_0.1/507_3D.npy')
pair_dist_3D = pairwise_distances(loc_3D, metric="euclidean")

# %%
import os
import torch
import torch.optim as optim
from tqdm import tqdm
model_save_dir = '/home/Code/STMSC/experiment'
model_save_path = os.path.join(model_save_dir, 'model_1_human_DLPFC_latent.pt')

X = torch.from_numpy(adata_st.X.toarray()).float().to(device)
A = torch.from_numpy(np.array(adata_st.obsm["graph"])).float().to(device)
Y = torch.from_numpy(np.array(adata_st.obsm["count"])).float().to(device)
lY = torch.from_numpy(np.array(adata_st.obs["library_size"].values.reshape(-1, 1))).float().to(device)
slice = torch.from_numpy(np.array(adata_st.obs["slice"].values)).long().to(device)
basis = torch.from_numpy(np.array(adata_basis.X)).float().to(device)


seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


hidden_dims = [adata_st.shape[1], 512, 128]
n_celltype = adata_basis.shape[0]
n_slices = len(sorted(set(adata_st.obs["slice"].values)))
n_heads = 1
slice_emb_dim = 16
coef_fe = 0.1
sf = hidden_dims[0]


model1 = networks.DeconvNet1(hidden_dims=hidden_dims,
                             n_celltypes=n_celltype,
                             n_slices=n_slices,
                             n_heads=n_heads,
                             slice_emb_dim=slice_emb_dim,
                             coef_fe=coef_fe).to(device)



model1.load_state_dict(torch.load(model_save_path))


model1.eval()

with torch.no_grad():  
    node_feats_recon, Z = model1.forward(adj_matrix=A,
                                         node_feats=X,
                                         count_matrix=Y,
                                         library_size=lY,
                                         slice_label=slice,
                                         basis=basis)

    
    features_loss = torch.mean(torch.sqrt(torch.sum(torch.pow(X - node_feats_recon, 2), axis=1)))
    print(f"Features Loss: {features_loss.item()}")


print("Evaluation completed.")
# %%
embeddings = Z.detach().cpu().numpy()
cell_reps = pd.DataFrame(embeddings)
cell_reps.index = adata_st.obs.index
adata_st.obsm['latent'] = cell_reps.loc[adata_st.obs_names, ].values
output1 = adata_st.obsm['latent']

# %%
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
gm = GaussianMixture(n_components=7, covariance_type='tied', init_params='kmeans')
y = gm.fit_predict(output1, y=None)

order = [2,4,6,0,3,5,1] # reordering cluster labels
pre_label = [order[label] for label in y]
wd = []
for i in range(4):
    wd.append(adata_st_list[i].shape[0])

Result_ARI_0 = ari_score(adata_st_list_raw[0].obs['layer'], pre_label[0:wd[0]])
Result_ARI_1 = ari_score(adata_st_list_raw[1].obs['layer'], pre_label[wd[0]:wd[0]+wd[1]])
Result_ARI_2 = ari_score(adata_st_list_raw[2].obs['layer'], pre_label[wd[0]+wd[1]:wd[0]+wd[1]+wd[2]])
Result_ARI_3 = ari_score(adata_st_list_raw[3].obs['layer'], pre_label[wd[0]+wd[1]+wd[2]:wd[0]+wd[1]+wd[2]+wd[3]])
Result_NMI_0 = nmi_score(adata_st_list_raw[0].obs['layer'], pre_label[0:wd[0]])
Result_NMI_1 = nmi_score(adata_st_list_raw[1].obs['layer'], pre_label[wd[0]:wd[0]+wd[1]])
Result_NMI_2 = nmi_score(adata_st_list_raw[2].obs['layer'], pre_label[wd[0]+wd[1]:wd[0]+wd[1]+wd[2]])
Result_NMI_3 = nmi_score(adata_st_list_raw[3].obs['layer'], pre_label[wd[0]+wd[1]+wd[2]:wd[0]+wd[1]+wd[2]+wd[3]])


print('Gaussian, ARI = %01.4f' % Result_ARI_0)
print('Gaussian, ARI = %01.4f' % Result_ARI_1)
print('Gaussian, ARI = %01.4f' % Result_ARI_2)
print('Gaussian, ARI = %01.4f' % Result_ARI_3)
print('Gaussian, NMI = %01.4f' % Result_NMI_0)
print('Gaussian, NMI = %01.4f' % Result_NMI_1)
print('Gaussian, NMI = %01.4f' % Result_NMI_2)
print('Gaussian, NMI = %01.4f' % Result_NMI_3)
