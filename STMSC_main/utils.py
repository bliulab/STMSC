import cv2
import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
import scipy.sparse
import matplotlib
import matplotlib.pyplot as plt
from align_tools import *
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from matplotlib import cm
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
import random
seed = 666
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def align_spots_3D(adata_st_list_input, # list of spatial transcriptomics datasets
                method="icp", # "icp" or "paste"
                data_type="Visium", # a spot has six nearest neighborhoods if "Visium", four nearest neighborhoods otherwise
                coor_key="spatial", # "spatial" for visium; key for the spatial coordinates used for alignment
                tol=0.01, # parameter for "icp" method; tolerance level
                test_all_angles=False, # parameter for "icp" method; whether to test multiple rotation angles or not
                plot=False,
                paste_alpha=0.1,
                paste_dissimilarity="kl",
                ):
    # Align coordinates of spatial transcriptomics

    # The first adata in the list is used as a reference for alignment
    adata_st_list = adata_st_list_input.copy()

    if plot:
        # Choose colors
        cmap = cm.get_cmap('rainbow', len(adata_st_list))
        colors_list = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(len(adata_st_list))]

        # Plot spots before alignment
        plt.figure(figsize=(5, 5))
        plt.title("Before alignment")
        for i in range(len(adata_st_list)):
            plt.scatter(adata_st_list[i].obsm[coor_key][:, 0], 
                adata_st_list[i].obsm[coor_key][:, 1], 
                c=colors_list[i],
                label="Slice %d spots" % i, s=5., alpha=0.5)
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        plt.xticks([])
        plt.yticks([])
        plt.legend(loc=(1.02, .2), ncol=(len(adata_st_list)//13 + 1))
        plt.show()


    if (method == "icp") or (method == "ICP"):
        print("Using the Iterative Closest Point algorithm for alignment.")
        # Detect edges
        print("Detecting edges...")
        point_cloud_list = []
        for adata in adata_st_list:
            # Use in-tissue spots only
            if 'in_tissue' in adata.obs.columns:
                adata = adata[adata.obs['in_tissue'] == 1]
            if data_type == "Visium":

                loc = adata.obs['loc'].values
                if loc.ndim == 1:
                    loc = np.array([np.array(item) for item in loc])
                pairwise_loc_distsq = np.sum((loc.reshape([1,-1,3]) - loc.reshape([-1,1,3])) ** 2, axis=2)
                n_neighbors = np.sum(pairwise_loc_distsq < 5, axis=1) - 1
                edge = ((n_neighbors > 1) & (n_neighbors < 5)).astype(np.float32)
            else:

                loc = adata.obs['loc'].values
                if loc.ndim == 1:
                    loc = np.array([np.array(item) for item in loc])
                pairwise_loc_distsq = np.sum((loc.reshape([1,-1,3]) - loc.reshape([-1,1,3])) ** 2, axis=2)
                min_distsq = np.sort(np.unique(pairwise_loc_distsq), axis=None)[1]
                n_neighbors = np.sum(pairwise_loc_distsq < (min_distsq * 3), axis=1) - 1
                edge = ((n_neighbors > 1) & (n_neighbors < 7)).astype(np.float32)
            point_cloud_list.append(np.array(adata.obs['loc'])[edge == 1].copy()) # Extract points marked as edges from the original data
            for isi in range(len(point_cloud_list)):
                arrays = [np.array(lst) for lst in point_cloud_list[isi]]
                # 垂直堆叠数组
                point_cloud_list[isi] = np.vstack(arrays)
        # Align edges
        print("Aligning edges...")
        trans_list = []
        adata_st_list[0].obsm["spatial_aligned"] = np.array(adata_st_list[0].obs['loc']).copy()
        # Calculate pairwise transformation matrices
        for i in range(len(adata_st_list) - 1):
            if test_all_angles == True:
                for angle in [0., np.pi * 1 / 3, np.pi * 2 / 3, np.pi, np.pi * 4 / 3, np.pi * 5 / 3]:
                    R = np.array([[np.cos(angle), np.sin(angle), 0], 
                                [-np.sin(angle), np.cos(angle), 0], 
                                [0, 0, 1]]).T
                    T, distances, _ = icp(transform(point_cloud_list[i+1], R), point_cloud_list[i], tolerance=tol)
                    if angle == 0:
                        loss_best = np.mean(distances)
                        angle_best = angle
                        R_best = R
                        T_best = T
                    else:
                        if np.mean(distances) < loss_best:
                            loss_best = np.mean(distances)
                            angle_best = angle
                            R_best = R
                            T_best = T
                T = T_best @ R_best
            else:
                T, _, _ = icp(point_cloud_list[i+1], point_cloud_list[i], tolerance=tol)
            trans_list.append(T)
        # Transform
        for i in range(len(adata_st_list) - 1):
            # point_cloud_align = adata_st_list[i+1].obsm[coor_key].copy()
            point_cloud_align = np.array(adata_st_list[i+1].obs['loc']).copy()
            arrays = [np.array(lst) for lst in point_cloud_align]
            point_cloud_align = np.vstack(arrays)
            for T in trans_list[:(i+1)][::-1]:
                point_cloud_align = transform_3D(point_cloud_align, T)
            adata_st_list[i+1].obsm["spatial_aligned"] = point_cloud_align

    if plot:
        plt.figure(figsize=(5, 5))
        plt.title("After alignment")
        for i in range(len(adata_st_list)):
            plt.scatter(adata_st_list[i].obsm["spatial_aligned"][:, 0], 
                adata_st_list[i].obsm["spatial_aligned"][:, 1], 
                c=colors_list[i],
                label="Slice %d spots" % i, s=5., alpha=0.5)
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        plt.xticks([])
        plt.yticks([])
        plt.legend(loc=(1.02, .2), ncol=(len(adata_st_list)//13 + 1))
        plt.show()

    return adata_st_list

def align_spots(adata_st_list_input, # list of spatial transcriptomics datasets
                method="icp", # "icp" or "paste"
                data_type="Visium", # a spot has six nearest neighborhoods if "Visium", four nearest neighborhoods otherwise
                coor_key="spatial", # "spatial" for visium; key for the spatial coordinates used for alignment
                tol=0.01, # parameter for "icp" method; tolerance level
                test_all_angles=False, # parameter for "icp" method; whether to test multiple rotation angles or not
                plot=False,
                paste_alpha=0.1,
                paste_dissimilarity="kl"
                ):
    # Align coordinates of spatial transcriptomics

    # The first adata in the list is used as a reference for alignment
    adata_st_list = adata_st_list_input.copy()

    if plot:
        # Choose colors
        cmap = cm.get_cmap('rainbow', len(adata_st_list))
        colors_list = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(len(adata_st_list))]

        # Plot spots before alignment
        plt.figure(figsize=(5, 5))
        plt.title("Before alignment")
        for i in range(len(adata_st_list)):
            plt.scatter(adata_st_list[i].obsm[coor_key][:, 0], 
                adata_st_list[i].obsm[coor_key][:, 1], 
                c=colors_list[i],
                label="Slice %d spots" % i, s=5., alpha=0.5)
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        plt.xticks([])
        plt.yticks([])
        plt.legend(loc=(1.02, .2), ncol=(len(adata_st_list)//13 + 1))
        plt.show()


    if (method == "icp") or (method == "ICP"):
        print("Using the Iterative Closest Point algorithm for alignemnt.")
        # Detect edges
        print("Detecting edges...")
        point_cloud_list = []
        for adata in adata_st_list:
            # Use in-tissue spots only
            if 'in_tissue' in adata.obs.columns:
                adata = adata[adata.obs['in_tissue'] == 1]
            if data_type == "Visium":
                loc_x = adata.obs.loc[:, ["array_row"]]
                loc_x = np.array(loc_x) * np.sqrt(3)
                loc_y = adata.obs.loc[:, ["array_col"]]
                loc_y = np.array(loc_y)
                loc = np.concatenate((loc_x, loc_y), axis=1)
                pairwise_loc_distsq = np.sum((loc.reshape([1,-1,2]) - loc.reshape([-1,1,2])) ** 2, axis=2)
                n_neighbors = np.sum(pairwise_loc_distsq < 5, axis=1) - 1
                edge = ((n_neighbors > 1) & (n_neighbors < 5)).astype(np.float32)
            else:
                loc_x = adata.obs.loc[:, ["array_row"]]
                loc_x = np.array(loc_x)
                loc_y = adata.obs.loc[:, ["array_col"]]
                loc_y = np.array(loc_y)
                loc = np.concatenate((loc_x, loc_y), axis=1)
                pairwise_loc_distsq = np.sum((loc.reshape([1,-1,2]) - loc.reshape([-1,1,2])) ** 2, axis=2)
                min_distsq = np.sort(np.unique(pairwise_loc_distsq), axis=None)[1]
                n_neighbors = np.sum(pairwise_loc_distsq < (min_distsq * 3), axis=1) - 1
                edge = ((n_neighbors > 1) & (n_neighbors < 7)).astype(np.float32)
            point_cloud_list.append(adata.obsm[coor_key][edge == 1].copy())

        # Align edges
        print("Aligning edges...")
        trans_list = []
        adata_st_list[0].obsm["spatial_aligned"] = adata_st_list[0].obsm[coor_key].copy()
        # Calculate pairwise transformation matrices
        for i in range(len(adata_st_list) - 1):
            if test_all_angles == True:
                for angle in [0., np.pi * 1 / 3, np.pi * 2 / 3, np.pi, np.pi * 4 / 3, np.pi * 5 / 3]:
                    R = np.array([[np.cos(angle), np.sin(angle), 0], 
                                  [-np.sin(angle), np.cos(angle), 0], 
                                  [0, 0, 1]]).T
                    T, distances, _ = icp(transform(point_cloud_list[i+1], R), point_cloud_list[i], tolerance=tol)
                    if angle == 0:
                        loss_best = np.mean(distances)
                        angle_best = angle
                        R_best = R
                        T_best = T
                    else:
                        if np.mean(distances) < loss_best:
                            loss_best = np.mean(distances)
                            angle_best = angle
                            R_best = R
                            T_best = T
                T = T_best @ R_best
            else:
                T, _, _ = icp(point_cloud_list[i+1], point_cloud_list[i], tolerance=tol)
            trans_list.append(T)
        # Tranform
        for i in range(len(adata_st_list) - 1):
            point_cloud_align = adata_st_list[i+1].obsm[coor_key].copy()
            for T in trans_list[:(i+1)][::-1]:
                point_cloud_align = transform(point_cloud_align, T)
            adata_st_list[i+1].obsm["spatial_aligned"] = point_cloud_align

    elif (method == "paste") or (method == "PASTE"):
        print("Using PASTE algorithm for alignemnt.")
        # Align spots
        print("Aligning spots...")
        pis = []
        # Calculate pairwise transformation matrices
        for i in range(len(adata_st_list) - 1):
            pi = pairwise_align_paste(adata_st_list[i], adata_st_list[i+1], coor_key=coor_key,
                                      alpha = paste_alpha, dissimilarity = paste_dissimilarity)
            pis.append(pi)
        # Tranform
        S1, S2  = generalized_procrustes_analysis(adata_st_list[0].obsm[coor_key], 
                                                  adata_st_list[1].obsm[coor_key], 
                                                  pis[0])
        adata_st_list[0].obsm["spatial_aligned"] = S1
        adata_st_list[1].obsm["spatial_aligned"] = S2
        for i in range(1, len(adata_st_list) - 1):
            S1, S2 = generalized_procrustes_analysis(adata_st_list[i].obsm["spatial_aligned"], 
                                                     adata_st_list[i+1].obsm[coor_key], 
                                                     pis[i])
            adata_st_list[i+1].obsm["spatial_aligned"] = S2

    if plot:
        plt.figure(figsize=(5, 5))
        plt.title("After alignment")
        for i in range(len(adata_st_list)):
            plt.scatter(adata_st_list[i].obsm["spatial_aligned"][:, 0], 
                adata_st_list[i].obsm["spatial_aligned"][:, 1], 
                c=colors_list[i],
                label="Slice %d spots" % i, s=5., alpha=0.5)
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        plt.xticks([])
        plt.yticks([])
        plt.legend(loc=(1.02, .2), ncol=(len(adata_st_list)//13 + 1))
        plt.show()

    return adata_st_list

def preprocess(adata_st_list_input, # list of spatial transcriptomics (ST) anndata objects
               adata_ref_input, # reference single-cell anndata object
               celltype_ref_col="celltype", # column of adata_ref_input.obs for cell type information
               sample_col=None, # column of adata_ref_input.obs for batch labels
               celltype_ref=None, # specify cell types to use for deconvolution
               n_hvg_group=500, # number of highly variable genes for reference anndata
               three_dim=False, # if not None, use existing 3d coordinates in shape [# of total spots, 3]
               coor_key="spatial_aligned", # "spatial_aligned" by default
               rad_cutoff=None, # cutoff radius of spots for building graph
               rad_coef=1.1, # if rad_cutoff=None, rad_cutoff is the minimum distance between spots multiplies rad_coef
               slice_dist_micron=None, # pairwise distances in micrometer for reconstructing z-axis 
               prune_graph_cos=False, # prune graph connections according to cosine similarity
               cos_threshold=0.5, # threshold for pruning graph connections
               c2c_dist=100, # center to center distance between nearest spots in micrometer
               ):

    adata_st_list = adata_st_list_input.copy()

    print("Finding highly variable genes...")
    adata_ref = adata_ref_input.copy()
    adata_ref.var_names_make_unique()    # 使基因名唯一
    # Remove mt-genes 去除线粒体基因
    adata_ref = adata_ref[:, np.array(~adata_ref.var.index.isna())
                          & np.array(~adata_ref.var_names.str.startswith("mt-"))
                          & np.array(~adata_ref.var_names.str.startswith("MT-"))]
    if celltype_ref is not None:   # 筛选细胞类型
        if not isinstance(celltype_ref, list):
            raise ValueError("'celltype_ref' must be a list!")
        else:
            adata_ref = adata_ref[[(t in celltype_ref) for t in adata_ref.obs[celltype_ref_col].values.astype(str)], :]
    else:
        celltype_counts = adata_ref.obs[celltype_ref_col].value_counts()
        celltype_ref = list(celltype_counts.index[celltype_counts > 1])
        adata_ref = adata_ref[[(t in celltype_ref) for t in adata_ref.obs[celltype_ref_col].values.astype(str)], :]

    # Remove cells and genes with 0 counts
    sc.pp.filter_cells(adata_ref, min_genes=1)
    sc.pp.filter_genes(adata_ref, min_cells=1)

    # Concatenate ST adatas
    for i in range(len(adata_st_list)):
        adata_st_new = adata_st_list[i].copy()
        adata_st_new.var_names_make_unique()
        # Remove mt-genes
        adata_st_new = adata_st_new[:, (np.array(~adata_st_new.var.index.str.startswith("mt-")) 
                                    & np.array(~adata_st_new.var.index.str.startswith("MT-")))]
        adata_st_new.obs.index = adata_st_new.obs.index + "-slice%d" % i
        adata_st_new.obs['slice'] = i
        if i == 0:
            adata_st = adata_st_new
        else:
            genes_shared = list(set(adata_st.var.index) & set(adata_st_new.var.index))
            adata_st = adata_st[:, genes_shared].concatenate(adata_st_new[:, genes_shared], index_unique=None)

    adata_st.obs["slice"] = adata_st.obs["slice"].values.astype(int)

    # Take gene intersection
    genes = list(set(adata_st.var.index) & set(adata_ref.var.index))
    adata_ref = adata_ref[:, genes]
    adata_st = adata_st[:, genes]

    # Select hvgs
    adata_ref_log = adata_ref.copy()
    sc.pp.log1p(adata_ref_log)
    hvgs = select_hvgs(adata_ref_log, celltype_ref_col=celltype_ref_col, num_per_group=n_hvg_group)

    print("%d highly variable genes selected." % len(hvgs))
    adata_ref = adata_ref[:, hvgs]

    print("Calculate basis for deconvolution...")
    sc.pp.filter_cells(adata_ref, min_genes=1)
    sc.pp.normalize_total(adata_ref, target_sum=1)
    celltype_list = list(sorted(set(adata_ref.obs[celltype_ref_col].values.astype(str))))

    basis = np.zeros((len(celltype_list), len(adata_ref.var.index)))
    if sample_col is not None:
        sample_list = list(sorted(set(adata_ref.obs[sample_col].values.astype(str))))
        for i in range(len(celltype_list)):
            c = celltype_list[i]
            tmp_list = []
            for j in range(len(sample_list)):
                s = sample_list[j]
                tmp = adata_ref[(adata_ref.obs[celltype_ref_col].values.astype(str) == c) & 
                                (adata_ref.obs[sample_col].values.astype(str) == s), :].X
                if scipy.sparse.issparse(tmp):
                    tmp = tmp.toarray()
                if tmp.shape[0] >= 3:
                    tmp_list.append(np.mean(tmp, axis=0).reshape((-1)))
            tmp_mean = np.mean(tmp_list, axis=0)
            if scipy.sparse.issparse(tmp_mean):
                tmp_mean = tmp_mean.toarray()
            print("%d batches are used for computing the basis vector of cell type <%s>." % (len(tmp_list), c))
            basis[i, :] = tmp_mean
    else:
        for i in range(len(celltype_list)):
            c = celltype_list[i]
            tmp = adata_ref[adata_ref.obs[celltype_ref_col].values.astype(str) == c, :].X
            if scipy.sparse.issparse(tmp):
                tmp = tmp.toarray()
            basis[i, :] = np.mean(tmp, axis=0).reshape((-1))

    adata_basis = ad.AnnData(X=basis)
    df_gene = pd.DataFrame({"gene": adata_ref.var.index})
    df_gene = df_gene.set_index("gene")
    df_celltype = pd.DataFrame({"celltype": celltype_list})
    df_celltype = df_celltype.set_index("celltype")
    adata_basis.obs = df_celltype
    adata_basis.var = df_gene
    adata_basis = adata_basis[~np.isnan(adata_basis.X[:, 0])]

    print("Preprocess ST data...")
    # Store counts and library sizes for Poisson modeling
    st_mtx = adata_st[:, hvgs].X.copy()
    if scipy.sparse.issparse(st_mtx):
        st_mtx = st_mtx.toarray()
    adata_st.obsm["count"] = st_mtx
    st_library_size = np.sum(st_mtx, axis=1)
    adata_st.obs["library_size"] = st_library_size

    # Normalize ST data
    sc.pp.normalize_total(adata_st, target_sum=1e4)
    sc.pp.log1p(adata_st)
    adata_st = adata_st[:, hvgs]
    if scipy.sparse.issparse(adata_st.X):
        adata_st.X = adata_st.X.toarray()

    return adata_st, adata_basis, adata_ref


def select_hvgs(adata_ref, celltype_ref_col, num_per_group=200):
    sc.tl.rank_genes_groups(adata_ref, groupby=celltype_ref_col, method="t-test", key_added="ttest", use_raw=False)
    markers_df = pd.DataFrame(adata_ref.uns['ttest']['names']).iloc[0:num_per_group, :]
    genes = sorted(list(np.unique(markers_df.melt().value.values)))
    return genes


def calculate_impubasis(adata_st_input, #st anndata object (should be one of the output from STitch3D.utils.preprocess)
                        adata_ref_input, # reference single-cell anndata object (raw data)
                        celltype_ref_col="celltype", # column of adata_ref_input.obs for cell type information
                        sample_col=None, # column of adata_ref_input.obs for batch labels
                        celltype_ref=None, # specify cell types to use for deconvolution
                        ):
    
    adata_ref = adata_ref_input.copy()
    adata_ref.var_names_make_unique()
    # Remove mt-genes
    adata_ref = adata_ref[:, np.array(~adata_ref.var.index.isna())
                          & np.array(~adata_ref.var_names.str.startswith("mt-"))
                          & np.array(~adata_ref.var_names.str.startswith("MT-"))]
    if celltype_ref is not None:
        if not isinstance(celltype_ref, list):
            raise ValueError("'celltype_ref' must be a list!")
        else:
            adata_ref = adata_ref[[(t in celltype_ref) for t in adata_ref.obs[celltype_ref_col].values.astype(str)], :]
    else:
        celltype_counts = adata_ref.obs[celltype_ref_col].value_counts()
        celltype_ref = list(celltype_counts.index[celltype_counts > 1])
        adata_ref = adata_ref[[(t in celltype_ref) for t in adata_ref.obs[celltype_ref_col].values.astype(str)], :]

    # Remove cells and genes with 0 counts
    sc.pp.filter_cells(adata_ref, min_genes=1)
    sc.pp.filter_genes(adata_ref, min_cells=1)

    # Calculate single cell library sizes
    hvgs = adata_st_input.var.index
    adata_ref_ls = adata_ref[:, hvgs]
    sc.pp.filter_cells(adata_ref_ls, min_genes=1)
    adata_ref = adata_ref[adata_ref_ls.obs.index, :]
    # ref_ls: library size (only account for hvgs) of single cells in ref
    if scipy.sparse.issparse(adata_ref_ls.X):
        ref_ls = np.sum(adata_ref_ls.X.toarray(), axis=1).reshape((-1,1))
        adata_ref.obsm["forimpu"] = adata_ref.X.toarray() / ref_ls
    else:
        ref_ls = np.sum(adata_ref_ls.X, axis=1).reshape((-1,1))
        adata_ref.obsm["forimpu"] = adata_ref.X / ref_ls

    # Calculate basis for imputation
    celltype_list = list(sorted(set(adata_ref.obs[celltype_ref_col].values.astype(str))))
    basis_impu = np.zeros((len(celltype_list), len(adata_ref.var.index)))
    if sample_col is not None:
        sample_list = list(sorted(set(adata_ref.obs[sample_col].values.astype(str))))
        for i in range(len(celltype_list)):
            c = celltype_list[i]
            tmp_list = []
            for j in range(len(sample_list)):
                s = sample_list[j]
                tmp = adata_ref[(adata_ref.obs[celltype_ref_col].values.astype(str) == c) & 
                                (adata_ref.obs[sample_col].values.astype(str) == s), :].obsm["forimpu"]
                if scipy.sparse.issparse(tmp):
                    tmp = tmp.toarray()
                if tmp.shape[0] >= 3:
                    tmp_list.append(np.mean(tmp, axis=0).reshape((-1)))
            tmp_mean = np.mean(tmp_list, axis=0)
            if scipy.sparse.issparse(tmp_mean):
                tmp_mean = tmp_mean.toarray()
            print("%d batches are used for computing the basis vector of cell type <%s>." % (len(tmp_list), c))
            basis_impu[i, :] = tmp_mean
    else:
        for i in range(len(celltype_list)):
            c = celltype_list[i]
            tmp = adata_ref[adata_ref.obs[celltype_ref_col].values.astype(str) == c, :].obsm["forimpu"]
            if scipy.sparse.issparse(tmp):
                tmp = tmp.toarray()
            basis_impu[i, :] = np.mean(tmp, axis=0).reshape((-1))

    adata_basis_impu = ad.AnnData(X=basis_impu)
    df_gene = pd.DataFrame({"gene": adata_ref.var.index})
    df_gene = df_gene.set_index("gene")
    df_celltype = pd.DataFrame({"celltype": celltype_list})
    df_celltype = df_celltype.set_index("celltype")
    adata_basis_impu.obs = df_celltype
    adata_basis_impu.var = df_gene
    adata_basis_impu = adata_basis_impu[~np.isnan(adata_basis_impu.X[:, 0])]
    return adata_basis_impu

def Noise_Cross_Entropy(adata_st, pred_sp, emb_sp, device):
        '''\
        Calculate noise cross entropy. Considering spatial neighbors as positive pairs for each spot
            
        Parameters
        ----------
        pred_sp : torch tensor
            Predicted spatial gene expression matrix.
        emb_sp : torch tensor
            Reconstructed spatial gene expression matrix.

        Returns
        -------
        loss : float
            Loss value.

        '''
        
        mat = cosine_similarity(pred_sp, emb_sp) 
        k = torch.exp(mat).sum(axis=1) - torch.exp(torch.diag(mat, 0))
        
        # positive pairs
        p = torch.exp(mat)
        graph_neigh = torch.FloatTensor(adata_st.obsm['graph'].copy() + np.eye(adata_st.obsm['graph'].shape[0])).to(device)
        p = torch.mul(p, graph_neigh).sum(axis=1)
        
        ave = torch.div(p, k)
        loss = - torch.log(ave).mean()
        
        return loss

def cosine_similarity(pred_sp, emb_sp):  #pres_sp: spot x gene; emb_sp: spot x gene
    '''\
    Calculate cosine similarity based on predicted and reconstructed gene expression matrix.    
    '''
    
    M = torch.matmul(pred_sp, emb_sp.T)
    Norm_c = torch.norm(pred_sp, p=2, dim=1)
    Norm_s = torch.norm(emb_sp, p=2, dim=1)
    Norm = torch.matmul(Norm_c.reshape((pred_sp.shape[0], 1)), Norm_s.reshape((emb_sp.shape[0], 1)).T) + -5e-12
    M = torch.div(M, Norm)
    
    if torch.any(torch.isnan(M)):
        M = torch.where(torch.isnan(M), torch.full_like(M, 0.4868), M)

    return M        

def loss(adata_st, emb_sp, emb_sc, map_matrix, device):
    '''\
    Calculate loss

    Parameters
    ----------
    emb_sp : torch tensor
        Spatial spot representation matrix.
    emb_sc : torch tensor
        scRNA cell representation matrix.

    Returns
    -------
    Loss values.

    '''
    # cell-to-spot
    map_probs = F.softmax(map_matrix, dim=1)   # dim=0: normalization by cell
    pred_sp = torch.matmul(map_probs.t(), emb_sc)
        
    loss_recon = F.mse_loss(pred_sp, emb_sp, reduction='mean')
    loss_NCE = Noise_Cross_Entropy(adata_st, pred_sp, emb_sp, device)
    # loss_NCE = Noise_Cross_Entropy(pred_sp, emb_sp)
    return loss_recon, loss_NCE


def select_hvgs(adata_ref, celltype_ref_col, num_per_group=200):
    sc.tl.rank_genes_groups(adata_ref, groupby=celltype_ref_col, method="t-test", key_added="ttest", use_raw=False)
    markers_df = pd.DataFrame(adata_ref.uns['ttest']['names']).iloc[0:num_per_group, :]
    genes = sorted(list(np.unique(markers_df.melt().value.values)))
    return genes

def construct_spatial_graph(spatial,
                            mode="KNN",
                            k_cutoff=6, r_cutoff=None,
                            metric="euclidean", 
                            symmetric=True, sparse=False):
    if mode == "KNN":
        A = kneighbors_graph(spatial, k_cutoff+1, mode="connectivity", metric=metric, include_self=True)
    elif mode == "Radius":
        A = radius_neighbors_graph(spatial, r_cutoff, mode="connectivity", metric=metric, include_self=True)
    A = A.toarray()
    
    if symmetric:
        A = A + A.T * (A.T > A) - A * (A.T > A)
        
    if sparse:
        return A.nonzero()
    else:
        return A

def edge_index_to_adj_matrix(edge_index, num_nodes):
    # 创建一个N x N的零矩阵
    adj_matrix = np.zeros((num_nodes, num_nodes))

    # edge_index的第一行是源节点，第二行是目标节点
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]

    # 在邻接矩阵中对应位置置1
    adj_matrix[src_nodes, dst_nodes] = 1

    return adj_matrix

class MeanAct(torch.nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()
    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)

class DispAct(torch.nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()
    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)
    
class ZINBLoss(torch.nn.Module):
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0,    ridge_lambda=0.0):
        eps = 1e-10
        # scale_factor = scale_factor[:, None]
        # mean = mean * scale_factor
        
        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0-pi+eps)
        zero_nb = torch.pow(disp/(disp+mean+eps), disp)
        zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)
        
        if ridge_lambda > 0:
            ridge = ridge_lambda*torch.square(pi)
            result += ridge
        result = torch.mean(result)
        return result

def find_resolution(adata_, n_clusters, random_state): 
    adata = adata_.copy()
    obtained_clusters = -1
    iteration = 0
    resolutions = [0., 1000.]
    while obtained_clusters != n_clusters and iteration < 50:
        current_res = sum(resolutions)/2
        sc.tl.louvain(adata, resolution=current_res, random_state=random_state)
        labels = adata.obs['louvain']
        obtained_clusters = len(np.unique(labels))

        if obtained_clusters < n_clusters:
            resolutions[0] = current_res
        else:
            resolutions[1] = current_res
        iteration = iteration + 1
    final_cluster=len(np.unique(adata.obs['louvain']))

    return current_res

def setGraph(adata_st, output2, bl):
    pair_dist_map = pairwise_distances(output2, metric="euclidean")
    pair_dist = pairwise_distances(adata_st.obsm['spatial'], metric="euclidean")
    pair_dist_zong = pair_dist + bl * pair_dist_map
    desired_mean_neighbors = 9.5  # 您想要的平均邻居数
    tolerance = 0.5  # 允许的偏差
    min_rad_cutoff = 0  # 最小的rad_cutoff值
    max_rad_cutoff = 1000  # 最大的rad_cutoff值
    rad_cutoff = 1.1000
    while True:
        G = (pair_dist_zong < rad_cutoff).astype(float)
        mean_neighbors = np.mean(np.sum(G, axis=1)) - 1
        if desired_mean_neighbors - tolerance <= mean_neighbors <= desired_mean_neighbors + tolerance:
            rad_cutoff = mean_neighbors
            break
        elif mean_neighbors < desired_mean_neighbors - tolerance:
            min_rad_cutoff = rad_cutoff
            rad_cutoff = (min_rad_cutoff + max_rad_cutoff) / 2
        else:
            max_rad_cutoff = rad_cutoff
            rad_cutoff = (min_rad_cutoff + max_rad_cutoff) / 2
    # 将计算得到的rad_cutoff值应用于您的代码
    print("Radius for graph connection is %.4f." % rad_cutoff)
    print('%.4f neighbors per cell on average.' % mean_neighbors)
    return G

def set_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
