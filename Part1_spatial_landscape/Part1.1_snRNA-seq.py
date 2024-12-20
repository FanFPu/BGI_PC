import sys
import os
import scanpy as sc
import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt
import matplotlib
import anndata as ad
matplotlib.use("Agg")
from anndata import AnnData
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.serif'] = ['Arial']
sc.settings.set_figure_params(dpi=100,dpi_save=300,facecolor='white',frameon=False,fontsize=10,vector_friendly=True,figsize=(3,3))

def removeBiasGenes(adata):
    malat1 = adata.var_names.str.startswith('MALAT1')
    MTgenes = adata.var_names.str.startswith('MT')
    hb_genes = adata.var_names.str.contains('^HB[^(P)]')
    RPgenes = adata.var_names.str.startswith('RP') & adata.var_names.str.contains('-')
    RPgenes2 = adata.var_names.str.contains('^RP[SL]')
    CTCgenes = adata.var_names.str.startswith('CTC') & adata.var_names.str.contains('-')
    MIRgenes = adata.var_names.str.startswith('MIR')
    ACgenes = adata.var_names.str.contains('^AC[0-9]') & adata.var_names.str.contains('.')
    CTgenes = adata.var_names.str.startswith('CT') & adata.var_names.str.contains('-')
    LINCgenes = adata.var_names.str.contains('^LINC[0-9]')
    ALgenes = adata.var_names.str.contains('^AL') & adata.var_names.str.contains('.')
    remove_genes = malat1 | MTgenes | hb_genes | RPgenes | RPgenes2 | CTCgenes | MIRgenes | ACgenes | CTgenes | LINCgenes | ALgenes
    keep = np.invert(remove_genes)
    res = adata[:,keep]
    return res

#### Clustering of scRNA_seq data
od = './scRNA_seq/'
os.chdir(od)

adata = sc.read_h5ad('adata_raw.h5ad')
adata = removeBiasGenes(adata)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var.highly_variable]
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
for res in np.arange(0.1, 2.1, 0.1):
    sc.tl.leiden(
        adata, key_added=f"leiden_res_{res:4.2f}", resolution=res
    )
sc.tl.umap(adata)
adata.write_h5ad('humanSC.h5ad')


#### After annotation by considering cell-specific gene expression and calling CNV
genes_of_interest = ["KRT5","CASC15","LRP1B","KRT17",
                     "KRT19","KRT13","SHROOM1",
                     "COL1A1", "SFRP2", "DCN",
                     "MYH11","LMOD1","ACTA2",
                     "FCGR3A","CD68","CD163",
                     "CD8A","CD2","CD3G",
                     "MS4A1", "CD79A", "CD19",
                     "IGLC2", "IGHG3", "JCHAIN",
                     "SLC18A2", "HDC", "MS4A2",
                     "HLA-DPA1","HLA-DPB1","CD83",
                     "PECAM1", "LDB2", "PTPRB",
                     "PDGFRB", "MCAM", "RGS5"]

sc.pl.dotplot(
    adata,
    var_names=genes_of_interest,
    groupby='cellTypes_new',
    dendrogram=False,
    cmap="plasma",
    swap_axes=False,
    standard_scale="var",
    save='DEG_celltype.pdf',
    figsize=(10,4)
)


cell_colors = {
    "B cell": "#FFBBFF",
    "DCs": "#81d8ae",
    "Endothelial": "#607d3a",
    "Fibroblast": "#FFD700",
    "Macrophage": "#87CEFF",
    "Mastcell": "#AB82FF",
    "Myocyte": "#3399FF",
    "Pericyte": "#FFD39B",
    "Plasmocyte": "#6bd155",
    "T cell": "#B23AEE",
    "Tumor cell": "#CD3278",
    "Normal epithelial": "#663300"
}
plt.rcParams['figure.figsize'] = [8, 8]
sc.pl.umap(adata, color='cellTypes_new', use_raw=False, palette=cell_colors, save='celltype_umap.pdf')
