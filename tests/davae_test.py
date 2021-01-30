import scbean.model.davae as davae
import scbean.tools.utils as tl
import scanpy as sc
import matplotlib
from numpy.random import seed
import umap
seed(2021)
matplotlib.use('TkAgg')

r1 = "./data/mixed_cell_lines/mixed.h5ad"
r2 = "./data/mixed_cell_lines/293t.h5ad"
r3 = "./data/mixed_cell_lines/jurkat.h5ad"

adata_b1 = tl.read_sc_data(r1, batch_name='mix')
adata_b2 = tl.read_sc_data(r2, batch_name='293t')
adata_b3 = tl.read_sc_data(r3, batch_name='jurkat')

adata_all = tl.davae_preprocessing([adata_b1, adata_b2, adata_b3], n_top_genes=2000)
adata_integrate = davae.fit_integration(
    adata_all,
    batch_num=3,
    domain_lambda=3.0,
    epochs=25,
    sparse=True,
    hidden_layers=[64, 32, 6]
)
adata_integrate.obsm['X_umap']=umap.UMAP().fit_transform(adata_integrate.obsm['X_davae'])
sc.pl.umap(adata_integrate, color=['_batch', 'celltype'], s=3)

