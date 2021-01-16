import scbean.model.vipcca as vip
import scbean.tools.utils as tl
import scbean.tools.plotting as pl

# If your script depends on a specific backend you can use the use() function:
import matplotlib
matplotlib.use('TkAgg')

# read single-cell data.
adata_b1 = tl.read_sc_data("./data/mixed_cell_lines/293t.h5ad", batch_name="293t")
adata_b2 = tl.read_sc_data("./data/mixed_cell_lines/jurkat.h5ad", batch_name="jurkat")
adata_b3 = tl.read_sc_data("./data/mixed_cell_lines/mixed.h5ad", batch_name="mixed")

# pp.preprocessing include filteration, log-TPM normalization, selection of highly variable genes.
adata_all= tl.preprocessing([adata_b1, adata_b2, adata_b3])

# VIPCCA will train the neural network on the provided datasets.
handle = vp.VIPCCA(
							adata_all,
							res_path='./results/CVAE_5/',
							split_by="_batch",
							epochs=100,
							lambda_regulizer=5,
							)

# transform user's single-cell data into shared low-dimensional space and recover gene expression.
adata_transform=handle.fit_integrate()

# Visualization
pl.run_embedding(adata_transform, path='./results/CVAE_5/',method="umap")
pl.plotEmbedding(adata_transform, path='./results/CVAE_5/', method='umap', group_by="_batch",legend_loc="right margin")
pl.plotEmbedding(adata_transform, path='./results/CVAE_5/', method='umap', group_by="celltype",legend_loc="on data")
