import scbean.model.vipcca as vip
import scbean.tools.utils as tl
import scbean.tools.plotting as pl

# Please choose an appropiate matplotlib backend.
import matplotlib
# matplotlib.use('TkAgg')

# read single-cell data.
adata_b1 = tl.read_sc_data("./data/mixed_cell_lines/293t.h5ad", batch_name="293t")
adata_b2 = tl.read_sc_data("./data/mixed_cell_lines/jurkat.h5ad", batch_name="jurkat")
adata_b3 = tl.read_sc_data("./data/mixed_cell_lines/mixed.h5ad", batch_name="mixed")

# tl.preprocessing include filteration, log-TPM normalization, selection of highly variable genes.
adata_all= tl.preprocessing([adata_b1, adata_b2, adata_b3])

# Construct VIPCCA with specific setting.
handle = vip.VIPCCA(
							adata_all,
							res_path='./results/CVAE_5/',
							split_by="_batch",
							epochs=100,
							lambda_regulizer=5,
							)

# Training and integrating multiple single-cell datasets. The VIPCCA's output include cell representation in reduced dimensional space and recovered gene expression.
adata_integrate=handle.fit_integrate()

# Visualization
pl.run_embedding(adata_integrate, path='./results/CVAE_5/',method="umap")
pl.plotEmbedding(adata_integrate, path='./results/CVAE_5/', method='umap', group_by="_batch",legend_loc="right margin")
pl.plotEmbedding(adata_integrate, path='./results/CVAE_5/', method='umap', group_by="celltype",legend_loc="on data")
