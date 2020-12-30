import VIPCCA
import VIPCCA.preprocessing as pp
import VIPCCA.plotting as pl

adata_b1 = pp.read_sc_data("./data/mixed_cell_lines/293t.h5ad", batch_name="293t")
adata_b2 = pp.read_sc_data("./data/mixed_cell_lines/jurkat.h5ad", batch_name="jurkat")
adata_b3 = pp.read_sc_data("./data/mixed_cell_lines/mixed.h5ad", batch_name="mixed")

# pp.preprocessing include filteration, log-TPM normalization, selection of highly variable genes.
adata_all= pp.preprocessing([adata_b1, adata_b2, adata_b3])
handle = VIPCCA.VIPCCA(
							adata_all,
							res_path='./results/CVAE_5/',
							split_by="_batch",
							patience_es=50,
							patience_lr=20,
							lambda_regulizer=5,
							# uncomment the following line if a pretrained model was provided in the result folder.
							# model_file="model.h5" 
							)
adata_transform=handle.fit_transform()

pl.plotPrediction2(adata_transform.raw.X,adata_transform.X,result_path=test_result_path,rnum=2000,lim=20)
pl.run_embedding(adata_transform, path=test_result_path,method="umap")
pl.plotEmbedding(adata_transform, path=test_result_path, method='umap', group_by="_batch",legend_loc="right margin")
pl.plotEmbedding(adata_transform, path=test_result_path, method='umap', group_by="celltype",legend_loc="on data")
