import scanpy as sc
import preprocessing as pp
import plotting as pl
import vipcca
mode="CVAE"
lambda_regulizer=5
batch_input_size=128
batch_input_size2=16

# # "CVAE"
# mode=str(sys.argv[1])
# # 5
# lambda_regulizer=int(sys.argv[2])
# # 128
# batch_input_size=int(sys.argv[3])
# # 16
# batch_input_size2=int(sys.argv[4])

# test_result_path="./results/%s_%s_%s_%s/" % (str(sys.argv[1]).lower(),str(sys.argv[2]),str(sys.argv[3]),str(sys.argv[4]))
# test_result_path="./results/%s_%s_%s_%s/" % (str(sys.argv[1]).lower(),str(sys.argv[2]),str(sys.argv[3]),str(sys.argv[4]))
test_result_path = './results/CVAE_5/'
r1="./data/mixed_cell_lines/293t.h5ad"
r2="./data/mixed_cell_lines/jurkat.h5ad"
r4="./data/mixed_cell_lines/mixed.h5ad"

adata_b1 = pp.read_sc_data(r1, batch_name="293t")
adata_b2 = pp.read_sc_data(r2, batch_name="jurkat")
adata_b4 = pp.read_sc_data(r4, batch_name="mixed")

# Run vipcca. Here, rawdata is optional. It will be generated from datasets if not provided.
adata_all= pp.preprocessing([adata_b1, adata_b2, adata_b4], mt_ratio=0.8,)
handle = vipcca.VIPCCA(
							adata_all,
							res_path=test_result_path,
							mode=mode,
							split_by="_batch",
							patience_es=50,
							patience_lr=20,
							lambda_regulizer=lambda_regulizer,
							batch_input_size=batch_input_size,
							batch_input_size2=batch_input_size2,
							model_file="model.h5"
							)
adata_transform=handle.fit_transform()

pl.plotPrediction2(adata_transform.raw.X,adata_transform.X,result_path=test_result_path,rnum=2000,lim=20)
pl.run_embedding(adata_transform, path=test_result_path,method="umap")
pl.plotEmbedding(adata_transform, path=test_result_path, method='umap', group_by="_batch",legend_loc="right margin")
pl.plotEmbedding(adata_transform, path=test_result_path, method='umap', group_by="celltype",legend_loc="on data")