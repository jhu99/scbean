# VIPCCA
Badges-----------------

Variational inference of probabilistic canonical correlation analysis

introduction......

............

### Install VIPCCA
```shell
$ tar -zxvf scxx.x.x.x.tar.gz
$ pip install -e ./scxx/
```

```shell
$ pip install vipcca
```

**Note**: you need to make sure that the `pip` is for python3. Our package is suitable for tensorflow 1.13.1



### Usage

Read the doc url..........................

#### Quick Start

Download example data at http://141.211.10.196/result/test/papers/vipcca/data.tar.gz

```python
import VIPCCA as vp
from VIPCCA import preprocessing as pp
from VIPCCA import plotting as pl

# read single-cell data.
adata_b1 = pp.read_sc_data("./data/mixed_cell_lines/293t.h5ad", batch_name="293t")
adata_b2 = pp.read_sc_data("./data/mixed_cell_lines/jurkat.h5ad", batch_name="jurkat")
adata_b3 = pp.read_sc_data("./data/mixed_cell_lines/mixed.h5ad", batch_name="mixed")

# pp.preprocessing include filteration, log-TPM normalization, selection of highly variable genes.
adata_all= pp.preprocessing([adata_b1, adata_b2, adata_b3])

# VIPCCA will train the neural network on the provided datasets.
handle = vp.VIPCCA(
							adata_all,
							res_path='./results/CVAE_5/',
							split_by="_batch",
							patience_es=50,
							patience_lr=20,
							lambda_regulizer=5,
							# uncomment the following line if a pretrained model was provided in the result folder.
							# model_file="model.h5" 
							)

# transform user's single-cell data into shared low-dimensional space and recover gene expression.
adata_transform=handle.fit_transform()

# Visualization
pl.run_embedding(adata_transform, path=test_result_path,method="umap")
pl.plotEmbedding(adata_transform, path=test_result_path, method='umap', group_by="_batch",legend_loc="right margin")
pl.plotEmbedding(adata_transform, path=test_result_path, method='umap', group_by="celltype",legend_loc="on data")
```




#### reference
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

