# scbean.VIPCCA
[![Documentation Status](https://readthedocs.org/projects/vipcca/badge/?version=latest)](https://vipcca.readthedocs.io/en/latest/?badge=latest)
![PyPI](https://img.shields.io/pypi/v/scbean?color=blue)

Variational inference of probabilistic canonical correlation analysis

introduction......

............

### Create conda environment
For more information about conda environment, see this [tutorial](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html).
```shell
$ conda create -n scbean python=3.6
$ conda activate scbean
```


### Install VIPCCA from pypi

```shell
$ pip install scbean
```

### Install VIPCCA from GitHub source code
```shell

$ git clone https://github.com/jhu99/scbean.git
$ cd ./scbean/
$ pip install .
```

**Note**: Please make sure that the `pip` is for python>=3.6. The current release depends on tensorflow with version 2.4.0. Install tenserfolow-gpu if gpu is avialable on the machine.


### Usage

For detailed documentation, please check [here](https://vipcca.readthedocs.io/en/latest/).

#### Quick Start

Download the [data](http://141.211.10.196/result/test/papers/vipcca/data.tar.gz) of the sample we provided.

```python
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
handle = vip.VIPCCA(
							adata_all,
							res_path='./results/CVAE_5/',
							split_by="_batch",
							epochs=100,
							lambda_regulizer=5,
							)

# transform user's single-cell data into shared low-dimensional space and recover gene expression.
adata_integrate=handle.fit_integrate()

# Visualization
pl.run_embedding(adata_integrate, path='./results/CVAE_5/',method="umap")
pl.plotEmbedding(adata_integrate, path='./results/CVAE_5/', method='umap', group_by="_batch",legend_loc="right margin")
pl.plotEmbedding(adata_integrate, path='./results/CVAE_5/', method='umap', group_by="celltype",legend_loc="on data")
```


