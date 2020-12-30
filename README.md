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

```python
import vipcca
import preprocessing as pp

mode="CVAE"
lambda_regulizer=5
batch_input_size=128
batch_input_size2=16

# load data
test_result_path = './results/CVAE_5/'
r1="./data/mixed_cell_lines/293t.h5ad"
r2="./data/mixed_cell_lines/jurkat.h5ad"
r4="./data/mixed_cell_lines/mixed.h5ad"

adata_b1 = pp.read_sc_data(r1, batch_name="293t")
adata_b2 = pp.read_sc_data(r2, batch_name="jurkat")
adata_b4 = pp.read_sc_data(r4, batch_name="mixed")

# Preprocessing
adata_all = pp.preprocessing([adata_b1, adata_b2, adata_b4], mt_ratio=0.8,)

# Construct and train model
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
adata_integeation = handle.fit_transform()
```




#### reference
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

