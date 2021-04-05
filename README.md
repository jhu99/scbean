# Scbean

scbean integrates a range of models for single-cell data analysis, including dimensionality reduction, remvoing batch effects, and transferring well-annotated cell type labels from scRNA-seq to scATAC-seq. It is efficient and scalable for large-scale datasets.

[![Documentation Status](https://readthedocs.org/projects/scbean/badge/?version=latest)](https://scbean.readthedocs.io/en/latest/?badge=latest) [![Build Status](https://www.travis-ci.com/jhu99/scbean.svg?branch=main)](https://www.travis-ci.com/jhu99/scbean) ![PyPI](https://img.shields.io/pypi/v/scbean?color=blue) [![Downloads](https://pepy.tech/badge/scbean)](https://pepy.tech/project/scbean) ![GitHub Repo stars](https://img.shields.io/github/stars/jhu99/scbean?color=yellow)

## scbean.DAVAE

Domain-adversarial and variational approximation framework, DAVAE, can integrate multiple single-cell data across samples, technologies and modalities without any post hoc data processing.
DAVAE fit normalized gene expression into a non-linear model, which transforms a latent variable of a lower-dimension into expression space with a non-linear function, a KL regularizier and a domain-adversarial regularizer.

We will also provide more fundamental analyses for multi-modal data and spatial resoved transcriptomics in the future. The output can be easily used for downstream data analyses such as clustering, identification of cell subpopulations, differential gene expression, visualization using either [Seurat](https://satijalab.org/seurat/) or [Scanpy](https://scanpy-tutorials.readthedocs.io).

### Installation

- Create conda environment

  ```shell
  $ conda create -n scbean python=3.8
  $ conda activate scbean
  ```

- Install scbean from pypi

  ```shell
  $ pip install scbean
  ```

- Alternatively, install the develop version of scbean from GitHub source code

  ```shell
  $ git clone https://github.com/jhu99/scbean.git
  $ cd ./scbean/
  $ python -m pip install .
  ```

**Note**: Please make sure your python version >= 3.7, and install tensorflow-gpu if GPU is available on your your machine.

### Usage

For detailed guide about the usage of scbean, the tutorial and documentation were provided [here](https://scbean.readthedocs.io/en/latest/).

### Quick start with DAVAE

Download the [data](http://141.211.10.196/result/test/papers/vipcca/data.tar.gz) of the following test code.

```python
import scbean.model.davae as davae
import scbean.tools.utils as tl

# Please choose an appropiate matplotlib backend.
import matplotlib
matplotlib.use('TkAgg')

# read single-cell data.
adata_b1 = tl.read_sc_data("./data/mixed_cell_lines/293t.h5ad", batch_name="293t")
adata_b2 = tl.read_sc_data("./data/mixed_cell_lines/jurkat.h5ad", batch_name="jurkat")
adata_b3 = tl.read_sc_data("./data/mixed_cell_lines/mixed.h5ad", batch_name="mixed")

# tl.preprocessing include filteration, log-TPM normalization, selection of highly variable genes.
adata_all= tl.preprocessing([adata_b1, adata_b2, adata_b3])

# Training and integrating multiple single-cell datasets. The DAVAE's output include cell representation in 
# reduced dimensional space and recovered gene expression.
adata_integrate = davae.fit_integration(
    adata_all,
    batch_num=3,
    split_by='batch_label',
    domain_lambda=2.0,
    epochs=25,
    sparse=True,
    hidden_layers=[64, 32, 6]
)
# Visualization
sc.pp.neighbors(adata_integrate, use_rep='X_davae')
sc.tl.umap(adata_integrate)
sc.pl.umap(adata_integrate, color='batch')
```
