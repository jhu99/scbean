# Scbean

[![Documentation Status](https://readthedocs.org/projects/scbean/badge/?version=latest)](https://scbean.readthedocs.io/en/latest/?badge=latest) [![Build Status](https://www.travis-ci.com/jhu99/scbean.svg?token=wnxY2Jwmr9V1MufszFW4&branch=main)](https://www.travis-ci.com/jhu99/scbean) ![PyPI](https://img.shields.io/pypi/v/scbean?color=blue) [![Downloads](https://pepy.tech/badge/scbean)](https://pepy.tech/project/scbean) ![GitHub Repo stars](https://img.shields.io/github/stars/jhu99/scbean?color=yellow)

Scbean integrates a range of models for single-cell data analysis, including dimensionality reduction, remvoing batch effects, and transferring well-annotated cell type labels from scRNA-seq to scATAC-seq and spatial resoved transcriptomics. It is efficient and scalable for large-scale datasets.

Scbean will also provide more fundamental analyses for multi-modal data and spatial resoved transcriptomics in the future. 

The output of our integrated data can be easily used for downstream data analyses such as clustering, identification of cell subpopulations, differential gene expression, visualization using either [Seurat](https://satijalab.org/seurat/) or [Scanpy](https://scanpy-tutorials.readthedocs.io).

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

### Quick start with scbean

Download the [data](http://141.211.10.196/result/test/papers/vipcca/data.tar.gz) of the following test code. 
Please see the detailed [usage](https://scbean.readthedocs.io/en/latest/) in our tutorials of scbean.



