# Analysis of integrated result

We implement downstream analysis based on the R language and some R packages (Seurat, KBET), including evaluating the degree of mixing and gene differential expression analysis.

### Read h5ad file with R

Import packages

```R
library(Seurat)
library(SeuratData)
library(SeuratDisk)
```

Convert h5ad files to h5seurat files

```R
Convert("/Users/zhongyuanke/data/vipcca/tutorials/output.h5ad", dest = "h5seurat", overwrite = TRUE)
```

Read h5seurat file into a Seurat Object

```R
mixed_cell_lines <- LoadH5Seurat("/Users/zhongyuanke/data/vipcca/tutorials/output.h5seurat")
```



### Calculating kBET 

```
library(kBET)

```

