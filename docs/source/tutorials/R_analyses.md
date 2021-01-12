# Analysis of integrated result

We implement downstream analysis based on the R language and some R packages (Seurat, KBET), including evaluating the degree of mixing and gene differential expression analysis.

***

### Read h5ad file with R
**Import packages**

```R
library(Seurat)
library(SeuratDisk)
```

**Convert h5ad files to h5seurat files**
After executing this code, a file in h5seurat format will be generated in the path where the h5ad file is located

```R
Convert("/Users/zhongyuanke/data/vipcca/mixed_cell_lines_result/output_save.h5ad", dest = "h5seurat", overwrite = TRUE)
```

**Read h5seurat file into a Seurat Object**

```R
mixed_cell_lines <- LoadH5Seurat("/Users/zhongyuanke/data/vipcca/mixed_cell_lines_result/output_save.h5seurat")
```



### Calculating kBET 
**Preparing kBET data**

```R
celltype <- t(data.frame(mcl@meta.data[["celltype"]]))
vipcca_emb <- data.frame(mcl@reductions[["vipcca"]]@cell.embeddings)
batch <- mcl@meta.data[["X_batch"]]

# Split the batch ID and VIPCCA embedding result by cell type.
vipcca_emb <- split(vipcca_emb,celltype) 
batch <- split(batch,celltype) 
```

**Calculating kBET for 293T celltype.**

For detailed usage of kbet, please check the [kBET tutorial](https://github.com/theislab/kBET) .


```R
library(kBET)

subset_size <- 0.25 #subsample to 25% of the data.
subset_id <- sample.int(n = length(t(batch[["293t"]])), size = floor(subset_size * length(t(batch[["293t"]]))), replace=FALSE)
batch.estimate_1 <- kBET(data_mix[["293t"]][subset_id,], batch[["293t"]][subset_id])
```

<img src="" width="50%">



***



### Plotting Enhanced Volcano

```R
library(EnhancedVolcano)

```

