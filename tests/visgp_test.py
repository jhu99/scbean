import scbean.model.visgp as visgp
import pandas as pd
import multiprocessing as mp
import anndata as ad

# load data (example: MOB)
filepath = 'data/real_data/Rep11_MOB_count_matrix-1.tsv'
data = pd.read_csv(filepath, sep='\t')

# data preprocessing
position = pd.DataFrame(index=data.index)
position['x'] = data['Unnamed: 0'].str.split('x').str.get(0).map(float)
position['y'] = data['Unnamed: 0'].str.split('x').str.get(1).map(float)
data.drop('Unnamed: 0', axis=1, inplace=True)
data = data.T[data.sum(0) >= 10].T  # Filter practically unobserved genes
data = data.T  # genes * position
position = position[data.sum(0) >= 10]
data = data.T[data.sum(0) >= 10].T
obs = pd.DataFrame()
obs['gene_name'] = data.index.values

adata = ad.AnnData(data.values, obs=obs, var=position, dtype='float64')



#obj = visgp.VISGP(adata, processes=mp.cpu_count())
obj = visgp.VISGP(adata)
results = obj.run_visgp()
print(results)
