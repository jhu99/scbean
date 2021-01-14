import numpy as np
import math
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('Agg')
sns.set(style='white', rc={'figure.figsize':(5,5), 'figure.dpi':150})


def findNeighbors(adata):
	sc.pp.neighbors(adata, n_neighbors=50,use_rep='X_vipcca',n_pcs=16)
	ann=adata.obs
	ann['nid']=range(ann.shape[0])
	ct=ann[ann['tech']=="rna"]['celltype'].unique()
	ann_atac=ann[ann['tech']=="atac"]
	ann_rna=ann[ann['tech']=="rna"]
	weights=ann_rna['celltype'].value_counts()/ann_rna.shape[0]
	for item in ct:
		ann_atac[item]=0.0
	ann_atac["pred_max_score"]=0.0
	querycells = ann[ann['tech']=="atac"]['nid']
	dist=adata.uns['neighbors']['distances']
	weights[weights.index]=np.ones(len(weights))
	weights["Platelets"]=0.001
	
	for cell in querycells:
		cell_key=ann.index[cell]
		nbs=dist.getrow(cell).nonzero()[1]
		ann_nbs=ann.iloc[nbs,:]
		ann_nbs=ann_nbs[ann_nbs['tech']=='rna']
		if ann_nbs.empty is not True:
			for key in ann_nbs.index:
				nbid=ann_nbs.loc[key,'nid']
				ct=ann_nbs.at[key,'celltype']
				var=weights[ann_nbs.at[key,'celltype']]**2
				ann_atac.at[cell_key,ct]+=math.exp(-(dist[cell,nbid].tolist()**2)/(2*var))
			ann_atac.at[cell_key,'pred_max_score'] = max(ann_atac.loc[cell_key,][9:22])
	ann_atac.loc[:,'celltype']=ann_atac.iloc[:,9:22].idxmax(axis=1)
	return ann_atac


def findNeighbors2(adata):
	sc.pp.neighbors(adata, n_neighbors=50,use_rep='X_vipcca',n_pcs=16)
	ann=adata.obs
	ann['nid']=range(ann.shape[0])
	ct=ann[ann['tech']=="rna"]['celltype'].unique()
	ann_atac=ann[ann['tech']=="atac"]
	ann_rna=ann[ann['tech']=="rna"]
	querycells = ann[ann['tech']=="atac"]['nid']
	dist=adata.uns['neighbors']['distances']
	for cell in querycells:
		cell_key=ann.index[cell]
		if cell_key == "TCACAGAGTAACGGAC-1":
			print(cell_key)
		nbs=dist.getrow(cell).nonzero()[1]
		ann_nbs=ann.iloc[nbs,:]
		ann_nbs=ann_nbs[ann_nbs['tech']=='rna']
		if ann_nbs.empty is not True:
				ann_atac.at[cell_key,'celltype']=ann_nbs.celltype.value_counts().idxmax()
	return ann_atac
