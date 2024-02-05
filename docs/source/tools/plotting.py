import scanpy as sc
import pandas as pd
import matplotlib.pyplot as pl
import seaborn as sns
import numpy as np

fontsize=15
params = {'legend.fontsize': fontsize,
          'figure.figsize': (8, 8),
          'figure.dpi': 150,
         'axes.labelsize': fontsize,
         'axes.titlesize':fontsize,
         'xtick.labelsize':fontsize,
         'ytick.labelsize':fontsize}
pl.rcParams.update(params)
#sns.set(style='white', rc={'figure.figsize':(5,5), 'figure.dpi':150})

def run_embedding(adata,path="./",n_neighbors=15, init=None, method='umap', resolution=1.0, min_dist=0.5):
	sc.pp.neighbors(adata,use_rep='X_vipcca')
	sc.tl.louvain(adata, directed=False,resolution =resolution)
	if method in ("umap","all"):
		sc.tl.umap(adata,min_dist=min_dist)
	if method in ("tsne","all"):
		sc.tl.tsne(adata, use_rep="X_vipcca",n_jobs=20)

def plotEmbedding(adata,path,group_by="batch",ncol=5, method="umap",legend_loc=None,frameon=True,legend_fontsize=6,title=""):
	filename=path+"2dplot_"+group_by+"_"+method+".png"
	if method in ("umap","all"):
		sc.pl.umap(adata,frameon =frameon,color=group_by,show=False,use_raw=False,legend_loc=legend_loc,legend_fontsize=legend_fontsize)
		pl.title(title)
		pl.savefig(filename)
		pl.close()
			
	if method in ("tsne","all"):
		sc.pl.tsne(adata,frameon =frameon,color=group_by,show=False,legend_fontsize=legend_fontsize)
		pl.title("")
		pl.legend(loc=3,fontsize=legend_fontsize,mode="expand",bbox_to_anchor=(0.0, 1.01, 1, 0.2),ncol=ncol)
		pl.savefig(filename)
		pl.close()
	
	if method in ("location","all"):
		sc.pl.scatter(adata, x='xcoord',y='ycoord',frameon =frameon,color=group_by,show=False,use_raw=False,legend_loc="right margin",legend_fontsize=legend_fontsize)
		pl.title("")
		pl.savefig(filename)
		pl.close()
		
def plotDEG(adata,path,group_by="louvain",method="wilcoxon"):
	current_path=path+"deg_"+group_by+"_"+method+"_"
	sc.tl.pca(adata)
	sc.tl.dendrogram(adata,groupby=group_by,use_rep ="X_pca")
	sc.tl.rank_genes_groups(adata,groupby=group_by,use_raw=False,n_genes=100,method=method,rankby_abs=False,corr_method="benjamini-hochberg")
	sc.tl.filter_rank_genes_groups(adata, min_fold_change=3)
	sc.pl.rank_genes_groups(adata,show=False)
	pl.savefig(current_path+"rank_genes_groups.png")
	sc.pl.rank_genes_groups(adata, key='rank_genes_groups_filtered',show=False)
	pl.savefig(current_path+"rank_genes_groups_filtered.png")
	sc.pl.rank_genes_groups_dotplot(adata, key='rank_genes_groups_filtered')
	pl.savefig(current_path+"rank_genes_groups_filtered_dotplot.png")
	top_ranked_genes=pd.DataFrame(adata.uns['rank_genes_groups']['names'][range(1)])
	top_ranked_genes_index = pd.Index(top_ranked_genes.values.flatten()).drop_duplicates(keep='first')
	sc.pl.stacked_violin(adata,top_ranked_genes_index,groupby=group_by,use_raw=False,show=False)
	pl.savefig(current_path+"stacked_violin.png")
	sc.pl.heatmap(adata,top_ranked_genes_index,groupby=group_by,use_raw=False,show=False,swap_axes=False)
	pl.savefig(current_path+"headmap.png")
	pl.close()

def plotDEG2(adata,path,key_batch="batch",key_celltype="celltype",method="wilcoxon", mode=None):
	celltypes=adata.obs[key_celltype].cat.categories.values
	# sc.pp.scale(adata)
	for ct in celltypes:
		if np.sum(adata.obs[key_celltype]==ct)<10:
			continue
		adata_sub = adata[adata.obs[key_celltype]==ct]
		sc.tl.rank_genes_groups(adata_sub, groupby=key_batch, use_raw=False, n_genes=adata.shape[1], rankby_abs=False)
		catname=adata_sub.obs[key_batch].cat.categories.values[0]
		pvals_adj=adata_sub.uns['rank_genes_groups']['pvals_adj'][catname]
		logfc=adata_sub.uns['rank_genes_groups']['logfoldchanges'][catname]
		t=np.isnan(logfc)
		logfc=logfc[~t]
		pvals_adj=pvals_adj[~t]
		# f=np.absolute(logfc)<10
		# logfc=logfc[f]
		# pvals_adj=pvals_adj[f]
		indicator_neg=np.logical_and(logfc<0, pvals_adj<1e-50)
		indicator_pos=np.logical_and(logfc>0, pvals_adj<1e-50)
		ndeg=np.sum(indicator_neg)
		pdeg=np.sum(indicator_pos)
		
		cv=np.repeat('k', len(logfc))
		cv[indicator_pos]='r'
		cv[indicator_neg]='b'
		
		pl.scatter(logfc, -np.log10(pvals_adj+1e-300), c=cv, s=1)
		pl.axvline(x=0, linewidth=0.5, color='c')
		from adjustText import adjust_text
		text1=pl.text(-7.5, 305, '%d significant genes'%np.int(ndeg))
		text2=pl.text(2.5, 305, '%d significant genes'%np.int(pdeg))
		adjust_text([text1,text2])
		if mode is not None:
			tt="{} ({})".format(mode,ct)
		pl.title(tt)
		pl.xlabel(r'$log_2(FC)\ (ctr/stim)$')
		pl.ylabel(r'$-log_{10}(FDR\ adjusted\ p-value)$')
		pl.savefig(path+"dge_%s.png"%ct.replace('+','p'))
		pl.close()
		
def runGeoSketch(adata,N=10000,use_rep="X_pca"):
	from geosketch import gs
	sc.tl.pca(adata)
	sketch_index = gs(adata.obsm[use_rep], N, replace=False)
	adata.uns['geosketch']=adata.obs.index[sketch_index]
	subdata = adata[adata.obs.index[sketch_index]]
	return subdata

def plotQQdeg(adata,path,groupby="batch",method="wilcoxon"):
	sc.tl.rank_genes_groups(adata, groupby=groupby, method=method,n_genes=adata.shape[1],rankby_abs=True,use_raw=False)
	result = adata.uns['rank_genes_groups']
	groups = result['names'].dtype.names
	df = pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'pvals_adj']})
	df.to_csv(path+"markers_"+groupby+"_"+method+".csv")
	pvals=adata.uns['rank_genes_groups']['pvals_adj']['0']+1e-260


def plotQQdeg2(adata,path,groupby="batch",method="wilcoxon"):
	for c in adata.obs.cell_type.cat.categories.values:
		adatatemp= adata[adata.obs.cell_type==c,:]
		patht=path+"pvals"+c.replace("/","_")
		plotQQdeg(adatatemp,patht)


def plotPrediction(err,result_path):
	err=err[err<1000]
	x=range(len(err))
	pl.scatter(x,err,c='r',s=1)
	pl.savefig(result_path+"square_error.png")
	pl.close()


def plotCorrelation(y,y_pred, save=True, result_path='./', show=True, rnum=1e4, lim=20):
	"""\
	Plot correlation between original data and corrected data

	Parameters
	----------

	y: matrix or csr_matrix
		The original data matrix.

	y_pred: matrix or csr_matrix
		The data matrix integrated by vipcca.

	save: bool, optional (default: True)
		If True, save the figure into result_path.

	result_path: string, optional (default: './')
		The path for saving the figure.

	show: bool, optional (default: True)
		If True, show the figure.

	rnum: double, optional (default: 1e4)
		The number of points you want to sample randomly in the matrix.

	lim: int, optional (default: 20)
		the right parameter of matplotlib.pyplot.xlim(left, right)

	"""
	from scipy.sparse import csr_matrix
	if (isinstance(y, csr_matrix)):
		y = y.toarray()
	rx = np.random.choice(y.shape[0], np.int(rnum), replace=True)
	ry = np.random.choice(y.shape[1], np.int(rnum), replace=True)
	pl.rcParams['figure.figsize'] = (8, 7)
	pl.scatter(y[rx, ry], y_pred[rx, ry], c='r', s=1)
	pl.xlim(-1, lim)
	pl.ylim(-1, lim)
	pl.xlabel('uncorrected_x')
	pl.ylabel('corrected_x')
	if show:
		pl.show()
	if save:
		pl.savefig(result_path+"correlation.png")
	
	
	
