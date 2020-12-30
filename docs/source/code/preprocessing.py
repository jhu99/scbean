import scipy
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.preprocessing import QuantileTransformer
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('Agg')
sns.set(style='white', rc={'figure.figsize': (5, 5), 'figure.dpi': 150})


def read_sc_data(input_file, fmt='h5ad', backed=None, transpose=False, sparse=False, delimiter=" ", unique_name=True,
				batch_name=None, var_names="gene_symbols"):

	"""\
	Read single cell dataset

	Params
	------

	input_file : string
		The path of the file to be read.

	fmt : string, optional (default: 'h5ad')
		The file type of the file to be read.

	backed : Union[Literal[‘r’, ‘r+’], bool, None] (default: None)
		If 'r', load AnnData in backed mode instead of fully loading it into memory (memory mode).
		If you want to modify backed attributes of the AnnData object, you need to choose 'r+'.

	transpose: bool, optional (default: False)
		Whether to transpose the read data.

	sparse: bool, optional (default: False)
		Whether the data in the dataset is stored in sparse matrix format.

	delimiter: str,  optional (default: ' ')
		Delimiter that separates data within text file. If None, will split at arbitrary number of white spaces,
		which is different from enforcing splitting at single white space ' '.

	unique_name: bool, optional (default: False)
		If Ture, AnnData object execute var_names_make_unique() and obs_names_make_unique() functions.

	batch_name: string, optional (default: None)
		Batch name of current batch data

	var_names: Literal[‘gene_symbols’, ‘gene_ids’] (default: 'gene_symbols')
		The variables index when the file type is 'mtx'.

	Returns
	-------
	AnnData
	"""
	if fmt == '10x_h5':
		adata = sc.read_10x_h5(input_file)
	elif fmt == '10x_mtx':
		adata = sc.read_10x_mtx(input_file, var_names=var_names)
	elif fmt == "mtx":
		adata = sc.read_mtx(input_file)
	elif fmt == 'h5ad':
		adata = sc.read_h5ad(input_file, backed=backed)
	elif fmt == "csv":
		adata = sc.read_csv(input_file)
	elif fmt == "txt":
		adata = sc.read_text(input_file, delimiter=delimiter)
	elif fmt == "tsv":
		adata = sc.read_text(input_file, delimiter="\t")
	else:
		raise ValueError('`format` needs to be \'10x_h5\' or \'10x_mtx\'')
	if transpose:
		adata = adata.transpose()
	if sparse:
		adata.X = csr_matrix(adata.X, dtype='float32')
	if unique_name:
		adata.var_names_make_unique()
		adata.obs_names_make_unique()
	if batch_name is not None:
		adata.obs["_batch"]=batch_name
	return adata


def add_location(adata):
	adata_loc = AnnData(adata.obs[['xcoord','ycoord']])
	adata = adata.T.concatenate(adata_loc.T, index_unique=None).T
	adata.obs.rename(columns={'xcoord-0':'xcoord','ycoord-0':'ycoord'},inplace=True)
	return adata


def write2mtx(adata, path):
	if not os.path.exists(path):
		os.makedirs(path)
	gn=adata.var
	bc=adata.obs
	gn.to_csv(path+"genes.tsv",sep="\t",index=True,header=False)
	bc.to_csv(path+"annotation.tsv",sep="\t",index=True,header=False)
	bc.to_csv(path+"barcodes.tsv",index=True,header=False,columns=[])
	scipy.io.mmwrite(path+"matrix.mtx", adata.X.transpose().astype(int))


def recipe_vipcca(adata, n_top_genes=2000, ncounts=1e6, min_cells=10, min_genes=10, max_genes=2500,mt_ratio=0.05,lognorm=True,loc=False,filter=True,hvg=True):
	if filter:
			sc.pp.filter_genes(adata,min_cells=min_cells)
			sc.pp.filter_cells(adata,min_genes=min_genes)
			mito_genes = adata.var_names.str.upper().str.startswith('MT-')
			adata.obs['percent_mito'] = np.sum(adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
			adata = adata[adata.obs['n_genes'] < max_genes, :]
			adata = adata[adata.obs['percent_mito'] < mt_ratio, :]
	if lognorm:
		adata_norm = logNormalization(adata, loc=loc)
	sc.pp.highly_variable_genes(adata_norm, flavor='seurat', n_top_genes =n_top_genes , inplace=True)
	if loc:
		adata_norm.var.loc[['xcoord','ycoord'],'highly_variable']=[True,True]
	features = adata_norm.var_names[adata_norm.var['highly_variable']]
	adata_norm = adata_norm[:,features]
	sc.pp.filter_cells(adata_norm, min_genes=5)
	adata = adata[adata_norm.obs.index,:]
	adata = adata[:,features]
	adata_norm.raw = adata.copy()
	return adata_norm, adata


def preprocessing(datasets, min_cells=1, min_genes=1, n_top_genes=2000, mt_ratio=0.8, lognorm=True, hvg=True,
				index_unique=None):

	"""\
	Preprocess and merge data sets from different batches

	Parameters
	----------

	datasets: list, optional (default: None)
		the list of anndata objects from different batches

	min_cells: int, optional (default: 1)
		Minimum number of counts required for a cell to pass filtering.

	min_genes: int, optional (default: 1)
		Minimum number of counts required for a gene to pass filtering.

	n_top_genes: int, optional (default: 2000)
		Number of highly-variable genes to keep.

	mt_ratio: double, optional (default: 0.8)
		Maximum proportion of mito genes for a cell to pass filtering.

	lognorm: bool, optional (default: True)
		If True, execute lognorm() function.

	hvg: bool, optional (default: True)
		If True, choose hypervariable genes for AnnData object.

	index_unique: string, optional (default: None)
		Make the index unique by joining the existing index names with the batch category, using
		index_unique='-', for instance. Provide None to keep existing indices.

	Returns
	-------
	AnnData
	"""
	if lognorm:
		for i in range(len(datasets)):
			sc.pp.filter_genes(datasets[i],min_cells=min_cells)
			sc.pp.filter_cells(datasets[i],min_genes=min_genes)
			mito_genes = datasets[i].var_names.str.upper().str.startswith('MT-')
			datasets[i].obs['percent_mito'] = np.sum(datasets[i][:, mito_genes].X, axis=1).A1 / np.sum(datasets[i].X, axis=1).A1
			datasets[i] = datasets[i][datasets[i].obs['percent_mito'] < mt_ratio, :]
			datasets[i]=logNormalization(datasets[i])
	all_features=pd.Index([])
	for i in range(len(datasets)):
		all_features=all_features.union(datasets[i].var_names)
	common_features=all_features
	for i in range(len(datasets)):
		common_features=common_features.intersection(datasets[i].var.index)
	df_var = pd.DataFrame(index=all_features)
	df_var['selected']=0
	if hvg:
		for i in range(len(datasets)):
			sc.pp.highly_variable_genes(datasets[i], flavor='seurat', n_top_genes =n_top_genes , inplace=True)
			features = datasets[i].var_names[datasets[i].var['highly_variable']]
			df_var.loc[features]+=1
		df_var=df_var.loc[common_features]
		df_var.sort_values(by="selected", ascending=False, inplace=True)
		selected_features=df_var.index[range(n_top_genes)]
	else:
		selected_features=common_features
	for i in range(len(datasets)):
		datasets[i] = datasets[i][:,selected_features]
		if i==0:
			adata=datasets[i]
		else:
			adata=adata.concatenate(datasets[i],index_unique =index_unique)
	return adata


def split_object(adata,by="batch"):
	adata.obs[by]=adata.obs[by].astype("category")
	index_cat = adata.obs[by].cat.categories
	datasets=[]
	for cat in index_cat:
		data=adata[adata.obs[by]==cat,:]
		datasets.append(data)
	return datasets


def scale(adata,max_value=None,use_rep=None):
	if use_rep is None:
		sc.pp.scale(adata,max_value=max_value)
	else:
		maxv=adata.obsm[use_rep].max(0)
		minv=adata.obsm[use_rep].min(0)
		adata.obsm[use_rep]=(adata.obsm[use_rep]-minv)/(maxv-minv)


def rank_normalization(adata,n_quantiles=2000):
	qt = QuantileTransformer(n_quantiles=n_quantiles,output_distribution='normal',ignore_implicit_zeros=False)
	adata.X = qt.fit_transform(adata.X.toarray())


def logNormalization(adata,loc=False):
	adata.obs['n_counts'] = adata.X.sum(axis=1).A1
	adata.obs['size_factor']=adata.obs['n_counts']/1e6
	adata.raw=adata.copy()
	sc.pp.normalize_total(adata,target_sum=1e6)
	if loc:
		adata_loc = add_location(adata)
		adata_loc.obs.rename(columns={'n_counts-0':'n_counts','size_factor-0':'size_factor'},inplace=True)
		sc.pp.log1p(adata_loc)
		return adata_loc
	else:
		sc.pp.log1p(adata)
		return adata


def createCoordination(adata, result_path=None, plot=False, resolution=1):
	xmax=np.ceil(adata.obs.xcoord.max())+resolution
	ymax=np.ceil(adata.obs.ycoord.max())+resolution
	nn=(xmax/resolution).astype("uint16")
	mm=(ymax/resolution).astype("uint16")
	img_size=np.int(np.ceil(np.max([nn,mm])/216)*216)
	mX=np.zeros(shape=(img_size,img_size),dtype="uint16")
	xcoord_img = (adata.obs.xcoord.values/resolution).astype("uint16")
	ycoord_img = (adata.obs.ycoord.values/resolution).astype("uint16")
	mX[ycoord_img,xcoord_img]+=1
	adata.obs["xcoord_img"]=xcoord_img
	adata.obs["ycoord_img"]=ycoord_img
	count = csr_matrix(([1]*len(xcoord_img),(xcoord_img,ycoord_img)),shape=(img_size,img_size),dtype="int")
	adata.obs["overlapped"]=count.toarray()[xcoord_img,ycoord_img]
	adata.X = csc_matrix((adata.X.T.todense()/adata.obs.overlapped.values).T)
	size_factor = np.zeros(shape=(img_size,img_size,1),dtype="float32")
	size_factor[xcoord_img,ycoord_img,0] = adata.obs["size_factor"]
	if plot:
		plt.scatter(xcoord_img,ycoord_img,s=1,c='r')
		plt.title("All cells in one image")
		plt.savefig(result_path+"image_cells.png")
		plt.close()
	return img_size, img_size, size_factor


def generate_img_from_genes(adata,img_size=648,batch_size=32, cols=[6,7]):
	while True:
		for i in range(adata.shape[1]):
			data=adata.X.getcol(i).data
			ind=adata.X.getcol(i).indices
			xx = adata.obs.iloc[ind,cols[0]].values
			yy = adata.obs.iloc[ind,cols[1]].values
			img = csr_matrix((data,(yy,xx)),shape=(img_size,img_size),dtype="f4").toarray().reshape(1,img_size,img_size,1)
			yield (img,img)


def generate_datasets(adata,img_size=648, cols=[6,7], count_only=False):
	n_genes=adata.shape[1]
	x=np.zeros(shape=(n_genes,img_size,img_size,1),dtype="float32")
	x_label=np.zeros_like(x)
	for i in range(n_genes):
		fi=i
		data=adata.X.getcol(fi).data
		ind=adata.X.getcol(fi).indices
		xx = adata.obs.ix[ind, 'xcoord_img'].values
		yy = adata.obs.ix[ind, 'ycoord_img'].values
		## accumalate cells in the same location
		x[i,]=csr_matrix((data,(xx,yy)),shape=(img_size,img_size),dtype="float32").toarray().reshape(img_size,img_size,1)
		if not count_only:
			rawdata=adata.raw.X.getcol(fi).data
			x_label[i,] = csr_matrix((rawdata,(xx,yy)),shape=(img_size,img_size),dtype="float32").toarray().reshape(img_size,img_size,1)
		else:
			x_label=x
	return x, x_label
	
## generate data used for fit_gen()
# x_train, x_train_label, x_val, x_val_label = pp.generate_datasets(adata,img_size=img_size,validation_split=validation_split)
# datagen = ImageDataGenerator()
# generator_train=datagen.flow(x_train, x_train_label, batch_size=batch_size)
# generator_val=datagen.flow(x_val, x_val_label, batch_size=batch_size)
# unet.fit_gen(generator_train, validation_data=generator_val)


def generate_datasets_ext(adata,img_size=648,validation_split=0.2):
	n_genes=adata.shape[1]
	samples_for_train=n_genes-np.int(n_genes*validation_split)
	id_list=np.random.permutation(n_genes)
	x_train=np.zeros((samples_for_train,img_size,img_size,1),dtype="f4")
	x_train_label=np.zeros_like(x_train)
	x_val=np.zeros((n_genes-samples_for_train,img_size,img_size,1),dtype="f4")
	x_val_label=np.zeros_like(x_val)
	p=0
	for i in id_list[range(samples_for_train)]:
		data=adata.X.getcol(i).data
		rawdata=adata.raw.X.getcol(i).data
		ind=adata.X.getcol(i).indices
		xx = adata.obs.iloc[ind,6].values
		yy = adata.obs.iloc[ind,7].values
		x_train[p,] = csr_matrix((data,(yy,xx)),shape=(img_size,img_size),dtype="f4").toarray().reshape(1,img_size,img_size,1)
		x_train_label[p,] = csr_matrix((rawdata,(yy,xx)),shape=(img_size,img_size),dtype="f4").toarray().reshape(1,img_size,img_size,1)
		p+=1
	p=0	
	for i in id_list[np.arange(samples_for_train,n_genes)]:
		data=adata.X.getcol(i).data
		rawdata=adata.raw.X.getcol(i).data
		ind=adata.X.getcol(i).indices
		xx = adata.obs.iloc[ind,6].values
		yy = adata.obs.iloc[ind,7].values
		x_val[p,] = csr_matrix((data,(yy,xx)),shape=(img_size,img_size),dtype="f4").toarray().reshape(1,img_size,img_size,1)
		x_val_label[p,] = csr_matrix((rawdata,(yy,xx)),shape=(img_size,img_size),dtype="f4").toarray().reshape(1,img_size,img_size,1)
		p+=1
	return x_train, x_train_label, x_val, x_val_label

