import pandas as pd
import numpy as np
import scanpy as sc
from keras.models import load_model
from network import VAE, CVAE, CVAE2, CVAE3
from preprocessing import logNormalization, read_sc_data, split_object, preprocessing
from plotting import plotPrediction2, run_embedding, plotEmbedding
import argparse


class Conf(object):
	pass

class VIPCCA(object):
	def __init__(self,
				 adata_all = None,
							patience_es=50,
							patience_lr=25,
							epochs=10000,
							res_path=None,
							split_by="_batch",
							method="lognorm",
							hvg=True,
							batch_input_size=16,
							batch_input_size2=8,
							activation="softplus",
							dropout_rate=0.01,
							hidden_layers=[128,64,32,16],
							lambda_regulizer=1.0,
							initializer="glorot_uniform",
							l1_l2=(0.0, 0.0),
							mode="CVAE",
							model_file=None,
							save=True):
		"""
		Initialize VIPCCA object
		:param patience_es: number of epochs with no improvement after which training will be stopped.
		:param patience_lr: number of epochs with no improvement after which learning rate will be reduced.
		:param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
		:param res_path: Folder path to save model training results model.h5 and output data adata.h5ad
		:param split_by: the obsm_name of obsm used to distinguish different batches.
		:param method: the normalization method for input data, one of {"qqnorm","count", other}
		:param batch_input_size: the length of the batch vector that concatenate with the input layer
		:param batch_input_size2: the length of the batch vector that concatenate with the latent layer
		:param activation: the activation function of
		:param dropout_rate:
		:param hidden_layers: Number of hidden layer neurons in the network
		:param lambda_regulizer: The coefficient multiplied by KL_loss
		:param initializer: Regularizer function applied to the kernel weights matrix.
		:param l1_l2: [L1 regularization factor, L2 regularization factor].
		:param mode:  one of {"CVAE", "CVAE2", "CVAE3"}
		:param model_file: The file name of the trained model, the default is None
		:param save: Whether to save output adata file.
		"""

		self.conf = Conf()
		self.conf.adata_all = adata_all
		self.conf.res_path=res_path
		self.conf.patience_es=patience_es
		self.conf.patience_lr=patience_lr
		self.conf.epochs=epochs
		self.conf.split_by=split_by
		self.conf.method=method
		self.conf.hvg=hvg
		self.conf.batch_input_size=batch_input_size
		self.conf.batch_input_size2=batch_input_size2
		self.conf.dropout_rate=dropout_rate
		self.conf.hidden_layers=hidden_layers
		self.conf.lambda_regulizer=lambda_regulizer
		self.conf.initializer=initializer
		self.conf.activation=activation
		self.conf.model_file=model_file
		self.conf.l1_l2=l1_l2
		self.conf.mode=mode
		self.conf.save=save
		self.preprocessing()

	def preprocessing(self):
		batch_int= self.conf.adata_all.obs[self.conf.split_by].astype("category").cat.codes.values
		np.random.seed(2019)
		batch_dic=np.random.randint(10, size=(np.max(batch_int)+1,self.conf.batch_input_size))
		X_batch=np.zeros((len(batch_int),self.conf.batch_input_size))
		batch_dic2=np.random.randint(10, size=(np.max(batch_int)+1,self.conf.batch_input_size2))
		X_batch2=np.zeros((len(batch_int),self.conf.batch_input_size2))
		for i in range(len(batch_int)):
			X_batch[i,:]=batch_dic[batch_int[i],:]
			X_batch2[i,:]=batch_dic2[batch_int[i],:]
		self.conf.adata_all.obsm["X_batch"]=X_batch
		self.conf.adata_all.obsm["X_batch2"]=X_batch2


	def build(self):
		if self.conf.mode=="CVAE":
			net = CVAE(input_size=self.conf.adata_all.shape[1],
					path=self.conf.res_path,
					batches=self.conf.batch_input_size,
					batches2=self.conf.batch_input_size2,
					patience_es=self.conf.patience_es,
					patience_lr=self.conf.patience_lr,
					activation=self.conf.activation,
					lambda_regulizer=self.conf.lambda_regulizer,
					hidden_layers=self.conf.hidden_layers,
					initializers=self.conf.initializer,
					dropout_rate=self.conf.dropout_rate,
					l1_l2=self.conf.l1_l2,
					method=self.conf.method)
		elif self.conf.mode=="CVAE2":
			net = CVAE2(input_size=self.conf.adata_all.shape[1],
					path=self.conf.res_path,
					batches=self.conf.batch_input_size,
					patience_es=self.conf.patience_es,
					patience_lr=self.conf.patience_lr,
					activation=self.conf.activation,
					lambda_regulizer=self.conf.lambda_regulizer,
					hidden_layers=self.conf.hidden_layers,
					initializers=self.conf.initializer,
					dropout_rate=self.conf.dropout_rate,
					l1_l2=self.conf.l1_l2,
					method=self.conf.method,
					save=self.conf.save)
		else:
			net = CVAE3(input_size=self.conf.adata_all.shape[1],
					path=self.conf.res_path,
					batches=self.conf.batch_input_size,
					patience_es=self.conf.patience_es,
					patience_lr=self.conf.patience_lr,
					activation=self.conf.activation,
					lambda_regulizer=self.conf.lambda_regulizer,
					hidden_layers=self.conf.hidden_layers,
					initializers=self.conf.initializer,
					dropout_rate=self.conf.dropout_rate,
					l1_l2=self.conf.l1_l2,
					method=self.conf.method)
		net.build()
		net.compile()
		self.conf.net=net
		
	def fit_transform(self):
		self.build()
		self.conf.net.train(self.conf.adata_all, epochs=self.conf.epochs, model_file=self.conf.model_file)
		return self.conf.net.integrate(self.conf.adata_all,save=self.conf.save)
		
	def plotEmbedding(self, eps=[], group_by="Batch", min_dist=0.5):
		self.build()
		for epoch in eps:
			test_result_path=self.conf.res_path+"e%04d_"%epoch
			self.conf.net.vae.load_weights(self.conf.res_path+"weights%04d.h5" %epoch)
			adata_transform=self.conf.net.integrate(self.conf.adata_all, save=False)
			plotPrediction2(adata_transform.raw.X,adata_transform.X,result_path=test_result_path)
			run_embedding(adata_transform,path=test_result_path,method="umap", min_dist=min_dist)
			plotEmbedding(adata_transform, path=test_result_path, method='umap', group_by=group_by,legend_loc="right margin")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--epochs', type=int, default=50)
	parser.add_argument('--batch_size', type=int, default=100)
	conf = parser.parse_args()

