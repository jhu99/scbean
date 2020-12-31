import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc
import os
import os.path
import datetime
import tensorflow as tf
from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout, Lambda, Concatenate, concatenate, Add
from keras.models import Model, Sequential, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras import optimizers, regularizers, initializers
from keras.utils import plot_model
from keras.losses import mse, binary_crossentropy
from scipy.sparse import csr_matrix


def sampling(args):
	z_mean, z_log_var = args
	batch = K.shape(z_mean)[0]
	dim = K.int_shape(z_mean)[1]
	epsilon = K.random_normal(shape=(batch, dim))
	return z_mean + K.exp(0.5 * z_log_var) * epsilon

MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e5)
		
class VAE:
	def __init__(self,input_size,path="./",
							y_size=10,
							method="lognorm",
							validation_split=0.0,
							patience_es=200,
							patience_lr=100,
							activation="softplus",
							lambda_regulizer=2.0,
							initializers="glorot_uniform",
							dropout_rate=0.01,
							hidden_layers=[128,64,32],
							l1_l2=(0.0,0.0),
							deterministic=False,
							save=True):
		self.input_size=input_size
		self.vae=None
		self.inputs=None
		self.outputs=None
		self.path=path
		self.lambda_regulizer=lambda_regulizer
		self.initializers=initializers
		self.method=method
		self.optimizer=optimizers.Adam(lr=0.01)
		self.y_size=y_size
		self.dropout_rate=dropout_rate
		self.hidden_layers=hidden_layers
		self.kernel_regularizer=regularizers.l1_l2(l1=l1_l2[0], l2=l1_l2[1])
		self.validation_split=validation_split
		self.activation=activation
		self.deterministic=deterministic
		self.save=save
		callbacks = []
		checkpointer = ModelCheckpoint(filepath=path+"model_{epoch:04d}.h5", verbose=1, save_best_only=True, save_weights_only=False,period=100)
		reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=patience_lr, min_lr=0.0001)
		early_stop = EarlyStopping(monitor='loss', patience=patience_es)
		tensor_board = TensorBoard(log_dir=path+"logs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S/"))
		callbacks.append(checkpointer)
		callbacks.append(reduce_lr)
		callbacks.append(early_stop)
		callbacks.append(tensor_board)
		self.callbacks = callbacks
	def build(self):
		# build encoder
		Relu="relu"
		inputs = Input(shape=(self.input_size,), name='encoder_input')
		x = Dense(128, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers, bias_initializer='zeros', name='en_hidden_layer_x1')(inputs)
		x = BatchNormalization(center=True,scale=False)(x)
		x = Activation(Relu)(x)
		x = Dropout(self.dropout_rate)(x)
		x = Dense(64, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers, name='en_hidden_layer_x2')(x)
		x = BatchNormalization(center=True,scale=False)(x)
		x = Activation(Relu)(x)
		x = Dropout(self.dropout_rate)(x)
		
		z_mean = Dense(32, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers, name="encoder_mean")(x)
		z_log_var = Dense(32, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers, name="encoder_log_var")(x)
		z = Lambda(sampling, output_shape=(32,), name='hidden_var_z')([z_mean, z_log_var])
		encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder_mlp')

		latent_inputs = Input(shape=(32,), name='z_sampling')
		x = Dense(64, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(latent_inputs)
		x = BatchNormalization(center=True,scale=False)(x)
		x = Activation(Relu)(x)
		x = Dropout(self.dropout_rate)(x)
		x = Dense(128, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(x)
		x = BatchNormalization(center=True,scale=False)(x)
		x = Activation(Relu)(x)
		x = Dropout(self.dropout_rate)(x)
		if self.method=="count":
			outputs = Dense(self.input_size, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers, activation="linear")(x)
		elif self.method=="qqnorm":
			outputs = Dense(self.input_size, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers, activation="linear")(x)
		else:
			outputs = Dense(self.input_size, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers, activation="softplus")(x)	
		decoder = Model(latent_inputs,outputs, name='decoder_mlp')
		if self.deterministic:
			outputs = decoder(encoder(inputs)[0])
		else:
			outputs = decoder(encoder(inputs)[2])
		vae = Model(inputs, outputs, name='vae_mlp')
		reconstruction_loss = mse(inputs, outputs)
		reconstruction_loss *= self.input_size
		kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
		kl_loss = -0.5*K.sum(kl_loss, axis=-1)
		vae_loss = K.mean(reconstruction_loss + kl_loss)
		vae.add_loss(vae_loss)
		self.vae = vae
		self.encoder = encoder
		self.decoder = decoder
	def compile(self):
		self.vae.compile(optimizer=self.optimizer)
		self.vae.summary()
	def train(self, adata, batch_size=256, epochs=300):
		if os.path.isfile(self.path+"model.h5"):
			#self.vae=load_model(self.path+"model.h5")
			self.vae.load_weights(self.path+"model.h5")
		else:
			self.vae.fit(adata.X, epochs=epochs, batch_size=batch_size, callbacks=self.callbacks, validation_split=self.validation_split, shuffle=True)
	def integrate(self, xadata, save=True, use_mean=True):
		[z_mean, z_log_var, z_batch] = self.encoder.predict(xadata.X)
		if use_mean:
			y_mean = self.decoder.predict(z_mean)
		else:
			y_mean = self.decoder.predict(z_batch)
		yadata = AnnData(X=y_mean, obs=xadata.obs, var=xadata.var)
		yadata.raw=AnnData(X=xadata.raw.X,var=xadata.raw.var)
		if save:
			yadata.write(self.path+"output.h5ad")
		yadata.obsm['X_vipcca']=z_mean
		return yadata

class CVAE(VAE):
	def __init__(self, batches=16, batches2=8, **kwargs):
		super().__init__(**kwargs)
		self.batches=batches
		self.batches2=batches2
	
	def build(self):
		Relu="relu"
		inputs = Input(shape=(self.input_size,), name='encoder_input')
		inputs_batch = Input(shape=(self.batches,), name = 'batch_input1')
		y = Dense(self.batches, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(inputs_batch)
		y = BatchNormalization(center=True,scale=False)(y)
		y = Activation(Relu)(y)
		y = Dense(self.batches, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(y)
		y = BatchNormalization(center=True,scale=False)(y)
		y = Activation(Relu)(y)
		x = concatenate([inputs,y])
		
		en_ly_size=len(self.hidden_layers)
		for i in range(en_ly_size):
			if i==en_ly_size-1:
				break
			ns = self.hidden_layers[i]
			x = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(x)
			x = BatchNormalization(center=True,scale=False)(x)
			x = Activation(Relu)(x)
			x = Dropout(self.dropout_rate)(x)
		ns = self.hidden_layers[i]
		z_mean = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers, name="encoder_mean")(x)
		z_log_var = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers, name="encoder_log_var")(x)
		z = Lambda(sampling, output_shape=(ns,), name='hidden_var_z')([z_mean, z_log_var])
		encoder = Model([inputs,inputs_batch], [z_mean, z_log_var, z], name='encoder_mlp')
		
		latent_inputs = Input(shape=(ns,), name='z_sampling')
		inputs_batch2= Input(shape=(self.batches2,), name='batch_input2')
		y = Dense(self.batches2, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(inputs_batch2)
		y = BatchNormalization(center=True,scale=False)(y)
		y = Activation(Relu)(y)
		y = Dense(self.batches, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(y)
		y = BatchNormalization(center=True,scale=False)(y)
		y = Activation(Relu)(y)
		v = concatenate([latent_inputs, y])
		x = latent_inputs
		for i in range(en_ly_size-1,0,-1):
			ns = self.hidden_layers[i-1]
			v = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(v)
			v = BatchNormalization(center=True,scale=False)(v)
			v = Activation(Relu)(v)
			v = Dropout(self.dropout_rate)(v)
			
			x = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(x)
			x = BatchNormalization(center=True,scale=False)(x)
			x = Activation(Relu)(x)
			x = Dropout(self.dropout_rate)(x)

		v = Dense(self.input_size, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers, activation=self.activation)(v)	
		x = Dense(self.input_size, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers, activation="hard_sigmoid")(x)	
		x = Add()([v,x])
		outputs = Activation(Relu)(x)
		decoder = Model([latent_inputs,inputs_batch2],outputs, name='decoder_mlp')
		
		if self.deterministic:
			outputs = decoder([encoder([inputs,inputs_batch])[0],inputs_batch2])
		else:
			outputs = decoder([encoder([inputs,inputs_batch])[2],inputs_batch2])
		vae = Model([inputs,inputs_batch,inputs_batch2], outputs, name='vae_mlp')
		#
		noise = tf.math.subtract(inputs,outputs)
		var = tf.math.reduce_variance(noise)
		reconstruction_loss = mse(inputs, outputs)
		reconstruction_loss *= (0.5*self.input_size)/var
		reconstruction_loss += (0.5*self.input_size)*tf.math.log(var)
		kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
		kl_loss = -0.5*K.sum(kl_loss, axis=-1)
		vae_loss = K.mean(2*reconstruction_loss + self.lambda_regulizer*kl_loss)
		vae.add_loss(vae_loss)
		self.vae = vae
		self.encoder = encoder
		self.decoder = decoder
		
	def train(self, adata, batch_size=256, epochs=300, model_file=None):
		if model_file is not None:
			# self.vae=load_model(self.path+model_file)
			self.vae.load_weights(self.path+model_file)
		else:
			self.vae.fit([adata.X, adata.obsm['X_batch'], adata.obsm['X_batch2']], epochs=epochs, batch_size=batch_size, callbacks=self.callbacks, validation_split=self.validation_split, shuffle=True)
			self.vae.save(self.path+"model.h5")
	
	def integrate(self, xadata, save=True, use_mean=True):
		[z_mean, z_log_var, z_batch] = self.encoder.predict([xadata.X, xadata.obsm['X_batch']])
		if use_mean:
			z_samples=z_mean
		else:
			z_samples=z_batch
			
		y_mean = self.decoder.predict([z_samples,xadata.obsm['X_batch2']])
		yadata = AnnData(X=y_mean, obs=xadata.obs, var=xadata.var)
		yadata.raw=xadata.copy()
		yadata.obsm['X_vipcca']=z_mean
		
		i_mean = self.decoder.predict([z_samples,np.tile(xadata.obsm['X_batch2'][1],(xadata.shape[0],1))])
		iadata = AnnData(X=i_mean, obs=xadata.obs, var=xadata.var)
		i_mean[i_mean<0.1]=0
		iadata.raw = AnnData(X=csr_matrix(i_mean), obs=xadata.obs, var=xadata.var)
		iadata.obsm['X_vipcca']=z_mean

		if save:
			iadata.write(self.path+"output.h5ad")
		return yadata

class CVAE2(VAE):
	def __init__(self, batches=2, **kwargs):
		super().__init__(**kwargs)
		self.batches=batches
		
	def build(self):
		Relu="relu"
		inputs = Input(shape=(self.input_size,), name='encoder_input')
		en_ly_size=len(self.hidden_layers)
		for i in range(en_ly_size):
			ns = self.hidden_layers[i]
			if i==0:
				x = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(inputs)
			elif i==en_ly_size-1:
				break
			else:
				x = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(x)
			x = BatchNormalization(center=True,scale=False)(x)
			x = Activation(Relu)(x)
			x = Dropout(self.dropout_rate)(x)
		ns = self.hidden_layers[i]
		z_mean = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers, name="encoder_mean")(x)
		z_log_var = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers, name="encoder_log_var")(x)
		z = Lambda(sampling, output_shape=(ns,), name='hidden_var_z')([z_mean, z_log_var])
		encoder = Model([inputs], [z_mean, z_log_var, z], name='encoder_mlp')
		
		latent_inputs = Input(shape=(ns,), name='z_sampling')
		inputs_batch = Input(shape=(self.batches,), name = 'batch_input')
		y = Dense(self.batches, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(inputs_batch)
		y = BatchNormalization(center=True,scale=False)(y)
		y = Activation(Relu)(y)
		y = Dense(self.batches, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(y)
		y = BatchNormalization(center=True,scale=False)(y)
		y = Activation(Relu)(y)
		v = concatenate([latent_inputs, y])
		x = latent_inputs
		for i in range(en_ly_size-1,0,-1):
			ns = self.hidden_layers[i-1]
			v = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(v)
			v = BatchNormalization(center=True,scale=False)(v)
			v = Activation(Relu)(v)
			v = Dropout(self.dropout_rate)(v)
			
			x = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(x)
			x = BatchNormalization(center=True,scale=False)(x)
			x = Activation(Relu)(x)
			x = Dropout(self.dropout_rate)(x)
		
		v = Dense(self.input_size, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers, activation=self.activation)(v)	
		x = Dense(self.input_size, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers, activation="hard_sigmoid")(x)	
		x = Add()([v,x])
		outputs = Activation(Relu)(x)
		decoder = Model([latent_inputs,inputs_batch],outputs, name='decoder_mlp')
		
		if self.deterministic:
			outputs = decoder([encoder([inputs])[0],inputs_batch])
		else:
			outputs = decoder([encoder([inputs])[2],inputs_batch])
		vae = Model([inputs,inputs_batch], outputs, name='vae_mlp')
		
		noise = tf.math.subtract(inputs,outputs)
		var = tf.math.reduce_variance(noise)
		reconstruction_loss = mse(inputs, outputs)
		reconstruction_loss *= (0.5*self.input_size)/var
		reconstruction_loss += (0.5*self.input_size)*tf.math.log(var)
		kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
		kl_loss = -0.5*K.sum(kl_loss, axis=-1)
		vae_loss = K.mean(2*reconstruction_loss + self.lambda_regulizer*kl_loss)
		vae.add_loss(vae_loss)
		self.vae = vae
		self.encoder = encoder
		self.decoder = decoder
		
	def train(self, adata, batch_size=256, epochs=300, model_file=None):
		if model_file is not None:
			# self.vae=load_model(self.path+model_file)
			self.vae.load_weights(self.path+model_file)
		else:
			self.vae.fit([adata.X, adata.obsm['X_batch']], epochs=epochs, batch_size=batch_size, callbacks=self.callbacks, validation_split=self.validation_split, shuffle=True)
			self.vae.save(self.path+"model.h5")
	
	def integrate(self, xadata, use_mean=True, return_corrected_y=True):
		[z_mean, z_log_var, z_batch] = self.encoder.predict([xadata.X])
		if use_mean:
			z_samples=z_mean
		else:
			z_samples=z_batch
		
		if return_corrected_y:
			i_mean = self.decoder.predict([z_samples,np.tile(xadata.obsm['X_batch'][1],(xadata.shape[0],1))])
			iadata = AnnData(X=i_mean, obs=xadata.obs, var=xadata.var)
			i_mean[i_mean<0.1]=0
			iadata.raw = AnnData(X=csr_matrix(i_mean), obs=xadata.obs, var=xadata.var)
			iadata.obsm['X_vipcca']=z_mean
			iadata.write(self.path+"output.h5ad")
		else:
			i_mean = self.decoder.predict([z_samples,xadata.obsm['X_batch']])
			iadata = AnnData(X=i_mean, obs=xadata.obs, var=xadata.var)
			i_mean[i_mean<0.1]=0
			iadata.raw = AnnData(X=csr_matrix(i_mean), obs=xadata.obs, var=xadata.var)
			iadata.obsm['X_vipcca']=z_mean
			iadata.write(self.path+"output_dsy.h5ad")
		return iadata

class CVAE3(VAE):
	def __init__(self, batches=2, **kwargs):
		super().__init__(**kwargs)
		self.batches=batches
		
	def build(self):
		Relu="relu"
		inputs = Input(shape=(self.input_size,), name='encoder_input')
		x = Dense(128, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers, bias_initializer='zeros', name='en_hidden_layer_x1')(inputs)
		x = BatchNormalization(center=True,scale=False)(x)
		x = Activation(Relu)(x)
		x = Dropout(self.dropout_rate)(x)
		x = Dense(64, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers, name='en_hidden_layer_x2')(x)
		x = BatchNormalization(center=True,scale=False)(x)
		x = Activation(Relu)(x)
		x = Dropout(self.dropout_rate)(x)
		
		z_mean = Dense(32, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers, name="encoder_mean")(x)
		z_log_var = Dense(32, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers, name="encoder_log_var")(x)
		z = Lambda(sampling, output_shape=(32,), name='hidden_var_z')([z_mean, z_log_var])
		encoder = Model([inputs], [z_mean, z_log_var, z], name='encoder_mlp')
		
		latent_inputs = Input(shape=(32,), name='z_sampling')
		inputs_batch = Input(shape=(self.batches,), name = 'batch_input')
		x = Dense(self.batches, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(inputs_batch)
		x = BatchNormalization(center=True,scale=False)(x)
		x = Activation(Relu)(x)
		y = Dense(32, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(latent_inputs)
		y = BatchNormalization(center=True,scale=False)(y)
		y = Activation(Relu)(y)
		xy = concatenate([y, x])
		z = Dense(32, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(xy)
		z = BatchNormalization(center=True,scale=False)(z)
		z = Activation(Relu)(z)
		z_mean_discriminator = Dense(32, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(z)
				
		translator = Model([latent_inputs, inputs_batch], z_mean_discriminator, name="translator_mlp")
		
		z_prime_input = Input(shape=(32,), name='discriminator')
		x = Dense(64, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(z_prime_input)
		x = BatchNormalization(center=True,scale=False)(x)
		x = Activation(Relu)(x)
		x = Dropout(self.dropout_rate)(x)
		x = Dense(128, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(x)
		x = BatchNormalization(center=True,scale=False)(x)
		x = Activation(Relu)(x)
		x = Dropout(self.dropout_rate)(x)
		
		if self.method=="count":
			outputs = Dense(self.input_size, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers, activation="linear")(x)
		elif self.method=="qqnorm":
			outputs = Dense(self.input_size, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers, activation="linear")(x)
		else:
			outputs = Dense(self.input_size, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers, activation=self.activation)(x)
		decoder = Model(z_prime_input, outputs, name='decoder_mlp')
		
		if self.deterministic:
			outputs = decoder(translator([encoder([inputs])[0],inputs_batch]))
		else:
			outputs = decoder(translator([encoder([inputs])[2],inputs_batch]))
		vae = Model([inputs,inputs_batch], outputs, name='cvae_mlp')
		reconstruction_loss = mse(inputs, outputs)
		reconstruction_loss *= self.input_size
		kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
		kl_loss = -0.5*K.sum(kl_loss, axis=-1)
		vae_loss = K.mean(reconstruction_loss + self.lambda_regulizer*kl_loss)
		vae.add_loss(vae_loss)
		self.vae = vae
		self.encoder = encoder
		self.decoder = decoder
		self.translator = translator
		
	def train(self, adata, batch_size=256, epochs=300, model_file=None):
		if model_file is not None:
			if os.path.exists(self.path+model_file):
				self.vae.load_weights(self.path+model_file)
			else:
				raise ValueError('The specified model file does not exist.')
		else:
			self.vae.fit([adata.X, adata.obsm['X_batch']], epochs=epochs, batch_size=batch_size, callbacks=self.callbacks, validation_split=self.validation_split, shuffle=True)
			self.vae.save(self.path+"model.h5")
	
	def integrate(self, xadata, save=True, use_mean=True):
		[z_mean, z_log_var, z_batch] = self.encoder.predict([xadata.X])
		if use_mean:
			z_samples=z_mean
		else:
			z_samples=z_batch
		z_mean_prime = self.translator.predict([z_samples, xadata.obsm['X_batch']])
		
		y_mean = self.decoder.predict([z_mean_prime])
		yadata = AnnData(X=y_mean, obs=xadata.obs, var=xadata.var)
		yadata.raw=xadata.copy()
		yadata.obsm['X_vipcca']=z_mean
		yadata.obsm['X_vipcca']=z_mean_prime
		p_mean = self.decoder.predict([z_mean_prime])
		padata = AnnData(X=y_mean, obs=xadata.obs, var=xadata.var)
		padata.raw=xadata.copy()
		padata.obsm['X_vipcca']=z_mean
		padata.obsm['X_vipcca']=z_mean_prime
		if save:
			yadata.write(self.path+"output.h5ad")
			padata.write(self.path+"integrated.h5ad")
		return yadata
