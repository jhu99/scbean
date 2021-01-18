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
    def __init__(self, input_size, path="./",
                 y_size=10,
                 method="lognorm",
                 validation_split=0.0,
                 patience_es=200,
                 patience_lr=100,
                 activation="softplus",
                 lambda_regulizer=2.0,
                 initializers="glorot_uniform",
                 dropout_rate=0.01,
                 hidden_layers=[128, 64, 32],
                 l1_l2=(0.0, 0.0),
                 deterministic=False,
                 save=True):
        self.input_size = input_size
        self.vae = None
        self.inputs = None
        self.outputs = None
        self.path = path
        self.lambda_regulizer = lambda_regulizer
        self.initializers = initializers
        self.method = method
        self.optimizer = optimizers.Adam(lr=0.01)
        self.y_size = y_size
        self.dropout_rate = dropout_rate
        self.hidden_layers = hidden_layers
        self.kernel_regularizer = regularizers.l1_l2(l1=l1_l2[0], l2=l1_l2[1])
        self.validation_split = validation_split
        self.activation = activation
        self.deterministic = deterministic
        self.save = save
        callbacks = []
        checkpointer = ModelCheckpoint(filepath=path + "model_{epoch:04d}.h5", verbose=1, save_best_only=False,
                                       save_weights_only=False, period=100)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=patience_lr, min_lr=0.0001)
        early_stop = EarlyStopping(monitor='loss', patience=patience_es)
        tensor_board = TensorBoard(log_dir=path + "logs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S/"))
        callbacks.append(checkpointer)
        callbacks.append(reduce_lr)
        callbacks.append(early_stop)
        callbacks.append(tensor_board)
        self.callbacks = callbacks

    def build(self):
        # build encoder
        Relu = "relu"
        inputs = Input(shape=(self.input_size,), name='encoder_input')
        x = Dense(128, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                  bias_initializer='zeros', name='en_hidden_layer_x1')(inputs)
        x = BatchNormalization(center=True, scale=False)(x)
        x = Activation(Relu)(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(64, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                  name='en_hidden_layer_x2')(x)
        x = BatchNormalization(center=True, scale=False)(x)
        x = Activation(Relu)(x)
        x = Dropout(self.dropout_rate)(x)

        z_mean = Dense(32, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                       name="encoder_mean")(x)
        z_log_var = Dense(32, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                          name="encoder_log_var")(x)
        z = Lambda(sampling, output_shape=(32,), name='hidden_var_z')([z_mean, z_log_var])
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder_mlp')

        latent_inputs = Input(shape=(32,), name='z_sampling')
        x = Dense(64, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(latent_inputs)
        x = BatchNormalization(center=True, scale=False)(x)
        x = Activation(Relu)(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(128, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(x)
        x = BatchNormalization(center=True, scale=False)(x)
        x = Activation(Relu)(x)
        x = Dropout(self.dropout_rate)(x)
        if self.method == "count":
            outputs = Dense(self.input_size, kernel_regularizer=self.kernel_regularizer,
                            kernel_initializer=self.initializers, activation="linear")(x)
        elif self.method == "qqnorm":
            outputs = Dense(self.input_size, kernel_regularizer=self.kernel_regularizer,
                            kernel_initializer=self.initializers, activation="linear")(x)
        else:
            outputs = Dense(self.input_size, kernel_regularizer=self.kernel_regularizer,
                            kernel_initializer=self.initializers, activation="softplus")(x)
        decoder = Model(latent_inputs, outputs, name='decoder_mlp')
        if self.deterministic:
            outputs = decoder(encoder(inputs)[0])
        else:
            outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae_mlp')
        reconstruction_loss = mse(inputs, outputs)
        reconstruction_loss *= self.input_size
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        self.vae = vae
        self.encoder = encoder
        self.decoder = decoder

    def compile(self):
        self.vae.compile(optimizer=self.optimizer)
        self.vae.summary()

    def train(self, adata, batch_size=256, epochs=300):
        if os.path.isfile(self.path + "model.h5"):
            # self.vae=load_model(self.path+"model.h5")
            self.vae.load_weights(self.path + "model.h5")
        else:
            self.vae.fit(adata.X, epochs=epochs, batch_size=batch_size, callbacks=self.callbacks,
                         validation_split=self.validation_split, shuffle=True)

    def integrate(self, xadata, save=True, use_mean=True):
        [z_mean, z_log_var, z_batch] = self.encoder.predict(xadata.X)
        if use_mean:
            y_mean = self.decoder.predict(z_mean)
        else:
            y_mean = self.decoder.predict(z_batch)
        yadata = AnnData(X=y_mean, obs=xadata.obs, var=xadata.var)
        yadata.raw = AnnData(X=xadata.raw.X, var=xadata.raw.var)
        if save:
            yadata.write(self.path + "output.h5ad")
        yadata.obsm['X_vipcca'] = z_mean
        return yadata


class CVAE(VAE):
    def __init__(self, batches=16, batches2=8, **kwargs):
        super().__init__(**kwargs)
        self.batches = batches
        self.batches2 = batches2

    def build(self):
        Relu = "relu"
        inputs = Input(shape=(self.input_size,), name='encoder_input')
        inputs_batch = Input(shape=(self.batches,), name='batch_input1')
        y = Dense(self.batches, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(
            inputs_batch)
        y = BatchNormalization(center=True, scale=False)(y)
        y = Activation(Relu)(y)
        y = Dense(self.batches, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(y)
        y = BatchNormalization(center=True, scale=False)(y)
        y = Activation(Relu)(y)
        x = concatenate([inputs, y])

        en_ly_size = len(self.hidden_layers)
        for i in range(en_ly_size):
            if i == en_ly_size - 1:
                break
            ns = self.hidden_layers[i]
            x = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(x)
            x = BatchNormalization(center=True, scale=False)(x)
            x = Activation(Relu)(x)
            x = Dropout(self.dropout_rate)(x)
        ns = self.hidden_layers[i]
        z_mean = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                       name="encoder_mean")(x)
        z_log_var = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                          name="encoder_log_var")(x)
        z = Lambda(sampling, output_shape=(ns,), name='hidden_var_z')([z_mean, z_log_var])
        encoder = Model([inputs, inputs_batch], [z_mean, z_log_var, z], name='encoder_mlp')

        latent_inputs = Input(shape=(ns,), name='z_sampling')
        inputs_batch2 = Input(shape=(self.batches2,), name='batch_input2')
        y = Dense(self.batches2, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(
            inputs_batch2)
        y = BatchNormalization(center=True, scale=False)(y)
        y = Activation(Relu)(y)
        y = Dense(self.batches, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(y)
        y = BatchNormalization(center=True, scale=False)(y)
        y = Activation(Relu)(y)
        v = concatenate([latent_inputs, y])
        x = latent_inputs
        for i in range(en_ly_size - 1, 0, -1):
            ns = self.hidden_layers[i - 1]
            v = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(v)
            v = BatchNormalization(center=True, scale=False)(v)
            v = Activation(Relu)(v)
            v = Dropout(self.dropout_rate)(v)

            x = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(x)
            x = BatchNormalization(center=True, scale=False)(x)
            x = Activation(Relu)(x)
            x = Dropout(self.dropout_rate)(x)

        v = Dense(self.input_size, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                  activation=self.activation)(v)
        x = Dense(self.input_size, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                  activation="hard_sigmoid")(x)
        x = Add()([v, x])
        outputs = Activation(Relu)(x)
        decoder = Model([latent_inputs, inputs_batch2], outputs, name='decoder_mlp')

        if self.deterministic:
            outputs = decoder([encoder([inputs, inputs_batch])[0], inputs_batch2])
        else:
            outputs = decoder([encoder([inputs, inputs_batch])[2], inputs_batch2])
        vae = Model([inputs, inputs_batch, inputs_batch2], outputs, name='vae_mlp')
        #
        noise = tf.math.subtract(inputs, outputs)
        var = tf.math.reduce_variance(noise)
        reconstruction_loss = mse(inputs, outputs)
        reconstruction_loss *= (0.5 * self.input_size) / var
        reconstruction_loss += (0.5 * self.input_size) * tf.math.log(var)
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
        vae_loss = K.mean(2 * reconstruction_loss + self.lambda_regulizer * kl_loss)
        vae.add_loss(vae_loss)
        self.vae = vae
        self.encoder = encoder
        self.decoder = decoder

    def train(self, adata, batch_size=256, epochs=300, model_file=None):
        if model_file is not None:
            # self.vae=load_model(self.path+model_file)
            self.vae.load_weights(self.path + model_file)
        else:
            self.vae.fit([adata.X.A, adata.obsm['X_batch'], adata.obsm['X_batch2']], epochs=epochs, batch_size=batch_size,
                         callbacks=self.callbacks, validation_split=self.validation_split, shuffle=True)
            self.vae.save(self.path + "model.h5")

    def integrate(self, xadata, save=True, use_mean=True):
        [z_mean, z_log_var, z_batch] = self.encoder.predict([xadata.X.A, xadata.obsm['X_batch']])
        if use_mean:
            z_samples = z_mean
        else:
            z_samples = z_batch

        y_mean = self.decoder.predict([z_samples, xadata.obsm['X_batch2']])
        yadata = AnnData(X=y_mean, obs=xadata.obs, var=xadata.var)
        yadata.raw = xadata.copy()
        yadata.obsm['X_vipcca'] = z_mean

        i_mean = self.decoder.predict([z_samples, np.tile(xadata.obsm['X_batch2'][1], (xadata.shape[0], 1))])
        iadata = AnnData(X=i_mean, obs=xadata.obs, var=xadata.var)
        i_mean[i_mean < 0.1] = 0
        iadata.raw = AnnData(X=csr_matrix(i_mean), obs=xadata.obs, var=xadata.var)
        iadata.obsm['X_vipcca'] = z_mean

        if save:
            iadata.write(self.path + "output.h5ad")
        return yadata


class CVAE2(VAE):
    def __init__(self, batches=2, **kwargs):
        super().__init__(**kwargs)
        self.batches = batches

    def build(self):
        Relu = "relu"
        inputs = Input(shape=(self.input_size,), name='encoder_input')
        en_ly_size = len(self.hidden_layers)
        for i in range(en_ly_size):
            ns = self.hidden_layers[i]
            if i == 0:
                x = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(inputs)
            elif i == en_ly_size - 1:
                break
            else:
                x = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(x)
            x = BatchNormalization(center=True, scale=False)(x)
            x = Activation(Relu)(x)
            x = Dropout(self.dropout_rate)(x)
        ns = self.hidden_layers[i]
        z_mean = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                       name="encoder_mean")(x)
        z_log_var = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                          name="encoder_log_var")(x)
        z = Lambda(sampling, output_shape=(ns,), name='hidden_var_z')([z_mean, z_log_var])
        encoder = Model([inputs], [z_mean, z_log_var, z], name='encoder_mlp')

        latent_inputs = Input(shape=(ns,), name='z_sampling')
        inputs_batch = Input(shape=(self.batches,), name='batch_input')
        y = Dense(self.batches, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(
            inputs_batch)
        y = BatchNormalization(center=True, scale=False)(y)
        y = Activation(Relu)(y)
        y = Dense(self.batches, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(y)
        y = BatchNormalization(center=True, scale=False)(y)
        y = Activation(Relu)(y)
        v = concatenate([latent_inputs, y])
        x = latent_inputs
        for i in range(en_ly_size - 1, 0, -1):
            ns = self.hidden_layers[i - 1]
            v = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(v)
            v = BatchNormalization(center=True, scale=False)(v)
            v = Activation(Relu)(v)
            v = Dropout(self.dropout_rate)(v)

            x = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(x)
            x = BatchNormalization(center=True, scale=False)(x)
            x = Activation(Relu)(x)
            x = Dropout(self.dropout_rate)(x)

        v = Dense(self.input_size, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                  activation=self.activation)(v)
        x = Dense(self.input_size, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                  activation="hard_sigmoid")(x)
        x = Add()([v, x])
        outputs = Activation(Relu)(x)
        decoder = Model([latent_inputs, inputs_batch], outputs, name='decoder_mlp')

        if self.deterministic:
            outputs = decoder([encoder([inputs])[0], inputs_batch])
        else:
            outputs = decoder([encoder([inputs])[2], inputs_batch])
        vae = Model([inputs, inputs_batch], outputs, name='vae_mlp')

        noise = tf.math.subtract(inputs, outputs)
        var = tf.math.reduce_variance(noise)
        reconstruction_loss = mse(inputs, outputs)
        reconstruction_loss *= (0.5 * self.input_size) / var
        reconstruction_loss += (0.5 * self.input_size) * tf.math.log(var)
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
        vae_loss = K.mean(2 * reconstruction_loss + self.lambda_regulizer * kl_loss)
        vae.add_loss(vae_loss)
        self.vae = vae
        self.encoder = encoder
        self.decoder = decoder

    def train(self, adata, batch_size=256, epochs=300, model_file=None):
        if model_file is not None:
            # self.vae=load_model(self.path+model_file)
            self.vae.load_weights(self.path + model_file)
        else:
            self.vae.fit([adata.X, adata.obsm['X_batch']], epochs=epochs, batch_size=batch_size,
                         callbacks=self.callbacks, validation_split=self.validation_split, shuffle=True)
            self.vae.save(self.path + "model.h5")

    def integrate(self, xadata, use_mean=True, return_corrected_y=True):
        [z_mean, z_log_var, z_batch] = self.encoder.predict([xadata.X])
        if use_mean:
            z_samples = z_mean
        else:
            z_samples = z_batch

        if return_corrected_y:
            i_mean = self.decoder.predict([z_samples, np.tile(xadata.obsm['X_batch'][1], (xadata.shape[0], 1))])
            iadata = AnnData(X=i_mean, obs=xadata.obs, var=xadata.var)
            i_mean[i_mean < 0.1] = 0
            iadata.raw = AnnData(X=csr_matrix(i_mean), obs=xadata.obs, var=xadata.var)
            iadata.obsm['X_vipcca'] = z_mean
            iadata.write(self.path + "output.h5ad")
        else:
            i_mean = self.decoder.predict([z_samples, xadata.obsm['X_batch']])
            iadata = AnnData(X=i_mean, obs=xadata.obs, var=xadata.var)
            i_mean[i_mean < 0.1] = 0
            iadata.raw = AnnData(X=csr_matrix(i_mean), obs=xadata.obs, var=xadata.var)
            iadata.obsm['X_vipcca'] = z_mean
            iadata.write(self.path + "output_dsy.h5ad")
        return iadata


class CVAE3(VAE):
    def __init__(self, batches=2, **kwargs):
        super().__init__(**kwargs)
        self.batches = batches

    def build(self):
        Relu = "relu"
        inputs = Input(shape=(self.input_size,), name='encoder_input')
        x = Dense(128, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                  bias_initializer='zeros', name='en_hidden_layer_x1')(inputs)
        x = BatchNormalization(center=True, scale=False)(x)
        x = Activation(Relu)(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(64, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                  name='en_hidden_layer_x2')(x)
        x = BatchNormalization(center=True, scale=False)(x)
        x = Activation(Relu)(x)
        x = Dropout(self.dropout_rate)(x)

        z_mean = Dense(32, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                       name="encoder_mean")(x)
        z_log_var = Dense(32, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                          name="encoder_log_var")(x)
        z = Lambda(sampling, output_shape=(32,), name='hidden_var_z')([z_mean, z_log_var])
        encoder = Model([inputs], [z_mean, z_log_var, z], name='encoder_mlp')

        latent_inputs = Input(shape=(32,), name='z_sampling')
        inputs_batch = Input(shape=(self.batches,), name='batch_input')
        x = Dense(self.batches, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(
            inputs_batch)
        x = BatchNormalization(center=True, scale=False)(x)
        x = Activation(Relu)(x)
        y = Dense(32, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(latent_inputs)
        y = BatchNormalization(center=True, scale=False)(y)
        y = Activation(Relu)(y)
        xy = concatenate([y, x])
        z = Dense(32, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(xy)
        z = BatchNormalization(center=True, scale=False)(z)
        z = Activation(Relu)(z)
        z_mean_discriminator = Dense(32, kernel_regularizer=self.kernel_regularizer,
                                     kernel_initializer=self.initializers)(z)

        translator = Model([latent_inputs, inputs_batch], z_mean_discriminator, name="translator_mlp")

        z_prime_input = Input(shape=(32,), name='discriminator')
        x = Dense(64, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(z_prime_input)
        x = BatchNormalization(center=True, scale=False)(x)
        x = Activation(Relu)(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(128, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(x)
        x = BatchNormalization(center=True, scale=False)(x)
        x = Activation(Relu)(x)
        x = Dropout(self.dropout_rate)(x)

        if self.method == "count":
            outputs = Dense(self.input_size, kernel_regularizer=self.kernel_regularizer,
                            kernel_initializer=self.initializers, activation="linear")(x)
        elif self.method == "qqnorm":
            outputs = Dense(self.input_size, kernel_regularizer=self.kernel_regularizer,
                            kernel_initializer=self.initializers, activation="linear")(x)
        else:
            outputs = Dense(self.input_size, kernel_regularizer=self.kernel_regularizer,
                            kernel_initializer=self.initializers, activation=self.activation)(x)
        decoder = Model(z_prime_input, outputs, name='decoder_mlp')

        if self.deterministic:
            outputs = decoder(translator([encoder([inputs])[0], inputs_batch]))
        else:
            outputs = decoder(translator([encoder([inputs])[2], inputs_batch]))
        vae = Model([inputs, inputs_batch], outputs, name='cvae_mlp')
        reconstruction_loss = mse(inputs, outputs)
        reconstruction_loss *= self.input_size
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
        vae_loss = K.mean(reconstruction_loss + self.lambda_regulizer * kl_loss)
        vae.add_loss(vae_loss)
        self.vae = vae
        self.encoder = encoder
        self.decoder = decoder
        self.translator = translator

    def train(self, adata, batch_size=256, epochs=300, model_file=None):
        if model_file is not None:
            if os.path.exists(self.path + model_file):
                self.vae.load_weights(self.path + model_file)
            else:
                raise ValueError('The specified model file does not exist.')
        else:
            self.vae.fit([adata.X, adata.obsm['X_batch']], epochs=epochs, batch_size=batch_size,
                         callbacks=self.callbacks, validation_split=self.validation_split, shuffle=True)
            self.vae.save(self.path + "model.h5")

    def integrate(self, xadata, save=True, use_mean=True):
        [z_mean, z_log_var, z_batch] = self.encoder.predict([xadata.X])
        if use_mean:
            z_samples = z_mean
        else:
            z_samples = z_batch
        z_mean_prime = self.translator.predict([z_samples, xadata.obsm['X_batch']])

        y_mean = self.decoder.predict([z_mean_prime])
        yadata = AnnData(X=y_mean, obs=xadata.obs, var=xadata.var)
        yadata.raw = xadata.copy()
        yadata.obsm['X_vipcca'] = z_mean
        yadata.obsm['X_vipcca'] = z_mean_prime
        p_mean = self.decoder.predict([z_mean_prime])
        padata = AnnData(X=y_mean, obs=xadata.obs, var=xadata.var)
        padata.raw = xadata.copy()
        padata.obsm['X_vipcca'] = z_mean
        padata.obsm['X_vipcca'] = z_mean_prime
        if save:
            yadata.write(self.path + "output.h5ad")
            padata.write(self.path + "integrated.h5ad")
        return yadata


class Conf(object):
    pass


class VIPCCA(object):
    """
    Initialize VIPCCA object

    Parameters

    ----------

    patience_es: int, optional (default: 50)
        number of epochs with no improvement after which training will be stopped.

    patience_lr: int, optional (default: 25)
        number of epochs with no improvement after which learning rate will be reduced.

    epochs: int, optional (default: 1000)
        Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.

    res_path: string, (default: None)
        Folder path to save model training results model.h5 and output data adata.h5ad.

    split_by: string, optional (default: '_batch')
        the obsm_name of obsm used to distinguish different batches.

    method: string, optional (default: 'lognorm')
        the normalization method for input data, one of {"qqnorm","count", other}.

    batch_input_size: int, optional (default: 128)
        the length of the batch vector that concatenate with the input layer.

    batch_input_size2: int, optional (default: 16)
        the length of the batch vector that concatenate with the latent layer.

    activation: string, optional (default: "softplus")
        the activation function of hidden layers.

    dropout_rate: double, optional (default: 0.01)
        the dropout rate of hidden layers.

    hidden_layers: list, optional (default: [128,64,32,16])
        Number of hidden layer neurons in the model

    lambda_regulizer: double, optional (default: 5.0)
        The coefficient multiplied by KL_loss

    initializer: string, optional (default: "glorot_uniform")
        Regularizer function applied to the kernel weights matrix.

    l1_l2: tuple, optional (default: (0.0, 0.0))
        [L1 regularization factor, L2 regularization factor].

    mode: string,  optional (default: 'CVAE')
        one of {"CVAE", "CVAE2", "CVAE3"}

    model_file: string, optional (default: None)
        The file name of the trained model, the default is None

    save: bool, optional (default: True)
        If true, save output adata file.

    """

    def __init__(self,
                 adata_all=None,
                 patience_es=50,
                 patience_lr=25,
                 epochs=1000,
                 res_path=None,
                 split_by="_batch",
                 method="lognorm",
                 hvg=True,
                 batch_input_size=128,
                 batch_input_size2=16,
                 activation="softplus",
                 dropout_rate=0.01,
                 hidden_layers=[128, 64, 32, 16],
                 lambda_regulizer=5.0,
                 initializer="glorot_uniform",
                 l1_l2=(0.0, 0.0),
                 mode="CVAE",
                 model_file=None,
                 save=True):
        self.conf = Conf()
        self.conf.adata_all = adata_all
        self.conf.res_path = res_path
        self.conf.patience_es = patience_es
        self.conf.patience_lr = patience_lr
        self.conf.epochs = epochs
        self.conf.split_by = split_by
        self.conf.method = method
        self.conf.hvg = hvg
        self.conf.batch_input_size = batch_input_size
        self.conf.batch_input_size2 = batch_input_size2
        self.conf.dropout_rate = dropout_rate
        self.conf.hidden_layers = hidden_layers
        self.conf.lambda_regulizer = lambda_regulizer
        self.conf.initializer = initializer
        self.conf.activation = activation
        self.conf.model_file = model_file
        self.conf.l1_l2 = l1_l2
        self.conf.mode = mode
        self.conf.save = save
        self.vipcca_preprocessing()

    def vipcca_preprocessing(self):
        """
        Generate the required random batch id for the VIPCCA model

        """
        batch_int = self.conf.adata_all.obs[self.conf.split_by].astype("category").cat.codes.values
        np.random.seed(2019)
        batch_dic = np.random.randint(10, size=(np.max(batch_int) + 1, self.conf.batch_input_size))
        X_batch = np.zeros((len(batch_int), self.conf.batch_input_size))
        batch_dic2 = np.random.randint(10, size=(np.max(batch_int) + 1, self.conf.batch_input_size2))
        X_batch2 = np.zeros((len(batch_int), self.conf.batch_input_size2))
        for i in range(len(batch_int)):
            X_batch[i, :] = batch_dic[batch_int[i], :]
            X_batch2[i, :] = batch_dic2[batch_int[i], :]
        self.conf.adata_all.obsm["X_batch"] = X_batch
        self.conf.adata_all.obsm["X_batch2"] = X_batch2


    def build(self):
        """
        build VIPCCA model
        """
        if self.conf.mode == "CVAE":
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
        elif self.conf.mode == "CVAE2":
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
        self.conf.net = net

    def fit_integrate(self):
        """
        Train the constructed VIPCCA model, integrate the data with the trained model,
        and return the integrated anndata object

        Returns
        -------
        :class:`~anndata.AnnData`
		    adata produced by function self.conf.net.integrate(self.conf.adata_all, save=self.conf.save)
        """
        self.build()
        self.conf.net.train(self.conf.adata_all, epochs=self.conf.epochs, model_file=self.conf.model_file)
        return self.conf.net.integrate(self.conf.adata_all, save=self.conf.save)

    # def plotEmbedding(self, eps=[], group_by="Batch", min_dist=0.5):
    #     self.build()
    #     for epoch in eps:
    #         test_result_path = self.conf.res_path + "e%04d_" % epoch
    #         self.conf.net.vae.load_weights(self.conf.res_path + "weights%04d.h5" % epoch)
    #         adata_transform = self.conf.net.integrate(self.conf.adata_all, save=False)
    #         plotCorrelation(adata_transform.raw.X, adata_transform.X, result_path=test_result_path)
    #         run_embedding(adata_transform, path=test_result_path, method="umap", min_dist=min_dist)
    #         plotEmbedding(adata_transform, path=test_result_path, method='umap', group_by=group_by,
    #                       legend_loc="right margin")

#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--epochs', type=int, default=50)
#     parser.add_argument('--batch_size', type=int, default=100)
#     conf = parser.parse_args()

