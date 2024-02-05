import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras import optimizers, regularizers, initializers
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Dropout, Lambda, Concatenate, concatenate
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import os.path

import matplotlib.pyplot as plt
import numpy as np


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e5)


class VIMCCA:
    def __init__(self, input_size_x, inputs_size_y,
                 hidden_layers,
                 path="./",
                 dropout_rate_big=0.05,
                 dropout_rate_small=0.0,
                 validation_split=0.0,
                 patience=60,
                 weight = 5,
                 deterministic=False):
        self.input_size_x = input_size_x
        self.input_size_y = inputs_size_y
        self.hidden_layers = hidden_layers
        self.dropout_rate_big = dropout_rate_big
        self.dropout_rate_small = dropout_rate_small
        self.weight = weight
        self.VIMCCA = None
        self.inputs = None
        self.outputs = None
        self.path = path
        self.initializers = "glorot_uniform"
        self.optimizer = optimizers.Adam(beta_1=0.9, beta_2=0.999, epsilon=10e-8)
        self.kernel_regularizer = regularizers.l1_l2(l1=0.01, l2=0.01)
        self.validation_split = validation_split
        self.deterministic = deterministic
        callbacks = []
        checkpointer = ModelCheckpoint(filepath=path + "vae_weights.h5", verbose=1, save_best_only=False,
                                       save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=10, min_lr=0.0001)
        early_stop = EarlyStopping(monitor='loss', patience=patience)
        tensor_board = TensorBoard(log_dir=path + 'logs/')
        callbacks.append(checkpointer)
        callbacks.append(reduce_lr)
        callbacks.append(early_stop)
        callbacks.append(tensor_board)
        self.callbacks = callbacks

    def build(self):
        en_ly_size = len(self.hidden_layers)
        Relu = "relu"
        inputs_x = Input(shape=(self.input_size_x,), name='inputs_x')
        inputs_y = Input(shape=(self.input_size_y,), name='inputs_y')
        z = inputs_x
        for i in range(en_ly_size):
            if i == en_ly_size - 1:
                break
            ns = self.hidden_layers[i]
            z = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(z)
            z = BatchNormalization(center=True, scale=False)(z)
            z = Activation(Relu)(z)
            z = Dropout(self.dropout_rate_small)(z)
        latent_z_size = self.hidden_layers[i]

        z_mean = Dense(latent_z_size, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                        bias_initializer='zeros', name="z_mean")(z)
        z_log_var = Dense(latent_z_size, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                           bias_initializer='zeros', name="z_log_var")(z)
        z = Lambda(sampling, output_shape=(latent_z_size,), name='z')([z_mean, z_log_var])
        encoder_z = Model(inputs_x, [z_mean, z_log_var, z], name='encoder_z')

        latent_inputs_x = Input(shape=(latent_z_size,), name='x')
        latent_inputs_y = Input(shape=(latent_z_size,), name='y')
        x = latent_inputs_x
        y = latent_inputs_y
        for i in range(en_ly_size - 1, 0, -1):
            ns = self.hidden_layers[i - 1]
            x = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(x)
            x = BatchNormalization(center=True, scale=False)(x)
            x = Activation(Relu)(x)
            x = Dropout(self.dropout_rate_big)(x)

            y = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(y)
            y = BatchNormalization(center=True, scale=False)(y)
            y = Activation(Relu)(y)
            y = Dropout(self.dropout_rate_big)(y)

        outputs_x = Dense(self.input_size_x, kernel_regularizer=self.kernel_regularizer,
                        kernel_initializer=self.initializers, activation="linear")(x)
        decoder_x = Model(latent_inputs_x, outputs_x, name='decoder_x')

        outputs_y = Dense(self.input_size_y, kernel_regularizer=self.kernel_regularizer,
                            kernel_initializer=self.initializers, activation="linear", name='outputs_y')(y)
        decoder_y = Model(latent_inputs_y, outputs_y, name='decoder_y')

        outputs_x = decoder_x(encoder_z(inputs_x)[2])
        outputs_y = decoder_y(encoder_z(inputs_x)[2])
        VIMCCA = Model([inputs_x, inputs_y], [outputs_x, outputs_y], name='VIMCCA_mlp')

        reconstruction_loss_x = mse(inputs_x, outputs_x)
        reconstruction_loss_y = mse(inputs_y, outputs_y)
        # reconstruction_loss_x = K.sum(binary_crossentropy(inputs_x, outputs_x))
        # reconstruction_loss_y = K.sum(binary_crossentropy(inputs_y, outputs_y))
        # reconstruction_loss_x*=self.input_size_x
        # reconstruction_loss_y*=self.input_size_y
        noise_x = tf.math.subtract(inputs_x, outputs_x)
        var_x = tf.math.reduce_variance(noise_x)
        reconstruction_loss_x *= (0.5 * self.input_size_x) / var_x
        reconstruction_loss_x += (0.5 * self.input_size_x) / var_x * tf.math.log(var_x)
        #
        noise_y = tf.math.subtract(inputs_y, outputs_y)
        var_y = tf.math.reduce_variance(noise_y)
        reconstruction_loss_y *= (0.5 * self.input_size_y) / var_y
        reconstruction_loss_y += (0.5 * self.input_size_y) / var_y * tf.math.log(var_y)
        # reconstruction_loss_y*=5
        # reconstruction_loss_x /= 2000
        # reconstruction_loss_y
        # reconstruction_loss_x = mse(inputs_x, outputs_x)*2000
        # reconstruction_loss_y = mse(inputs_y, outputs_y)*2000
        kl_loss_z = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        # reconstruction_loss_x = tf.reduce_mean(reconstruction_loss_x)
        # reconstruction_loss_y = tf.reduce_mean(reconstruction_loss_y)
        # kl_loss_z/=self.hidden_layers[-1]

        vae_loss = K.mean(reconstruction_loss_x + self.weight*reconstruction_loss_y + kl_loss_z)

        VIMCCA.add_loss(vae_loss)
        self.VIMCCA = VIMCCA
        self.encoder_z = encoder_z
        self.decoder_x = decoder_x
        self.decoder_y = decoder_y

    def compile(self):
        self.VIMCCA.compile(optimizer=self.optimizer)
        self.VIMCCA.summary()

    def train(self, x, y, batch_size=1, epochs=300):
        # if os.path.isfile(self.path + "vae_weights.h5"):
        #     self.vae.load_weights(self.path + "vae_weights.h5")
        # else:
        #
        history = self.VIMCCA.fit({'inputs_x': x, 'inputs_y': y}, epochs=epochs, batch_size=batch_size,
                         validation_split=self.validation_split, shuffle=True)

        return history

    def integrate(self, x):
        [z_mean, z_log_var, z] = self.encoder_z.predict(x)
        return z_mean

    def integrate_compose(self, x):
        [z_mean, z_log_var, z] = self.encoder_z.predict(x)
        return z_mean

    def get_output(self, x):
        [z_mean, z_log_var, z] = self.encoder_z.predict(x)
        output_x = self.decoder_x.predict(z_mean)
        output_y = self.decoder_y.predict(z_mean)
        return output_x, output_y

    def plot_loss(self, history):
        print(len(history.history['loss']))
        plt.plot(history.history['loss'])
        plt.show()


# class PVIMCCA:
#     def __init__(self, input_size_x, inputs_size_y,
#                  hidden_layers=[128,64,32,5],
#                  dropout_rate_small=0.01,
#                  dropout_rate_big=0.05,
#                  path="./",
#                  latent_xy_size=2,
#                  batch_num=2,
#                  validation_split=0.0,
#                  patience=60,
#                  deterministic=False):
#         self.input_size_x = input_size_x
#         self.input_size_y = inputs_size_y
#         self.VIMCCA = None
#         self.inputs = None
#         self.outputs = None
#         self.path = path
#         self.initializers = "glorot_normal"
#         self.optimizer = optimizers.Adam(beta_1=0.9, beta_2=0.999, epsilon=10e-8)
#         self.hidden_layers = hidden_layers
#         self.latent_xy_size = latent_xy_size
#         self.dropout_rate_small = dropout_rate_small
#         self.dropout_rate_big = dropout_rate_big
#         self.kernel_regularizer = regularizers.l1_l2(l1=0.01, l2=0.01)
#         self.validation_split = validation_split
#         self.deterministic = deterministic
#
#         self.batch_num = batch_num
#         callbacks = []
#         checkpointer = ModelCheckpoint(filepath=path + "vae_weights.h5", verbose=1, save_best_only=False,
#                                        save_weights_only=True)
#         reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=10, min_lr=0.0001)
#         early_stop = EarlyStopping(monitor='loss', patience=patience)
#         tensor_board = TensorBoard(log_dir=path + 'logs/')
#         callbacks.append(checkpointer)
#         callbacks.append(reduce_lr)
#         callbacks.append(early_stop)
#         callbacks.append(tensor_board)
#         self.callbacks = callbacks
#
#     def build(self):
#         en_ly_size = len(self.hidden_layers)
#         Relu = "relu"
#         inputs_x = Input(shape=(self.input_size_x,), name='inputs_x')
#         inputs_y = Input(shape=(self.input_size_y,), name='inputs_y')
#
#         hx = inputs_x
#         hy = inputs_y
#         for i in range(en_ly_size):
#             if i == en_ly_size - 1:
#                 break
#             ns = self.hidden_layers[i]
#             hx = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(hx)
#             hx = BatchNormalization(center=True, scale=False)(hx)
#             hx = Activation(Relu)(hx)
#             hx = Dropout(self.dropout_rate_small)(hx)
#
#             hy = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(hy)
#             hy = BatchNormalization(center=True, scale=False)(hy)
#             hy = Activation(Relu)(hy)
#             hy = Dropout(self.dropout_rate_small)(hy)
#
#         hx_mean = Dense(self.latent_xy_size, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
#                        name="hx_mean")(hx)
#         hx_log_var = Dense(self.latent_xy_size, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
#                           name="hx_log_var")(hx)
#         hx_z = Lambda(sampling, output_shape=(self.latent_xy_size,), name='hx_z')([hx_mean, hx_log_var])
#         encoder_hx = Model(inputs_x, [hx_mean, hx_log_var, hx_z], name='encoder_hx')
#
#         hy_mean = Dense(self.latent_xy_size, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
#                         name="hy_mean")(hy)
#         hy_log_var = Dense(self.latent_xy_size, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
#                            name="hy_log_var")(hy)
#         hy_z = Lambda(sampling, output_shape=(self.latent_xy_size,), name='hy_z')([hy_mean, hy_log_var])
#         encoder_hy = Model(inputs_y, [hy_mean, hy_log_var, hy_z], name='encoder_hy')
#
#         z = inputs_x
#         for i in range(en_ly_size):
#             if i == en_ly_size - 1:
#                 break
#             ns = self.hidden_layers[i]
#             z = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(z)
#             z = BatchNormalization(center=True, scale=False)(z)
#             z = Activation(Relu)(z)
#             z = Dropout(self.dropout_rate_small)(z)
#         latent_z_size = self.hidden_layers[i]
#
#         z_mean = Dense(latent_z_size, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
#                         name="z_mean")(z)
#         z_log_var = Dense(latent_z_size, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
#                            name="z_log_var")(z)
#         z = Lambda(sampling, output_shape=(latent_z_size,), name='z')([z_mean, z_log_var])
#         encoder_z = Model(inputs_x, [z_mean, z_log_var, z], name='encoder_z')
#
#         latent_inputs_x = Input(shape=(self.latent_xy_size,), name='x')
#         latent_inputs_xz = Input(shape=(latent_z_size,), name='z_x')
#         latent_x = concatenate([latent_inputs_x, latent_inputs_xz], axis=1)
#         # x = Dense(self.latent_xy_size, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
#         #           activation="linear")(latent_inputs_x)
#         #
#         # lz = Dense(self.latent_z_size, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
#         #           activation="linear")(latent_inputs_xz)
#         #
#         # x = concatenate([x, lz], axis=1)
#         x = latent_x
#         for i in range(en_ly_size - 1, 0, -1):
#             ns = self.hidden_layers[i - 1]
#             x = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(x)
#             x = BatchNormalization(center=True, scale=False)(x)
#             x = Activation(Relu)(x)
#             x = Dropout(self.dropout_rate_big)(x)
#
#         outputs_x = Dense(self.input_size_x, kernel_regularizer=self.kernel_regularizer,
#                         kernel_initializer=self.initializers, activation="softplus")(x)
#         decoder_x = Model([latent_inputs_x, latent_inputs_xz], outputs_x, name='decoder_x')
#
#         latent_inputs_y = Input(shape=(self.latent_xy_size,), name='y')
#         latent_inputs_yz = Input(shape=(latent_z_size,), name='z_y')
#         latent_y = concatenate([latent_inputs_y, latent_inputs_yz], axis=1)
#         y = latent_y
#         for i in range(en_ly_size - 1, 0, -1):
#             ns = self.hidden_layers[i - 1]
#             y = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(y)
#             y = BatchNormalization(center=True, scale=False)(y)
#             y = Activation(Relu)(y)
#             y = Dropout(self.dropout_rate_big)(y)
#
#         outputs_y = Dense(self.input_size_y, kernel_regularizer=self.kernel_regularizer,
#                             kernel_initializer=self.initializers, activation="softplus", name='outputs_y')(y)
#         decoder_y = Model([latent_inputs_y, latent_inputs_yz], outputs_y, name='decoder_y')
#         # decoder_y = Model(latent_inputs, outputs_y, name='decoder_y')
#         # decoder_x = Model(latent_inputs, outputs_x, name='decoder_x')
#
#         # outputs_x = decoder_x(encoder(inputs_x)[0])
#         # outputs_y = decoder_y(encoder(inputs_x)[0])
#         outputs_x = decoder_x([encoder_hx(inputs_x)[2], encoder_z(inputs_x)[2]])
#         outputs_y = decoder_y([encoder_hy(inputs_y)[2], encoder_z(inputs_x)[2]])
#         VIMCCA = Model([inputs_x, inputs_y], [outputs_x, outputs_y], name='vae_mlp')
#         reconstruction_loss_x = K.sum(mse(inputs_x, outputs_x), axis=-1)*10
#         reconstruction_loss_y = K.sum(mse(inputs_y, outputs_y), axis=-1)*2000
#         # reconstruction_loss_x = mse(inputs_x, outputs_x) * 2000
#         # reconstruction_loss_y = mse(inputs_y, outputs_y)*2000
#         kl_loss_z = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
#         kl_loss_hx = -0.5 * K.sum(1 + hx_log_var - K.square(hx_mean) - K.exp(hx_log_var), axis=-1)
#         kl_loss_hy = -0.5 * K.sum(1 + hy_log_var - K.square(hy_mean) - K.exp(hy_log_var), axis=-1)
#
#         vae_loss = K.mean(reconstruction_loss_x + reconstruction_loss_y + kl_loss_z + kl_loss_hx +
#                           kl_loss_hy)
#
#         VIMCCA.add_loss(vae_loss)
#         self.VIMCCA = VIMCCA
#         self.encoder_z = encoder_z
#         self.encoder_hx = encoder_hx
#         self.encoder_hy = encoder_hy
#         self.decoder_x = decoder_x
#         self.decoder_y = decoder_y
#
#     def compile(self):
#         self.VIMCCA.compile(optimizer=self.optimizer)
#         self.VIMCCA.summary()
#
#     def train(self, x, y, batch_size=100, epochs=300):
#         # if os.path.isfile(self.path + "vae_weights.h5"):
#         #     self.vae.load_weights(self.path + "vae_weights.h5")
#         # else:
#         #
#         history = self.VIMCCA.fit({'inputs_x': x, 'inputs_y': y}, epochs=epochs, batch_size=batch_size,
#                          validation_split=self.validation_split, shuffle=True)
#             # self.vae.model.save(self.path + "/model_best.h5")
#         return history
#
#     def integrate(self, x, save=True, use_mean=True):
#         [z_mean, z_log_var, z] = self.encoder_z.predict(x)
#         return z_mean
#
#     def integrate_compose(self, x, y, save=True, use_mean=True):
#         [z_mean, z_log_var, z] = self.encoder_z.predict(x)
#         [hx_mean, hx_log_var, hx_z] = self.encoder_hx.predict(x)
#         [hy_mean, hy_log_var, hy_z] = self.encoder_hy.predict(y)
#         # res = concatenate([z_mean, hx_mean, hy_mean],axis=1)
#         # res = np.array(res)
#         return z_mean
#
#     def get_output(self, x, y):
#         [z_mean, z_log_var, z] = self.encoder_z.predict(x)
#         [hx_mean, hx_log_var, hx_z] = self.encoder_hx.predict(x)
#         [hy_mean, hy_log_var, hy_z] = self.encoder_hy.predict(y)
#         output_x = self.decoder_x.predict([hx_z, z])
#         output_y = self.decoder_y.predict([hy_z, z])
#         return output_x, output_y
#
#     def plot_loss(self, history):
#         print(len(history.history['loss']))
#         plt.plot(history.history['loss'])
#         plt.show()


def fit_integration(adata_x, adata_y, hidden_layers=[128,64,32,5], epochs=30,weight=5,
                    sparse_x=False, sparse_y=False, batch_size=128):
    """/
    Build VIMCCA model and fit the data to the model for training.

    Parameters
    ----------
    adata_x: AnnData
        AnnData object for one of the modals.

    adata_y: AnnData
        AnnData object for another of the modals.

    epochs: int, optional (default: 200)
        Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.

    batch_size: int or None, optional (default: 128)
        Number of samples per gradient update. If unspecified, batch_size will default to 128.

    weight: double, optional (default: 5.0)
        The weights of the reconstruction loss for the second modality data.

    sparse_x: bool, optional (default: False)
        If True, Matrix X in the AnnData object is stored as a sparse matrix.

    sparse_y: bool, optional (default: False)
        If True, Matrix Y in the AnnData object is stored as a sparse matrix.

    hidden_layers: list of integers, (default: [128,64,32,5])
        Number of hidden layer neurons in the model.

    Returns
    ----------
        :class:`~Numpy array(s)`
            z
    """

    if sparse_x:
        x = adata_x.X.A
    else:
        x = adata_x.X
    if sparse_y:
        y = adata_y.X.A
    else:
        y = adata_y.X
    net = VIMCCA(input_size_x=x.shape[1], inputs_size_y=y.shape[1], hidden_layers=hidden_layers,weight=weight)
    net.build()
    net.compile()
    his = net.train(x, y, epochs=epochs, batch_size=128)
    z = net.integrate(x)
    return z