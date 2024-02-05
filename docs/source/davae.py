import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Dropout, Lambda, Concatenate, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
import numpy as np
from tensorflow.python.keras.layers import Layer
import anndata
import scbean.tools.utils as tl


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

#
# def reverse_gradient(X, hp_lambda):
#     '''Flips the sign of the incoming gradient during training.'''
#     try:
#         reverse_gradient.num_calls += 1
#     except AttributeError:
#         reverse_gradient.num_calls = 1
#
#     grad_name = "GradientReversal%d" % reverse_gradient.num_calls
#
#     @tf.RegisterGradient(grad_name)
#     def _flip_gradients(op, grad):
#         return [tf.negative(grad) * hp_lambda]
#
#     g = tf.compat.v1.get_default_graph()
#     with g.gradient_override_map({'Identity': grad_name}):
#         y = tf.identity(X)
#
#     return y
#
#
# class GradientReversal(Layer):
#     '''Flip the sign of gradient during training.'''
#     def __init__(self, hp_lambda, **kwargs):
#         super(GradientReversal, self).__init__(**kwargs)
#         self.supports_masking = False
#         self.hp_lambda = hp_lambda
#
#     def build(self, input_shape):
#         self._trainable_weights = []
#
#     def call(self, x, mask=None):
#         return reverse_gradient(x, self.hp_lambda)
#
#     def get_output_shape_for(self, input_shape):
#         return input_shape
#
#     def get_config(self):
#         config = {'hp_lambda': self.hp_lambda}
#         base_config = super(GradientReversal, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy
    return y, custom_grad


class GradReverse(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return grad_reverse(x)


class DAVAE:
    def __init__(self, input_size, batches=2, domain_scale_factor=1.0, 
                 hidden_layers=[128, 64, 32, 5], path=''):
        self.input_size = input_size
        self.path = path
        self.dann_vae = None
        self.inputs = None
        self.outputs_x = None
        self.initializers = "glorot_uniform"
        self.optimizer = optimizers.Adam(lr=0.01)
        self.hidden_layers = hidden_layers
        self.domain_scale_factor = domain_scale_factor
        self.dropout_rate_small = 0.01
        self.dropout_rate_big = 0.05
        self.kernel_regularizer = regularizers.l1_l2(l1=0.00, l2=0.00)
        self.validation_split = 0.0
        self.batches = batches
        self.dropout_rate = 0.01
        callbacks = []
        checkpointer = ModelCheckpoint(filepath=path + "vae_weights.h5", verbose=1, save_best_only=False,
                                       save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=100, min_lr=0.0001)
        early_stop = EarlyStopping(monitor='loss', patience=200)
        tensor_board = TensorBoard(log_dir=path + 'logs/')
        callbacks.append(checkpointer)
        callbacks.append(reduce_lr)
        callbacks.append(early_stop)
        callbacks.append(tensor_board)
        self.callbacks = callbacks

    def build(self):
        Relu = "relu"
        en_ly_size = len(self.hidden_layers)
        z_size = self.hidden_layers[en_ly_size - 1]

        inputs_x = Input(shape=(self.input_size,), name='inputs')
        inputs_batch = Input(shape=(self.batches,), name='inputs_batch')
        inputs_loss_weights = Input(shape=(1,), name='inputs_weights')
        x = inputs_x
        for i in range(en_ly_size):
            if i == en_ly_size - 1:
                break
            ns = self.hidden_layers[i]
            x = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(x)
            x = BatchNormalization(center=True, scale=False)(x)
            x = Activation(Relu)(x)
            x = Dropout(self.dropout_rate)(x)

        hx_mean = Dense(z_size, kernel_regularizer=self.kernel_regularizer,
                        kernel_initializer=self.initializers,
                        name="hx_mean")(x)
        hx_log_var = Dense(z_size, kernel_regularizer=self.kernel_regularizer,
                           kernel_initializer=self.initializers,
                           name="hx_log_var")(x)
        hx_z = Lambda(sampling, output_shape=(z_size,), name='hx_z')([hx_mean, hx_log_var])
        encoder_hx = Model(inputs_x, [hx_mean, hx_log_var, hx_z], name='encoder_hx')

        latent_inputs_x = Input(shape=(z_size,), name='latent')
        x = latent_inputs_x
        for i in range(en_ly_size - 1, 0, -1):
            ns = self.hidden_layers[i - 1]
            x = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(x)
            x = BatchNormalization(center=True, scale=False)(x)
            x = Activation(Relu)(x)
            x = Dropout(self.dropout_rate_big)(x)

        outputs_x = Dense(self.input_size, kernel_regularizer=self.kernel_regularizer,
                          kernel_initializer=self.initializers, activation="softplus")(x)
        decoder_x = Model(latent_inputs_x, outputs_x, name='decoder_x')

        latent_inputs_batch = Input(shape=(z_size,), name='latent_domain')
        Flip = GradReverse()
        d = Flip(latent_inputs_batch)
        d = Dense(16, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(d)
        d = BatchNormalization(center=True, scale=False)(d)
        d = Activation(Relu)(d)
        d = Dropout(self.dropout_rate_big)(d)

        d = Dense(self.batches, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                  activation="softmax")(d)
        domian_classifier = Model(latent_inputs_batch, d, name='domain_classifier')

        outputs_x = decoder_x(encoder_hx(inputs_x)[2])
        domain_pred = domian_classifier(encoder_hx(inputs_x)[2])

        dann_vae = Model([inputs_x, inputs_batch, inputs_loss_weights], [outputs_x, domain_pred], name='vae_mlp')

        inputs_x = tf.multiply(inputs_x, inputs_loss_weights)
        outputs_x = tf.multiply(outputs_x, inputs_loss_weights)

        reconstruction_loss = mse(inputs_x, outputs_x)
        # reconstruction_loss = mse(inputs_x, outputs_x)

        noise = tf.math.subtract(inputs_x, outputs_x)
        var = tf.math.reduce_variance(noise)
        reconstruction_loss *= (0.5*self.input_size)/var
        reconstruction_loss += (0.5*self.input_size)/var*tf.math.log(var)

        kl_loss_z = -0.5 * K.sum(1 + hx_log_var - K.square(hx_mean) - K.exp(hx_log_var), axis=-1)

        pred_loss = K.categorical_crossentropy(inputs_batch, domain_pred)*self.input_size*self.domain_scale_factor
        vae_loss = K.mean(reconstruction_loss + kl_loss_z + pred_loss)

        dann_vae.add_loss(vae_loss)
        self.dann_vae = dann_vae
        self.encoder = encoder_hx
        self.decoder = decoder_x

    def compile(self):
        self.dann_vae.compile(optimizer=self.optimizer)
        self.dann_vae.summary()

    def train(self, x, batch, loss_weights, batch_size=100, epochs=300):
        history = self.dann_vae.fit({'inputs': x, 'inputs_batch': batch, 'inputs_weights': loss_weights},
                                    epochs=epochs, batch_size=batch_size,
                                    validation_split=self.validation_split, shuffle=True)
        return history

    def get_output(self, x, batches,):
        [z_mean, z_log_var, z] = self.encoder.predict(x)
        output_x = self.decoder.predict(z_mean)
        return z_mean, output_x


class DACVAE:
    def __init__(self, input_size, batches=2, domain_scale_factor=1.0, hidden_layers=[128, 64, 32, 5], path=''):
        self.input_size = input_size
        self.path = path
        self.dann_vae = None
        self.inputs = None
        self.outputs_x = None
        self.initializers = "glorot_uniform"
        self.optimizer = optimizers.Adam(lr=0.01)
        self.hidden_layers = hidden_layers
        self.dropout_rate_small = 0.01
        self.dropout_rate_big = 0.1
        self.kernel_regularizer = regularizers.l1_l2(l1=0.00, l2=0.00)
        self.domain_scale_factor = domain_scale_factor
        self.validation_split = 0.0
        self.batches = batches
        self.dropout_rate = 0.01
        callbacks = []
        checkpointer = ModelCheckpoint(filepath=path + "vae_weights.h5", verbose=1, save_best_only=False,
                                       save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=100, min_lr=0.0001)
        early_stop = EarlyStopping(monitor='loss', patience=200)
        tensor_board = TensorBoard(log_dir=path + 'logs/')
        callbacks.append(checkpointer)
        callbacks.append(reduce_lr)
        callbacks.append(early_stop)
        callbacks.append(tensor_board)
        self.callbacks = callbacks

    def build(self):
        Relu = "relu"
        en_ly_size = len(self.hidden_layers)
        z_size = self.hidden_layers[en_ly_size-1]

        inputs_x = Input(shape=(self.input_size,), name='inputs')
        inputs_batch = Input(shape=(self.batches,), name='inputs_batch')
        inputs_loss_weights = Input(shape=(1,), name='inputs_weights')
        inputs = concatenate([inputs_x, inputs_batch])
        x = inputs
        for i in range(en_ly_size):
            if i == en_ly_size - 1:
                break
            ns = self.hidden_layers[i]
            x = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(x)
            x = BatchNormalization(center=True, scale=False)(x)
            x = Activation(Relu)(x)
            x = Dropout(self.dropout_rate_big)(x)

        hx_mean = Dense(z_size, kernel_regularizer=self.kernel_regularizer,
                        kernel_initializer=self.initializers,
                        name="hx_mean")(x)
        hx_log_var = Dense(z_size, kernel_regularizer=self.kernel_regularizer,
                           kernel_initializer=self.initializers,
                           name="hx_log_var")(x)
        hx_z = Lambda(sampling, output_shape=(z_size,), name='hx_z')([hx_mean, hx_log_var])
        encoder_hx = Model([inputs_x, inputs_batch], [hx_mean, hx_log_var, hx_z], name='encoder_hx')

        latent_inputs_x = Input(shape=(z_size,), name='latent')
        latent_inputs_batch = Input(shape=(self.batches,), name='latent_batch')
        latent_inputs = concatenate([latent_inputs_x, latent_inputs_batch])
        x=latent_inputs
        for i in range(en_ly_size - 1, 0, -1):
            ns = self.hidden_layers[i - 1]
            x = Dense(ns, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(x)
            x = BatchNormalization(center=True, scale=False)(x)
            x = Activation(Relu)(x)
            x = Dropout(self.dropout_rate_big)(x)

        outputs_x = Dense(self.input_size, kernel_regularizer=self.kernel_regularizer,
                          kernel_initializer=self.initializers, activation="softplus")(x)
        decoder_x = Model([latent_inputs_x, latent_inputs_batch], outputs_x, name='decoder_x')

        latent_inputs_domain = Input(shape=(z_size,), name='latent_domain')
        Flip = GradReverse()
        # d= grad_reverse(latent_inputs_batch)
        d = Flip(latent_inputs_domain)
        d = Dense(16, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(d)
        d = BatchNormalization(center=True, scale=False)(d)
        d = Activation(Relu)(d)
        d = Dropout(self.dropout_rate_small)(d)

        d = Dense(self.batches, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                  activation="softmax")(d)
        domian_classifier = Model(latent_inputs_domain, d, name='domain_classifier')

        outputs_x = decoder_x([encoder_hx([inputs_x, inputs_batch])[2], inputs_batch])
        domain_pred = domian_classifier(encoder_hx([inputs_x, inputs_batch])[2])

        dann_vae = Model([inputs_x, inputs_batch, inputs_loss_weights], [outputs_x, domain_pred], name='vae_mlp')

        inputs_x = tf.multiply(inputs_x, inputs_loss_weights)
        outputs_x = tf.multiply(outputs_x, inputs_loss_weights)
        reconstruction_loss = mse(inputs_x, outputs_x)
        # hx_log_var = tf.multiply(hx_log_var, inputs_loss_weights)
        # hx_mean = tf.multiply(hx_mean, inputs_loss_weights)
        # reconstruction_loss = mse(inputs_x, outputs_x)

        noise = tf.math.subtract(inputs_x, outputs_x)
        var = tf.math.reduce_variance(noise)
        reconstruction_loss *= (0.5*self.input_size)/var
        reconstruction_loss += (0.5*self.input_size)/var*tf.math.log(var)

        kl_loss_z = -0.5 * K.sum(1 + hx_log_var - K.square(hx_mean) - K.exp(hx_log_var), axis=-1)

        pred_loss = K.categorical_crossentropy(inputs_batch, domain_pred)*self.input_size*self.domain_scale_factor
        vae_loss = K.mean(reconstruction_loss + kl_loss_z + pred_loss)

        dann_vae.add_loss(vae_loss)
        self.dann_vae = dann_vae
        self.encoder = encoder_hx
        self.decoder = decoder_x

    def compile(self):
        self.dann_vae.compile(optimizer=self.optimizer)
        self.dann_vae.summary()

    def train(self, x, batch, loss_weights, batch_size=100, epochs=300):

        history = self.dann_vae.fit({'inputs': x, 'inputs_batch': batch, 'inputs_weights': loss_weights},
                                        epochs=epochs, batch_size=batch_size,
                                        validation_split=self.validation_split, shuffle=True)
        return history

    def get_output(self, x, batches,):
        [z_mean, z_log_var, z] = self.encoder.predict([x, batches])
        output_x = self.decoder.predict([z_mean, batches])
        return z_mean, output_x




def fit_integration(adata, batch_num=2, mode='DACVAE', split_by='batch_label', epochs=20, batch_size=128,
                    domain_lambda=1.0, sparse=True, hidden_layers=[128,64,32,5]):
    """/
    Build DAVAE model and fit the data to the model for training.

    Parameters
    ----------
    adata: AnnData
        AnnData object need to be integrated.

    batch_num: int, optional (default: 2)
        Number of batches of datasets to be integrated.

    mode: string, optional (default: 'DACVAE')
        if 'DACVAE', construct a DACVAE model
        if 'DAVAE', construct a DAVAE model

    split_by: string, optional (default: '_batch')
        the obsm_name of obsm used to distinguish different batches.

    epochs: int, optional (default: 200)
        Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.

    batch_size: int or None, optional (default: 256)
        Number of samples per gradient update. If unspecified, batch_size will default to 32.

    domain_lambda: double, optional (default: 1.0)
        The coefficient multiplied by the loss value of the domian classifier of DAVAE model.

    sparse: bool, optional (default: True)
        If True, Matrix X in the AnnData object is stored as a sparse matrix.

    hidden_layers: list of integers, (default: [128,64,32,5])
        Number of hidden layer neurons in the model.

    Returns
    -------
    :class:`~anndata.AnnData`
        out_adata
    """

    batch = adata.obs[split_by]
    batch = np.array(batch.values, dtype=int)
    loss_weight = adata.obs['loss_weight']
    # orig_batch, batch_num = tl.generate_batch_code(batch, batch_num)
    orig_batch  = to_categorical(batch)
    if sparse:
        orig_data = adata.X.A
    else:
        orig_data = adata.X
    data, batch, loss_weight = shuffle(orig_data, orig_batch, loss_weight, random_state=0)
    if mode=='DAVAE':
        net = DAVAE(input_size=data.shape[1], batches=batch_num, domain_scale_factor=domain_lambda,
                    hidden_layers=hidden_layers)
    else:
        net = DACVAE(input_size=data.shape[1],  batches=batch_num, domain_scale_factor=domain_lambda,
                     hidden_layers=hidden_layers)
    net.build()
    net.compile()
    net.train(data, batch, loss_weight, batch_size=batch_size, epochs=epochs)
    latent_z, output_x = net.get_output(orig_data, orig_batch)
    out_adata = anndata.AnnData(X=output_x, obs=adata.obs, var=adata.var)
    out_adata.obsm['X_davae'] = latent_z
    out_adata.raw = adata.copy()
    return out_adata

