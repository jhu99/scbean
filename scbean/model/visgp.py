import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
import multiprocessing
from tqdm import tqdm
import fastlmm.util.stats.quadform as qf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
dtype = np.float64

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'  # not GPU

def covariance_matrix(X, length):
    """
    Calculate the covariance matrix.
    :param X: Represents the spatial position coordinates(two dimensional array)
    :param length: kernel parameter
    :return: Covariance matrix
    """
    Xsq = np.sum(np.square(X), 1)
    R2 = -2. * np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
    R2 = np.clip(R2, 1e-12, np.inf)
    K = np.exp(-R2 / (2 * length ** 2))
    return K

def score_test(K, n, y, acc=1e-7):
    """
    Score statistics test.
    :param K: Covariance matrix
    :param n: Number of spots
    :param y: A vector representing the expression value of a gene
    :return: P value of a gene
    """
    # Perform the eigendecomposition
    eigenvalues = np.linalg.eigvalsh(K)
    # Round to zero
    eigenvalues[eigenvalues < 10e-5] = 0
    eigenvalues = np.sort(eigenvalues)[::-1]
    # Calculate k = rank(K)
    k = int(sum(eigenvalues > 0))
    # Calculate q = dim(ker(K) & col(I))
    q = int(n - sum(eigenvalues > 10e-5))
    r = n * np.dot(np.dot(y.T, K), y) / np.square(y).sum()
    alphars = np.concatenate((eigenvalues[:k] - float(r) / n, np.ones(q) * -float(r) / n), axis=0)
    return qf.qf(0, alphars, acc=acc)[0]

def qvalue(pv):
    """
    Calculate Q values using BH adjustment.
    :param pv: P values of all genes
    :return: Q values of all genes
    """
    original_shape = pv.shape
    pv = pv.ravel()
    m = float(len(pv))
    p_ordered = np.argsort(pv)
    pv = pv[p_ordered]
    qv = np.zeros_like(pv)
    qv[-1] = pv[-1]
    for i in range(len(pv) - 2, -1, -1):
        qv[i] = min(m * pv[i] / (i + 1), pv[i + 1])
    qv_temp = qv.copy()
    qv = np.zeros_like(qv)
    qv[p_ordered] = qv_temp
    qv = qv.reshape(original_shape)
    return qv

def build(k, X, y, inducing_points=20, iters=1000):
    """
    Build and training model.
    :param k: Used to mark a gene
    :param X: The spatial position coordinates(array)
    :param y: A vector representing the expression value of a gene
    :param inducing_points: The number of inducing points
    :param iters: The number of iters
    :return: k and p-value
    """
    # Create kernel parameters, and observation noise variance variable
    amplitude = tfp.util.TransformedVariable(1., tfb.Softplus(), dtype=dtype, name='amplitude')
    length_scale = tfp.util.TransformedVariable(1., tfb.Softplus(), dtype=dtype, name='length_scale')
    # k(x, y) = amplitude**2 * exp(-||x - y||**2 / (2 * length_scale**2))
    kernel = tfk.ExponentiatedQuadratic(amplitude=amplitude, length_scale=length_scale)
    observation_noise_variance = tfp.util.TransformedVariable(
        1., tfb.Softplus(), dtype=dtype, name='observation_noise_variance')
    # Create trainable inducing point locations and variational parameters.
    num_inducing_points_ = inducing_points
    inducing_mean = np.array([0, 0])
    inducing_conv = np.array([[1, 0.0], [0.0, 1]])
    inducing_index_points = tf.Variable(
        np.random.multivariate_normal(mean=inducing_mean, cov=inducing_conv, size=num_inducing_points_),
        dtype=dtype, name='inducing_index_points')
    variational_inducing_observations_loc = tf.Variable(
        np.zeros([num_inducing_points_], dtype=dtype),
        name='variational_inducing_observations_loc')
    variational_inducing_observations_scale = tf.Variable(
        np.eye(num_inducing_points_, dtype=dtype),
        name='variational_inducing_observations_scale')
    index_points_ = X
    num_points_ = index_points_.shape[0]
    # Construct our variational GP Distribution instance.
    vgp = tfd.VariationalGaussianProcess(
        kernel,
        index_points=index_points_,
        inducing_index_points=inducing_index_points,
        variational_inducing_observations_loc=variational_inducing_observations_loc,
        variational_inducing_observations_scale=variational_inducing_observations_scale,
        observation_noise_variance=observation_noise_variance)
    optimizer = tf.keras.optimizers.Adam(learning_rate=.1)

    @tf.function
    def optimize(x_train_batch, y_train_batch):
        with tf.GradientTape() as tape:
            # Create the loss function we want to optimize.
            loss = vgp.variational_loss(
                observations=y_train_batch,
                observation_index_points=x_train_batch,
                kl_weight=float(y.shape[0]) / float(num_points_))
        grads = tape.gradient(loss, vgp.trainable_variables)
        optimizer.apply_gradients(zip(grads, vgp.trainable_variables))
        return loss

    num_iters = iters
    for i in range(num_iters):
        optimize(X, y)
    K = covariance_matrix(X, length_scale.numpy())
    p_value = score_test(K, num_points_, y)
    return k, p_value

def run(X, Y, processes=1, inducing_points=20, iters=1000):
    """
    :param X: The spatial position coordinates(array)
    :param Y: The expression values of genes(DataFrame, genes * spots)
    :param processes: Number of concurrent processes
    :return: results
    """
    names = Y.index.values
    y_all_genes = Y.values
    y_all_genes = y_all_genes.T
    y_all_genes = preprocessing.scale(y_all_genes)
    y_all_genes = y_all_genes.T
    num_genes = y_all_genes.shape[0]
    results = pd.DataFrame(columns=['gene', 'p_value', 'q_value'])

    if processes == 1:
        for k in tqdm(range(num_genes)):
            k, p_value = build(k, X, y_all_genes[k], inducing_points, iters)
            results.loc[k, 'gene'] = names[k]
            results.loc[k, 'p_value'] = p_value
    else:
        args = []
        pool = multiprocessing.Pool(processes)
        for k in range(num_genes):
            args.append((k, X, y_all_genes[k], inducing_points, iters))
        args = tqdm(args)
        result = pool.starmap(build, args)
        pool.close()
        pool.join()
        print("Results....................")
        for k in tqdm(range(num_genes)):
            results.loc[k, 'gene'] = names[result[k][0]]
            results.loc[k, 'p_value'] = result[k][1]

    q_value = qvalue(results['p_value'].values)
    for j in range(num_genes):
        results.loc[j, 'q_value'] = q_value[j]
    return results
