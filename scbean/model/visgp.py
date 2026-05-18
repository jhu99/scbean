import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
import multiprocessing
from tqdm import tqdm
#import fastlmm.util.stats.quadform as qf
import fastlmmclib.quadform as qf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
dtype = np.float64

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'  # not GPU

class VISGP(object):
    """
    Initialize VISGP object
    
    Parameters
    
    ----------
    
    adata: anndata, (default: None)
        The input data for the model, including gene expression levels (adata.X, genes*spots),
        spatial location coordinates (adata.var) and genes name (adata.obs).

    inducing_points: int, optional (default: 20)
        The number of inducing points.

    iters: int, optional (default: 1000)
        The number of iters.

    processes: int, optional (default: 1)
        The number of concurrent processes.
        
    kernel_type: str, optional (default: 'rbf')
        Type of kernel to use: 'rbf', 'matern', 'periodic', 'anisotropic', or 'multi'.
        
    nu: float, optional (default: 1.5)
        Shape parameter for Matérn kernel (1.5, 2.5, or inf).
    """
    def __init__(self, adata=None, inducing_points=20, iters=1000, processes=1, 
                 kernel_type='rbf', nu=1.5):
        self.adata = adata
        self.inducing_points = inducing_points
        self.iters = iters
        self.processes = processes
        self.acc = 1e-7
        self.kernel_type = kernel_type
        self.nu = nu

    def covariance_matrix(self, length, kernel_type=None):
        """
        Calculate the covariance matrix using specified kernel.
        
        Parameters
        ----------
        length: float or dict
            kernel parameter (float for most kernels, dict for anisotropic)
        kernel_type: str, optional
            Type of kernel. If None, uses self.kernel_type
        
        Returns
        ----------
        :class:`~Numpy array(s)`
            Covariance matrix
        """
        if kernel_type is None:
            kernel_type = self.kernel_type
            
        if kernel_type == 'rbf':
            return self._rbf_kernel(length)
        elif kernel_type == 'matern':
            return self._matern_kernel(length)
        elif kernel_type == 'periodic':
            return self._periodic_kernel(length)
        elif kernel_type == 'anisotropic':
            return self._anisotropic_kernel(length)
        elif kernel_type == 'multi':
            return self._multi_kernel(length)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

    def _rbf_kernel(self, length):
        """
        RBF (Squared Exponential) kernel.
        K(x, y) = exp(-||x - y||^2 / (2 * length^2))
        """
        Xsq = np.sum(np.square(self.adata.var), 1)
        R2 = -2. * np.dot(self.adata.var, self.adata.var.T) + (Xsq[:, None] + Xsq[None, :])
        R2 = np.clip(R2, 1e-12, np.inf)
        K = np.exp(-R2 / (2 * length ** 2))
        return K

    def _matern_kernel(self, length):
        """
        Matérn kernel with shape parameter nu.
        More flexible than RBF, less smooth for smaller nu values.
        nu: 1.5, 2.5, or inf (inf = RBF)
        """
        Xsq = np.sum(np.square(self.adata.var), 1)
        R2 = -2. * np.dot(self.adata.var, self.adata.var.T) + (Xsq[:, None] + Xsq[None, :])
        R = np.sqrt(np.clip(R2, 1e-12, np.inf))
        
        if self.nu == 1.5:
            K = (1 + np.sqrt(3) * R / length) * np.exp(-np.sqrt(3) * R / length)
        elif self.nu == 2.5:
            K = (1 + np.sqrt(5) * R / length + 5 * R**2 / (3 * length**2)) * np.exp(-np.sqrt(5) * R / length)
        else:  # nu = inf, equivalent to RBF
            K = np.exp(-R2 / (2 * length ** 2))
        
        return K

    def _periodic_kernel(self, length):
        """
        Periodic kernel for capturing periodic spatial patterns.
        K(x, y) = exp(-2 * sin^2(pi * ||x - y|| / period) / length^2)
        """
        Xsq = np.sum(np.square(self.adata.var), 1)
        R2 = -2. * np.dot(self.adata.var, self.adata.var.T) + (Xsq[:, None] + Xsq[None, :])
        R = np.sqrt(np.clip(R2, 1e-12, np.inf))
        
        # Use length as both period and lengthscale
        period = length * 2.0
        K = np.exp(-2.0 * np.sin(np.pi * R / period) ** 2 / (length ** 2))
        
        return K

    def _anisotropic_kernel(self, length_dict):
        """
        Anisotropic kernel with different lengthscales for each dimension.
        Useful for capturing directional spatial patterns.
        
        Parameters
        ----------
        length_dict: dict
            Dictionary with 'scales' key containing array of lengthscales per dimension
        """
        if isinstance(length_dict, (int, float)):
            # Fallback to RBF if scalar is provided
            return self._rbf_kernel(length_dict)
            
        scales = length_dict.get('scales', np.array([1.0, 1.0]))
        
        X = self.adata.var.values
        n = X.shape[0]
        K = np.zeros((n, n))
        
        # Scale coordinates by lengthscales
        X_scaled = X / scales[np.newaxis, :]
        
        # Compute RBF kernel on scaled coordinates
        Xsq = np.sum(np.square(X_scaled), 1)
        R2 = -2. * np.dot(X_scaled, X_scaled.T) + (Xsq[:, None] + Xsq[None, :])
        R2 = np.clip(R2, 1e-12, np.inf)
        K = np.exp(-R2 / 2.0)
        
        return K

    def _multi_kernel(self, length):
        """
        Multi-kernel approach: weighted combination of different kernels.
        Improves robustness across diverse spatial patterns.
        K_multi = w1*RBF + w2*Matérn + w3*Periodic
        """
        # Weights for kernel combination
        w_rbf = 0.5
        w_matern = 0.3
        w_periodic = 0.2
        
        K_rbf = self._rbf_kernel(length)
        K_matern = self._matern_kernel(length)
        K_periodic = self._periodic_kernel(length)
        
        K = w_rbf * K_rbf + w_matern * K_matern + w_periodic * K_periodic
        
        # Normalize to ensure valid covariance matrix
        K = K / (w_rbf + w_matern + w_periodic)
        
        return K

    def score_test(self, K, y):
        """
        Score statistics test.
        
        Parameters
        ----------
        K: 
            Covariance matrix
        y: 
            A vector representing the expression value of a gene
        
        Returns
        ----------
        :class:`~float`
            P value of a gene
        """
        n = self.adata.n_vars
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
        return qf.qf(0, alphars, acc=self.acc)[0]

    def qvalue(self, pv):
        """
        Calculate Q values using BH adjustment.
        
        Parameters
        ----------
        pv:             P values of all genes
        
        Returns
        -------
        :class:`~Numpy array(s)`
            Q values of all genes
        """
        original_shape = pv.shape
        pv = pv.ravel()
        m = float(len(pv))
        p_ordered = np.argsort(pv)
        pv = pv[p_ordered]
        qv = np.zeros_like(pv)
        qv[-1] = min(pv[-1], 1.0)
        
        for i in range(len(pv) - 2, -1, -1):
            qv[i] = min(m * pv[i] / (i + 1), qv[i + 1])
        
        qv_temp = np.zeros_like(qv)
        qv_temp[p_ordered] = qv
        qv_temp = qv_temp.reshape(original_shape)
        return qv_temp

    def build(self, k, y):
        """
        Build and training model.
        
        Parameters
        ----------
        k: int
            Used to mark a gene
        y: 
            A vector representing the expression value of a gene
        
        Returns
        ----------
        :class:`~int` and `~float`
            k and p-value
        """
        # Create kernel parameters, and observation noise variance variable
        amplitude = tfp.util.TransformedVariable(1., tfb.Softplus(), dtype=dtype, name='amplitude')
        length_scale = tfp.util.TransformedVariable(1., tfb.Softplus(), dtype=dtype, name='length_scale')
        
        # Create kernel based on kernel_type
        if self.kernel_type in ['rbf', 'matern', 'periodic']:
            kernel = tfk.ExponentiatedQuadratic(amplitude=amplitude, length_scale=length_scale)
        elif self.kernel_type == 'anisotropic':
            # For anisotropic, still use ExponentiatedQuadratic but we'll handle scaling in covariance_matrix
            kernel = tfk.ExponentiatedQuadratic(amplitude=amplitude, length_scale=length_scale)
        elif self.kernel_type == 'multi':
            kernel = tfk.ExponentiatedQuadratic(amplitude=amplitude, length_scale=length_scale)
        else:
            kernel = tfk.ExponentiatedQuadratic(amplitude=amplitude, length_scale=length_scale)
        
        observation_noise_variance = tfp.util.TransformedVariable(
            1., tfb.Softplus(), dtype=dtype, name='observation_noise_variance')
        # Create trainable inducing point locations and variational parameters.
        num_inducing_points_ = self.inducing_points
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
        index_points_ = self.adata.var.values
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

        for i in range(self.iters):
            optimize(index_points_, y)
        
        K = self.covariance_matrix(length_scale.numpy())
        p_value = self.score_test(K, y)
        return k, p_value

    def run(self):
        """
        Run VISGP.
        
        Returns
        ----------
        :class:`~pandas.core.frame.DataFrame`
            results, including ['gene', 'p_value', 'q_value']
        """
        names = self.adata.obs
        y_all_genes = self.adata.X  # genes*spots(ndarray)
        y_all_genes = y_all_genes.T
        y_all_genes = preprocessing.scale(y_all_genes)
        y_all_genes = y_all_genes.T
        num_genes = self.adata.n_obs
        results = pd.DataFrame(columns=['gene', 'p_value', 'q_value'])

        if self.processes == 1:
            for k in tqdm(range(num_genes)):
                k, p_value = self.build(k, y_all_genes[k])
                results.loc[k, 'gene'] = names.iloc[k, 0]
                results.loc[k, 'p_value'] = p_value
        else:
            args = []
            pool = multiprocessing.Pool(self.processes)
            for k in range(num_genes):
                args.append((k, y_all_genes[k]))
            args = tqdm(args)
            result = pool.starmap(self.build, args)
            pool.close()
            pool.join()
            print("Results....................")
            for k in tqdm(range(num_genes)):
                results.loc[k, 'gene'] = names.iloc[result[k][0], 0]
                results.loc[k, 'p_value'] = result[k][1]

        q_value = self.qvalue(results['p_value'].values)
        for j in range(num_genes):
            results.loc[j, 'q_value'] = q_value[j]
        return results
