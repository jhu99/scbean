from scipy.stats import norm
from scipy.stats import rankdata
import numpy as np

def cacu_theory_quantiles(data, vx, scale=1):
    if vx > 10:
        sx = (data-0.5)/vx
    else:
        sx = (data-0.375)/(vx+0.25)
    sy = norm.ppf(sx, scale=scale)
    return sy

def rank(X, method="average", axis=0):
    rm=np.ndarray(shape=X.shape)
    for i in range(X.shape[1-axis]):
        if axis==0:
            data = X[:,i]
            rm[:,i]=rankdata(data, method=method)
        else:
            data = X[i,:]
            rm[i,:]=rankdata(data, method=method)
    return rm

def qqnorm(X, method='average', axis=0, scale=1):
    rm=rank(X, method=method, axis=axis)
    n_quantiles=rm.max(axis=axis)
    X_new = np.ndarray(shape=X.shape)
    for i in range(len(n_quantiles)):
        n_quantile=n_quantiles[i]
        if axis==0:
            X_new[:,i]=cacu_theory_quantiles(rm[:,i],n_quantile, scale=scale)
        else:
            X_new[i,:]=cacu_theory_quantiles(rm[i,:],n_quantile, scale=scale)
    return X_new
