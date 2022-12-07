import numpy as np
from tqdm import tqdm

def fit_kalman_filter(sigma_sq, Q, R, y, Z, T, init_val):
    ndim = Z.shape[1]
    n_it = y.shape[1]
    alpha = np.zeros((n_it, ndim))
    alpha[0] = init_val
    P = np.zeros((n_it, ndim, ndim))
    P[0] = np.eye(ndim) * 1000
    for it in tqdm(range(n_it)):
        vt = y[it] - Z[it]@alpha[it, :]
        Ft = Z[it]@P[it]@Z[it].T + sigma_sq
        Kt = T@P[it]@Z[it].T * 1/Ft
        Lt = T - Kt@Z[it]
        alpha[it + 1] = T@alpha[it] + Kt@vt
        P[it + 1] = T@P[it]@Lt.T + R@Q@R.T
    return alpha, P









