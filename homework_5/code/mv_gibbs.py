from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.stats import multivariate_normal, invwishart, dirichlet
from tqdm import tqdm


class MvGibbsSampler:
    def __init__(self, X, k=2, n_it=1000, mu_0=0, nu_0=10, Sig_0=10, Psi_0=5, alpha=2):
        self.n, self.ndim = X.shape
        self.X = X
        self.k = k
        self.n_it = n_it

        gm = GaussianMixture(n_components=k).fit(X)
        self.weights = np.array([gm.weights_] * n_it)
        self.means = np.array([gm.means_] * n_it)
        self.covs = np.array([gm.covariances_] * n_it)

        self.mu_0 = np.ones(self.ndim) * mu_0
        self.Sig_0 = np.eye(self.ndim) * Sig_0
        self.Psi_0 = np.eye(self.ndim) * Psi_0
        self.nu_0 = nu_0
        self.alpha = alpha

    def _draw_z(self, pi_ik):
        z = np.array([np.random.multinomial(n=1, pvals=pi_ik[i, :]) for i in range(self.n)])
        return z

    def _calculate_pi_ik(self, weights, means, covs):
        pi_ik = np.zeros([self.n, self.k])
        for k in range(self.k):
            pi_ik[:, k] = weights[k] * multivariate_normal.pdf(x=self.X, mean=means[k], cov=covs[k],
                                                               allow_singular=True)
        return pi_ik / np.sum(pi_ik, axis=1)[:, None]

    def _update_means(self, covs_k, z):
        means = np.zeros((self.k, self.ndim))
        for k in range(self.k):
            inv_cov = np.linalg.inv(covs_k[k])
            cov = np.linalg.inv(np.sum(z[:, k]) * inv_cov + np.linalg.inv(self.Sig_0))
            mus = cov @ (inv_cov @ np.sum(self.X * z[:, k][:, None], axis=0) + np.linalg.inv(self.Sig_0) @ self.mu_0)
            means[k] = multivariate_normal(mus, cov).rvs()
        return means

    def _update_covs(self, z, mus):
        covs = np.zeros((self.k, self.ndim, self.ndim))
        nu = np.sum(z, axis=0) + self.nu_0
        for k in range(self.k):
            terms = np.array(
                [(self.X[i, :] - mus[k]).reshape(-1, 1) @ (self.X[i, :] - mus[k]).reshape(-1, 1).T * z[i, k] for i in
                 range(self.n)])
            Psi_prime = np.sum(terms, axis=0)
            covs[k] = invwishart(nu[k], Psi_prime + self.Psi_0).rvs()
        return covs

    def _update_weights(self, z):
        n_k = np.sum(z, axis=0)
        weights = dirichlet(alpha=self.alpha + n_k).rvs().flatten()
        return weights

    def _remove_burn(self, arr, n_burn, n_thin):
        return arr[n_burn:][::n_thin]

    def fit(self, n_burn=500, n_thin=2):
        for it in tqdm(range(self.n_it - 1)):
            pi_i_k = self._calculate_pi_ik(self.weights[it], self.means[it], self.covs[it])
            z = self._draw_z(pi_i_k)
            self.means[it + 1] = self._update_means(self.covs[it], z)
            self.covs[it + 1] = self._update_covs(z, self.means[it + 1])
            self.weights[it + 1] = self._update_weights(z)

        self.weights = self._remove_burn(self.weights, n_burn, n_thin)
        self.covs = self._remove_burn(self.covs, n_burn, n_thin)
        self.means = self._remove_burn(self.means, n_burn, n_thin)
