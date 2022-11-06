import numpy as np
from scipy.stats import norm, multinomial, multivariate_normal
from tqdm import tqdm
import copy
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


class EmAlgorithm:
    def __init__(self, y, k, n_wait=20, pooled_sigma=False, tol=1e-12, max_iter=5000):
        self.k = k
        self.data = y
        self.tol = tol
        self.max_iter = max_iter
        self.pooled_sigma = pooled_sigma
        self._initialize_sigmas(pooled_sigma)
        self._initialize_arrays()
        self.n_wait = n_wait

    def _initialize_arrays(self):
        self.weights = np.ones((self.max_iter, self.k)) * 1 / self.k
        self.mus = np.ones((self.max_iter, self.k)) * np.random.uniform(np.min(self.data), np.max(self.data),
                                                                        size=self.k)
        self.weights_i = np.ones((self.max_iter, len(self.data), self.k)) * 1 / self.k
        self.weights_i[0] = self._update_weights_i(self.weights[0], self.mus[0], self.std[0])
        self.log_likelihoods = np.ones(self.max_iter)
        self.log_likelihoods[0] = self._calculate_log_like(self.mus[0], self.std[0], self.weights[0])

    def _initialize_sigmas(self, pooled_sigma):
        if pooled_sigma:
            self.std = np.ones((self.max_iter, 1)) * np.random.uniform(0.5, 10, size=1)
        else:
            self.std = np.ones((self.max_iter, self.k)) * np.random.uniform(0.5, 10, size=self.k)

    def _update_weights_i(self, current_weights, current_mus, current_sigma):
        normal_draws = norm(loc=current_mus, scale=current_sigma).pdf(np.vstack([self.data] * self.k).T)
        numerator = current_weights * normal_draws
        denominator = np.sum(numerator, axis=1)[:, None]
        return numerator / denominator

    @staticmethod
    def _update_weights(weights_i):
        un_normalized_weights = np.sum(weights_i, axis=0)
        return un_normalized_weights / np.sum(un_normalized_weights)

    def _update_mus(self, weights_i):
        mus = np.sum(self.data[:, None] * weights_i, axis=0) / np.sum(weights_i, axis=0)
        return mus

    def _update_sigmas_not_pooled(self, weights_i, mus):
        numerator = weights_i * (self.data[:, None] - mus) ** 2
        denominator = np.sum(weights_i, axis=0)
        return np.sqrt(np.sum(numerator / denominator, axis=0))

    def _update_sigmas_pooled(self, weights_i, mus):
        numerator = np.sum(weights_i * (self.data[:, None] - mus) ** 2)
        denominator = np.sum(weights_i)
        return np.sqrt(numerator / denominator)

    def _check_convergence(self, it):
        return np.abs((self.log_likelihoods[it + 1] - self.log_likelihoods[it]) / self.log_likelihoods[it]) < self.tol

    def _calculate_log_like(self, current_mus, current_sigma, current_weights):
        densities = np.sum(current_weights * norm(current_mus, current_sigma).pdf(np.vstack([self.data] * self.k).T),
                           axis=1)
        return np.sum(np.log(densities))

    def _update_sigmas(self, weights_i, mus):
        if self.pooled_sigma:
            return self._update_sigmas_pooled(weights_i, mus)
        else:
            return self._update_sigmas_not_pooled(weights_i, mus)

    def fit(self):
        for it in tqdm(range(self.max_iter - 1)):
            self.weights[it + 1] = self._update_weights(self.weights_i[it])
            self.weights_i[it + 1] = self._update_weights_i(self.weights[it + 1], self.mus[it], self.std[it])
            if it <= self.n_wait:
                self.mus[it + 1] = copy.copy(self.mus[it])
            else:
                self.mus[it + 1] = self._update_mus(self.weights_i[it + 1])
            self.std[it + 1] = self._update_sigmas(self.weights_i[it + 1], self.mus[it + 1])
            self.log_likelihoods[it + 1] = self._calculate_log_like(self.mus[it + 1], self.std[it + 1],
                                                                    self.weights[it + 1])
            if self._check_convergence(it):
                return {'mus': self.mus[it + 1], 'std': self.std[it + 1], 'weights': self.weights[it + 1],
                        'loglike': self.log_likelihoods[it + 1]}
        return 'not converged'


class StochasticEmAlgorithm(EmAlgorithm):

    def _do_multinomial_draw(self, it):
        # [[multinomial(1, self.weights_i[it, i, :]).rvs().flatten(), print(self.weights_i[it, i, :])] for i in
        # range(len(self.data))]
        draws = np.array([np.random.multinomial(1, self.weights_i[it, i, :]) for i in range(len(self.data))])
        if np.all(draws.sum(axis=0)) > 0:
            return draws
        else:
            # draws[-1, :] = np.ones(self.k)
            return self._do_multinomial_draw(it)

    def fit(self):
        for it in tqdm(range(self.max_iter - 1)):
            self.weights_i[it] = np.around(self.weights_i[it], 8)
            self.weights_i[it] = self.weights_i[it] / self.weights_i[it].sum(axis=1)[:, None]
            draws = self._do_multinomial_draw(it)
            self.weights[it + 1] = self._update_weights(draws)
            if it <= self.n_wait:
                self.mus[it + 1] = copy.copy(self.mus[it])
            else:
                self.mus[it + 1] = self._update_mus(draws)
            self.std[it + 1] = self._update_sigmas(draws, self.mus[it + 1])
            self.log_likelihoods[it + 1] = self._calculate_log_like(self.mus[it + 1], self.std[it + 1],
                                                                    self.weights[it + 1])
            self.weights_i[it + 1] = self._update_weights_i(self.weights[it + 1], self.mus[it + 1], self.std[it + 1])
            self.weights_i[it + 1][np.isnan(self.weights_i[it + 1])] = .1
            if self._check_convergence(it):
                return {'mus': self.mus[it + 1], 'std': self.std[it + 1], 'weights': self.weights[it + 1],
                        'loglike': self.log_likelihoods[it + 1]}
        return 'not converged'


class EmBivariateNormal:
    def __init__(self, data, k, tol=1e-6, max_iter=1000):
        self.data = data
        self.k = k
        self.n_params = self.data.shape[1]
        self.max_iter = max_iter
        self.tol = tol
        self.means = np.ones((self.k, self.n_params)) * np.random.uniform(np.min(data), np.max(data),
                                                                          size=(self.k, self.n_params))
        self.covs = [np.eye(self.n_params)] * self.k
        self.weights = np.ones(self.k) * 1 / self.k
        self.W = np.zeros((self.data.shape[0], self.k))
        self.log_likes = []

    def fit(self):
        for it in tqdm(range(self.max_iter)):
            self.W = self._update_W(self.weights, self.means, self.covs)
            self.log_likes.append(self._calculate_log_likes(self.weights, self.means, self.covs))
            if it < 10:
                self.means = self._update_means(self.W)
            self.covs = self._update_covs(self.W, self.means)
            self.weights = self._update_weights(self.W)
            if it > 1:
                converged = self._check_convergence(it)
                if converged:
                   return 'converged'
        return 'not converged'

    def _check_convergence(self, it):
        return np.abs((self.log_likes[it] - self.log_likes[it - 1]) / self.log_likes[it - 1]) < self.tol

    def _update_means(self, W):
        return np.sum(W[:, None, :] * self.data[:, :, None], axis=0).T / np.sum(W, axis=0)[:, None]

    def _calculate_log_likes(self, weights, means, covs):
        log_likes = np.sum(np.array([weights[k] * multivariate_normal.pdf(self.data, mean=means[k], cov=covs[k], allow_singular=True) for k in range(self.k)]), axis=0)
        return np.sum(np.log(log_likes))

    def _update_W(self, weights, means, covs):
        W_un_normalized = np.zeros((self.data.shape[0], self.k))
        for k in range(self.k):
            W_un_normalized[:, k] = weights[k] * multivariate_normal.pdf(self.data, mean=means[k], cov=covs[k],
                                                                         allow_singular=True)
        W_normalized = (W_un_normalized.T / W_un_normalized.sum(axis=1)).T
        return W_normalized

    def _update_covs(self, W, means):
        W_s = np.sum(W, axis=0)
        covs = [np.eye(self.n_params)] * self.k
        for k in range(self.k):
            covs[k] = ((W[:, k] * (self.data - means[k]).T) @ (self.data - means[k])) / W_s[k]
        return covs

    @staticmethod
    def _update_weights(W):
        return np.mean(W, axis=0)


if __name__ == '__main__':
    seeds = np.random.uniform(0, 100000, 5000)
    for seed in seeds:
        try:

            np.random.seed(int(seed))
            df = pd.read_csv('../../homework_4/data/faithful.csv')
            data = df[['eruptions', 'waiting']].to_numpy()
            em = EmBivariateNormal(data, k=5, max_iter=50)
            print(em.fit())
            print(int(seed))
            print('--')

          #  if em.fit() == 'converged':
               # print(int(seed))
        except:
            pass

