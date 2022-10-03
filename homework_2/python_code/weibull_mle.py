""" . """
import numpy as np
import warnings


class WeibullMle:
    """
    Fit k and lambda via MLE
    """

    def __init__(self, data, max_iter=1000, lr=.001, tol=1e-5, init_params=None):
        """

        Parameters
        ----------
        data
        max_iter
        lr
        tol
        """
        self.data = data
        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.k = None
        self.lam = None
        self.n = len(data)
        self.init_params = init_params

    def initialize_opt(self):
        """

        Parameters
        ----------

        Returns
        -------

        """
        if self.init_params is None:
            self.init_params = [200, 10]
        params = np.zeros((2, self.max_iter))
        log_likelihoods = np.zeros(self.max_iter)
        params[:, 0] = self.init_params
        log_likelihoods[0] = self.calculate_log_likelihood(*params[:, 0])
        return params, log_likelihoods

    def calculate_hessian(self, lam, k):
        """

        Parameters
        ----------
        lam
        k

        Returns
        -------

        """

        d_k_lam = np.sum(((self.data / lam) ** k + k * (self.data / lam) ** k * np.log(self.data / lam) - 1) / lam)
        d_k_k = np.sum((self.data/lam)**k * (-np.log(self.data/lam)**2)-1/k**2)
        d_lam_lam = - np.sum((k*(k*(self.data/lam)**k+(self.data/lam)**k-1))/lam**2)
        return np.array(([d_lam_lam, d_k_lam], [d_k_lam, d_k_k]))

    def calculate_derivative_log_likelihood(self, lam, k):
        """

        Parameters
        ----------
        lam
        k

        Returns
        -------

        """
        d_k = np.sum(1 / k - ((self.data / lam) ** k - 1) * np.log(self.data / lam))
        d_lam = np.sum(k / lam * ((self.data / lam) ** k - 1))
        return np.array([d_lam, d_k])

    def calculate_log_likelihood(self, lam, k):
        """

        Parameters
        ----------
        lam
        k

        Returns
        -------

        """
        log_likelihood = self.n * np.log(k) - self.n * np.log(lam) + (k - 1) * np.sum(np.log(self.data / lam)) - np.sum(
            (self.data / lam) ** k)
        return log_likelihood

    def check_convergence(self, it, log_likelihoods):
        """

        Parameters
        ----------
        it
        log_likelihoods

        Returns
        -------

        """
        return np.abs((log_likelihoods[it + 1] - log_likelihoods[it]) / log_likelihoods[it]) < self.tol

    def fit(self):
        """

        Returns
        -------

        """

        params, log_likelihoods = self.initialize_opt()
        for it in range(self.max_iter - 1):
            hessian = self.calculate_hessian(*params[:, it])
            new_params = params[:, it] - np.linalg.inv(hessian) \
                @ self.calculate_derivative_log_likelihood(*params[:, it])
            params[:, it+1] = np.array([max(new_params[0], 1e-6), max(new_params[1], 1e-6)])
            log_likelihoods[it + 1] = self.calculate_log_likelihood(*params[:, it + 1])
            if self.check_convergence(it, log_likelihoods):
                self.lam, self.k = params[:, it + 1]
                return self.lam, self.k, log_likelihoods[it + 1]
        warnings.warn(f'not converged')
        self.lam, self.k = params[:, -1]
        return self.lam, self.k, log_likelihoods[-1]


if __name__ == '__main__':
    obs_data = np.array([225, 171, 198, 189, 189, 135, 162, 135, 117, 162])
    opt = WeibullMle(obs_data)
    opt.fit()
    print(opt.lam, opt.k)
    pass
