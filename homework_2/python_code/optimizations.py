""" . """
import numpy as np
import warnings


class GradientAscent:
    """
    do gradient ascent
    """

    def __init__(self, y_obs, lr=.01, max_iter=5000, tol=1e-5):
        """

        Parameters
        ----------
        y_obs: array like
            observed data
        lr: float
            learning rate
        max_iter: int
            max number of iterations
        tol: float
            tolerance
        """
        self.lr = lr
        self.max_iter = max_iter
        self.init_points = np.linspace(y_obs.min(), y_obs.max(), 50)
        self.tol = tol
        self.optimal_theta = None
        self.it_to_convergence = None
        self.y_obs = y_obs

    def initialize_params(self, init_theta, y_obs):
        """

        Parameters
        ----------
        y_obs
        init_theta: float
            initial theta guess

        Returns
        -------

        """
        thetas = np.zeros(self.max_iter)
        thetas[0] = init_theta
        log_likelihoods = np.zeros(self.max_iter)
        log_likelihoods[0] = self.calculate_log_likelihood(thetas[0], y_obs)
        return thetas, log_likelihoods

    @staticmethod
    def calculate_log_likelihood(theta, y_obs):
        """

        Parameters
        ----------
        y_obs
        theta: float
            theta to evaluate log like at

        Returns
        -------

        """
        return np.sum(np.log(1 / np.pi * 1 / (1 + (y_obs - theta) ** 2)))

    @staticmethod
    def calculate_derivative_log_likelihood(theta, y_obs):
        """

        Parameters
        ----------
        y_obs
        theta: float
            theta to evaluate d/d theta log like at

        Returns
        -------

        """
        return np.sum(2 * (y_obs - theta) / (1 + (theta - y_obs) ** 2))

    def check_convergence(self, it, log_likelihoods):
        """

        Parameters
        ----------
        it: int
            iteration number
        log_likelihoods: array like
            array of thetas


        Returns
        -------

        """
        return np.abs((log_likelihoods[it + 1] - log_likelihoods[it]) / log_likelihoods[it]) < self.tol

    def fit_one_initial_point(self, init_theta):
        """

        Parameters
        ----------
        init_theta: float
            initial theta guess

        Returns
        -------

        """
        thetas, log_likelihoods = self.initialize_params(init_theta, self.y_obs)
        for it in range(self.max_iter - 1):
            thetas[it + 1] = thetas[it] + self.lr * self.calculate_derivative_log_likelihood(thetas[it], self.y_obs)
            log_likelihoods[it + 1] = self.calculate_log_likelihood(thetas[it + 1], self.y_obs)
            if self.check_convergence(it, log_likelihoods):
                return thetas[it + 1], log_likelihoods[it + 1], it
        warnings.warn(f'not converged for init point {init_theta}')
        return thetas[-1], log_likelihoods[-1], self.max_iter

    def fit(self):
        """

        Returns
        -------
        MLE for theta

        """
        optimal_thetas = np.zeros(len(self.init_points))
        maximum_likelihoods = np.zeros(len(self.init_points))
        n_iter = np.zeros(len(self.init_points))
        for n, point in enumerate(self.init_points):
            optimal_thetas[n], maximum_likelihoods[n], n_iter[n] = self.fit_one_initial_point(point)
        self.optimal_theta = optimal_thetas[np.nanargmax(maximum_likelihoods)]
        self.it_to_convergence = n_iter[np.nanargmax(maximum_likelihoods)]
        return {'optimal theta': self.optimal_theta, 'n_iter': self.it_to_convergence}


class StochasticGradientAscent(GradientAscent):
    """
        inherits gradient ascent and does sga
    """

    def fit_one_initial_point(self, init_theta):
        """

        Parameters
        ----------
        init_theta: float
            initial theta guess

        Returns
        -------

        """
        thetas, log_likelihoods = self.initialize_params(init_theta, self.y_obs)
        for it in range(self.max_iter - 1):
            selected_obs = np.random.choice(a=self.y_obs, size=1, replace=False)
            thetas[it + 1] = thetas[it] + self.lr * self.calculate_derivative_log_likelihood(thetas[it], selected_obs)
            log_likelihoods[it + 1] = self.calculate_log_likelihood(thetas[it + 1], self.y_obs)
            if self.check_convergence(it, log_likelihoods):
                return thetas[it + 1], log_likelihoods[it + 1], it
        warnings.warn(f'not converged for init point {init_theta}')
        return thetas[-1], log_likelihoods[-1], self.max_iter


class NewtonMethod(GradientAscent):
    """
    Newton's method
    """

    def fit_one_initial_point(self, init_theta):
        """

        Parameters
        ----------
        init_theta: float
            initial theta guess

        Returns
        -------

        """
        thetas, log_likelihoods = self.initialize_params(init_theta, self.y_obs)
        for it in range(self.max_iter - 1):
            thetas[it + 1] = thetas[it] - 1 / self.calculate_hessian(thetas[it], self.y_obs) * self.calculate_derivative_log_likelihood(thetas[it], self.y_obs)
            log_likelihoods[it + 1] = self.calculate_log_likelihood(thetas[it + 1], self.y_obs)
            if self.check_convergence(it, log_likelihoods):
                return thetas[it + 1], log_likelihoods[it + 1], it
        #warnings.warn(f'not converged for init point {init_theta}')
        return thetas[-1], log_likelihoods[-1], self.max_iter

    @staticmethod
    def calculate_hessian(theta, y_obs):
        """

        Parameters
        ----------
        theta
        y_obs

        Returns
        -------

        """
        hessian = np.sum(2*((theta-y_obs)**2 - 1) / (1 + (theta-y_obs)**2)**2)
        return hessian


if __name__ == '__main__':
    observed_data = np.array(
        [7.52, 9.92, 9.52, 21.97, 8.39, 8.09, 9.22, 9.37, 7.33, 15.32, 1.08, 8.51, 17.73, 11.20, 8.33, 10.83, 12.40,
         14.49, 9.44, 3.67])

    ga = GradientAscent(y_obs=observed_data)
    ga.fit()
    sga = StochasticGradientAscent(y_obs=observed_data)
    sga.fit()
    nt = NewtonMethod(y_obs=observed_data)
    nt.fit()
    print('Gradient Ascent iter :', ga.it_to_convergence)
    print('Stochastic gradient Ascent iter :', sga.it_to_convergence)
    print('Newton Ascent iter :', nt.it_to_convergence)
    pass
