import numpy as np


class MonitoringPopulation:
    def __init__(self, problem, population):
        self.problem = problem
        self.dim = problem.dim
        self.population = population
        self.lb = self.problem.f.bounds.lb
        self.ub = self.problem.f.bounds.ub

    def variance(self, population):
        # Return: mean per component, var per component, averaged population variance
        s = np.sum([el.x for el in population], axis=0)
        s2 = np.sum([el.x * el.x for el in population], axis=0)
        var = s2 / len(population) - (s / len(population)) ** 2
        return s / len(population), var, var.mean()

    def covariance(self, population):
        mean = np.zeros(self.dim)
        for ind in range(self.dim):
            for el in population:
                mean[ind] += el.x[ind]
            mean[ind] /= len(population)
        cov = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(i, self.dim):
                for el in population:
                    cov[i, j] += (el.x[i] - mean[i]) * (el.x[j] - mean[j])
                cov[i, j] /= len(population)
                cov[j, i] = cov[i, j]  # use symmetry to fill the lower triangle
        return cov

    def hessian(self, trial, h=1e-5):
        hessian = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(i, self.dim):
                # Calculate second-order partial derivatives
                hessian[i, j] = hessian[j, i] = (self.problem.eval(trial + h * (i == j)) -
                                                 2 * self.problem.eval(trial) +
                                                 self.problem.eval(trial - h * (i == j))) / (h ** 2)

        return hessian

    def frobenius_norm(self, matrix1, matrix2):
        # Check if the matrices have the same shape
        if matrix1.shape != matrix2.shape:
            raise ValueError("Matrices must have the same shape for Frobenius norm calculation.")

        # Initialize the sum of squared differences to 0
        sum_of_squared_differences = 0

        # Iterate over each element in both matrices
        for i in range(matrix1.shape[0]):
            for j in range(matrix1.shape[1]):
                diff = matrix1[i, j] - matrix2[i, j]
                sum_of_squared_differences += diff ** 2

        # Take the square root of the sum to compute the Frobenius norm
        frobenius_norm = np.sqrt(sum_of_squared_differences)

        return frobenius_norm

    def mahalanobis_distance(self, x, mu, inv_cov_matrix):
        diff = x - mu
        dist = np.sqrt(diff.T.dot(inv_cov_matrix).dot(diff))
        return dist












