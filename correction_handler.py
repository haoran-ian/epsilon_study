import numpy as np
from scipy.stats import beta
from scipy.spatial import distance
from scipy.optimize import minimize



# Constant definitions
METHOD_SATURATION = 0
METHOD_MIDPOINT_TARGET = 1
METHOD_MIDPOINT_BEST = 2
METHOD_UNIF = 3
METHOD_BETA = 4
METHOD_MIRROR = 5
METHOD_TOROIDAL = 6
METHOD_EXPC_R = 7
METHOD_EXPC_TARGET = 8
METHOD_EXPC_BEST = 9
METHOD_VECTOR_R = 10
METHOD_VECTOR_TARGET = 11
METHOD_VECTOR_BEST = 12
METHOD_MAHALANOBIS = 13

NONE = -1

corr_names={METHOD_SATURATION : 'sat',
METHOD_MIDPOINT_TARGET : 'midT',
METHOD_MIDPOINT_BEST : 'midB',
METHOD_UNIF : 'unif',
METHOD_BETA : 'beta',
METHOD_MIRROR : 'mir',
METHOD_TOROIDAL : 'tor',
METHOD_EXPC_R : 'expC_R',
METHOD_EXPC_TARGET : 'expC_T',
METHOD_EXPC_BEST : 'expC_B',
METHOD_VECTOR_R : 'vectR',
METHOD_VECTOR_TARGET : 'vectT',
METHOD_VECTOR_BEST : 'vectB',
METHOD_MAHALANOBIS : 'mahalanobis'}


class CorrectionHandler:
    def __init__(self, problem):
        self.problem = problem
        # self.lb = self.problem.f.bounds.lb
        # self.ub = self.problem.f.bounds.ub
        self.lb = self.problem.lb
        self.ub = self.problem.ub
        self.R = 0.5 * (np.array(self.lb) + np.array(self.ub))
        self.aBeta = [1] * problem.dim
        self.bBeta = [1] * problem.dim
        self.gamma = -1

    def correction_component(self, method, lower, upper, component, target, best, R, aBeta, bBeta, population_mean):
        """Applies correction to a component based on the specified method if it is out of bounds.
            Args:
                method (str): The correction method.
                lower (float): The lower bound.
                upper (float): The upper bound.
                component (float): The component from element.
                target (float): The component from target.
                best (float): The component from the best.
                R (float): The component from the Reference point R.
                aBeta (float): First parameter from beta distribution.
                bBeta (float): Second parameter from beta distribution.
            Returns:
                tuple: Corrected component value and a boolean indicating if it was repaired.
                """
        #print(f"Method: {method}, Component: {component}, Lower: {lower}, Upper: {upper}")

        if component < lower:
            return self.correct_lower(method, lower, upper, component, target, best, R, aBeta, bBeta, population_mean)
        elif component > upper:
            return self.correct_upper(method, lower, upper, component, target, best, R, aBeta, bBeta, population_mean)
        else:
            return component, False

    def correct_lower(self, method, lower, upper,component, target, best, R, aBeta, bBeta, population_mean):
        """Applies correction to a component which exceed its lower bound."""
        repaired = True
        component_correction_methods = {
            METHOD_SATURATION: lambda: lower,
            METHOD_MIDPOINT_TARGET: lambda: (lower + target) / 2,
            METHOD_MIDPOINT_BEST: lambda: (lower + best)/2,
            METHOD_EXPC_R: lambda: lower - np.log(1 + np.random.uniform(0, 1) * (np.exp(lower - R)-1)),
            METHOD_EXPC_TARGET: lambda: lower - np.log(1 + np.random.uniform(0, 1) * (np.exp(lower - target)-1)),
            METHOD_EXPC_BEST: lambda: lower - np.log(1 + np.random.uniform(0, 1) * (np.exp(lower - best)-1)),
            METHOD_UNIF: lambda: np.random.uniform(lower,upper),
            METHOD_MIRROR: lambda: 2*lower-component,
            METHOD_TOROIDAL: lambda: upper-lower+component,
            METHOD_BETA: lambda: (lower + beta.rvs(aBeta, bBeta, size=1) * (
                        upper - lower)) if aBeta > 0 and bBeta > 0 else population_mean
        }
        # Defensive Lambda Execution
        #print(f"Infeasible component", {component})
        if aBeta <= 0 or bBeta <= 0:
            print(f"Parametrii Beta trebuie să fie pozitivi. aBeta: {aBeta}, bBeta: {bBeta}")
        component = component_correction_methods.get(method, lambda: component)()
        #print(f"Corrected component", {component})
        return component, repaired

    def correct_upper(self, method, lower, upper, component, target, best, R, aBeta, bBeta, population_mean):
        """Applies correction to a component which exceed its upper bound."""
        repaired = True
        component_correction_methods = {
            METHOD_SATURATION: lambda: upper,
            METHOD_MIDPOINT_TARGET: lambda: (upper + target) / 2,
            METHOD_MIDPOINT_BEST: lambda: (upper + best) / 2,
            METHOD_EXPC_R: lambda: upper + np.log(1 + (1 - np.random.uniform(0, 1)) * (np.exp(R - upper) - 1)),
            METHOD_EXPC_TARGET: lambda: upper + np.log(1 + (1 - np.random.uniform(0, 1)) * (np.exp(target - upper) - 1)),
            METHOD_EXPC_BEST: lambda: upper + np.log(1 + (1 - np.random.uniform(0, 1)) * (np.exp(best - upper) - 1)),
            METHOD_UNIF: lambda: np.random.uniform(lower, upper),
            METHOD_MIRROR: lambda: 2*upper-component,
            METHOD_TOROIDAL: lambda: lower-upper+component,
            METHOD_BETA: lambda: (lower + beta.rvs(aBeta, bBeta, size=1) * (
                    upper - lower)) if aBeta > 0 and bBeta > 0 else population_mean
        }

        if aBeta <= 0 or bBeta <= 0:
            print(f"Parametrii Beta trebuie să fie pozitivi. aBeta: {aBeta}, bBeta: {bBeta}")
        if lower >= upper:
            raise ValueError(
                f"Limita inferioară trebuie să fie mai mică decât limita superioară. Lower: {lower}, Upper: {upper}")
        component = component_correction_methods.get(method, lambda: component)()
        return component, repaired

    def minimize_mahalanobis_distance_vectorial(self, mutant, cov, lower, upper):
        """Find the element that minimizes the sum of squared Mahalanobis distances using optimization."""

        if np.linalg.det(cov) != 0:
            inv_cov = np.linalg.inv(cov)
        else:
            # Use the identity matrix if cov is not invertible
            inv_cov = np.identity(cov.shape[0])

        def objective_function(x):
            """Objective function to minimize the sum of squared Mahalanobis distances."""
            #modified_mutant = np.full_like(mutant, x)
            diff = x - mutant
            mahalanobis_dist = np.dot(diff.T, np.dot(inv_cov, diff)) #ce de intampla daca inlocuiesc cu matricea identitate?
            #mahalanobis_dist = np.dot(diff.T, np.dot(np.eye(self.problem.dim), diff)) #=>SATURATION
            return mahalanobis_dist

        bounds=[]
        for i in range(len(lower)):
            # Define bounds for the feasible element
            bounds.append((lower[i], upper[i]))

        # Initial guess for the feasible element (e.g., midpoint)
        initial_guess = (lower + upper) / 2

        # Minimize the objective function
        result = minimize(objective_function, initial_guess, bounds=bounds)

        # The optimal feasible element is the result.x
        optimal_feasible_element = result.x

        return optimal_feasible_element

    def correction_mahalanobis_vectorial(self, lower, upper, trial_vector, cov):
        repaired = [False] * len(trial_vector)
        for j in range(len(trial_vector)):
            if trial_vector[j] < lower[j] or trial_vector[j] > upper[j]:
                repaired[j] = True
        if any(repaired):
            # print("Infeasible:", trial_vector)
            trial_vector = self.minimize_mahalanobis_distance_vectorial(trial_vector, cov, lower, upper)
            # print("Corrected:", trial_vector)
        return trial_vector, repaired


    def correction_vectorial(self, lower, upper, trial_vector, R):
        """
        Applies correction on the entire vector, including the feasible components.
        Args:
            lower (float): The lower bound.
            upper (float): The upper bound.
            trial_vector (ndarray): The trial vector to be corrected.
            R (ndarray): The reference vector.
        Returns:
            tuple: A tuple containing the corrected vector, a boolean array indicating the repaired indices,
                   and the gamma value.
        """
        alpha = np.zeros(self.problem.dim)
        repaired = [False] * self.problem.dim
        eps0 = 10 ** (-8)
        for j in range(self.problem.dim):
            if trial_vector[j] < lower[j]:# -eps0:
                # print("lower:", lower[j])
                # print("R:", R[j])
                if R[j] == trial_vector[j]:
                    alpha[j] = 1
                else:
                    alpha[j] = round((R[j] - lower[j]) / (R[j] - trial_vector[j]), 10)
                repaired[j] = True
            elif trial_vector[j] > upper[j]:  # +eps0:
                # print("upper:", upper[j])
                # print("R:", R[j])
                if R[j] == trial_vector[j]:
                    alpha[j] = 1
                else:
                    alpha[j] = round((upper[j] - R[j]) / (trial_vector[j] - R[j]), 10)
                repaired[j] = True
            else:
                alpha[j] = 1
        self.gamma = alpha.min()

        if (self.gamma < 0) or (self.gamma > 1):
            print("!!! gamma=", self.gamma, " alpha=", alpha)
            if (self.gamma < 0):
                self.gamma = 0
            else:
                self.gamma = 1

        repaired_vector = self.gamma * trial_vector + (1 - self.gamma) * R  # repaired vector
        # for i in range(len(trial_vector)):
        #     repaired_vector[i] = round(self.gamma * trial_vector[i] + (1 - self.gamma) * R[i], 5)  # repaired vector
        #
        # if np.any(repaired):
        #     print('infeasible', trial_vector)
        #     print('repaired', repaired_vector)

        for j in range(self.problem.dim):
            if repaired_vector[j] < lower[j]:
                # print("projected on lower")
                repaired_vector[j] = lower[j]
            if repaired_vector[j] > upper[j]:
                # print("projected on upper")
                repaired_vector[j] = upper[j]



        return repaired_vector, repaired, self.gamma

    # def computation_beta_parameters(self):
    #     meanPopVector, varPopVector, varPop = self.measures.variance()
    #     # mean and variance after linear transformation [lower, upper] -> [0,1]
    #     meanPopVector = (meanPopVector - self.problem.lower) / (self.problem.upper - self.problem.lower)
    #     varPopVector = varPopVector / (
    #                 (self.problem.upper - self.problem.lower) * (self.problem.upper - self.problem.lower))
    #     # Computation of the Beta parameters = mean and variance of current population
    #     for i in range(self.problem.dim):
    #         if varPopVector[i] < self.epsVar:
    #             varPopVector[i] = self.epsVar
    #         self.aBeta[i] = meanPopVector[i] * (
    #                 meanPopVector[i] - meanPopVector[i] * meanPopVector[i] - varPopVector[i]) / varPopVector[i]
    #         self.bBeta[i] = ((1 - meanPopVector[i]) / meanPopVector[i]) * self.aBeta[i]
