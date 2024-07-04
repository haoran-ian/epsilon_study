from time import time
import ioh
import copy
from scipy.integrate import quad
from correction_handler import *
from monitoring_population import *
from sklearn.neighbors import KernelDensity
import numpy as np
from scipy.stats import beta
from scipy.special import kl_div
from scipy.stats import entropy
from scipy.stats import uniform
import pandas as pd


def time_checker(method):
    def timed(*args, **kw):
        ts = time()
        result = method(*args, **kw)
        te = time()
        ms = round(te - ts)
        m, s = divmod(ms, 60)
        h, m = divmod(m, 60)
        print(f'Execution time: {h} h {m} m {s} s')
        return result

    return timed


class ContinuousOptimization:
    def __init__(self, dim, fct_name, inst): #, near_bounds, index_params):
        """Constructor for the ContinuousOptimization class.
            Args:
                dim (int): The dimensionality of the optimization problem.
                fct_name (str): The name of the evaluation function to be used.
        """
        self.dim = dim
        self.eval_fct = fct_name
        self.inst=inst
        self.lb = np.array([-5.] * dim)
        self.ub = np.array([5.] * dim)

        self.f = ioh.get_problem(self.eval_fct, instance=inst, dimension=dim, problem_class=ioh.ProblemClass.BBOB)
        # self.f = ioh.get_problem(self.eval_fct, instance=inst, dimension=dim, problem_class=ioh.ProblemClass.SBOX)
        # print("Global optimum clasic pos:", fct_name, "=", self.f.optimum.x)
        # print("Global optimum clasic val:", fct_name, "=", self.f.optimum.y)


    def set_bounds(self, lb, ub):
        if len(lb) != self.dim or len(ub) != self.dim:
            raise ValueError("Bounds must match the dimensionality of the problem.")
        self.lb = np.array(lb)
        self.ub = np.array(ub)

    def generate_random_elem(self):
        """Generates a random element within the bounds of the optimization domain.
           Returns:
               numpy.ndarray: A random element within the domain.
        """
        #return self.f.bounds.lb + np.random.random(self.dim) * (self.f.bounds.ub - self.f.bounds.lb)
        return np.array(self.lb) + np.random.random(self.dim) * (self.ub-self.lb)

    def eval(self, element):
        """Evaluates the given element using the specified evaluation function.
            Args:
                element (numpy.ndarray): The element from the population.
            Returns:
                float: The fitness value.
        """
        return self.f(element)
        #  return getattr(self, self.eval_fct)(element)


class Element:
    def __init__(self, x):
        """Constructor for the Element class.
            Args:
                x (numpy.ndarray): This element represents an individual in the population.
        """
        self.x = x
        self.cost = 0
        self.repaired = [False] * len(x)  # Correction indicator for each component


    def copy(self):
        """Creates a copy of this element.
            Returns:
                Element: A new Element object with the same values.
        """
        new_el = Element(self.x.copy())
        new_el.cost = self.cost
        new_el.repaired = self.repaired.copy()
        return new_el

    def update(self, el, cost, repaired):
        """Updates the values of this element with the values from another element.
            Args:
                el (numpy.ndarray): The new values for x.
                cost (float): The new cost value.
                repaired (list): The new repaired value.
        """
        self.x = el.copy()
        self.cost = cost
        self.repaired = repaired.copy()

    def __repr__(self):
        """Returns:
            str: A string representation of this element.
        """
        return "cost: {} element: {}".format(self.cost, self.x)


class OptimizationAlgorithm:
    def __init__(self, problem, probabilities_update_strategy, learning_period, corr_type, corr_method, adaptive, candidateBCHM):
        """Initializes the OptimizationAlgorithm instance.
            Args:
            problem (ContinuousOptimization): The optimization problem to be solved.
            probabilities_update_strategy (int): Strategy for updating the distribution probability for
            adaptive correction operator
            learning_period (int): Update the distribution probability every LP generations
        """
        self.problem = problem
        self.population = []
        self.nr_infeasible = 0  # Number of infeasible components
        self.nr_mutated = 0  # Number of mutated elements
        self.correction_handler = CorrectionHandler(problem)
        self.generation_counter = 0  # Generation counter
        self.probabilities_update_strategy = probabilities_update_strategy
        self.learning_period = learning_period
        self.learning_period_min = 50
        # Adaptive correction operator initialization
        # self.bchms = [name for name in globals() if
        #               name.startswith("METHOD_")]  # Correction names instead of indexes for explicity
        self.bchms = candidateBCHM
        # Initial use a uniform probability distribution for selecting corrections
        self.probabilities = np.ones(len(self.bchms)) / len(self.bchms)
        # Scores for successful and unsuccessful uses
        self.successes = np.zeros(len(self.bchms))
        self.failures = np.zeros(len(self.bchms))
        self.alfa = 0.5  # Parameter used in the adjustment of correction distribution
        self.eps = 0.01  # Smoothing factor for the distribution of corrections
        self.epsilon = 1e-10  # Prevent zero division for violation probability
        self.epsVar = 10 ** (-15)  # Variance error
        self.aBeta = [1] * problem.dim  # First parameter of beta distribution
        self.bBeta = [1] * problem.dim  # Second parameter of beta distribution
        self.gamma = -1  # No vectorial correction used
        # Correction initialization
        self.corr_type = corr_type
        self.corr_method = corr_method
        self.adaptive = adaptive
        self.BBlower = self.problem.lb
        self.BBupper = self.problem.ub
        self.lb = self.problem.lb
        self.ub = self.problem.ub
        # self.lb = np.ones(20)*(-4)
        # self.ub = np.ones(20)*(4)


    def initialize_population(self, pop_size):
        """Initializes the population with random elements.
          Args:
              pop_size (int): The size of the population.
          """
        self.population = [Element(self.problem.generate_random_elem()) for _ in range(pop_size)]

    def evaluate_population(self):
        """Evaluates the population using the evaluation function defined in the ContinuousOptimization class."""
        for element in self.population:
            element.cost = self.problem.eval(element.x)

    def non_adaptive_correction_method(self):
        # if self.corr_type == "component":
        #     if self.corr_method is None:
        #         raise ValueError("For component correction, a correction method index must be selected!")
        #     return self.corr_method
        # elif self.corr_type in [METHOD_VECTOR_TARGET, METHOD_VECTOR_BEST, METHOD_VECTOR_R, METHOD_MAHALANOBIS]:
        #     return {'vectR': 9, 'vectT': 10, 'vectB': 11}[self.corr_type]
        # else:
        #     return 100  # mahalanobis
        return self.corr_method

    def select_correction_method(self):
        """Proportional selection """
        return np.random.choice(len(self.bchms), p=self.probabilities)

    def update_probability_distribution(self):
        if self.probabilities_update_strategy == 0:
            self.linear_combination_update()
        elif self.probabilities_update_strategy == 1:
            self.beta_update()
        else:
            pass

    def linear_combination_update(self):
        """Linear combination between anterior probability distribution and current distribution based on scores"""

        # Success score of corrections is based on their success rate relative to their total usage
        self.realtive_success_score = self.successes / (self.successes + self.failures + self.eps)
        # if self.realtive_success_score.all():
        #     self.realtive_success_score = self.eps
        self.total_score = np.sum(self.realtive_success_score) + len(self.bchms) * self.eps

        # Update the probabilities
        self.probabilities = self.alfa * self.probabilities + (1 - self.alfa) * (
                    self.realtive_success_score + self.eps) / self.total_score
        # Normalize the probabilities
        self.probabilities = self.probabilities / self.probabilities.sum()

    def beta_update(self):
        # Implementarea strategiei cu model beta
        alfa = np.array(self.successes)
        beta = np.array(self.failures)
        new_distribution = np.random.beta(alfa + 1, beta + 1)
        #print('alfa', alfa)
        self.probabilities = new_distribution / new_distribution.sum()

    def update_correction_method_scores(self, method, success):
        if success:
            self.successes[method] += 1
        else:
            self.failures[method] += 1

    def correct_component(self, method, component, target, best, R, aBeta, bBeta, lower, upper, population_mean):
        """Returns: component, repaired"""
        return self.correction_handler.correction_component(
            method, lower, upper,
            component, target, best, R, aBeta, bBeta, population_mean)

    def correct_vector(self, vector, R, lower, upper):
        """Returns: repaired_vector, repaired, gamma"""
        return self.correction_handler.correction_vectorial(lower, upper, vector, R)

    def correct_mahalanobis_vectorial(self, trial, cov, lower, upper):
        return self.correction_handler.correction_mahalanobis_vectorial(lower, upper, trial, cov)

    def bounding_box(self):
        min_vals = np.full(self.problem.dim, self.ub)
        max_vals = np.full(self.problem.dim, self.lb)

        for el in self.population:
            min_vals = np.minimum(min_vals, el.x)
            max_vals = np.maximum(max_vals, el.x)

        self.BBlower = min_vals
        self.BBupper = max_vals

    def bounding_box_ratios(self):
        self.bounding_box()
        BBmin, BBmax = self.BBlower, self.BBupper
        ratios_min = np.zeros(self.problem.dim)
        ratios_max = np.zeros(self.problem.dim)
        for i in range(self.problem.dim):
            if BBmin[i] != self.lb[i]:
                ratios_min[i] = (BBmax[i] - BBmin[i]) / (BBmin[i] - self.lb[i])
            else:
                ratios_min[i] = np.inf

            if BBmax[i] != self.ub[i]:
                ratios_max[i] = (BBmax[i] - BBmin[i]) / (self.ub[i] - BBmax[i])
            else:
                ratios_max[i] = np.inf

        return ratios_min, ratios_max

    def bounding_box_ratios_closeness(self):
        self.bounding_box()
        BBmin, BBmax = self.BBlower, self.BBupper
        ratios_min = np.zeros(self.problem.dim)
        ratios_max = np.zeros(self.problem.dim)
        out_lower = []  # Lista pentru a stoca care latura a 'cutiei fezabile' este mai apropiata pentru fiecare dimensiune
        out_upper = []

        for i in range(self.problem.dim):
            if BBmin[i] != self.lb[i]:
                ratios_min[i] = (BBmax[i] - BBmin[i]) / (BBmin[i] - self.lb[i])
            else:
                ratios_min[i] = np.inf

            if BBmax[i] != self.ub[i]:
                ratios_max[i] = (BBmax[i] - BBmin[i]) / (self.ub[i] - BBmax[i])
            else:
                ratios_max[i] = np.inf

            # Calculeaza distantele de la BBlower si BBupper la limitele problemelor lb si ub
            distance_to_lower_bound = BBmin[i] - self.lb[i]
            distance_to_upper_bound = self.ub[i] - BBmax[i]

            # Determina care latura este mai apropiata
            if distance_to_lower_bound < distance_to_upper_bound:
                out_lower.append((distance_to_lower_bound, 'Lower'))
            else:
                out_upper.append((distance_to_upper_bound, 'Upper'))

        return (out_lower, out_upper)

    def bounding_box_proportion(self):
        self.bounding_box()
        BBmin, BBmax = self.BBlower, self.BBupper
        ratio = np.zeros(self.problem.dim)
        for i in range(self.problem.dim):
            ratio[i] = (BBmax[i] - BBmin[i]) / (self.ub[i] - self.lb[i])
        return ratio

    def compute_extension(self):  # use sqrt
        return (np.prod((self.BBupper - self.BBlower) / (self.ub - self.lb)))**(1/self.problem.dim)

    def compute_density(self):  # move if before division; extract sqrt from volume
        volume = np.prod(self.BBupper - self.BBlower)
        return len(self.population) / np.sqrt(volume) if volume != 0 else 0

    def compute_shape(self):
        extension = self.BBupper - self.BBlower
        return np.min(extension) / np.max(extension)

    def compute_eccentricity(self):  # imparte la diagonala
        # Compute the centroid of the population
        centroid = np.mean([el.x for el in self.population], axis=0)

        # Compute the middle point of the search space
        middle = (self.BBupper + self.BBlower) / 2

        # Compute the distance from the centroid to the middle
        distance = np.linalg.norm(centroid - middle)

        # Compute the length of the diagonal of the search space
        diagonal_length = np.linalg.norm(self.BBupper - self.BBlower)

        # Normalize the distance by the length of the diagonal
        eccentricity = distance / diagonal_length

        return eccentricity

    def compute_uniformity(self, m):  # m=popSize
        uniformity = np.zeros(self.problem.dim)
        for j in range(self.problem.dim):
            segment_length = (self.BBupper[j] - self.BBlower[j]) / m
            segment_counts = np.zeros(m)
            for i in range(m):
                lower_bound = self.BBlower[j] + i * segment_length
                upper_bound = lower_bound + segment_length
                segment_counts[i] = sum(lower_bound <= el.x[j] < upper_bound for el in self.population)
            p_ij = segment_counts / len(self.population)
            # Avoid division by zero and log(0)
            p_ij = p_ij[p_ij > 0]
            uniformity[j] = -np.sum(p_ij * np.log(p_ij))  # if p=0 do not compute log(0) set 0
        return uniformity


    def kl_divergence_kde_beta(self, population, mean_vector, variance_vector, bandwidth=0.1): #do not call function for var <10^-10
        # Extracting the numerical data from the population of Element objects
        # Assuming each Element's 'x' attribute is the numerical data of interest
        numeric_population_total = np.array([el.x for el in population])
        kl_div_vector = np.zeros(len(mean_vector))
        kl_div_unif = np.zeros(len(mean_vector))
        for j in range(len(mean_vector)):
            numeric_population = numeric_population_total[:, j]
            # Normalizarea populației numerice în [0, 1]
            min_val = np.min(numeric_population)
            max_val = np.max(numeric_population)
            if (max_val > min_val):
                normalized_population = (numeric_population - min_val) / (max_val - min_val)
                normalized_population = normalized_population.reshape(-1, 1)  # Transform for KDE
                mean_vector[j] = (mean_vector[j] - min_val) / (max_val - min_val)
                variance_vector[j] = variance_vector[j] / ((max_val - min_val) ** 2)
                # Estimarea KDE
                kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
                kde.fit(normalized_population)
                # Puncte de evaluare pentru KDE și distribuția Beta
                x_grid = np.linspace(0.0001, 0.9999, 1000)[:, np.newaxis]
                log_pdf_kde = kde.score_samples(x_grid)
                pdf_kde = np.exp(log_pdf_kde)  # Densitatea estimată KDE
                # Normalizare mean_vector și variance_vector
                mean_vector = (mean_vector - self.lb) / (self.ub - self.lb)
                variance_vector = variance_vector / ((self.ub - self.lb) ** 2)
                alpha_j = mean_vector[j] * (mean_vector[j] - mean_vector[j] * mean_vector[j] - variance_vector[j]) / \
                          variance_vector[j]
                beta_j = ((1 - mean_vector[j]) / mean_vector[j]) * alpha_j
                # print("alpha_j:", alpha_j, "beta_j:", beta_j)
                # print("lower:", self.lb, "upper:", self.ub)
                # print("mean:", mean_vector[j], "variance:", variance_vector[j])
                pdf_beta = beta.pdf(x_grid.ravel(), alpha_j, beta_j)
                pdf_unif = np.ones_like(x_grid).ravel()
                epsilon = 1e-10  # O valoare mică, pozitivă
                # Normalizare pentru a asigura că sumele sunt egale cu 1, epsilon pentru a evita log(0)
                pdf_kde = (pdf_kde / np.sum(pdf_kde)) + epsilon
                pdf_beta = (pdf_beta / np.sum(pdf_beta)) + epsilon
                pdf_unif = (pdf_unif / np.sum(pdf_unif)) + epsilon
                # Calculul divergenței KL
                # kl_div_value = np.sum(kl_div(pdf_kde, pdf_beta))
                kl_div_vector[j] = entropy(pdf_kde, pdf_beta)
                kl_div_unif[j] = entropy(pdf_kde, pdf_unif)
            else:
                kl_div_vector[j] = np.nan
                kl_div_unif[j] = np.nan
        return kl_div_vector, kl_div_unif  # kl_div_value

        # # Normalizare pentru a asigura că sumele sunt egale cu 1
        # pdf_kde /= np.sum(pdf_kde)
        # pdf_beta /= np.sum(pdf_beta)
        #
        # # Calculul divergenței KL
        # kl_div_value = np.sum(kl_div(pdf_kde, pdf_beta))

        return kl_div_value

    def compute_distance_to_optimum(self):
        distances = []
        for i in range(self.problem.dim):
            if self.problem.f.optimum.x[i] < self.BBlower[i]:
                distance = self.BBlower[i] - self.problem.f.optimum.x[i]
            elif self.problem.f.optimum.x[i] > self.BBupper[i]:
                distance = self.problem.f.optimum.x[i] - self.BBupper[i]
            else:
                distance = 0  # Optimum is inside the bounding box
            distances.append(distance)
        return np.linalg.norm(distances)

    def optimize(self):
        """Placeholder method for optimization. Should be implemented by subclasses."""
        raise NotImplementedError("The subclasses must implement optimize method!")

class AdaptationOfParameters(OptimizationAlgorithm):
    def __init__(self, problem, pop_size, sizeH, NFE_max, corr_type, corr_method,
                 adaptive, candidateBCHM, probabilities_update_strategy, learning_period, N_min=None):
        super().__init__(problem, probabilities_update_strategy, learning_period, corr_type, corr_method, adaptive, candidateBCHM)
        self.dim = problem.dim
        self.initialize_population(pop_size)
        self.evaluate_population()
        self.measures = MonitoringPopulation(problem, self.population)
        self.global_best = min(self.population, key=lambda element: element.cost)
        self.crossover_points = np.zeros(self.dim, dtype=bool)
        self.repaired = [False] * self.dim
        # Archives for success factors initialization
        self.sizeH = sizeH
        self.NFE_max = NFE_max
        self.memF = [0.5] * sizeH  # initial archive for F
        self.memCR = [0.5] * sizeH  # initial archive for CR
        self.k = 1  # index for success control parameters values actualization
        # Adaptive population size initializations
        self.N_min = N_min  # smallest population size
        self.Ninit = 18 * self.dim  # initial population size
        self.pop_size = self.Ninit
        self.pmin = 2 / self.pop_size
        # Memory initialization
        self.memA = copy.deepcopy(self.population)
        self.pop_size = pop_size
        self.avgF = 0
        self.stdF = 0
        self.avgCR = 0
        self.stdCR = 0
        self.adaptive = adaptive
        self.candidateBCHM=candidateBCHM

    def generate_F_CR_p_values(self, sizeH, memF, memCR, pmin):
        # sizeH  - size of the archive used for F and CR
        # memF - archive for F
        # memCR - archive for CR
        # pmin - minimum proportion of population
        idxH = np.random.randint(0, sizeH)  # random index in the archives
        muF = memF[idxH]  # mean of the Cauchy distribution for F
        muCR = memCR[idxH]  # mean of the Gaussian distribution for CR
        sd = 0.1  # standard deviations of the distributions used to generate F and CR
        # Fi = np.random.normal(muF, sd)
        Fi = np.random.standard_cauchy() * sd + muF
        while Fi <= 0:
            # Fi = np.random.normal(muF, sd)
            Fi = np.random.standard_cauchy() * sd + muF
        if Fi > 1:
            Fi = 1
        CRi = np.random.normal(muCR, sd)
        CRi = np.clip(CRi, 0, 1)
        pi = np.random.rand() * (0.2 - pmin) + pmin
        return Fi, CRi, pi

    def update_mem_F_CR(self, SF, SCR, improvements):
        total = np.sum(improvements)
        # Total might be 0 when selection accepts elements with the same fitness value

        if (total > 0):
            weights = improvements / total
        else:
            weights = np.array([1 / len(SF)] * len(SF))
        Fnew = np.sum(weights * SF * SF) / np.sum(weights * SF)  # Lehmer mean
        Fnew = np.clip(Fnew, 0, 1)
        CRnew = np.sum(weights * SCR)  # weighted mean
        CRnew = np.clip(CRnew, 0, 1)
        return Fnew, CRnew

    def limit_memory(self, memory, memorySize):
        """
        Limit the memory to  the memorySize by removing randomly selected elements
        """
        if len(memory) > memorySize:
            indexes = np.random.permutation(len(memory))[:memorySize]
            # memory = memory[indexes]
            return [memory[index] for index in indexes]
        else:
            return memory

    def update_F_CR_archives(self, SCR, SF, improvements):
        # Update MemF and MemCR
        if len(SCR) > 0 and len(SF) > 0:  # at least one successful trial vector
            Fnew, CRnew = self.update_mem_F_CR(SF, SCR, improvements)
            self.memF[self.k] = Fnew
            self.memCR[self.k] = CRnew
            self.k = (self.k + 1) % self.sizeH  # limit the memory - old values are overwritten
            self.avgF = np.mean(self.memF)
            self.avgCR = np.mean(self.memCR)
            self.stdF = np.std(self.memF)
            self.stdCR = np.std(self.memCR)

    def mutation(self, el):
        Fi, CRi, p = self.generate_F_CR_p_values(self.sizeH, self.memF, self.memCR,
                                                 self.pmin)  # for each trial vector a new F, CR and p value is generate
        maxbest = int(p * len(self.population))
        idx_pBest = np.random.randint(low=0, high=maxbest + 1)
        # sorted_population = sorted(self.population, key=lambda element: element.cost)
        # pbest = sorted_population[idx_pBest]
        pbest = self.population[idx_pBest]
        r1, r2 = -1, -1  # Initialize to invalid values

        while True:
            r1 = np.random.randint(low=0, high=len(self.population))
            r2 = np.random.randint(low=0, high=len(self.memA))

            # Break out of loop if both r1 and r2 are different and
            # the x values of the selected elements are also different
            if r1 != r2 and not np.all(self.population[r1].x == self.memA[r2].x):
                break

        # mutantVector = el.x + Fi * (pbest.x - el.x) + Fi * (self.population[r1].x - self.memA[r2].x)
        diff1= pbest.x - el.x
        diff2= self.population[r1].x - self.memA[r2].x
        mutantVector = el.x + Fi * diff1 + Fi * diff2

        return mutantVector, Fi, CRi, diff1, diff2

    def crossover(self, mutantVector, el, CR):
        self.crossover_points = np.random.rand(self.dim) < CR
        # Test if there is the possibility of performing crossover, if not, randomly create one
        if not np.any(self.crossover_points):
            self.crossover_points[np.random.randint(0, self.dim)] = True
        # If crossover_points=true, return the element from the mutant, otherwise,
        # return the element from the current population
        trialVector = np.where(self.crossover_points, mutantVector, el.x)
        self.nr_mutated += np.sum(self.crossover_points)
        return trialVector

    def generate_trial_vector(self, el, cov, population_mean):
        self.nr_infeasible = 0  # Number of infeasible components
        self.nr_mutated = 0  # Number of mutated elements
        mutantVector, F, CR, diff1, diff2 = self.mutation(el)
        trialVector = self.crossover(mutantVector, el, CR)
        idx_bchm=0

        if self.adaptive:
            idx_bchm = self.select_correction_method() #indice in lista selectat
            bchm = self.candidateBCHM[idx_bchm] # valoarea numerica asociata metodei in lista globala

        else:
            # bchm = self.non_adaptive_correction_method()
            bchm = self.corr_method

        if bchm == METHOD_VECTOR_R:
            trialVector, self.repaired, gamma = self.correct_vector(
                vector=trialVector,
                R=self.correction_handler.R,
                lower=self.lb,
                upper=self.ub
            )
            for i in range(self.dim):
                if self.repaired[i] is True:
                    self.nr_infeasible += 1

        elif bchm == METHOD_VECTOR_BEST:
            trialVector, self.repaired, gamma = self.correct_vector(
                vector=trialVector,
                R=self.global_best.x,
                lower=self.lb,
                upper=self.ub
            )
            for i in range(self.dim):
                if self.repaired[i] is True:
                    self.nr_infeasible += 1

        elif bchm == METHOD_VECTOR_TARGET:
            trialVector, self.repaired, gamma = self.correct_vector(
                vector=trialVector,
                R=el.x,
                lower=self.lb,
                upper=self.ub
            )
            for i in range(self.dim):
                if self.repaired[i] is True:
                    self.nr_infeasible += 1


        elif bchm == METHOD_MAHALANOBIS:
            trialVector, self.repaired = self.correct_mahalanobis_vectorial(
                trial=trialVector,
                cov=cov,
                lower=self.lb,
                upper=self.ub
            )
            for i in range(self.dim):
                if self.repaired[i] is True:
                    self.nr_infeasible += 1

        else:
            for i in range(self.dim):
                trialVector[i], self.repaired[i] = self.correct_component(
                    # method=self.corr_method,
                    method=bchm,
                    component=trialVector[i],
                    target=el.x[i],
                    best=self.global_best.x[i],
                    R=self.correction_handler.R[i],
                    aBeta=self.aBeta[i],
                    bBeta=self.bBeta[i],
                    lower=self.lb[i],
                    upper=self.ub[i],
                    population_mean=population_mean
                )
                if self.repaired[i] is True:
                    self.nr_infeasible += 1

        return trialVector, bchm, idx_bchm,F, CR, diff1, diff2