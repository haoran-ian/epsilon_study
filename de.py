from core import OptimizationAlgorithm, Element
from core import time_checker
import json
from correction_handler import *
import random
import pandas as pd
from monitoring_population import *



class DifferentialEvolution(OptimizationAlgorithm):
    def __init__(self, problem, pop_size, F, CR, NFE_max, mut_scheme, crossover_strategy, corr_type, corr_method,
                 adaptive, probabilities_update_strategy, learning_period, run_number):
        """Initializes the DifferentialEvolution instance.
            Args:
                problem (ContinuousOptimization): The optimization problem to be solved.
                pop_size (int): The size of the population.
                F (float): The scaling factor.
                CR (float): The crossover rate.
                NFE_max (int): The maximum number of function evaluations.
                mut_scheme (str): Mutation scheme.
                crossover_strategy (str): Crossover type
                corr_type (str): Correction level application: component or vectorial
                corr_method (int): Index of BCHM as defined in corection_handler.py
                adaptive (bool): Determine if the adaptive correction operator is active
                probabilities_update_strategy (int) (OptimizationAlgorithm): Strategy for updating the
                distribution probability for correction operator
                learning_period (int) (OptimizationAlgorithm): Update the distribution probability every LP generations

        """
        super().__init__(problem, probabilities_update_strategy, learning_period, corr_type, corr_method, adaptive)
        self.initialize_population(pop_size)
        self.evaluate_population()
        self.measures = MonitoringPopulation(problem, self.population)
        self.global_best = min(self.population, key=lambda element: element.cost)
        self.F = F
        self.CR = CR
        self.dim = problem.dim
        self.NFE_max = NFE_max
        self.mut_scheme = mut_scheme
        self.crossover_strategy = crossover_strategy
        self.crossover_points = np.zeros(self.dim, dtype=bool)
        self.repaired = [False] * self.dim
        self.pop_size = pop_size
        self.run_number = run_number

    def generate_csv_filename(self):
        alg_name = "DE"
        func_name = f"f{self.problem.eval_fct}"
        dim = f"D{self.problem.dim}"
        pop_size = f"N{self.pop_size}"
        F = f"F{self.F}"
        CR = f"CR{self.CR}"
        # correction = f"{self.corr_type}"
        if self.adaptive != 0:
            prob_update_strategy = "adaptive_linear" if self.probabilities_update_strategy == 0 else "adaptive_beta"
            correction = ""
        else:
            prob_update_strategy = ""
            if self.corr_type == "mahalanobis" or "vectT" or "vectB" or "vectR":
                correction = f"{self.corr_type}"
            else:
                correction = f"{self.corr_method}"
        filename_parts = [alg_name, prob_update_strategy, correction, func_name, pop_size, dim, F, CR,
                          f"run{self.run_number}"]
        filename = "_".join(part for part in filename_parts if part)  # Exclude componentele goale
        filepath = f"D:\\DE\\D{self.dim}\\{filename}.csv"
        return filepath

    def generate_csv_BBfilename(self):
        alg_name = "DE"
        func_name = f"f{self.problem.eval_fct}"
        dim = f"D{self.problem.dim}"
        pop_size = f"N{self.pop_size}"
        F = f"F{self.F}"
        CR = f"CR{self.CR}"
        # correction = f"{self.corr_type}"
        if self.adaptive != 0:
            prob_update_strategy = "adaptive_linear" if self.probabilities_update_strategy == 0 else "adaptive_beta"
            correction = ""
        else:
            prob_update_strategy = ""
            if self.corr_type == "mahalanobis" or "vectT" or "vectB" or "vectR":
                correction = f"{self.corr_type}"
            else:
                correction = f"{self.corr_method}"
        filename_parts = [alg_name, prob_update_strategy, correction, func_name, pop_size, dim, F, CR,
                          f"run{self.run_number}"]
        filename = "_".join(part for part in filename_parts if part)  # Exclude componentele goale
        filepath = f"D:\\DE\\D{self.dim}\\{filename}_BB.csv"
        return filepath

    def mutation(self):
        mutantVector = np.zeros(self.dim)
        if self.mut_scheme == "rand1":
            r1, r2, r3 = random.sample(range(len(self.population)), 3)
            mutantVector = self.population[r1].x + self.F * (self.population[r2].x - self.population[r3].x)
        elif self.mut_scheme == "rand2":
            r1, r2, r3, r4, r5 = random.sample(range(len(self.population)), 5)
            mutantVector = (self.population[r1].x + self.F * (self.population[r2].x - self.population[r3].x) +
                            self.F * (self.population[r4].x - self.population[r5].x))
        elif self.mut_scheme == "best1":
            r1, r2 = random.sample(range(len(self.population)), 2)
            mutantVector = self.global_best.x + self.F * (self.population[r1].x - self.population[r2].x)

        return mutantVector

    def crossover(self, mutantVector, el):
        if self.crossover_strategy == 'bin':
            self.crossover_points = np.random.rand(self.dim) < self.CR
            # Test if there is the possibility of performing crossover, if not, randomly create one
            if not np.any(self.crossover_points):
                self.crossover_points[np.random.randint(0, self.dim)] = True
            # If crossover_points=true, return the element from the mutant, otherwise,
            # return the element from the current population
            trialVector = np.where(self.crossover_points, mutantVector, el.x)
        else:  # exponential crossover
            self.crossover_points = np.full(self.dim, False)
            trialVector = np.copy(el.x)
            k = np.random.randint(self.dim)
            l = 0
            while True:
                trialVector[k] = mutantVector[k]
                self.crossover_points[k] = True
                k = (k + 1) % self.dim
                l += 1
                if np.random.rand() >= self.CR or l >= self.dim:
                    break
        self.nr_mutated += np.sum(self.crossover_points)

        return trialVector

    def generate_trial_vector(self, el, cov):
        self.nr_infeasible = 0  # Number of infeasible components
        self.nr_mutated = 0  # Number of mutated elements
        mutantVector = self.mutation()
        trialVector = self.crossover(mutantVector, el)
        if self.adaptive:
            bchm = self.select_correction_method()
        else:
            bchm = self.non_adaptive_correction_method()

        if bchm == 2 or bchm == METHOD_VECTOR_BEST:
            trialVector, self.repaired, gamma = self.correct_vector(
                vector=trialVector,
                R=self.correction_handler.R,
                lower=self.lb,
                upper=self.ub
            )
            for i in range(self.dim):
                if self.repaired[i] is True:
                    self.nr_infeasible += 1
        # elif bchm==10 or bchm==METHOD_VECTOR_TARGET:
        #     trialVector, self.repaired, gamma = self.correct_vector(
        #         vector=trialVector,
        #         R=el.x,
        #         lower=self.lb,
        #         upper=self.ub
        #     )
        #     for i in range(self.dim):
        #         if self.repaired[i] is True:
        #             self.nr_infeasible += 1

        elif bchm == 4 or bchm == METHOD_MAHALANOBIS:
            trialVector, self.repaired = self.correct_mahalanobis_vectorial(
                trial=trialVector,
                cov=cov,
                lower=self.lb,
                upper=self.ub
            )
            for i in range(self.dim):
                if self.repaired[i] is True:
                    self.nr_infeasible += 1

        elif bchm == 0:
            bchm = METHOD_SATURATION

        elif bchm == 1:
            bchm = METHOD_EXPC_BEST

        elif bchm == 3:
            bchm = METHOD_BETA

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
                upper=self.ub[i]
            )
            if self.repaired[i] is True:
                self.nr_infeasible += 1

        return trialVector, bchm

    @time_checker
    def optimize(self):
        """Placeholder method for optimization specific to Differential Evolution."""
        # Implementation of DE
        correction_probabilities = []
        NFE = 0  # Number of Function Evaluation
        NFS = 0  # Number of successful trial vectors/generation
        totalInfeasibleComponent = 0
        totalMutatedComponent = 0
        totalInfeasibleElement = 0
        rez = np.zeros(shape=(self.NFE_max // len(self.population) + 1, 6))
        BB_metrics = []
        meanPopVector, varPopVector, varPop = self.measures.variance()
        covPop = self.measures.covariance()
        # mean and variance after linear transformation [lower, upper] -> [0,1]
        meanPopVector = (meanPopVector - self.lb) / (self.ub - self.lb)
        varPopVector = varPopVector / (
                (self.ub - self.lb) * (self.ub - self.lb))
        # Computation of the Beta parameters = mean and variance of current population
        for i in range(self.problem.dim):
            if varPopVector[i] < self.epsVar:
                varPopVector[i] = self.epsVar
            self.aBeta[i] = meanPopVector[i] * (
                    meanPopVector[i] - meanPopVector[i] * meanPopVector[i] - varPopVector[i]) / varPopVector[i]
            self.bBeta[i] = ((1 - meanPopVector[i]) / meanPopVector[i]) * self.aBeta[i]

        while NFE < self.NFE_max and self.problem.f.optimum.y - self.global_best.cost != 0:
            # Increment the generation counter
            self.generation_counter += 1
            bb = self.bounding_box()
            bb_ratios = self.bounding_box_ratios()
            density = self.compute_density()
            dist_to_opt = self.compute_distance_to_optimum()
            shape = self.compute_shape()
            uniformity = self.compute_uniformity(m=50)
            extension = self.compute_extension()
            eccentricity = self.compute_eccentricity()
            bchms_used_this_gen = []
            gammaList = []
            genInfeasibleElement = 0  # number of infeasible elements
            genInfeasibleComponent = 0  # number of infeasible components
            genMutatedComponent = 0  # number of mutated components

            for el in self.population:
                # Generate a trial vector
                trial_vector, selected_bchm = self.generate_trial_vector(el, covPop)
                bchms_used_this_gen.append(selected_bchm)
                if self.gamma != -1:
                    gammaList.append(self.gamma)
                genMutatedComponent = genMutatedComponent + self.nr_mutated
                if self.nr_infeasible > 0:
                    genInfeasibleComponent = genInfeasibleComponent + self.nr_infeasible
                    genInfeasibleElement = genInfeasibleElement + 1

                # Evaluate the fitness of the trial vector
                trial_fitness = self.problem.eval(trial_vector)
                NFE += 1  # Increment the number of function evaluations

                # Update scores of the correction method based on success
                is_success = trial_fitness < el.cost
                if self.adaptive:
                    self.update_correction_method_scores(selected_bchm, is_success)

                # If generation number is a multiple of LP it is time to update the probability distribution and reset scores
                if self.adaptive and self.generation_counter % self.learning_period == 0:
                    # Get the last n values of "Prob_infeas"
                    # last_n_prob_infeas_values = [entry["Prob_infeas"] for entry in correction_probabilities[-self.learning_period:]]
                    # Calculate the average
                    # average_prob_infeas = sum(last_n_prob_infeas_values) / self.learning_period if self.learning_period > 0 else 0
                    # self.learning_period = max(round(self.learning_period_min * np.log(average_prob_infeas+self.epsilon)), self.learning_period_min)
                    # print("Generation:",self.generation_counter)
                    # print("LP:",self.learning_period)
                    self.update_probability_distribution()
                    # Reset successes and failures for next period
                    self.successes.fill(0)
                    self.failures.fill(0)

                # Selection: Compare the fitness of the trial vector to the current individual
                # and select the better one for the next generation
                if is_success:
                    el.update(trial_vector, trial_fitness, self.repaired)
                    NFS += 1

            if NFS > 1:
                # Update the global best solution
                self.global_best = min(self.population, key=lambda element: element.cost)
                meanPopVector, varPopVector, varPop = self.measures.variance()
                covPop = self.measures.covariance()

            totalInfeasibleComponent = totalInfeasibleComponent + genInfeasibleComponent
            totalMutatedComponent = totalMutatedComponent + genMutatedComponent
            totalInfeasibleElement = totalInfeasibleElement + genInfeasibleElement

            try:
                # estimation of ViolationProbability*MutationProbability (is counted only for components selected by crossover)
                prob_infeas = genInfeasibleComponent / genMutatedComponent
            except ZeroDivisionError:
                prob_infeas = genInfeasibleComponent / (genMutatedComponent + self.epsilon)

            # print(f'Generation: {generation}, Best Cost: {self.global_best.cost}')
            if self.adaptive:
                correction_methods = [self.bchms[i][self.bchms[i].find('_') + 1:] for i in set(bchms_used_this_gen)]
                scores = {self.bchms[idx]: {"Successes": self.successes[idx], "Failures": self.failures[idx],
                                            "Probability": self.probabilities[idx]} for idx in range(len(self.bchms))}
                correction_probabilities.append({
                    "Generation": self.generation_counter,
                    "Correction Methods": correction_methods,
                    "Scores": scores,
                    "Prob_infeas": prob_infeas
                })
            print(
                f"Correction methods used this generation: {[method_name[method_name.find('_') + 1:] for method_name in [self.bchms[i] for i in set(bchms_used_this_gen)]]}")
            print(f"Correction scores after generation {self.generation_counter}:")
            for idx in range(len(self.bchms)):
                print(
                    f"{self.bchms[idx]}: Successes - {self.successes[idx]}, Failures - {self.failures[idx]}, Probability - {self.probabilities[idx]:.10f}")
            print("\n" + "=" * 50 + "\n")
            rez[self.generation_counter - 1] = [self.generation_counter, self.global_best.cost,
                                                self.problem.f.optimum.y - self.global_best.cost, prob_infeas,
                                                genInfeasibleElement, varPop]
            # BB_metrics[self.generation_counter-1] = np.array([extension, density, shape, eccentricity, uniformity, dist_to_opt])
            BB_metrics.append({
                'extension': extension,
                'density': density,
                'shape': shape,
                'eccentricity': eccentricity,
                'uniformity': uniformity,
                'dist_to_opt': dist_to_opt
            })

            dfBB = pd.DataFrame(BB_metrics)

            # Convert NumPy arrays to lists and then to JSON strings
            for column in dfBB.columns:
                dfBB[column] = dfBB[column].apply(lambda x: np.array(x).tolist()).apply(json.dumps)

            # save the array with results in a csv file
            df = pd.DataFrame(rez, columns=['it', 'best', 'error', 'prob_infeas', 'genInfeasibleElement', 'varPop'])
            csv_filename = self.generate_csv_filename()
            csv_BBfilename = self.generate_csv_BBfilename()
            df.to_csv(csv_filename, index=False)
            dfBB.to_csv(csv_BBfilename, index=False)

        # if self.adaptive:
        #     with open(f'DE_adaptive_CorrectionProbabilities_{self.problem.eval_fct}_{self.run_number}.csv', 'w', newline='') as file:
        #         writer = csv.DictWriter(file, fieldnames=["Generation", "Correction Methods", "Scores", "Prob_infeas"])
        #         writer.writeheader()
        #         for row in correction_probabilities:
        #             writer.writerow(row)
        return self.global_best