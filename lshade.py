from core import AdaptationOfParameters, Element
from core import time_checker
import json
import pandas as pd
from correction_handler import *
import pickle


class LSHADE(AdaptationOfParameters):
    def __init__(self, problem, pop_size, sizeH, NFE_max, N_min, corr_type, corr_method,
                 adaptive, candidateBCHM, probabilities_update_strategy, learning_period, run_number, path, epsilon):
        super().__init__(problem, pop_size, sizeH, NFE_max, corr_type, corr_method,
                         adaptive, candidateBCHM, probabilities_update_strategy, learning_period, N_min)
        self.run_number = run_number
        self.path=path
        self.epsilon=epsilon
        # self.f_opt = self.problem.get_f_opt()

    def generate_csv_filename(self):
        alg_name = "LSHADE"
        func_name = f"f{self.problem.eval_fct}"
        dim = f"D{self.problem.dim}"
        alfa=""
        if self.adaptive != 0:
            prob_update_strategy = "adaptive_linear" if self.probabilities_update_strategy == 0 else "adaptive_beta"
            alfa = f"{self.alfa}" if self.probabilities_update_strategy == 0 else ""
            correction = ""
        else:
            prob_update_strategy = ""
            correction = corr_names[self.corr_method]

        filename_parts = [alg_name, prob_update_strategy, alfa, correction, func_name, dim, f"eps{self.epsilon}"
                          f"run{self.run_number}"]
        filename = "_".join(part for part in filename_parts if part)  # Exclude componentele goale
        filepath = f"{self.path}/{filename}_gen.csv"
        return filepath

    @time_checker
    def optimize(self):
        correction_probabilities = []  # Collecting probabilities for BCHM adaptive over generations
        NFE = 0  # Number of Function Evaluation
        NFS = 0  # Number of successful trial vectors/generation
        totalInfeasibleComponent = 0
        totalMutatedComponent = 0
        totalInfeasibleElement = 0
        rez = np.zeros(shape=(self.NFE_max // self.N_min+2000, 15))
        
        BB_metrics = []
        diff_strs = []
        meanPopVector, varPopVector, varPop = self.measures.variance(self.population)
        # mean and variance after linear transformation [lower, upper] -> [0,1]
        meanPopVector = (meanPopVector - self.lb) / (self.ub - self.lb)
        varPopVector = varPopVector / (
                (self.ub - self.lb) * (self.ub - self.lb))
        covPop = self.measures.covariance(self.population)

        # Computation of the Beta parameters = mean and variance of current population
        for i in range(self.problem.dim):
            if varPopVector[i] < self.epsVar:
                varPopVector[i] = self.epsVar
            self.aBeta[i] = meanPopVector[i] * (
                    meanPopVector[i] - meanPopVector[i] * meanPopVector[i] - varPopVector[i]) / varPopVector[i]
            self.bBeta[i] = ((1 - meanPopVector[i]) / meanPopVector[i]) * self.aBeta[i]

        #while (NFE < self.NFE_max):
        while (NFE < self.NFE_max) and (self.global_best.cost - self.problem.f.optimum.y >= 10**(-8)) and varPop > 10**(-10):
        # while ((self.ub-self.BBupper)<(self.BBupper-self.BBlower)).all() and ((self.BBlower-self.lb)<(self.BBupper-self.BBlower)).all():
            # Increment the generation counter
            difference1 = []
            difference2 = []
            self.generation_counter += 1
            self.bounding_box()
            bb_ratios = self.bounding_box_ratios()
            closeness = self.bounding_box_ratios_closeness()
            density = self.compute_density()
            dist_to_opt = self.compute_distance_to_optimum()
            shape = self.compute_shape()
            uniformity = self.compute_uniformity(m=self.pop_size)
            extension = self.compute_extension()
            eccentricity = self.compute_eccentricity()
            kl_beta_vector, kl_unif_vector = self.kl_divergence_kde_beta(self.population, meanPopVector, varPopVector)
            kl_beta = np.nanmean(kl_beta_vector)
            kl_unif = np.nanmean(kl_unif_vector)
            if (self.generation_counter % 100 == 0):
                print("KL div beta (mean,std):", np.nanmean(kl_beta_vector), " ",
                      np.nanstd(kl_beta_vector))  # ignore NaN when computing mean and std
                print("KL div unif (mean,std):", np.nanmean(kl_unif_vector), " ", np.nanstd(kl_unif_vector))
            bchms_used_this_gen = []
            gammaList = []
            genInfeasibleElement = 0  # number of infeasible elements
            genInfeasibleComponent = 0  # number of infeasible components
            genMutatedComponent = 0  # number of mutated components
            genSuccessMutants = 0
            meanImprovements=0
            meanImprovementsMut=0
            SF = []
            SCR = []
            improvements = []
            new_population=[]

            for el_idx, el in enumerate(self.population):
                # Generate a trial vector
                trial_vector, selected_bchm,idx_bchm, F, CR, diff1,diff2 = self.generate_trial_vector(el, covPop, np.mean(meanPopVector))
                difference1.append([round(x, 2) for x in diff1])
                difference2.append([round(x, 2) for x in diff2])
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
                    self.update_correction_method_scores(idx_bchm, is_success)

                # If generation number is a multiple of LP it is time to update the probability distribution and reset scores
                if self.adaptive and self.generation_counter % self.learning_period == 0:
                    # Get the last n values of "Prob_infeas"
                    # last_n_prob_infeas_values = [entry["Prob_infeas"] for entry in correction_probabilities[-self.learning_period:]]
                    # # Calculate the average
                    # average_prob_infeas = sum(last_n_prob_infeas_values) / self.learning_period if self.learning_period > 0 else 0
                    # self.learning_period = max(round(self.learning_period_min * np.log(average_prob_infeas+self.epsilon)), self.learning_period_min)
                    self.update_probability_distribution()
                    # Reset successes and failures for next period
                    self.successes.fill(0)
                    self.failures.fill(0)

                # Selection: Compare the fitness of the trial vector to the current individual
                # and select the better one for the next generation
                if is_success:
                    # el.update(trial_vector, trial_fitness, self.repaired)
                    new_population.append(Element(trial_vector))
                    new_population[el_idx].cost=trial_fitness
                    new_population[el_idx].repaired = self.repaired
                    NFS += 1  # number of successful mutations
                    genSuccessMutants +=1
                    SF.append(F)
                    SCR.append(CR)
                    improvements.append(el.cost - trial_fitness)
                    self.memA.append(el)
                else:
                    new_population.append(el)

            if improvements:
                meanImprovements = sum(improvements) / len(improvements)
                meanImprovementsMut = sum(improvements) / genSuccessMutants
            else:
                meanImprovements = 0
                meanImprovementsMut =0

            if NFS > 1:
                # Update the global best solution
                new_population=sorted(new_population, key= lambda element: element.cost)
                for el_idx, el in enumerate(self.population):
                    # el.update(new_population[el_idx].x, new_population[el_idx].cost, new_population[el_idx].repaired)
                    self.population[el_idx]=new_population[el_idx].copy()

                self.global_best = min(self.population, key=lambda element: element.cost)
                meanPopVector, varPopVector, varPop = self.measures.variance(self.population)
                meanPopVector = (meanPopVector - self.lb) / (self.ub - self.lb)
                # varPopVector = varPopVector / (
                #         (self.ub - self.lb) * (self.ub - self.lb))
                varPopVector = varPopVector / ((self.ub - self.lb) * (self.ub - self.lb))

                #print(f"meanPopVector: {meanPopVector}, varPopVector: {varPopVector}")
                covPop = self.measures.covariance(self.population)

                for i in range(self.problem.dim):
                    if varPopVector[i] < self.epsVar:
                        varPopVector[i] = self.epsVar
                    self.aBeta[i] = meanPopVector[i] * (
                            meanPopVector[i] - meanPopVector[i] * meanPopVector[i] - varPopVector[i]) / varPopVector[i]
                    self.bBeta[i] = ((1 - meanPopVector[i]) / meanPopVector[i]) * self.aBeta[i]
                    #print(f"aBeta: {self.aBeta[i]}, bBeta: {self.bBeta[i]}")


            totalInfeasibleComponent = totalInfeasibleComponent + genInfeasibleComponent
            totalMutatedComponent = totalMutatedComponent + genMutatedComponent
            totalInfeasibleElement = totalInfeasibleElement + genInfeasibleElement

            #print(f"New pop:{new_population} , \n , Self:{self.population}")

            self.memA = self.limit_memory(self.memA, len(self.population))
            self.update_F_CR_archives(SCR, SF, improvements)
            newPopSize = round(self.Ninit - NFE / self.NFE_max * (self.Ninit - self.N_min))
            self.population = self.population[0:newPopSize]
            self.pop_size = newPopSize

            try:
                # estimation of ViolationProbability*MutationProbability (is counted only for components selected by crossover)
                prob_infeas = genInfeasibleComponent / genMutatedComponent
            except ZeroDivisionError:
                prob_infeas = genInfeasibleComponent / (genMutatedComponent + self.epsilon)

            if self.adaptive:
                # correction_methods = [self.candidateBCHM[i][self.candidateBCHM[i].find('_') + 1:] for i in set(bchms_used_this_gen)]
                correction_methods = set(bchms_used_this_gen)
                scores = {self.bchms[idx]: {"Successes": self.successes[idx], "Failures": self.failures[idx],
                                            "Probability": self.probabilities[idx]} for idx in range(len(self.bchms))}
                correction_summary = {
                    'Generation': self.generation_counter,
                    'Run': self.run_number,
                    'Fun': str(self.problem.f.meta_data),
                    'Best Cost': self.global_best.cost,
                    'Correction methods used this generation': set(bchms_used_this_gen),
                    'Correction scores': [{self.bchms[idx]: {"Successes": self.successes[idx],
                                                             "Failures": self.failures[idx],
                                                             "Probability": self.probabilities[idx]} for idx in
                                           range(len(self.bchms))}],
                    'Prob_infeas': prob_infeas
                }
                filepath = f"{self.path}/correction_methods_summary.pkl"
                with open(filepath, 'ab') as file:  # Modul 'ab' pentru a adăuga la conținutul existent
                    pickle.dump(correction_summary, file)

                # correction_probabilities.append({
                #     "Generation": self.generation_counter,
                #     "Correction Methods": correction_methods,
                #     "Scores": scores,
                #     "Prob_infeas": prob_infeas
                # })
                # print(f'Generation: {self.generation_counter}, Best Cost: {self.global_best.cost}')
                # print(
                #     f"Correction methods used this generation: {correction_methods}")
                # print(f"Correction scores after generation {self.generation_counter}:")
                # for idx in range(len(self.bchms)):
                #     print(
                #         f"{self.bchms[idx]}: Successes - {self.successes[idx]}, Failures - {self.failures[idx]}, Probability - {self.probabilities[idx]:.10f}")
                # print("\n" + "=" * 50 + "\n")
            # print('prob_infeas=', prob_infeas)

            rez[self.generation_counter - 1] = [self.generation_counter, self.pop_size, self.global_best.cost,
                                                self.global_best.cost - self.problem.f.optimum.y, prob_infeas,
                                                genInfeasibleElement, genMutatedComponent, genSuccessMutants/self.pop_size, meanImprovements, meanImprovementsMut, varPop, self.avgF, self.stdF, self.avgCR,
                                                self.stdCR]

            BB_metrics.append({
                'bb_ratios': bb_ratios,
                'closeness': closeness,
                'extension': extension,
                'density': density,
                'shape': shape,
                'eccentricity': eccentricity,
                'uniformity': uniformity,
                'dist_to_opt': dist_to_opt,
                'kl_beta':kl_beta,
                'kl_unif':kl_unif
            })

        # Save the array with results in a csv file
        df_rez = pd.DataFrame(rez,
                          columns=['it', 'pop_size', 'best', 'error', 'prob_infeas', 'genInfeasibleElement', 'genMutatedComponent', 'ratio', 'meanImprovements', 'meanImprovementsMut', 'varPop',
                                   "avgF", "stdF", "avgCR", "stdCR"])

        df_BB = pd.DataFrame(BB_metrics)
        df_final = pd.concat([df_rez, df_BB], axis=1)



        csv_filename = self.generate_csv_filename()
        df_final.to_csv(csv_filename, index=False)
        return self.global_best