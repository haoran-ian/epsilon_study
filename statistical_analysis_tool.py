import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, wilcoxon
import itertools
import seaborn as sns
import scikit_posthocs as sp
import numpy as np
from scipy.stats import friedmanchisquare
import os

class AlgorithmAnalyzer:
    def __init__(self, averaged_path, ranking_path, algorithms_info, dim):
        self.averaged_path = averaged_path
        self.ranking_path = ranking_path
        self.algorithms_info = algorithms_info
        self.dim = dim

    def calculate_and_save_averages(self):
        for legend_name, (alg_name, source_path) in self.algorithms_info.items():
            function_data = pd.DataFrame()
            for j in range(5):
                filename = f"{source_path}\\{alg_name}_D{self.dim}_run{j+1}.csv"
                data = pd.read_csv(filename)
                function_data = pd.concat([function_data, data])
            mean_values = function_data.groupby('it')[['best','error', 'prob_infeas', 'varPop']].mean()
            mean_values = mean_values.loc[~(mean_values == 0).all(axis=1)]
            mean_values_abs = mean_values.abs()

            output_filename = os.path.join(self.averaged_path, f"{legend_name}_{alg_name}_D{self.dim}.csv")
            mean_values_abs.to_csv(output_filename)
    def compare_and_rank_algorithms(self):
        ranking_results = []
        # Iterate through each function
        pbs = [13, 14, 17, 18, 4, 12, 3, 23, 16, 5]
        for i in pbs:
            errors = {}  # Dict for last error of each algorithm
            for legend_name, (alg_name, source_path) in self.algorithms_info.items():
                data = pd.read_csv(os.path.join(self.averaged_path, f"{legend_name}_mean_values_f{i}.csv"))
                last_error = data['error'].iloc[-1]  # Get the last error value
                errors[legend_name] = last_error

            sorted_algs = sorted(errors.items(), key=lambda x: x[1])
            ranking_results.append([i] + [alg for alg, error in sorted_algs])

        num_rank_columns = len(sorted_algs)  # Number of algorithms ranked
        columns = ['Function'] + [f'Rank_{j + 1}' for j in range(num_rank_columns)]

        ranking_df = pd.DataFrame(ranking_results, columns=columns)
        output_filename = os.path.join(self.ranking_path, "algorithm_rankings.csv")
        ranking_df.to_csv(output_filename, index=False)
        print(f"Rankings saved to {output_filename}")

    def compare_statistical_significance(self, statistical_tests):
        pbs = [13, 14, 17, 18, 4, 12, 3, 23, 16, 5]
        ranking_data = pd.read_csv(f"{self.ranking_path}\\algorithm_rankings.csv")
        error_means = {alg_name: [] for alg_name in self.algorithms_info}

        for i in pbs:
            for legend_name, (alg_name, _) in self.algorithms_info.items():
                filename = os.path.join(self.averaged_path, f"{legend_name}_mean_values_f{i}.csv")
                data = pd.read_csv(filename)
                last_error_mean = data['error'].iloc[-1]
                error_means[legend_name].append(last_error_mean)

        friedman_input = pd.DataFrame(error_means)
        num_algorithms = len(self.algorithms_info)
        if num_algorithms == 2:
            # Perform Wilcoxon test directly if there are only two algorithms
            alg1, alg2 = self.algorithms_info.keys()
            differences = friedman_input[alg1] - friedman_input[alg2]
            if np.all(differences == 0):
                print(f"No variability in differences between {alg1} and {alg2}. Wilcoxon test not applicable.")
                results = [{
                    'Algorithm_1': alg1,
                    'Algorithm_2': alg2,
                    'Wilcoxon_Statistic': 'N/A',
                    'P_val': 'N/A',
                    'Significant': 'N/A'
                }]
            else:
                stat, p_value = wilcoxon(differences)
                results = [{
                    'Algorithm_1': alg1,
                    'Algorithm_2': alg2,
                    'Wilcoxon_Statistic': stat,
                    'P_val': p_value,
                    'Significant': p_value < 0.05
                }]
        else:
            # Friedman test
            friedman_stat, friedman_p_value = friedmanchisquare(*friedman_input.values.T)
            print("Friedman Test:", "Statistic =", friedman_stat, "P-value =", friedman_p_value)

            if friedman_p_value < 0.05:
                print("Significant differences found in algorithm performances.")
                # Pairwise - Wilcoxon test
                results = []
                for i, (alg1, _) in enumerate(self.algorithms_info.items()):
                    for j, (alg2, _) in enumerate(self.algorithms_info.items()):
                        if j > i:
                            stat, p_value = wilcoxon(friedman_input[alg1], friedman_input[alg2])
                            # Bonferroni posthoc
                            num_algorithms = len(self.algorithms_info)
                            alpha = 0.05 / (num_algorithms * (num_algorithms - 1) / 2)
                            significance = p_value < alpha

                            results.append({
                                'Algorithm_1': alg1,
                                'Algorithm_2': alg2,
                                'Wilcoxon_Statistic': stat,
                                'P_val': p_value,
                                'Significant_After_Bonferroni': significance,
                            })
            else:
                print("No significant differences found in algorithm performances.")
                results = []

        results_df = pd.DataFrame(results)
        results_file = os.path.join(self.ranking_path, "Wilcoxon_Test_Results.csv")
        results_df.to_csv(results_file, index=False)
        print(f"Test results saved to {results_file}")


class StatisticalTests:
    @staticmethod
    def test_normality(data):
        stat, p_value = stats.shapiro(data)
        is_normal = p_value > 0.05
        return is_normal, p_value

    @staticmethod
    def test_heteroskedasticity(data1, data2):
        stat, p_value = stats.levene(data1, data2)
        is_homoscedastic = p_value > 0.05
        return is_homoscedastic, p_value

    @staticmethod
    def calculate_nemenyi_distances(data, group_col, value_col):
        posthoc_results = sp.posthoc_nemenyi(data, val_col=value_col, group_col=group_col)
        return posthoc_results


class DataVisualization:
    @staticmethod
    def plot_nemenyi_distances(data, filename):
        plt.figure(figsize=(10, 8))
        sns.heatmap(data, annot=True, cmap="coolwarm", fmt=".3f", cbar_kws={'label': 'Nemenyi Distance'})
        plt.title("Nemenyi Distances")
        plt.savefig(filename)
        plt.show()


averaged_path = "D:\\BBOB_LSHADE\\adaptive\\"
ranking_path = "D:\\BBOB_LSHADE\\adaptive\\"
algorithms_info = {
    'LSHADE_adaptive_betaVT_LP50': ['LSHADE_adaptive_beta', 'D:\\BBOB_LSHADE\\adaptive\\Adaptive_betaVT_LP50'],
    'LSHADE_adaptive_linVT_alpha05_LP50': ['LSHADE_adaptive_linear', 'D:\\BBOB_LSHADE\\adaptive\\Adaptive_linearVT_a05_LP50']
    # 'LSHADE_adaptive_linM_alpha05_LP50': ['LSHADE_adaptive_linear', 'D:\\BBOB_LSHADE\\adaptive\\Adaptive_linearM_a05_LP50']

}
analyzer = AlgorithmAnalyzer(averaged_path, ranking_path, algorithms_info)
statistical_tests = StatisticalTests()
visualization = DataVisualization()

# Perform the analysis
analyzer.calculate_and_save_averages()
analyzer.compare_and_rank_algorithms()
analyzer.compare_statistical_significance(statistical_tests)

print(f"Comparison results saved to {ranking_path}")
