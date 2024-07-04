from core import ContinuousOptimization
from lshade import LSHADE
from correction_handler import *
import random
import os

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Define pb dimensions, function name, and instance
correction_methods = [
    METHOD_VECTOR_BEST,
    METHOD_VECTOR_TARGET,
    METHOD_VECTOR_R,
    METHOD_SATURATION,
    METHOD_MIDPOINT_TARGET,
    METHOD_MIDPOINT_BEST,
    METHOD_MAHALANOBIS,
    METHOD_EXPC_R,
    METHOD_EXPC_TARGET,
    METHOD_EXPC_BEST,
    METHOD_UNIF,
    METHOD_MIRROR,
    METHOD_TOROIDAL,
    METHOD_BETA
]
d=20
num_runs = 10
pbs=[23,1,3,4,5,16]
instance_ids=[2, 3, 4, 5]
random.seed(42)
np.random.seed(42)
epsilons = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.5, ""]
os.makedirs("data/search_region", exist_ok=True)
for iid in instance_ids:
    if not os.path.exists(f"data/instance_{iid}/"):
        os.makedirs(f"data/instance_{iid}/", exist_ok=True)
for problem_id in pbs:
    for instance_id in instance_ids:
        prob = ContinuousOptimization(dim=d, fct_name=problem_id, inst=instance_id)
        xopt = prob.f.optimum.x
        lb = [-5. for _ in range(d)]
        ub = [5. for _ in range(d)]
        abs_diff_with_neg5 = np.abs(xopt - lb)
        abs_diff_with_5 = np.abs(xopt - ub)
        min_abs_diff = np.minimum(abs_diff_with_neg5, abs_diff_with_5)
        sorted_indices = np.argsort(min_abs_diff)
        sorted_values = min_abs_diff[sorted_indices]
        sorted_xopt = xopt[sorted_indices]
        # components = sorted_indices[:k_components]
        for epsilon in epsilons[:-1]:
            search_region = np.array([[lb[i], ub[i]] for i in range(d)])
            for j in sorted_indices:
                if np.abs(xopt[j] - lb[j]) <= np.abs(xopt[j] - ub[j]):
                    search_region[j][0] = xopt[j] - \
                                          (ub[j] - xopt[j]) * epsilon / (1 - epsilon)
                    search_region[j][1] = ub[j]
                else:
                    search_region[j][0] = lb[j]
                    search_region[j][1] = xopt[j] + \
                                          (xopt[j] - lb[j]) * epsilon / (1 - epsilon)
            np.savetxt(f"data/search_region/{problem_id}_{instance_id}_{epsilon}.txt",
                       search_region)

for bchm in correction_methods:
    for pb in pbs:
        for epsilon in epsilons:
            for instance_id in instance_ids:
                problem = ContinuousOptimization(dim=d, fct_name=pb, inst=instance_id)
                print("old lb:", problem.lb)
                if epsilon != "":
                    try:
                        bounds = np.loadtxt(f"data/search_region/{pb}_{instance_id}_{epsilon}.txt")
                        problem.set_bounds(lb=bounds.T[0], ub=bounds.T[1])
                        print("new lb:", problem.lb)
                    except Exception as e:
                        print(f"Error setting bounds for problem {pb}, instance {1}: {e}")
                for run in range(num_runs):
                    lshade = LSHADE(
                        problem=problem,
                        pop_size=18*d,
                        sizeH=6,
                        NFE_max=d * 5000,
                        N_min=10,
                        corr_type=None,
                        corr_method=bchm,
                        adaptive=0,
                        candidateBCHM=[METHOD_SATURATION, METHOD_EXPC_BEST, METHOD_BETA, METHOD_VECTOR_BEST,
                                    METHOD_VECTOR_TARGET],
                        probabilities_update_strategy=0,
                        learning_period=50,
                        run_number=run+1,
                        path=f"data/instance_{instance_id}",
                        epsilon=epsilon
                    )
                    best_solution = lshade.optimize()
                    print(f"Problem {pb}, corr_method {bchm}, instance {instance_id}, run {run+1}, epsilon {epsilon}: {best_solution}")
    print("Best solution:", best_solution)



