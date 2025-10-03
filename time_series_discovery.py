import os
import sys
import pandas as pd
import numpy as np
# import matplotlib
# from matplotlib import pyplot as plt
# plt.style.use('ggplot')
# import sklearn

# import tigramite
from tigramite import data_processing as pp
# from tigramite.toymodels import structural_causal_processes as toys

# from tigramite import plotting as tp
# from tigramite.pcmci import PCMCI
from tigramite.lpcmci import LPCMCI

from tigramite.independence_tests.parcorr import ParCorr
# from tigramite.independence_tests.robust_parcorr import RobustParCorr
# from tigramite.independence_tests.parcorr_wls import ParCorrWLS
# from tigramite.independence_tests.gpdc import GPDC
# from tigramite.independence_tests.cmiknn import CMIknn
# from tigramite.independence_tests.cmisymb import CMIsymb
# from tigramite.independence_tests.gsquared import Gsquared
# from tigramite.independence_tests.regressionCI import RegressionCI

instance_ids = [2, 3, 4, 5]
problem_ids = [1, 3, 4, 5, 16, 23]
epsilons = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.5, ""]
bchms = ["beta", "expC_B", "expC_R", "expC_T", "mahalanobis", "midB",
         "midT", "mir", "sat", "tor", "unif", "vectB", "vectR", "vectT"]


def build_atom_data(iid, pid, bchm, eps):
    dfs = []
    for run in range(1, 11):
        f_path = f"data/instance_{iid}/LSHADE_{bchm}_f{pid}_D20_eps{eps}run{run}_gen.csv"
        if os.path.exists(f_path):
            df = pd.read_csv(f_path)
            first_col = df.columns[0]
            df = df[df[first_col] != 0.0]
            df.drop("kl_beta", axis=1, inplace=True)
            df = df.select_dtypes(exclude=['object'])
            dataframe = pp.DataFrame(df.values,
                                     datatime={0: np.arange(len(df))},
                                     var_names=df.columns)
            dfs += [dataframe]
        else:
            print(f"Missing file: {f_path}")
    return dfs, df.columns


def discovery(dataframe):
    parcorr = ParCorr(significance='analytic')
    lpcmci = LPCMCI(dataframe=dataframe,
                    cond_ind_test=parcorr,
                    verbosity=1)
    tau_max = 3
    pc_alpha = 0.01
    # Run LPCMCI
    results = lpcmci.run_lpcmci(tau_max=tau_max,
                                pc_alpha=pc_alpha)
    val_matrix = results['val_matrix']
    graph = results['graph']
    return graph, val_matrix


iid = int(sys.argv[1])
pid = int(sys.argv[2])
epsilon = epsilons[int(sys.argv[3])]
bchm = bchms[int(sys.argv[4])]
print(f"Processing {iid} {pid} {epsilon} {bchm}")
dfs, var_names = build_atom_data(iid, pid, bchm, epsilon)
for i in range(len(dfs)):
    # flag_path = f"data/graphs/{iid}_{pid}_{epsilon}_{bchm}_{i}.npy"
    # if os.path.exists(flag_path):
    #     continue
    graph, val_matrix = discovery(dfs[i])
    # discovery(dfs[i])
    # print(graph)
    np.save(
        f"data/LPCMCI/graphs/{iid}_{pid}_{epsilon}_{bchm}_{i}.npy", graph)
    np.save(
        f"data/LPCMCI/val_matrix/{iid}_{pid}_{epsilon}_{bchm}_{i}.npy", val_matrix)
