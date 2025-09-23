import os
import sys
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
# plt.style.use('ggplot')
import sklearn

import tigramite
from tigramite import data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys

from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.lpcmci import LPCMCI

from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.independence_tests.parcorr_wls import ParCorrWLS
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.cmisymb import CMIsymb
from tigramite.independence_tests.gsquared import Gsquared
from tigramite.independence_tests.regressionCI import RegressionCI

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
            # print(df.columns)
            dfs += [dataframe]
        else:
            print(f"Missing file: {f_path}")
    return dfs, df.columns


def discovery(dataframe, var_names):
    parcorr = ParCorr(significance='analytic')
    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=parcorr,
        verbosity=1)
    # correlations = pcmci.get_lagged_dependencies(
    #     tau_max=20, val_only=True)['val_matrix']
    pcmci.verbosity = 1
    results = pcmci.run_pcmci(tau_max=8, pc_alpha=None, alpha_level=0.01)
    q_matrix = pcmci.get_corrected_pvalues(
        p_matrix=results['p_matrix'], tau_max=8, fdr_method='fdr_bh')
    pcmci.print_significant_links(
        p_matrix=q_matrix,
        val_matrix=results['val_matrix'],
        alpha_level=0.01)
    graph = pcmci.get_graph_from_pmatrix(p_matrix=q_matrix, alpha_level=0.01,
                                         tau_min=0, tau_max=8, link_assumptions=None)
    val_matrix = results['val_matrix']
    results['graph'] = graph
    return graph, val_matrix

iid = int(sys.argv[1])
pid = int(sys.argv[2])
epsilon = epsilons[int(sys.argv[3])]
bchm = bchms[int(sys.argv[4])]
print(f"Processing {iid} {pid} {epsilon} {bchm}")
# for iid in instance_ids:
#     for pid in problem_ids:
#         for epsilon in epsilons:
#             for bchm in bchms:
dfs, var_names = build_atom_data(iid, pid, bchm, epsilon)
for i in range(len(dfs)):
    # flag_path = f"data/graphs/{iid}_{pid}_{epsilon}_{bchm}_{i}.npy"
    # if os.path.exists(flag_path):
    #     continue
    graph, val_matrix = discovery(dfs[i], var_names)
    print(graph)
    np.save(
        f"data/graphs/{iid}_{pid}_{epsilon}_{bchm}_{i}.npy", graph)
    np.save(
        f"data/val_matrix/{iid}_{pid}_{epsilon}_{bchm}_{i}.npy", val_matrix)
