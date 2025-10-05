import os
import sys
import pandas as pd
import numpy as np

from tigramite import data_processing as pp
from tigramite.lpcmci import LPCMCI
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.parcorr import ParCorr

instance_ids = [2, 3, 4, 5]
problem_ids = [1, 3, 4, 5, 16, 23]
epsilons = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.5, ""]
bchms = ["beta", "expC_B", "expC_R", "expC_T", "mahalanobis", "midB",
         "midT", "mir", "sat", "tor", "unif", "vectB", "vectR", "vectT"]


def build_atom_data(iid, pid, bchm, eps):
    dfs = []
    for run in range(1, 11):
        f_path = f"data/instance_{iid}/LSHADE_{bchm}_f{pid}_D20_eps{eps}run{run}_gen.csv"
        print(f_path)
        if os.path.exists(f_path):
            df = pd.read_csv(f_path)
            first_col = df.columns[0]
            df = df[df[first_col] != 0.0]
            df.drop("kl_beta", axis=1, inplace=True)
            df = df.select_dtypes(exclude=['object'])
            dfs += [df]
        else:
            print(f"Missing file: {f_path}")
    return dfs, df.columns


def discovery(df):
    dataframe = pp.DataFrame(df.values, var_names=df.columns)
    cond_ind_test = ParCorr(significance='analytic')
    lpcmci = LPCMCI(dataframe=dataframe,
                    cond_ind_test=cond_ind_test,
                    verbosity=1)
    tau_max = 1
    pc_alpha = 0.01
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
if os.path.exists(f"data/LPCMCI/graphs/{iid}_{pid}_{epsilon}_{bchm}_9.npy"):
    sys.exit(0)
dfs, var_names = build_atom_data(iid, pid, bchm, epsilon)

for i in range(len(dfs)):
    if os.path.exists(f"data/LPCMCI/graphs/{iid}_{pid}_{epsilon}_{bchm}_{i}.npy"):
        continue
    graph, val_matrix = discovery(dfs[i])
    print(graph)
    np.save(
        f"data/LPCMCI/graphs/{iid}_{pid}_{epsilon}_{bchm}_{i}.npy", graph)
    np.save(
        f"data/LPCMCI/val_matrix/{iid}_{pid}_{epsilon}_{bchm}_{i}.npy", val_matrix)
