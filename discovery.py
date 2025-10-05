import re
import os
import sys
import pydot
# import cairosvg
import numpy as np
import pandas as pd
from pathlib import Path
from pycausal import prior as p
from pycausal import search as s
from pycausal.pycausal import pycausal as pc
# from IPython.display import SVG

instance_ids = [2, 3, 4, 5]
problem_ids = [1, 3, 4, 5, 16, 23]
epsilons = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.5, ""]
bchms = ["beta", "expC_B", "expC_R", "expC_T", "mahalanobis", "midB",
         "midT", "mir", "sat", "tor", "unif", "vectB", "vectR", "vectT"]


def are_headers_same(df_list):
    if not df_list:
        return True
    first_header = set(df_list[0].columns)
    return all(set(df.columns) == first_header for df in df_list[1:])


def merge_if_same_headers(df_list):
    if are_headers_same(df_list):
        return pd.concat(df_list, ignore_index=True)
    else:
        print("Warning: Headers of DataFrame are not the same, merge is not allowed.")
        return None


def build_atom_data(iid, pid, bchm, eps):
    dfs = []
    out = f"data/causal_discovery/{iid}_{pid}_{bchms.index(bchm)}_{eps}.csv"
    if os.path.exists(out):
        return
    for run in range(1, 11):
        f_path = f"data/instance_{iid}/LSHADE_{bchm}_f{pid}_D20_eps{eps}run{run}_gen.csv"
        if os.path.exists(f_path):
            df = pd.read_csv(f_path)
            first_col = df.columns[0]
            df = df[df[first_col] != 0.0]
            # df["it"] = df["it"].astype(int)
            # df["pop_size"] = df["pop_size"].astype(int)
            df.drop("kl_beta", axis=1, inplace=True)
            df = df.select_dtypes(exclude=['object'])
            new_columns = {
                "iid": [iid] * len(df),
                "pid": [pid] * len(df),
                "bchm": [float(bchms.index(bchm))] * len(df),
                "eps": [eps if eps != "" else -1] * len(df)
            }
            new_df = pd.DataFrame(new_columns)
            df = pd.concat([new_df, df], axis=1)
            dfs += [df]
        else:
            print(f"Missing file: {f_path}")
    df = merge_if_same_headers(dfs)
    df.to_csv(f"data/causal_discovery/{iid}_{pid}_{bchms.index(bchm)}_{eps}.csv",
              index=False)
    print(df.info())


def causal_discovery(iid, pid):
    if os.path.exists(f"results/causal_bchm_fci/{iid}_{pid}.svg"):
        return 0
    dfs = []
    for bchm in bchms:
        for eps in epsilons:
            f_path = f"data/causal_discovery/{iid}_{pid}_{bchms.index(bchm)}_{eps}.csv"
            df = pd.read_csv(f_path)
            df = df.iloc[:, 2:]
            dfs += [df]
    df = merge_if_same_headers(dfs)
    print(len(df))
    sample_size = min(len(df), 1000)
    df = df.sample(n=sample_size, random_state=42)
    tetrad = s.tetradrunner()
    tetrad.getAlgorithmParameters(
        algoId="gfci", testId="fisher-z-test", scoreId="sem-bic")
    tetrad.run(algoId='gfci', dfs=df, testId='fisher-z-test', scoreId='sem-bic',
               maxDegree=-1, maxPathLength=-1, completeRuleSetUsed=False,
               faithfulnessAssumed=True, verbose=False, numberResampling=5,
               resamplingEnsemble=1, addOriginalDataset=True)
    graph = tetrad.getTetradGraph()
    nodes = tetrad.getNodes()
    edges = tetrad.getEdges()
    dot_str = pc.tetradGraphToDot(graph)
    graphs = pydot.graph_from_dot_data(dot_str)
    svg_str = graphs[0].create_svg()
    f = open(f"data/raw_graph_pycausal/nodes/{iid}_{pid}.txt", "w")
    for n in nodes:
        f.write(n+"\n")
    f.close()
    f = open(f"data/raw_graph_pycausal/edges/{iid}_{pid}.txt", "w")
    for e in edges:
        f.write(e+"\n")
    f.close()
    f = open(f"data/raw_graph_pycausal/graphs/{iid}_{pid}.txt", "w")
    f.write(dot_str)
    f.close()
    # f = open(f"results/causal_bchm_fci/{iid}_{pid}.svg", "wb")
    # f.write(svg_str)


if __name__ == "__main__":
    iid = int(sys.argv[1])
    pid = int(sys.argv[2])
    # bchm_id = int(sys.argv[3])
    pc = pc()
    pc.start_vm()
    for bchm_id in range(14):
        for eps in epsilons:
            build_atom_data(iid, pid, bchms[bchm_id], eps)
    causal_discovery(iid, pid)
    pc.stop_vm()
