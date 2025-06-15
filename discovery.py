import re
import os
import numpy as np
import pandas as pd

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
    for run in range(1, 11):
        f_path = f"data/instance_{iid}/LSHADE_{bchm}_f{pid}_D20_eps{eps}run{run}_gen.csv"
        if os.path.exists(f_path):
            df = pd.read_csv(f_path)
            first_col = df.columns[0]
            df = df[df[first_col] != 0.0]
            df.drop("kl_beta", axis=1, inplace=True)
            df = df.select_dtypes(exclude=['object'])
            new_columns = {
                "iid": [iid] * len(df),
                "pid": [pid] * len(df),
                "bchm": [bchms.index(bchm)] * len(df),
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

if __name__ == "__main__":
    for eps in epsilons:
        build_atom_data(2, 1, bchms[0], eps)