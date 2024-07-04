import os
import pandas as pd

num_runs = 10
pbs = [23, 1, 3, 4, 5, 16]
instance_ids = [2, 3, 4, 5]
epsilons = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.5, ""]
correction_methods = ["vectB", "vectR", "vectT", "sat", "midT", "midB",
                      "mahalanobis", "expC_R", "expC_T", "expC_B",
                      "unif", "mir", "tor", "beta"]
raw_values = []
for instance_id in instance_ids:
    for bchm in correction_methods:
        for pb in pbs:
            for epsilon in epsilons:
                for run in range(1, num_runs+1):
                    raw_values.append([instance_id, bchm, pb, epsilon, run])
check = 0
index = []
for i in range(len(raw_values)):
    key = raw_values[i]
    file_path = f"data/instance_{key[0]}/LSHADE_{key[1]}_f{key[2]}" + \
        f"_D20_eps{key[3]}run{key[4]}_gen.csv"
    if os.path.exists(file_path):
        check += 1
        index += [i]
num_files = 0
for instance_id in instance_ids:
    files = os.listdir(f"data/instance_{instance_id}")
    num_files += len(files)
print(f"Total files needed: {len(raw_values)}, " +
      f"files found: {check}, files created: {num_files}")

values = []
for i in range(len(raw_values)):
    if i not in index:
        record = [raw_values[i][0], correction_methods.index(raw_values[i][1]),
                  raw_values[i][2], raw_values[i][3]]
        if record not in values:
            values.append(record)

df = pd.DataFrame(
    values, columns=["instance_id", "bchm", "pb", "epsilon"])
df.to_csv("data/missing_files.csv", index=False)
