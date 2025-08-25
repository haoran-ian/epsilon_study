import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

instance_ids = [2, 3, 4, 5]
problem_ids = [1, 3, 4, 5, 16, 23]
epsilons = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.5, ""]
bchms = ["beta", "expC_B", "expC_R", "expC_T", "mahalanobis", "midB",
         "midT", "mir", "sat", "tor", "unif", "vectB", "vectR", "vectT"]
labels = ["it", "pop_size", "best", "error", "prob_infeas",
          "genInfeasibleElement", "genMutatedComponent", "ratio",
          "meanImprovements", "meanImprovementsMut", "varPop", "avgF", "stdF",
          "avgCR", "stdCR", "extension", "density", "shape", "eccentricity",
          "dist_to_opt", "kl_unif"]


def build_GTO(iid, pid, epsilon, bchm):
    edge_types = ["", "-->", "o-o"]
    graph_GTO = [[[0 for _ in range(3)] for _ in range(21)] for _ in range(21)]
    for run in range(10):
        f_path = f"data/graphs/{iid}_{pid}_{epsilon}_{bchm}_{run}.npy"
        if not os.path.exists(f_path):
            continue
        graph = np.load(f_path)
        for i in range(graph.shape[0]):
            for j in range(graph.shape[1]):
                for k in range(graph.shape[2]):
                    graph_GTO[i][j][edge_types.index(graph[i][j][k])] += 1
    return graph_GTO


def plot_GTO(graph, iid, pid, epsilon, bchm):
    data_normalized = graph / graph.sum(axis=2, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 21)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks(np.arange(0, 21, 1))
    ax.set_yticks(np.arange(0, 21, 1))
    ax.grid(True, linestyle="-", linewidth=0.5, color="gray")
    colors = ["#556270", "#FF6B6B", "#4ECDC4"]
    for i in range(21):
        for j in range(21):
            values = data_normalized[i, j]
            x_start = j
            y_start = i
            width_accum = 0
            for k in range(3):
                width = values[k]
                rect = Rectangle((x_start + width_accum, y_start),
                                 width,
                                 1,
                                 facecolor=colors[k],
                                 edgecolor="none")
                ax.add_patch(rect)
                width_accum += width
    
    ax.set_xticks(np.arange(0, 21, 1))  # 将标签放在方格中心
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)

    # 设置Y轴刻度位置和标签
    ax.set_yticks(np.arange(0, 21, 1))
    ax.set_yticklabels(labels, fontsize=9)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], label="no edge"),
        Patch(facecolor=colors[1], label="-->"),
        Patch(facecolor=colors[2], label="o-o")
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    plt.title(f"iid: {iid}, fid: {pid}, epsilon: {epsilon}, bchm: {bchm}")
    plt.tight_layout()
    plt.savefig(f"results/graph_GTO/{iid}_{pid}_{epsilon}_{bchm}.png")


iid = int(sys.argv[1])
pid = int(sys.argv[2])
epsilon = epsilons[int(sys.argv[3])]
bchm = bchms[int(sys.argv[4])]
f_path = f"data/graphs/{iid}_{pid}_{epsilon}_{bchm}_1.npy"
if not os.path.exists(f_path):
    exit
print(f"Processing {iid} {pid} {epsilon} {bchm}")
# for iid in instance_ids:
#     for pid in problem_ids:
#         for epsilon in epsilons:
#             for bchm in bchms:
graph_GTO = build_GTO(iid, pid, epsilon, bchm)
graph_GTO = np.array(graph_GTO)
if np.sum(graph_GTO) == 0:
    exit
plot_GTO(graph_GTO, iid, pid, epsilon, bchm)
