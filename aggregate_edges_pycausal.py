import pydot
import numpy as np

iids = [2, 3, 4, 5]
pids = [1, 3, 4, 5, 16, 23]
edge_types = ["<->", "-->", "o->", "o-o", "no edge"]
nodes = ["avgCR", "avgF", "bchm", "best", "density", "dist_to_opt",
         "eccentricity", "eps", "error", "extension", "genInfeasibleElement",
         "genMutatedComponent", "it", "kl_unif", "meanImprovements",
         "meanImprovementsMut", "pop_size", "prob_infeas", "ratio", "shape",
         "stdCR", "stdF", "varPop"]


def parse(line, strength_count):
    results = []
    # edge_types = ["<->", "-->", "o->", "o-o", "no edge"]
    line = line.split("\n")[0]
    probs = line.split(";")[:-1]
    probs[0] = "[" + probs[0].split("[")[1]
    vote = line.split("[")[0][:-1].split(" ")
    var1 = nodes.index(vote[0])
    var2 = nodes.index(vote[2])
    strength_count[var1, var2] += 1
    for prob in probs:
        p = float(prob.split(":")[1])
        vote = prob.split("]")[0][1:].split(" ")
        if len(vote) == 2:
            results += [[4, var1, var2, p], [4, var2, var1, p]]
            continue
        temp_var1 = nodes.index(vote[0])
        temp_var2 = nodes.index(vote[2])
        if vote[1] == "<--":
            results += [[1, temp_var2, temp_var1, p]]
        elif vote[1] == "<-o":
            results += [[2, temp_var2, temp_var1, p]]
        elif vote[1] == "-->":
            results += [[1, temp_var1, temp_var2, p]]
        elif vote[1] == "o->":
            results += [[2, temp_var1, temp_var2, p]]
        else:
            results += [[edge_types.index(vote[1]), temp_var1, temp_var2, p],
                        [edge_types.index(vote[1]), temp_var2, temp_var1, p]]
    return results, strength_count


def aggregate(iid, pid, strength_count):
    # <->, -->, o->, o-o, no edge
    edges = np.zeros((5, 23, 23))
    edge_file = f"data/raw_graph_pycausal/edges/{iid}_{pid}.txt"
    f = open(edge_file, "r")
    lines = f.readlines()
    for line in lines:
        results, strength_count = parse(line, strength_count)
        for e in results:
            edges[e[0], e[1], e[2]] += e[3]
    return edges


def normalize_array(arr):
    result = arr.copy().astype(float)
    for i in range(arr.shape[1]):
        for j in range(arr.shape[2]):
            values = arr[:, i, j]
            if np.all(values == 0):
                continue
            else:
                sum_values = np.sum(values)
                result[:, i, j] = values / sum_values
    return result


def format_edges(aggregate_edges, strength_count):
    edges_str = ""
    for i in range(aggregate_edges.shape[1]):
        for j in range(aggregate_edges.shape[2]):
            if strength_count[i, j] <= 2:
                continue
            values = aggregate_edges[:, i, j]
            if np.all(values == 0):
                continue
            else:
                edge_type = edge_types[np.argmax(values)]
                line = f"{nodes[i]} {edge_type} {nodes[j]} "
                for k in range(values.shape[0]):
                    if values[k] == 0:
                        continue
                    if k == 4:
                        line += f"[no edge]:{values[k]:.4f};"
                        continue
                    line += f"[{nodes[i]} {edge_types[k]} {nodes[j]}]:{values[k]:.4f};"
                line += "\n"
            edges_str += line
    return edges_str


if __name__ == "__main__":
    for pid in pids:
        aggregate_edges = np.zeros((5, 23, 23))
        strength_count = np.zeros((23, 23))
        for iid in iids:
            edges = aggregate(iid, pid, strength_count)
            aggregate_edges = aggregate_edges + edges
        aggregate_edges = normalize_array(aggregate_edges)
        edges_str = format_edges(aggregate_edges, strength_count)
        f = open(f"data/raw_graph_pycausal/{pid}_edges.txt", "w")
        f.write(edges_str)
        f.close()
        # read edge str to create plot
        f = open(f"data/raw_graph_pycausal/{pid}_edges.txt", "r")
        edges = f.readlines()
        for i in range(len(edges)):
            edges[i] = edges[i][:-1]
        known_nodes = set()
        edge_data = {}
        for line in edges:
            parts = line.split(' [')
            node_part = parts[0]
            probabilities_part = parts[1].rstrip(';')
            node1, edge_type, node2 = node_part.split(' ')
            known_nodes.add(node1)
            known_nodes.add(node2)
            probabilities = {}
            for prob_item in probabilities_part.split(';'):
                if prob_item:
                    edge_desc, prob = prob_item.split(':')
                    probabilities[edge_desc] = float(prob)
            
            edge_data[(node1, node2)] = probabilities
        graph = pydot.Dot(graph_type='digraph', rankdir='LR', splines='true')
        for node in nodes:
            graph.add_node(pydot.Node(node, shape='ellipse'))
        edge_styles = {
            '<->': {'style': 'solid', 'arrowhead': 'normal', 'arrowtail': 'normal', 'dir': 'both'},
            '-->': {'style': 'solid', 'arrowhead': 'normal'},
            'o->': {'style': 'solid', 'arrowhead': 'normal', 'arrowtail': 'odot', 'dir': 'both'},
            'o-o': {'style': 'dashed', 'arrowhead': 'odot', 'arrowtail': 'odot', 'dir': 'both'}
        }
        for (node1, node2), probabilities in edge_data.items():
            max_prob = -1
            best_edge_type = None
            for edge_type, prob in probabilities.items():
                if edge_type != 'no edge' and prob > max_prob:
                    max_prob = prob
                    best_edge_type = edge_type
            if best_edge_type:
                actual_edge_type = best_edge_type.split(' ')[1]
                if actual_edge_type == '<->':
                    edge = pydot.Edge(node1, node2, **edge_styles[actual_edge_type])
                elif actual_edge_type == '-->':
                    edge = pydot.Edge(node1, node2, **edge_styles[actual_edge_type])
                elif actual_edge_type == 'o->':
                    edge = pydot.Edge(node1, node2, **edge_styles[actual_edge_type])
                elif actual_edge_type == 'o-o':
                    edge = pydot.Edge(node1, node2, **edge_styles[actual_edge_type])
                edge.set_label(f"{max_prob:.3f}")
                graph.add_edge(edge)
        graph.write_svg(f'results/aggregate_pycausal/{pid}.svg')