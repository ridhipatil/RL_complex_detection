import pickle
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import logging
from pickle import load as pickle_load
from pickle import dump as pickle_dump
import scipy.interpolate
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import cpu_count as mul_cpu_count
from glob import glob
import time
from argparse import ArgumentParser as argparse_ArgumentParser
import os

def interval(graph):
    # intervals for states to better organize and observe data
    intervals = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
                 0.95, 1]
    d = nx.density(graph)
    for i in intervals:
        if d <= i:
            d = i
            break
        else:
            continue
    return d


# if previous training method did not encounter a state, the value function of that state is interpolated
def interpolate(value_functions, dens):
    d = list(value_functions.keys())
    v = list(value_functions.values())
    interp = scipy.interpolate.interp1d(d, v)
    new_vf = interp(dens)
    return new_vf

def pred_complex(n, nodes_list, G, gg, value_functions, intervals,args):
    # make sure n is not a node floating around
    neighb_n = list(G.neighbors(str(n)))
    if len(neighb_n) == 0:
        return

    # seed edge with node that gives max edge weight
    x = [(neib, G.get_edge_data(n, neib)) for neib in neighb_n]
    n2 = max(x, key=lambda x: x[1]['weight'])[0]

    # create seed edge
    temp_weight = G.get_edge_data(n, n2)
    nx.add_path(gg, [n, n2], weight=temp_weight.get('weight'))
    nodes_order = [n, n2]
    val_fns = []

    # value iteration
    while True:
        # Initial value functions of states are 0
        curr_nodes = gg.nodes()  # all current nodes
        neighb_val_fns = {}

        # get neighbors
        neighbors = []
        for k in curr_nodes:
            neighbors = neighbors + list(G.neighbors(k))
        neighbors = list(set(neighbors) - set(curr_nodes))
        for m in neighbors:
            for k in curr_nodes:
                curr_nb = list(G.neighbors(k))
                if m in curr_nb:
                    # density of adding temporary node
                    temp_weight = G.get_edge_data(k, m)
                    nx.add_path(gg, [k, m], weight=temp_weight.get('weight'))
                    temp_dens = interval(gg)
                    gg.remove_node(m)  # remove node

                    # check if state is encountered in training, if not, interpolate for value function
                    if temp_dens in value_functions:
                        curr_val_fn = value_functions[temp_dens]
                    else:
                        curr_val_fn = interpolate(value_functions, temp_dens)
                        value_functions[temp_dens] = curr_val_fn
                    neighb_val_fns[m] = curr_val_fn

        # find the node that has the highest value function
        if len(neighbors) != 0:
            added_n = max(neighb_val_fns, key=neighb_val_fns.get)
            # add node to graph
            for k in list(curr_nodes):
                neighbors = list(G.neighbors(k))
                if added_n in neighbors:
                    temp_weight = G.get_edge_data(added_n, k)
                    nx.add_path(gg, [added_n, k], weight=temp_weight.get('weight'))
            val_fns.append(neighb_val_fns[added_n])  # max, get index
            nodes_order = list(gg.nodes())
        else:
            final_dens = interval(gg)
            cmplx_val_fn = value_functions[final_dens]
            break

        # if value function of complex is decreasing, stop updating
        if val_fns[len(val_fns) - 2] > val_fns[len(val_fns) - 1]:
            cmplx_val_fn = val_fns[len(val_fns) - 1]
            break
        else:
            for k in list(curr_nodes):
                neighbors = list(G.neighbors(k))
                if added_n in neighbors:
                    temp_weight = G.get_edge_data(added_n, k)
                    nx.add_path(gg, [added_n, k], weight=temp_weight.get('weight'))

    tup_cmplx = (nodes_order, cmplx_val_fn)
    #args.pred_results = "../../humap2/pred_results"

   # args.pred_results = "../results/pred_results"
    file = args.pred_results + '/nodes_complexes/'
    with open(file + str(n), 'wb') as f:
        pickle_dump(tup_cmplx, f)
    with open(file + str(n), 'rb') as f:
        pickle_load(f)


def network(G, gg, nodes, intervals, value_functions,args):
    ## input data
    nodes_list = list(nodes)
    # make sure all intervals are accounted for
    for i in intervals:
        if i not in value_functions:
            val_fn = interpolate(value_functions, i)
            value_functions[i] = val_fn
    filename = args.pred_results + '/value_fns_pred.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(value_functions, f)
    fname = args.pred_results + '/value_fns_interp.txt'
    with open(fname, 'w') as f:
        f.write(str(value_functions))

    # parallel running
    num_cores = mul_cpu_count()
    Parallel(n_jobs=num_cores, backend='loky')(
        delayed(pred_complex)(node, nodes_list, G, gg, value_functions, intervals,args) for node in tqdm(nodes_list))

    pred_comp_list = []
    sdndap = pred_comp_list.append
    allfiles = args.pred_results + '/nodes_complexes/*'
    for fname in glob(allfiles, recursive=True):
        with open(fname, 'rb') as f:
            pred_comp_tmp = pickle_load(f)
        sdndap(pred_comp_tmp)
    file = args.pred_results + '/predicted_complexes.pkl'
    with open(file, 'wb') as f:
        pickle_dump(pred_comp_list, f)



def main():
    start_time = time.time()
    matplotlib.use('Agg')
    logging.basicConfig(level=logging.WARNING)
    # input data
    parser = argparse_ArgumentParser("Input parameters")
    parser.add_argument("--graph_file", default="", help="Graph edges file path")
    parser.add_argument("--train_results", default="", help="Directory for training results")
    parser.add_argument("--pred_results", default="", help="Directory for main results")
    parser.add_argument("--out_dir_name", default = "", help = 'Main output directory')
    args = parser.parse_args()
    os.makedirs(args.pred_results + '/nodes_complexes', exist_ok=True)

   # args.graph_file = "../hu.MAP_network_experiments/input_data/humap_network_weighted_edge_lists.txt"
   # args.train_results = "../results/train_results"
   # args.pred_results = "../results/pred_results"
    # dictionary containing states (density) and corresponding value functions
    file = args.train_results + '/value_fn_dens_dict.pkl'
    with open(file, 'rb') as f:
        value_functions = pickle_load(f)
    value_functions = dict(value_functions)

    # edges data
    #fileName = "../../humap_network_weighted_edge_lists.txt"
    filename = args.graph_file

    G = nx.read_weighted_edgelist(filename, nodetype=str)
    f.close()
    # remove duplicate edges and none
    G.remove_edges_from(nx.selfloop_edges(G))
    for i in list(G.nodes()):
        if i.isnumeric() is False:
            G.remove_node(i)
    # humap nodes
    nodes = G.nodes()
    # new graph
    gg = nx.Graph()
    # i.e interval 1 is for states in between 0.05-1, interval 0.95 is for states in between 0.9-0.95, etc.
    intervals = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
                 0.95, 1]

    network(G, gg, nodes, intervals, value_functions,args)
    file = args.pred_results + "/humap_CORUM_complexes_node_lists.pkl"
    with open(file, 'wb') as f:
        pickle_dump(list(nodes), f)

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
