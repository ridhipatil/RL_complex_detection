import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import csv
import logging
import scipy as sp
from pickle import load as pickle_load
from pickle import dump as pickle_dump
from collections import Counter
import scipy.interpolate
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import cpu_count as mul_cpu_count
from glob import glob

matplotlib.use('Agg')
logging.basicConfig(level=logging.WARNING)

with open('Value Functions Intervals.txt','rb') as f:
    value_functions = pickle_load(f)
value_functions = dict(value_functions)
#print(value_functions)
# edges data
fileName = "humap_network_weighted_edge_lists.txt"
G = nx.Graph()
G = nx.read_weighted_edgelist(fileName, nodetype=str)
# remove duplicate edges and none
G.remove_edges_from(nx.selfloop_edges(G))
for i in list(G.nodes()):
    if i.isnumeric() is False:
        G.remove_node(i)
# humap nodes
nodes = G.nodes()
gg = nx.Graph()

all_lists = []
cmplx_info = []
intervals = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
def interpolate(value_functions,dens):
    d = list(value_functions.keys())
    v = list(value_functions.values())
    interp = scipy.interpolate.interp1d(d,v)
    new_vf = interp(dens)
    return new_vf

def pred_complex(n,nodes_list):
    # make sure n is not a node floating around
    neighb_n = list(G.neighbors(str(n)))
    if len(neighb_n) == 0:
        return
    x = [(neib, G.get_edge_data(n, neib)) for neib in neighb_n]
    n2 = max(x, key=lambda x: x[1]['weight'])[0]
    nx.add_path(gg, [n, n2])
    nodes_order = [n, n2]
    val_fns = []
    logging.warning('Added Node to test value function')
    logging.warning(gg.nodes())
    # empty dictionary, add key in loop if not in there already (if it is just update current)
    # value iteration

    while True:
        # Initial value functions of states are 0
        curr_nodes = gg.nodes  # all current nodes
        neighb_val_fns = {}  ##
        logging.warning("The current nodes in gg are")
        logging.warning(curr_nodes)
        d = nx.density(gg)
        # get neighbors
        neighbors = []
        cmplx_val_fn = 0
        final_dens = 0
        for k in curr_nodes:
            neighbors = neighbors + list(G.neighbors(k))
        neighbors = list(set(neighbors) - set(curr_nodes))
        for m in neighbors:
            for k in curr_nodes:
                curr_nb = list(G.neighbors(k))
                if m in curr_nb:
                    # density of adding temporary node
                    nx.add_path(gg, [k, m])
                    temp_dens = nx.density(gg)
                    logging.warning('Temp all nodes')
                    logging.warning(gg.nodes)
                    logging.warning('Temporary Density')
                    logging.warning(temp_dens)
                    for i in intervals:
                        if temp_dens <= i:
                            temp_dens = i
                            break
                        else:
                            continue
                    gg.remove_node(m)  # remove node
                    logging.warning(temp_dens)
                    if temp_dens in value_functions:
                        curr_val_fn = value_functions[temp_dens]
                    else:
                        curr_val_fn = interpolate(value_functions, temp_dens)
                        value_functions[temp_dens] = curr_val_fn
                    logging.warning(curr_val_fn)
                    logging.warning('Corresponding set value function')
                    logging.warning(curr_val_fn)
                    neighb_val_fns[m] = curr_val_fn
        # find the node that has the highest value function
        logging.warning("find max val fn in list")
        logging.warning(neighb_val_fns)
        if len(neighbors) != 0:
            added_n = max(neighb_val_fns, key=neighb_val_fns.get)
            logging.warning("added node")
            logging.warning(added_n)
            # add node to graph
            for k in list(curr_nodes):
                # k = str(k)
                neighbors = list(G.neighbors(k))
                #  logging.warning(neighbors)
                if added_n in neighbors:
                    nx.add_path(gg, [added_n, k])
            val_fns.append(neighb_val_fns[added_n])  # max, get index
            nodes_order = list(gg.nodes())
            # nodes_order.append(added_n)# list(gg.nodes)
            logging.warning(nodes_order)
        else:
            logging.warning("No neighbors left so stop")
            final_dens = nx.density(gg)
            for i in intervals:
                if final_dens <= i:
                    final_dens = i
                    break
                else:
                    continue
            cmplx_val_fn = value_functions[final_dens]
            break
        if val_fns[len(val_fns) - 2] > val_fns[len(val_fns) - 1]:
            logging.warning('The current value function is')
            logging.warning(neighb_val_fns[added_n])
            cmplx_val_fn = val_fns[len(val_fns) - 1]
            final_dens = nx.density(gg)
            logging.warning("value function is decreasing so stop")
            logging.warning(val_fns)
            break
        else:
            logging.warning("if not decreasing then get neighbors of current nodes")
            logging.warning(curr_nodes)
            for k in list(curr_nodes):
                # k = str(k)
                neighbors = list(G.neighbors(k))
                #  logging.warning(neighbors)
                if added_n in neighbors:
                    #     logging.warning(added_n)
                    nx.add_path(gg, [added_n, k])
            # logging.warning(value_dict)
    tup_cmplx = (nodes_order, cmplx_val_fn)
 #   logging.warning('Total list of all complexes')
 #   logging.warning(all_lists)
    with open('./nodes_complexes/'+str(n), 'wb') as f:
        pickle_dump(tup_cmplx, f)
    with open('./nodes_complexes/'+str(n),'rb') as f:
        pickle_load(f)

def network():
    nodes_list = list(nodes)
    # parallel running
    num_cores = mul_cpu_count()
    Parallel(n_jobs=num_cores, backend='loky')(
        delayed(pred_complex)(node, nodes_list) for node in tqdm(nodes_list))

    pred_comp_list = []
    sdndap = pred_comp_list.append
    allfiles = './nodes_complexes/*'
    for fname in glob(allfiles, recursive=True):
        with open(fname, 'rb') as f:
            pred_comp_tmp = pickle_load(f)
        sdndap(pred_comp_tmp)
    with open('predicted_complexes humap.txt', 'w') as f:
        f.writelines([''.join(str(comp)) + '\n' for comp in pred_comp_list])
    with open('predicted_complexes humap.pkl', 'wb') as f:
        pickle_dump(pred_comp_list,f)
network()

#print(cmplx_info)
with open('humap_CORUM_complexes_node_lists.pkl','wb') as f:
    pickle_dump(list(nodes),f)

#with open('predicted_complexes density and value function humap.pkl','wb') as f:
 #   pickle_dump(cmplx_info)
# histogram of densities and value functions
densities = [d[1] for d in cmplx_info]
vals = [v[2] for v in cmplx_info]
d_freq = Counter(densities)
d_freq = [d[1] for d in d_freq.items()]
v_freq = Counter(vals)
v_freq = [v[1] for v in v_freq.items()]

plt.figure()
plt.hist(densities, bins = 'auto',label='density')
plt.savefig('Histogram of Density humap')
plt.figure()
plt.hist(vals,bins = 'auto',label='value functions')
plt.savefig('Histogram of VF humap')

x = densities #density
y = vals #value function
plt.figure()
plt.plot(x,y,'o')
plt.xlabel('Density')
plt.ylabel('Value Function')
plt.title('Value Function and Density Relationship')
plt.savefig('Value Function and Density Relationship Predicted humap')