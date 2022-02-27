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
matplotlib.use('Agg')
logging.basicConfig(level=logging.WARNING)

fileName = "testing_CORUM_complexes_node_lists.txt"
fileObj = open(fileName, "r") #opens the file in read mode
complexes = fileObj.read().splitlines()
for c in range(len(complexes)):
    complexes[c] = complexes[c].split()

with open('Value Functions.txt','rb') as f:
    value_functions = pickle_load(f)
value_functions = dict(value_functions)
print(value_functions)
# edges data
fileName = "humap_network_weighted_edge_lists.txt"
G = nx.Graph()
G = nx.read_weighted_edgelist(fileName, nodetype=str)
fileObj.close()
# remove duplicate edges and none
G.remove_edges_from(nx.selfloop_edges(G))
for i in list(G.nodes()):
    if i.isnumeric() is False:
        G.remove_node(i)

# subgraphs
subgraphs = []
for s in range(len(complexes)):
    comp_list = complexes[s]
    for b in range(len(comp_list)):
        if comp_list[b] not in G.nodes():
            G.add_nodes_from(complexes[s])
    sub = G.subgraph(complexes[s])
    subgraphs.append(complexes[s])

val_fns = []
all_orders = []
def network():
    global imag_n
    visited = set()
    for graph in subgraphs:
        all_nodes = list(G.nodes())
        sub = G.subgraph(graph)
        nodes_list = list(sub.nodes())
        logging.warning('Current graph')
        logging.warning(nodes_list)
        for n in nodes_list:
            if n in visited:
                continue
            else:
                visited.add(n)
        # make sure n is not a node floating around
            neighb_n = list(G.neighbors(str(n)))
            while len(neighb_n) == 0:
                i = nodes_list.index(str(n))+1
                n = nodes_list[i]
                if len(neighb_n) != 0:
                    break
                i += 1
            # new graph to store new complexes
            gg = nx.Graph()
            x = [(neib,G.get_edge_data(n,neib)) for neib in neighb_n]
            n2 = max(x, key=lambda x: x[1]['weight'])[0]
            nx.add_path(gg, [n, n2])
            logging.warning('Added Node to test value function')
            logging.warning(gg.nodes())
            imag_n = 0
            d = nx.density(gg)
            # empty dictionary, add key in loop if not in there already (if it is just update current)
            # value iteration
            while True:
                # Initial value functions of states are 0
                curr_nodes = gg.nodes # all current nodes
                neighb_val_fns = {}
                nodes_order = [n,n2]
                logging.warning("The current nodes in gg are")
                logging.warning(curr_nodes)
                d = nx.density(gg)
                # get neighbors
                neighbors = []
                imag_n = 0
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
                            gg.remove_node(m)  # remove node
                            curr_val_fn = value_functions[temp_dens]
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
                            #     logging.warning(added_n)
                            nx.add_path(gg, [added_n, k])
                    val_fns.append(neighb_val_fns[added_n]) # max, get index
                    nodes_order.append(added_n)# list(gg.nodes)
                    logging.warning(nodes_order)
                else:
                    logging.warning("No neighbors left so stop")
                    break
                if val_fns[len(val_fns)-2] > val_fns[len(val_fns)-1]:
                   logging.warning('The current value function is')
                   logging.warning(neighb_val_fns[added_n])
                   logging.warning('The old value function is')
                   logging.warning(value_functions[d])
                   logging.warning("value function is decreasing so stop")
                   break
                else:
                   logging.warning("if not decreasing then get neighbors of current nodes")
                   logging.warning(curr_nodes)
                   for k in list(curr_nodes):
                        #k = str(k)
                        neighbors = list(G.neighbors(k))
                      #  logging.warning(neighbors)
                        if added_n in neighbors:
                      #     logging.warning(added_n)
                           nx.add_path(gg, [added_n, k])
                           logging.warning(gg.nodes)

                  # logging.warning(value_dict)
            all_orders.append(nodes_order)
            logging.warning('Total list of all complexes')
            logging.warning(all_orders)
    return gg
network()
# remove redudancy
#uniq_comps = set()
#for comp in all_orders:
 #   if set(comp) not in uniq_comps:
  #      uniq_comps.add(set(comp))

with open('predicted_complexes.txt','w') as f:
    f.writelines([' '.join(comp) + '\n' for comp in all_orders])
