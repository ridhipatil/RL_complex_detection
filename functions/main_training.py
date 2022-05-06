import pickle
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import logging
import time

import numpy as np

start_time = time.time()
matplotlib.use('Agg')
logging.basicConfig(level=logging.WARNING)
gamma = 0.5
matplotlib.use( 'tkagg' )

# get training data
with open("../../training_CORUM_complexes_node_lists.txt") as f:
    complexes = f.read().splitlines()
for c in range(len(complexes)):
    complexes[c] = complexes[c].split()
weight = []

# get edges data
fileName = "../../humap_network_weighted_edge_lists.txt"
G = nx.Graph()
G = nx.read_weighted_edgelist(fileName, nodetype=str)
f.close()
# remove duplicate edges and none
G.remove_edges_from(nx.selfloop_edges(G))
for i in list(G.nodes()):
    if i.isnumeric() is False:
        G.remove_node(i)

# create subgraphs from training complexes
subgraphs = []
for s in range(len(complexes)):
    sub = G.subgraph(complexes[s])
    subgraphs.append(sub)


action_returns = []
node_order = []
value_dict = {}
reward_dict = {}
gg = nx.Graph()
dens = nx.density(gg)
value_dict[dens] = 0
dens_counter = {}
valuefn_update = {}

def interval(graph):
    # intervals for states to better organize and observe data
    # i.e interval 1 is for states in between 0.05-1, interval 0.95 is for states in between 0.9-0.95, etc.
    intervals = list(np.arange(0.05,1.05,0.05))

    d = nx.density(graph)
    for i in intervals:
        if d <= i:
            d = i
            break
        else:
            continue
    return d
def network():
    global imag_n
    iteration = 0
    reward_dict = {}
    update_list = []
    epochs = 1
    e = 1
    while e <= epochs:
        # run for each graph in subgraph
        for graph in subgraphs:
            all_nodes = list(G.nodes())
            iteration = iteration + 1
            sub = graph
            nodes_list = list(sub.nodes())
            logging.warning('Current graph')
            logging.warning(nodes_list)
            for n in nodes_list:
                # create rewards dictionary to assign for nodes inside and outside of complex
                for i in nodes_list:
                    reward_dict[i] = 0.2
                str_list = [str(n) for n in nodes_list]
                nodes_list_set = set(str_list)
                all_nodes_set = set(all_nodes)
                remaining_n = all_nodes_set - nodes_list_set
                for i in remaining_n:
                    reward_dict[i] = -0.2

                # make sure n is not a node floating around and has neighbors
                neighb_n = list(G.neighbors(n))
                while len(neighb_n) == 0:
                    i = nodes_list.index(n) + 1
                    n = nodes_list[i]
                    if len(neighb_n) != 0:
                        break
                    i += 1

                # new graph to store new complexes
                gg = nx.Graph()
                x = [(neib, G.get_edge_data(n, neib)) for neib in neighb_n]

                # add neighbor with edge that gives maximum edge weight
                n2 = max(x, key=lambda x: x[1]['weight'])[0]
                max_weight = G.get_edge_data(n, n2)
                nx.add_path(gg, [n, n2], weight = max_weight.get('weight'))

                # value iteration
                while True:
                    # Initial value functions of states are 0
                    curr_nodes = gg.nodes  # all current nodes

                    # get neighbors of current nodes
                    neighbors = []
                    imag_n = 0
                    neighb_val = {}
                    for k in curr_nodes:
                        neighbors = neighbors + list(G.neighbors(k))
                    neighbors = list(set(neighbors) - set(curr_nodes))
                    for m in neighbors:
                        for k in curr_nodes:
                            curr_nb = list(G.neighbors(k))
                            if m in curr_nb:
                                temp_weight = G.get_edge_data(k, m)
                                nx.add_path(gg, [k, m], weight = temp_weight.get('weight'))
                                gg.remove_node(m)  # remove node

                                # get intervals for density
                                temp_dens = interval(gg)

                                # new state if new density encountered
                                if temp_dens not in value_dict:
                                    logging.warning("Value function of new density")
                                    # find corresponding reward
                                    reward = reward_dict[m]
                                    update = reward + gamma * 0
                                    imag_n = 0  # add imaginary node value function to stop program

                                # if density encountered before, update VF
                                else:
                                    logging.warning("Updating value function of density")
                                    # get value function of neighbor
                                    old_val = value_dict[temp_dens]
                                    reward = reward_dict[m]
                                    update = reward + gamma * old_val
                                # add imaginary node value function
                                imag_n = 0
                                neighb_val[m] = update

                    # find the node that has the highest value function
                    neighb_val[2] = imag_n
                    if len(neighbors) != 0:
                        added_n = max(neighb_val, key=neighb_val.get)  # max, get index
                    else:
                        added_n = 2
                    if added_n == 2:
                        break
                    else:
                        # Value function is not less than 0, continue adding max node

                        # add node with maximum VF to subgraph
                        for k in list(curr_nodes):
                            neighbors = list(G.neighbors(k))
                            if added_n in neighbors:
                                ed_weight = G.get_edge_data(added_n, k)
                                nx.add_path(gg, [added_n, k], weight = ed_weight.get('weight'))
                                d = interval(gg)
                                value_dict[d] = neighb_val[added_n]

                            # frequency of encountering value functions
                            if d not in valuefn_update:
                                update_list = [0]
                                update_list.append(neighb_val[added_n])
                                valuefn_update[d] = update_list
                                dens_counter[d] = 1
                            else:
                                update_list = valuefn_update[d]
                                update_list.append(neighb_val[added_n])
                                valuefn_update[d] = update_list
                                dens_counter[d] += 1
        return gg
    e += 1
def main():
    network()
    # save value function scores in dictionary
    file = open("../Value_dictionary.txt", "w")
    value_dict_sorted = sorted(value_dict.items())
    #value_dict_sort = {keys[i]: vals[i] for i in range(len(keys))}
    str_dictionary = repr(value_dict_sorted)
    file.write(str_dictionary + "\n")
    file.close()
    with open('../Value_Dictionary.pkl', 'wb') as f:
        pickle.dump(value_dict_sorted, f)

    # Frequency of density visited
    file = open("../../Density Frequency.txt", "w")
    str_dictionary = repr(dens_counter)
    file.write("density  = " + str_dictionary + "\n")
    file.close()

    # plotting Value Function vs Density
    keys = [key[0] for key in value_dict_sorted]
    vals = [val[1] for val in value_dict_sorted]
    x = keys #density
    y = vals #value function
    plt.figure()
    plt.plot(x,y,'o-')
    plt.xlabel('Density')
    plt.ylabel('Value Function')
    plt.title('Value Function and Density Relationship')
    plt.savefig('Value Function and Density Relationship ')
    print(value_dict_sorted)

    # plot updating value fns for each one
    keys = valuefn_update.keys()
    for i in keys:
        y = valuefn_update[i]
        print(i,": ",y[len(y)-1])
        plt.figure()
        plt.plot(y,'o-')
        plt.xlabel('Time')
        plt.ylabel('Value Function')
        str_key = str(i)
        title = 'Value Function and Density Over Time for '+ str_key
        plt.title(title)
        plt.savefig(title+'.png')
    file = open("../Value Function and Density Over Time.txt", "w")
    str_dictionary = repr(valuefn_update)
    file.write(str_dictionary)
    file.close()

print("--- %s seconds ---" % (time.time() - start_time))

