import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import csv
import logging
import scipy as sp

matplotlib.use('Agg')
logging.basicConfig(level=logging.WARNING)
gamma = 0.5
matplotlib.use( 'tkagg' )


## Using real Protein Complexes
# get data
fileName = "training_CORUM_complexes_node_lists.txt"
fileObj = open(fileName, "r") #opens the file in read mode
complexes = fileObj.read().splitlines()
for c in range(len(complexes)):
    complexes[c] = complexes[c].split()

#puts the file into an array
fileObj.close()
weight = []
# edges data
fileName = "humap_network_weighted_edge_lists.txt"
G = nx.Graph()
G = nx.read_weighted_edgelist(fileName, nodetype=str)
fileObj.close()
# remove duplicate edges and none
G.remove_edges_from(nx.selfloop_edges(G))
for i in list(G.nodes()):
    if i.isnumeric() == False:
        G.remove_node(i)
#print(G.nodes())
        #print(i)
# subgraphs
subgraphs = []
for s in range(len(complexes)):
    comp_list = complexes[s]
    for b in range(len(comp_list)):
        if comp_list[b] not in G.nodes():
            G.add_nodes_from(complexes[s])
    sub = G.subgraph(complexes[s])
    subgraphs.append(complexes[s])
#print(G.nodes())
logging.warning('Number of Nodes and Edges')
logging.warning(G.number_of_nodes())
logging.warning(G.number_of_edges())

# plot original raw data
#plt.figure()
#nx.draw(G, with_labels=True)
#plt.savefig("raw graph.png")

action_returns = []
node_order = []
value_dict = {}
reward_dict = {}
gg = nx.Graph()
dens = nx.density(gg)
value_dict[dens] = 0
dens_counter = {}
valuefn_change = {}


def network():
    global imag_n
    iteration = 0
    reward_dict = {}
    for graph in subgraphs:
        all_nodes = list(G.nodes())
        iteration = iteration + 1
        sub = G.subgraph(graph)
        nodes_list = list(sub.nodes())
        logging.warning('Current graph')
        logging.warning(nodes_list)
        for n in nodes_list:
            # create rewards dictionary
            for i in nodes_list:
               reward_dict[str(i)] = 0.2
            str_list = [str(int) for int in nodes_list]
            nodes_list_set = set(str_list)
            all_nodes_set = set(all_nodes)
            remaining_n = all_nodes_set - nodes_list_set
            #print(remaining_n)
            for i in remaining_n:
                reward_dict[i] = -0.2

        # make sure n is not a node floating around
            logging.warning('Current node')
            logging.warning(n)
            logging.warning('Neighbors')
            logging.warning(len(list(G.neighbors(str(n)))))
            while len(list(G.neighbors(str(n)))) == 0:
                i = nodes_list.index(str(n))+1
                logging.warning('Neighbors = 0')
                n = nodes_list[i]
                logging.warning(len(list(G.neighbors(str(n)))))
                if len(list(G.neighbors(str(n)))) != 0:
                    break
                i +=1
            # new graph to store new complexes
            gg = nx.Graph()
            nx.add_path(gg, [n])
            logging.warning('Added Node to test value function')
            logging.warning(gg.nodes())

            # empty dictionary, add key in loop if not in there already (if it is just update current)
            # value iteration
            while True:
                logging.warning('Value Dictionary')
                logging.warning(value_dict)
                # Initial value functions of states are 0
                curr_nodes = gg.nodes  # all current nodes
                logging.warning('Current Nodes in updating graph')
                logging.warning(curr_nodes)
                d = nx.density(gg)
                # get neighbors
                neighbors = []
                update = 0
                imag_n = 0
                neighb_val = {}
                for k in curr_nodes:
                    neighbors = neighbors + list(G.neighbors(k))
                neighbors = list(set(neighbors) - set(curr_nodes))
                logging.warning('Neighbors of current node')
                logging.warning(neighbors)
                for m in neighbors:
                    for k in curr_nodes:
                        curr_nb = list(G.neighbors(k))
                        if m in curr_nb:
                            logging.warning('Checking neighbors and temp density')
                        # density of adding temporary node
                            nx.add_path(gg, [k, m])
                            temp_dens = nx.density(gg)
                            logging.warning(gg.nodes())
                            gg.remove_node(m)  # remove node
                            logging.warning('Remove Node')
                            logging.warning(gg.nodes())
                            update_list = []

                            # add new state if new density
                            if temp_dens not in value_dict:
                                logging.warning("Value function of new density")
                               # dens_counter[temp_dens] = 1
                              #  logging.warning(temp_dens)
                            # find corresponding reward
                                reward = reward_dict[m]
                                logging.warning(m)
                                logging.warning(reward)
                                logging.warning(reward_dict[m])
                                update = reward + gamma * 0
                                logging.warning("reward:")
                                logging.warning(reward)
                                update_list.append(update)
                                valuefn_change[temp_dens] = update
                                imag_n = 0 + gamma * 0  # add imaginary node value function
                            else:
                                logging.warning("Updating value function of density")
                                logging.warning(dens_counter)
                              #  dens_counter[temp_dens] += 1
                               # curr_valfns = valuefn_change[temp_dens]
                               # get value function of neighbor
                                old_val = value_dict[temp_dens]
                                #print(temp_dens, ":", old_val)
                                reward = reward_dict[m]
                                update = reward + gamma * old_val
                                logging.warning(update)
                               # curr_valfns.append(update)
                                update_list.append(update)
                                valuefn_change[temp_dens] = update
                                logging.warning("reward:")
                                logging.warning(reward)
                            # add imaginary node value function
                                imag_n = 0 + gamma * old_val
                            logging.warning("update node and value fn in list:")
                            neighb_val[m] = update
                            logging.warning("dictionary to see how density value fns change over iterations")
                            logging.warning(valuefn_change)
                            neighb_val[2] = imag_n
            # find the node that has the highest value function
                logging.warning("find max val fn in list")
                logging.warning(neighb_val)
                if len(neighbors) != 0:
                    added_n = max(neighb_val, key=neighb_val.get)  # max, get index
                else:
                    added_n = 2
                logging.warning(added_n)
                logging.warning(neighb_val.get(added_n))
                if added_n == 2:
                   logging.warning("if imaginary node then stop")
                   break
                else:
                   logging.warning("if not imaginary then get neighbors of current nodes")
                   logging.warning(curr_nodes)
                   for k in list(curr_nodes):
                        #k = str(k)
                        neighbors = list(G.neighbors(k))
                        logging.warning(neighbors)
                        if added_n in neighbors:
                           logging.warning(added_n)
                           nx.add_path(gg, [added_n, k])
                           value_dict[d] = neighb_val[added_n]
                           logging.warning(d)
                   logging.warning(value_dict)
            file = open("Value_dictionary_protein.txt", "w")
            str_dictionary = repr(value_dict)
            file.write("value_dict  = " + str_dictionary + "\n")
            file.close()
    return gg
network()
print(dens_counter)
plt.figure()
nx.draw(gg, with_labels=True)
plt.savefig("final")
file = open("Value_dictionary Final.txt", "w")
str_dictionary = repr(value_dict)
file.write("value_dict  = " + str_dictionary + "\n")
file.close()

# Frequency of density visited
file = open("Density Frequency.txt", "w")
str_dictionary = repr(dens_counter)
file.write("density  = " + str_dictionary + "\n")
file.close()

# plotting Value Function vs Density
x = list(value_dict.keys())
x.pop(0)
y = list(value_dict.values())
y.pop(0)
print(x,y)
plt.figure()
plt.plot(x,y,'o-')
plt.xlabel('Density')
plt.ylabel('Value Function')
plt.title('Value Function and Density Relationship')
plt.savefig('Value Function and Density Relationship')
plt.show()