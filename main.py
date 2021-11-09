
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import csv
import logging
matplotlib.use('Agg')
logging.basicConfig(level=logging.WARNING)
gamma = 0.5
g = nx.Graph()
nx.add_path(g, [1])
# pentagon graph
K = nx.Graph()
nx.add_path(K, [1,2,3,4,5,1])
nx.add_path(K, [1, 6])
nx.add_path(K, [2, 7])
nx.add_path(K, [3, 8])
nx.add_path(K, [4, 9])
nx.add_path(K, [5, 10])
K.add_nodes_from([11,12,13,14,15])
nx.draw(K, with_labels=True)
plt.savefig("pentagon")
rewards_dict_K = {2:0.2,3: 0.2, 4: 0.2, 5: 0.2, 6: -0.2,
                7: -0.2, 8: -0.2, 9: -0.2,
                10: -0.2, 11: -0.2, 12: -0.2,
                13: -0.2, 14: -0.2, 15: -0.2}

# almost star shape
H = nx.Graph()
nx.add_path(H, [1,3,6,2,4,5,1])
nx.add_path(H, [5,7])
H.add_nodes_from([8,9])
plt.figure()
nx.draw(H, with_labels=True)
plt.savefig("almost_star")
rewards_dict_H = {2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2, 6: 0.2,
                7: -0.2, 8: -0.2, 9: -0.2}

# weird shape
I = nx.Graph()
nx.add_path(I, [1,5,7,6,4,5,2,3,8])
nx.add_path(I, [8,10])
nx.add_path(I, [3,13])
nx.add_path(I, [2,11])
nx.add_path(I, [11,12])
nx.add_path(I, [7,9])
plt.figure()
nx.draw(K, with_labels=True)
plt.savefig("scribble")
rewards_dict_I = {2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2, 6: 0.2,
                7: 0.2, 8: 0.2, 9: -0.2,
                10: -0.2, 11: -0.2, 12: -0.2,
                13: -0.2}
# kite
J = nx.Graph()
nx.add_path(J, [1,2,3,4,2,3,1,4,5,4,2,6])
nx.add_path(J, [5,3])
nx.add_path(J, [1,7])
nx.add_path(J, [2,8])
nx.add_path(J, [3,9])
plt.figure()
nx.draw(J, with_labels=True)
plt.savefig("pentagon")
rewards_dict_J = {2:0.2 ,3: 0.2, 4: 0.2, 5: -0.2, 6: 0.2,
                7: -0.2, 8: -0.2, 9: -0.2}

all_graphs = [K,H,I,J]
complex_graphs = []
complex_graphs_append = complex_graphs.append
for comp in all_graphs:
   Gsub = g.subgraph(comp)
   if Gsub not in complex_graphs:
      complex_graphs_append(Gsub)
#plt.figure()
#nx.draw(complex_graphs, with_labels=True)
#plt.savefig("all graphs")
#print(G.edges)
action_returns = []
node_order = []
value_dict = {}
rewards_dict = {}
dens = nx.density(g)
value_dict[dens] = 0

def network():
    global imag_n
    iteration = 0
    for graph in range(len(all_graphs)):
        iteration = iteration+1
        G = all_graphs[graph]
        logging.warning(G)
        if all_graphs[graph] == K:
            rewards_dict = rewards_dict_K
        if all_graphs[graph] == H:
            rewards_dict = rewards_dict_H
        if all_graphs[graph] == I:
            rewards_dict = rewards_dict_I
        else:
            rewards_dict = rewards_dict_K

    # empty dictionary, add key in loop if not in there already (if it is just update current)
        iter = 0
    # value iteration
        while True:
            logging.warning(value_dict)
            iter = iter+1
        # Initial value functions of states are 0
            all_nodes = list(g.nodes) # all current nodes
            logging.warning(all_nodes)
            d = nx.density(g)
        # get neighbors
            neighbors = []
            update = 0
            imag_n = 0
            neighb_val = {}
            for k in range(len(all_nodes)):
                neighbors = neighbors + list(G.neighbors(all_nodes[k]))
            neighbors = list(set(neighbors) - set(all_nodes))
            logging.warning(neighbors)
            for m in range(len(neighbors)):
                for k in range(len(all_nodes)):
                    curr_nb = G.neighbors(all_nodes[k])
                    if neighbors[m] in curr_nb:
                        logging.warning('Checking nieghbors and temp density')
                    # density of adding temporary node
                        nx.add_path(g, [all_nodes[k], neighbors[m]])
                        temp_dens = nx.density(g)
                        g.remove_node(neighbors[m]) # remove node
                    # add new state if new density
                        if temp_dens not in value_dict:
                            logging.warning("Value function of new density")
                        # find corresponding reward
                            reward = rewards_dict[neighbors[m]]
                            update = reward + gamma*0
                            logging.warning("reward:")
                            logging.warning(reward)
                            imag_n = 0 + gamma*0 # add imaginary node value function
                        else:
                            logging.warning("Updating value function of density")
                        # get value function of neighbor
                            old_val = value_dict[temp_dens]
                            reward = rewards_dict[neighbors[m]]
                            update = reward + gamma * old_val
                            logging.warning("reward:")
                            logging.warning(reward)
                        # add imaginary node value function
                            imag_n = 0 + gamma * old_val
                        logging.warning("update node and value fn in list:")
                        neighb_val[neighbors[m]] = update
                        neighb_val[16] = imag_n
        #find the node that has the highest value function
            logging.warning("find max val fn in list")
            logging.warning(neighb_val)
            added_n = max(neighb_val, key = neighb_val.get) # max, get index
            if added_n == 16:
                logging.warning("if imaginary node then stop")
                break
            else:
                for k in range(len(all_nodes)):
                    logging.warning("if not imaginary then get neighbors of current nodes")
                    neighbors = list(G.neighbors(all_nodes[k]))
                    for j in range(len(neighbors)):
                        if neighbors[j] == added_n:
                            logging.warning(added_n)
                            nx.add_path(g, [added_n, all_nodes[k]])
                            value_dict[d] = neighb_val[added_n]
            logging.warning(value_dict)
        if iteration == 1:
            file = open("Value_dictionary K.txt", "w")
            str_dictionary = repr(value_dict)
            file.write("value_dict  = " + str_dictionary + "\n")
            file.close()
        if iteration == 2:
            file = open("Value_dictionary H.txt", "w")
            str_dictionary = repr(value_dict)
            file.write("value_dict  = " + str_dictionary + "\n")
            file.close()
        if iteration == 3:
            file = open("Value_dictionary I.txt", "w")
            str_dictionary = repr(value_dict)
            file.write("value_dict  = " + str_dictionary + "\n")
            file.close()
        else:
            file = open("Value_dictionary J.txt", "w")
            str_dictionary = repr(value_dict)
            file.write("value_dict  = " + str_dictionary + "\n")
            file.close()
    return g
p = network()
plt.figure()
nx.draw(p, with_labels=True)
plt.savefig("final")
file = open("Value_dictionary Final.txt","w")
str_dictionary = repr(value_dict)
file.write("value_dict  = " + str_dictionary + "\n")
file.close()
