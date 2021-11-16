import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import csv
import logging

matplotlib.use('Agg')
logging.basicConfig(level=logging.WARNING)
gamma = 0.5
matplotlib.use( 'tkagg' )
# pentagon graph
K = nx.Graph()
nx.add_path(K, [1, 2, 3, 4, 5, 1])
nx.add_path(K, [1, 6])
nx.add_path(K, [2, 7])
nx.add_path(K, [3, 8])
nx.add_path(K, [4, 9])
nx.add_path(K, [5, 10])
K.add_nodes_from([11, 12, 13, 14, 15])
nx.draw(K, with_labels=True)
plt.savefig("pentagon")
rewards_dict_K = {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2, 6: -0.2,
                  7: -0.2, 8: -0.2, 9: -0.2,
                  10: -0.2, 11: -0.2, 12: -0.2,
                  13: -0.2, 14: -0.2, 15: -0.2, 16: -0.2}

# weird shape
I = nx.Graph()
nx.add_path(I, [16, 20, 22, 21, 19, 20, 17, 18, 23])
nx.add_path(I, [23, 25])
nx.add_path(I, [18, 28])
nx.add_path(I, [17, 26, 27])
nx.add_path(I, [22, 24])
plt.figure()
nx.draw(I, with_labels=True)
plt.savefig("scribble")
rewards_dict_I = {16: 0.2, 17: 0.2, 18: 0.2, 19: 0.2, 20: 0.2,
                  21: 0.2, 22: 0.2, 23: -0.2,
                  24: -0.2, 25: -0.2, 26: -0.2,
                  27: -0.2, 28: -0.2, 30: -0.2, 5: -0.2}
# kite
J = nx.Graph()
nx.add_path(J, [29, 30, 31, 32, 30, 31, 29, 32, 33, 32, 30, 34])
nx.add_path(J, [33, 31])
nx.add_path(J, [29, 35])
nx.add_path(J, [30, 36])
nx.add_path(J, [31, 37])
nx.add_path(J, [18, 30])
plt.figure()
nx.draw(J, with_labels=True)
plt.savefig("kite")
rewards_dict_J = {29: 0.2, 30: 0.2, 31: 0.2, 32: 0.2, 33: -0.2,
                  34: -0.2, 35: -0.2, 36: -0.2, 37: -0.2, 39: -0.2, 18: -0.2}

# almost star shape
H = nx.Graph()
nx.add_path(H, [38, 40, 43, 39, 41, 42, 38])
nx.add_path(H, [42, 44])
H.add_nodes_from([45, 46])
plt.figure()
nx.draw(H, with_labels=True)
plt.savefig("almost_star")
rewards_dict_H = {38: 0.2, 39: 0.2, 40: 0.2, 41: 0.2, 42: 0.2, 43: 0.2,
                  44: -0.2, 45: -0.2, 46: -0.2, 32: -0.2}
plt.figure()
all_graphs = [K, I, J, H]

# combine all complexes
allgraphs = nx.compose_all(all_graphs)
nx.add_path(allgraphs, [5, 16])
nx.add_path(allgraphs, [18, 30])
nx.add_path(allgraphs, [32, 39])
plt.figure()
nx.draw(allgraphs, with_labels=True)
plt.savefig("all complexes")
print(allgraphs.nodes())
# plt.figure()
# nx.draw(complex_graphs, with_labels=True)
# plt.savefig("all graphs")
# print(G.edges)
action_returns = []
node_order = []
value_dict = {}
rewards_dict = {}
g = nx.Graph()
dens = nx.density(g)
value_dict[dens] = 0
dens_counter = {}


def network():
    global imag_n
    iteration = 0
    for graph in range(len(all_graphs)):
        iteration = iteration + 1
        G = all_graphs[graph]
        nodes_list = list(G.nodes())
        n = random.choice(nodes_list)
        # make sure n is not a node floating around
        while len(list(allgraphs.neighbors(n))) == 0:
           n = random.choice(nodes_list)
           if allgraphs.neighbors(n) != 0:
             False
           else:
             continue
        g = nx.Graph()
        nx.add_path(g, [n])
        logging.warning(n)
        if all_graphs[graph] == K:
            rewards_dict = rewards_dict_K
        elif all_graphs[graph] == H:
            rewards_dict = rewards_dict_H
        elif all_graphs[graph] == I:
            rewards_dict = rewards_dict_I
        else:
            rewards_dict = rewards_dict_J

        # empty dictionary, add key in loop if not in there already (if it is just update current)
        iter = 0
        # value iteration
        while True:
            logging.warning(value_dict)
            iter = iter + 1
            # Initial value functions of states are 0
            all_nodes = list(g.nodes)  # all current nodes
            logging.warning(all_nodes)
            d = nx.density(g)
            # get neighbors
            neighbors = []
            update = 0
            imag_n = 0
            neighb_val = {}
            for k in range(len(all_nodes)):
                neighbors = neighbors + list(allgraphs.neighbors(all_nodes[k]))
            neighbors = list(set(neighbors) - set(all_nodes))
            logging.warning(neighbors)
            for m in range(len(neighbors)):
                for k in range(len(all_nodes)):
                    curr_nb = allgraphs.neighbors(all_nodes[k])
                    if neighbors[m] in curr_nb:
                        logging.warning('Checking neighbors and temp density')
                        # density of adding temporary node
                        nx.add_path(g, [all_nodes[k], neighbors[m]])
                        temp_dens = nx.density(g)
                        g.remove_node(neighbors[m])  # remove node
                        # add new state if new density
                        if temp_dens not in value_dict:
                            logging.warning("Value function of new density")
                            dens_counter[temp_dens] = 1
                            # find corresponding reward
                            logging.warning(rewards_dict)
                            reward = rewards_dict[neighbors[m]]
                            update = reward + gamma * 0
                            logging.warning("reward:")
                            logging.warning(reward)
                            imag_n = 0 + gamma * 0  # add imaginary node value function
                        else:
                            logging.warning("Updating value function of density")
                            dens_counter[temp_dens] += 1
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
                        neighb_val[50] = imag_n
            # find the node that has the highest value function
            logging.warning("find max val fn in list")
            logging.warning(neighb_val)
            added_n = max(neighb_val, key=neighb_val.get)  # max, get index
            if added_n == 50:
                logging.warning("if imaginary node then stop")
                break
            else:
                for k in range(len(all_nodes)):
                    logging.warning("if not imaginary then get neighbors of current nodes")
                    neighbors = list(allgraphs.neighbors(all_nodes[k]))
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
print(dens_counter)
plt.figure()
nx.draw(p, with_labels=True)
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
plt.plot(x,y)
plt.xlabel('Density')
plt.ylabel('Value Function')
plt.title('Value Function and Density Relationship')
plt.savefig('Value Function and Density Relationship')
plt.show()