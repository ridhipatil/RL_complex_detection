
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
nx.add_path(g, [1, 2])

nodes_left = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


G = nx.Graph()
nx.add_path(G, [1,2,3,4,5,1])

#G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
nx.add_path(G, [1, 6])
nx.add_path(G, [1, 2])
nx.add_path(G, [2, 7])
nx.add_path(G, [3, 8])
nx.add_path(G, [4, 9])
nx.add_path(G, [5, 10])
G.add_nodes_from([11,12,13,14,15])


print(nx.density(G))

rewards_dict = {3: 0.2, 4: 0.2, 5: 0.2, 6: -0.2,
                7: -0.2, 8: -0.2, 9: -0.2,
                10: -0.2, 11: -0.2, 12: -0.2,
                13: -0.2, 14: -0.2, 15: -0.2}

nx.draw(G, with_labels=True)
plt.savefig("pentagon")
#print(G.edges)
action_returns = []
node_order = []

def network():
    global imag_n
    value_dict = {}
    # empty dictionary, add key in loop if not in there already (if it is just update current)
    dens = nx.density(g)
    value_dict[dens] = 0

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
                            # last_value_dict =
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
        if iter > 9:
            break
    return g
p = network()
plt.figure()
nx.draw(p, with_labels=True)
plt.savefig("pentagon_res")