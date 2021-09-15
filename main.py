
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

matplotlib.use('Agg')

gamma = 0.5
g = nx.Graph()
nx.add_path(g, [1, 2])

nodes_left = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


G = nx.Graph()
nx.add_path(G, [1, 2, 3, 4, 5, 1])

#G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
nx.add_path(G, [1, 6])
nx.add_path(G, [2, 7])
nx.add_path(G, [3, 8])
nx.add_path(G, [4, 9])
nx.add_path(G, [5, 10])
G.add_nodes_from([11,12,13,14,15])


print(nx.density(G))

rewards_dict = {3: 0.2, 4: 0.2, 5:0.2, 6: -0.2,
                7: -0.2, 8: -0.2, 9: -0.2,
                10: -0.2, 11: -0.2, 12: -0.2,
                13: -0.2, 14: -0.2, 15: -0.2}

plt.figure()
nx.draw(G, with_labels=True)
plt.savefig("star")
#print(G.edges)
action_returns = []
node_order = []

def network():
    # value iteration
    while True:
        dens = []
        for n in range(len(nodes_left)):
            node = nodes_left[n]
            all_nodes = list(g.nodes)
            neighbors = list(G.neighbors(node))
        #add node and edge for node and connect to current nodes
            if len(neighbors) == 0:
               g.add_node(node)
               #print(g.nodes, ", ", nx.density(g))
            elif len(neighbors) == 1:
                for j in range(len(all_nodes)):
                    if neighbors[0] == all_nodes[j]:
                        nx.add_path(g, [node, neighbors[0]])
                    else:
                        g.add_node(node)
            else:
                for i in range(len(neighbors)):
                    print(node, ", ", neighbors)
                    for j in range(len(all_nodes)):
                        if neighbors[i] == all_nodes[j]:
                            nx.add_path(g, [node, neighbors[i]])
                            #print(node, ", ", neighbors[i])
               # print(g.nodes, ", ", nx.density(g))
            #find density of possible action
            new_dens = nx.density(g)
            # state value
            # get possible actions for state
            key_list = list(rewards_dict.keys())
            val_list = list(rewards_dict.values())
            reward = 0
            value_function = gamma * new_dens
            node_ = n
            for m in range(len(key_list)):
                if node_ == key_list[m]:
                    n_index = key_list.index(node_)
                    reward = val_list[n_index]
            action_returns.append(reward + value_function)
            #print(action_returns)
            # print(state_value)
            dens.append(new_dens)
            #print(node)
            #print(g.edges)
            nodes = list(g.nodes)
            #print(node, ", ",new_dens)
            #remove node to test other possibilities and repeat process
            if node in nodes:
                g.remove_node(node)
        #print(nx.density(G))
        #print(action_dict)
        #dict of nodes and corresponding value functions
        value_fn_dict = dict(zip(nodes_left, action_returns))
        v_fn = list(value_fn_dict.values())
        n_keys = list(value_fn_dict.keys())
        #sort dictionary from highest value of action
        action_returns.sort(reverse=True)
        sorted_vfn = action_returns
        sorted_nkeys = []
        for o in range(len(v_fn)):
           for l in range(len(v_fn)):
               if sorted_vfn[l] == v_fn[o]:
                  sorted_nkeys.append(n_keys[o])

        value_fn_dict = dict(zip(sorted_nkeys, sorted_vfn))
        # add highest value function node to graph
        new_node = sorted_nkeys[0]
        all_nodes = list(g.nodes)
        neighbors = list(G.neighbors(new_node))
        if len(neighbors) >= 3:
            print(new_node)
                    #   print(v_fn[j])
                        #print(new_node, ", ", neighbors)
             # add node and edge for node and connect to current nodes
        print(new_node)
        for k in range(len(neighbors)):
            for j in range(len(all_nodes)):
                if neighbors[k] == all_nodes[j]:
                   g.add_node(new_node)
                   g.add_edge(new_node, neighbors[k])
        # remove node from list and to order list
        if new_node in g.nodes:
           node_order.append(new_node)

        if node_order[len(node_order) - 1] in nodes_left:
            nodes_left.remove(node_order[len(node_order) - 1])
        if len(nodes_left) == 10:
            break
network()

plt.figure()

nx.draw(g, with_labels=True)
plt.savefig("goal_pic")
#(nx.density(g))
print(g.nodes, ", ", g.edges)

