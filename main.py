
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

matplotlib.use('Agg')


gamma = 0.5
g = nx.Graph()
nx.add_path(g, [1, 2, 5])

nodes_left = [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


G = nx.Graph()
nx.add_path(G, [1, 3, 5, 2, 4,1])

#G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
nx.add_path(G, [1, 6])
nx.add_path(G, [1, 2])
nx.add_path(G, [2, 7])
nx.add_path(G, [3, 8])
nx.add_path(G, [4, 9])
nx.add_path(G, [5, 10])
G.add_nodes_from([11,12,13,14,15])


print(nx.density(G))

rewards_dict = {3: 0.2, 4: 0.2, 6: 0.2,
                7: 0.2, 8: 0.2, 9: 0.2,
                10: 0.2, 11: -0.2, 12: -0.2,
                13: -0.2, 14: -0.2, 15: -0.2}

def network():
    # value iteration

    while True:
        node_order = []
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
                    for j in range(len(all_nodes)):
                        if neighbors[i] == all_nodes[j]:
                            nx.add_path(g, [node, neighbors[i]])
                            #print(node, ", ", neighbors[i] )
            #print(g.nodes, ", ", nx.density(g))
            new_dens = nx.density(g)
            dens.append(new_dens)
            #print(node)
            #print(g.edges)
            nodes = list(g.nodes)
            #print(nx.density(g))
            if node in nodes:
                g.remove_node(node)
        #print(nx.density(G))
        Density_dict = dict(zip(nodes_left, dens))

        # state value

        #print(dens)
        # get possible actions for state
        action_returns = []
        key_list = list(rewards_dict.keys())
        val_list = list(rewards_dict.values())
        reward = 0
        max_value = 0


        for s in range(len(dens)):
            value_function = gamma * dens[s]
            density = dens[s]
            node_ = nodes_left[s]
            for m in range(len(key_list)):
                if node_ == key_list[m]:
                   n_index = key_list.index(node_)
                   reward = val_list[n_index]
            action_returns.append(reward + value_function)
            max_value = np.max(action_returns)
            #print(state_value)

        #print(action_dict)
        #add node from highest value of action
        for i in range(len(action_returns)):
            if max_value == action_returns[i]:
               node__ = nodes_left[i]
               node_order.append(node__)
               all_nodes = list(g.nodes)
               neighbors = list(G.neighbors(node__))
             # print(node, ", ", neighbors)
             # add node and edge for node and connect to current nodes
               if len(neighbors) == 1:
                   for j in range(len(all_nodes)):
                       if neighbors[0] == all_nodes[j]:
                           nx.add_path(g, [node__, neighbors[0]])
               else:
                   for k in range(len(neighbors)):
                       for j in range(len(all_nodes)):
                          if neighbors[k] == all_nodes[j]:
                             nx.add_path(g, [node__, neighbors[k]])
        # remove node from list


        if node_order[len(node_order)-1] in nodes_left:
           nodes_left.remove(node_order[len(node_order) - 1])
        if len(nodes_left) == 0:
            break

# compute the optimal policy
    #policy = np.zeros(GOAL + 1)
    #for state in STATES[1:GOAL]:
     #   actions = np.arange(min(state, GOAL - state) + 1)
      #  action_returns = []
       # for action in actions:
        #    action_returns.append(
         #       HEAD_PROB * state_value[state + action] + (1 - HEAD_PROB) * state_value[state - action])

        # round to resemble the figure in the book, see
        # https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/issues/83
        #policy[state] = actions[np.argmax(np.round(action_returns[1:], 5)) + 1]
network()
nx.draw(g, with_labels=True)
plt.savefig("goal__")
print(nx.density(g))