import pickle
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import logging
import time
from argparse import ArgumentParser as argparse_ArgumentParser
from yaml import load as yaml_load, dump as yaml_dump, Loader as yaml_Loader


def network(G, gg, value_dict, dens_counter, valuefn_update, intervals, subgraphs):
    iteration = 0
    reward_dict = {}
    gamma = 0.5
    # run for each graph in subgraph
    for graph in subgraphs:
        all_nodes = list(G.nodes())
        iteration = iteration + 1
        sub = graph
        nodes_list = list(sub.nodes())
        for n in nodes_list:
            # create rewards dictionary to assign for nodes inside and outside of complex
            for i in nodes_list:
                reward_dict[str(i)] = 0.2
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
            nx.add_path(gg, [n, n2], weight=max_weight.get('weight'))

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
                            nx.add_path(gg, [k, m], weight=temp_weight.get('weight'))
                            temp_dens = nx.density(gg)
                            gg.remove_node(m)  # remove node
                            # get intervals for density
                            for i in intervals:
                                if temp_dens <= i:
                                    temp_dens = i
                                    break
                                else:
                                    continue

                            # new state if new density encountered
                            if temp_dens not in value_dict:
                                # find corresponding reward
                                reward = reward_dict[m]
                                update = reward + gamma * 0
                                valuefn_update[temp_dens] = [update]
                                imag_n = 0 + gamma*0
                            # if density encountered before, update VF
                            else:
                                # get value function of neighbor
                                old_val = value_dict[temp_dens]
                                reward = reward_dict[m]
                                update = reward + gamma * old_val
                                #vf_update = valuefn_update[temp_dens]
                                #vf_update.append(update)
                                # add imaginary node value function to stop program
                                imag_n = 0 + gamma*old_val
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
                    # If reward is positive value above 0, continue adding max node
                    d = nx.density(gg)
                    for i in intervals:
                        if d <= i:
                            d = i
                            break
                        else:
                            continue
                    # density frequency counter
                    if d not in value_dict.keys():
                        dens_counter[d] = 1
                    else:
                        dens_counter[d] += 1
                    # add node with maximum VF to subgraph
                    for k in list(curr_nodes):
                        neighbors = list(G.neighbors(k))
                        if added_n in neighbors:
                            ed_weight = G.get_edge_data(added_n, k)
                            nx.add_path(gg, [added_n, k], weight=ed_weight.get('weight'))
                            value_dict[d] = neighb_val[added_n]
    return gg
    # e += 1


def main():
    start_time = time.time()
    matplotlib.use('Agg')
    logging.basicConfig(level=logging.WARNING)
    matplotlib.use('tkagg')
    # input data
    parser = argparse_ArgumentParser("Input parameters")
    parser.add_argument("--input_training_file", default="", help="Training Complexes file path")
    parser.add_argument("--graph_file", default="", help="Graph edges file path")
    parser.add_argument("--train_results", default="", help="Directory for training results")
    args = parser.parse_args()

    # get training data
    file = args.input_training_file
    #file = "../../training_CORUM_complexes_node_lists.txt"
    with open(file) as f:
        complexes = f.read().splitlines()
    for c in range(len(complexes)):
        complexes[c] = complexes[c].split()

    # get edges data
    # filename = "../../humap_network_weighted_edge_lists.txt"
    filename = args.graph_file
    G = nx.read_weighted_edgelist(filename, nodetype=str)
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

    value_dict = {}
    gg = nx.Graph()
    dens = nx.density(gg)
    value_dict[dens] = 0
    dens_counter = {}  # frequency of each density encountered
    valuefn_update = {}  # shows how value fn changes over time for each density
    intervals = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
                 0.95, 1]

    network(G, gg, value_dict, dens_counter, valuefn_update, intervals, subgraphs)
    # save value function scores in dictionary
    fname = args.train_results + "/value_fn_dens_dict.txt"
    file = open(fname, "w")
    value_dict_sorted = sorted(value_dict.items())
    # value_dict_sort = {keys[i]: vals[i] for i in range(len(keys))}
    str_dictionary = repr(value_dict_sorted)
    file.write(str_dictionary + "\n")
    file.close()
    fname = args.train_results + "/value_fn_dens_dict.pkl"
    with open(fname, 'wb') as f:
        pickle.dump(value_dict_sorted, f)

    # Frequency of density visited
    fname = args.train_results + "/density_freq.txt"
    file = open(fname, "w")
    str_dictionary = repr(dens_counter)
    file.write("density  = " + str_dictionary + "\n")
    file.close()

    # plotting Value Function vs Density
    keys = [key[0] for key in value_dict_sorted]
    vals = [val[1] for val in value_dict_sorted]
    x = keys  # density
    y = vals  # value function
    plt.figure()
    plt.plot(x, y, 'o-')
    plt.xlabel('Density')
    plt.ylabel('Value Function')
    plt.title('Value Function and Density Relationship')
    plt.savefig(args.train_results + '/Value Function and Density Relationship.png')

    # plot updating value fns for each one
#    keys = valuefn_update.keys()
#    for i in keys:
#        y = valuefn_update[i]
#        plt.figure()
#        plt.plot(y, 'o-')
#        plt.xlabel('Time')
#        plt.ylabel('Value Function')
#        str_key = str(i)
#        title = 'Value Function and Density Over Time for ' + str_key
#        plt.title(title)
#        plt.savefig(args.train_results + '/' + title + '.png')
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
