import os
from pickle import load as pickle_load
from pickle import dump as pickle_dump
from yaml import load as yaml_load, dump as yaml_dump, Loader as yaml_Loader
import networkx as nx
from networkx import write_weighted_edgelist as nx_write_weighted_edgelist
from postprocess_sc import merge_filter_overlapped_score_qi
from convert_humap_ids2names import convert2names_wscores
from argparse import ArgumentParser as argparse_ArgumentParser
import time
import numpy as np

def main():
    start_time = time.time()
    # Input data
    parser = argparse_ArgumentParser("Input parameters")
    parser.add_argument("--input_file_name", default="", help="Input parameters file name")
    parser.add_argument("--graph_file", default="", help="Graph edges file path")
    parser.add_argument("--out_dir_name", default="", help="Output directory name")
    parser.add_argument("--pred_results", default="", help="Directory for prediction results")
    parser.add_argument("--train_results", default="", help="Directory for main results")
    parser.add_argument("--input_training_file", default="", help="Training Graph file path")
    parser.add_argument("--input_testing_file", default="", help="Testing Graph file path")
    #parser.add_argument("--threshold", default="", help="Qi or Jaccard threshold")
    parser.add_argument("--id_map_path", default="", help="Path for id to gene name file")

    args = parser.parse_args()
    #args.threshold = str(args.th)
    with open(args.input_file_name, 'r') as f:
        inputs = yaml_load(f, yaml_Loader)

    # make tuples of complex and corresponding value function
    fname = args.pred_results + "/predicted_complexes.pkl"
    with open(fname, 'rb') as f:
        all_lists = pickle_load(f)
    value_fns = [v[1] for v in all_lists]
    node_lists = [n[0] for n in all_lists]
    cmplx_tup = []
    for n in range(len(node_lists)):
        if isinstance(value_fns[n],np.ndarray) is True:
            value_fns[n] = float(value_fns[n])
        tup = (set(node_lists[n]), value_fns[n])  # make score the value function of complex
        cmplx_tup.append(tup)

    # postprocessing
    fileName = args.graph_file
    G = nx.read_weighted_edgelist(fileName, nodetype=str)
    # remove duplicate edges and none
    G.remove_edges_from(nx.selfloop_edges(G))
    for i in list(G.nodes()):
        if i.isnumeric() is False:
            G.remove_node(i)

    # Finding unique complexes
    file = args.pred_results + '/value_fns_pred.pkl'
    with open(file, 'rb') as f:
        value_fns_dict = pickle_load(f)
    fin_list_graphs = set([(frozenset(comp), score) for comp, score in cmplx_tup if len(comp) > 2])
    fin_list_graphs_orig = merge_filter_overlapped_score_qi(fin_list_graphs, inputs, G, value_fns_dict)
    fin_list_graphs_orig = [(set(comp), score) for comp, score in fin_list_graphs_orig]
    file = ''
    if inputs['dir_nm'] == 'toy_network':
        file = args.out_dir_name + '/qi_results'

        if not os.path.exists(args.out_dir_name + '/qi_results'):
            os.mkdir(args.out_dir_name + '/qi_results')
        filename = file + '/res'
    else:
        if inputs['overlap_method'] == 'qi':
           file = args.out_dir_name + '/qi_results'
           if not os.path.exists(args.out_dir_name + '/qi_results'):
               os.mkdir(args.out_dir_name + '/qi_results')
           #os.makedirs(args.out_dir_name + '/qi_results', exist_ok=True)
           filename = file + '/res'  # inputs['out_comp_nm']
           #os.makedirs(file + '/results_qi', exist_ok=True)
        elif inputs["overlap_method"] == '1':  # jaccard coeff
           file = args.out_dir_name + '/jacc_results'
           if not os.path.exists(file):
               os.mkdir(file)
           #os.makedirs(args.out_dir_name + '/jacc_results', exist_ok=True)
           filename = file + '/res'  # inputs['out_comp_nm']
           #os.makedirs(file + '/results_jacc', exist_ok=True)

    with open(filename + '_pred_complexes_pp.pkl', 'wb') as f:
        pickle_dump(fin_list_graphs_orig, f)
    with open(filename + '_pred_complexes_pp.txt', 'w') as f:
        f.writelines([''.join(str(comp)) + '\n' for comp in fin_list_graphs_orig])
    with open(filename + "_input_pp.yaml", 'w') as outfile:
        yaml_dump(inputs, outfile, default_flow_style=False)

    # write out protein names
    out_comp_nm = file + '/res' #inputs['out_comp_nm']
    if inputs['dir_nm'] == "humap2":  # humap
        convert2names_wscores(fin_list_graphs_orig, out_comp_nm + '_pred_names.out',
                              G, out_comp_nm + '_pred_edges_names.out', args.id_map_path)
    tot_pred_edges_unique_max_comp_prob = {}
    fin_list_graphs = sorted(fin_list_graphs_orig, key=lambda x: x[1], reverse=True)
    with open(args.input_testing_file, 'r') as f:
        testing = f.read().splitlines() #pickle_load(f)
    for c in range(len(testing)):
        testing[c] = testing[c].split()
    with open(args.input_training_file, 'r') as f:  # opens the file in read mode
        training = f.read().splitlines()
    for c in range(len(training)):
        training[c] = training[c].split()
    train_prot_list = [n for sublist in training for n in sublist]
    train_prot_list = set(train_prot_list)
    test_prot_list = [n for sublist in testing for n in sublist]
    test_prot_list = set(test_prot_list)
    known_complex_nodes_list = testing + training
    prot_list = [n for sublist in known_complex_nodes_list for n in sublist]
    prot_list = set(prot_list)
    with open(out_comp_nm + '_pred.out', "w") as fn:
        with open(out_comp_nm + '_pred_edges.out', "wb") as f_edges:
            fn_write = fn.write
            f_edges_write = f_edges.write
            for index in range(len(fin_list_graphs)):
                tmp_graph_nodes = fin_list_graphs[index][0]
                tmp_score = fin_list_graphs[index][1]
                for node in tmp_graph_nodes:
                    fn_write("%s " % node)
                fn_write("%.3f" % tmp_score)
                tmp_graph = G.subgraph(tmp_graph_nodes)
                nx_write_weighted_edgelist(tmp_graph, f_edges)
                tmp_graph_edges = tmp_graph.edges()

                for edge in tmp_graph_edges:
                    edge_set = frozenset([edge[0], edge[1]])
                    tmp_weight = G.get_edge_data(edge[0], edge[1]).get('weight')
                    if edge_set in tot_pred_edges_unique_max_comp_prob:
                        tot_pred_edges_unique_max_comp_prob[edge_set] = max(
                            tot_pred_edges_unique_max_comp_prob[edge_set], tmp_score)
                    else:
                        tot_pred_edges_unique_max_comp_prob[edge_set] = tmp_weight
                fn_write("\n")
                f_edges_write("\n".encode())

    with open(out_comp_nm + '_tot_pred_edges_unique_max_comp_prob.out', "w") as f:
        with open(out_comp_nm + '_tot_pred_edges_unique_max_comp_prob_inKnown.out', "w") as f_inKnown:
            with open(out_comp_nm + '_tot_pred_edges_unique_max_comp_prob_inKnown_train.out', "w") as f_inKnown_train:
                with open(out_comp_nm + '_tot_pred_edges_unique_max_comp_prob_inKnown_test.out', "w") as f_inKnown_test:
                    for edge_key in tot_pred_edges_unique_max_comp_prob:
                        edge = list(edge_key)
                        edge_score = tot_pred_edges_unique_max_comp_prob[edge_key]
                        f.write(edge[0] + "\t" + edge[1] + "\t" + "%.3f" % edge_score + "\n")

                        if edge[0] in prot_list and edge[1] in prot_list:
                            f_inKnown.write(edge[0] + "\t" + edge[1] + "\t" + "%.3f" % edge_score + "\n")
                        if edge[0] in train_prot_list and edge[1] in train_prot_list:
                            f_inKnown_train.write(edge[0] + "\t" + edge[1] + "\t" + "%.3f" % edge_score + "\n")
                        if edge[0] in test_prot_list and edge[1] in test_prot_list:
                            f_inKnown_test.write(edge[0] + "\t" + edge[1] + "\t" + "%.3f" % edge_score + "\n")
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()
