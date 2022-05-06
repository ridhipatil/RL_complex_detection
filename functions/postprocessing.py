from pickle import load as pickle_load
from pickle import dump as pickle_dump
from yaml import load as yaml_load, dump as yaml_dump, Loader as yaml_Loader
import networkx as nx
from postprocess import merge_filter_overlapped_score_qi
from convert_humap_ids2names import convert2names_wscores
from argparse import ArgumentParser as argparse_ArgumentParser

def post_process(cmplx_tup, inputs, G):
    fin_list_graphs = set([(frozenset(comp), score) for comp, score in cmplx_tup if len(comp) > 2])
    fin_list_graphs_orig = merge_filter_overlapped_score_qi(fin_list_graphs,inputs,G)
    out_comp_nm = inputs['dir_nm'] + inputs['out_comp_nm']
    if inputs['dir_nm'] == "humap": #humap
        convert2names_wscores(fin_list_graphs, out_comp_nm + '_pred_names.out',
                            G, out_comp_nm + '_pred_edges_names.out')

    return fin_list_graphs_orig

def main_humap():
    with open('./humap/predicted_complexes humap.pkl', 'rb') as f:
        all_lists = pickle_load(f)
    value_fns = [v[1] for v in all_lists]
    node_lists = [n[0] for n in all_lists]
    cmplx_tup = []
    for n in range(len(node_lists)):
        tup = (set(node_lists[n]), value_fns[n])  # make score the value function of complex
        cmplx_tup.append(tup)

    # postprocessing
    fileName = "../../humap_network_weighted_edge_lists.txt"
    G = nx.read_weighted_edgelist(fileName, nodetype=str)
    # remove duplicate edges and none
    G.remove_edges_from(nx.selfloop_edges(G))
    for i in list(G.nodes()):
        if i.isnumeric() is False:
            G.remove_node(i)

    # Finding unique complexes
    parser = argparse_ArgumentParser("Input parameters")
    parser.add_argument("--input_file_name", default="input_humap.yaml", help="Input parameters file name")
    parser.add_argument("--graph_files_dir", default="", help="Graph files' folder path")
    parser.add_argument("--out_dir_name", default="/results", help="Output directory name")
    args = parser.parse_args()
    with open(args.input_file_name, 'r') as f:
        inputs = yaml_load(f, yaml_Loader)

    fin_list_graphs_orig = post_process(cmplx_tup, inputs, G)
    fin_list_graphs_orig = [(set(comp), score) for comp, score in fin_list_graphs_orig]
    file = ''
    if inputs['overlap method'] == 'qi':
        file = '../qi_results'
    elif inputs['overlap method'] == '1':  #jaccard coeff
        file = '../jacc_results'
    filename = file + inputs['out_comp_nm']
    with open(filename + '/predicted_postprocess.pkl', 'wb') as f:
        pickle_dump(fin_list_graphs_orig,f)
    with open(filename + '/predicted_postprocess.txt', 'w') as f:
        f.writelines([''.join(str(comp)) + '\n' for comp in fin_list_graphs_orig])

    with open(inputs['dir_nm'] + inputs['out_comp_nm'] + "input_pp.yaml", 'w') as outfile:
        yaml_dump(inputs, outfile, default_flow_style=False)

    return fin_list_graphs_orig

def main_yeast():
    with open('yeast/training_MIPS_testing_TAPS/PC1_results/predicted_complexes yeast.pkl', 'rb') as f:
        all_lists = pickle_load(f)
    value_fns = [v[1] for v in all_lists]
    node_lists = [n[0] for n in all_lists]
    cmplx_tup = []
    for n in range(len(node_lists)):
        tup = (set(node_lists[n]), value_fns[n])  # make score the value function of complex
        cmplx_tup.append(tup)

    # postprocessing
    fileName = "../../yeast_DIP_network_experiments/input_data/dip_yeast_network_weighted_edge_lists.txt"
    G = nx.read_weighted_edgelist(fileName, nodetype=str)
    # remove duplicate edges and none
    G.remove_edges_from(nx.selfloop_edges(G))
    # Removing complexes with only two nodes
    # Finding unique complexes
    parser = argparse_ArgumentParser("Input parameters")
    parser.add_argument("--input_file_name", default="input_yeast_mips.yaml", help="Input parameters file name")
    parser.add_argument("--graph_files_dir", default="", help="Graph files' folder path")
    parser.add_argument("--out_dir_name", default="/results", help="Output directory name")
    args = parser.parse_args()
    with open(args.input_file_name, 'r') as f:
        inputs = yaml_load(f, yaml_Loader)

    fin_list_graphs_orig = post_process(cmplx_tup, inputs, G)
    fin_list_graphs_orig = [(set(comp), score) for comp, score in fin_list_graphs_orig]
    with open('yeast/training_MIPS_testing_TAPS/predicted_postprocess.pkl', 'wb') as f:
        pickle_dump(fin_list_graphs_orig,f)
    with open('yeast/training_MIPS_testing_TAPS/predicted_postprocess.txt', 'w') as f:
        f.writelines([''.join(str(comp)) + '\n' for comp in fin_list_graphs_orig])


    with open(inputs['dir_nm'] + inputs['out_comp_nm'] + "input_pp.yaml", 'w') as outfile:
        yaml_dump(inputs, outfile, default_flow_style=False)

    return fin_list_graphs_orig

def main_pp_final_convert2names():
    fileName = "../../humap_network_weighted_edge_lists.txt"
    G = nx.read_weighted_edgelist(fileName, nodetype=str)
    # remove duplicate edges and none
    G.remove_edges_from(nx.selfloop_edges(G))
    for i in list(G.nodes()):
        if i.isnumeric() is False:
            G.remove_node(i)
    parser = argparse_ArgumentParser("Input parameters")
    parser.add_argument("--input_file_name", default="input_humap.yaml", help="Input parameters file name")
    parser.add_argument("--graph_files_dir", default="", help="Graph files' folder path")
    parser.add_argument("--out_dir_name", default="/results", help="Output directory name")
    args = parser.parse_args()
    with open(args.input_file_name, 'r') as f:
         inputs = yaml_load(f, yaml_Loader)
    out_comp_nm = './humap/qi_results/results_qi0.325/'
    filename = out_comp_nm + 'predicted_postprocess.pkl'
    with open(filename, 'rb') as f:
        fin_list_graphs_orig = pickle_load(f)
    convert2names_wscores(fin_list_graphs_orig, out_comp_nm + 'res_pp_pred_names.out',
                                G, out_comp_nm + 'res_pp_pred_edges_names.out')

