from pickle import load as pickle_load
from pickle import dump as pickle_dump
from yaml import load as yaml_load, dump as yaml_dump, Loader as yaml_Loader
import networkx as nx
from humap.functions.postprocess_sc import merge_filter_overlapped_score_qi
from convert_humap_ids2names import convert2names_wscores
from argparse import ArgumentParser as argparse_ArgumentParser


def main():
    # Input data
    parser = argparse_ArgumentParser("Input parameters")
    parser.add_argument("--input_file_name", default="input_humap.yaml", help="Input parameters file name")
    parser.add_argument("--graph_files", default="", help="Graph files' folder path")
    parser.add_argument("--out_dir_name", default="/results", help="Output directory name")
    parser.add_argument("--pred_results", default="../pred_results", help="Directory for prediction results")
    parser.add_argument("--train_results", default="../train_results", help="Directory for main results")
    args = parser.parse_args()
    with open(args.input_file_name, 'r') as f:
        inputs = yaml_load(f, yaml_Loader)

    # make tuples of complex and corresponding value function
    fname = args.pred_results + "/predicted_complexes humap.pkl"
    with open(fname, 'rb') as f:
        all_lists = pickle_load(f)
    value_fns = [v[1] for v in all_lists]
    node_lists = [n[0] for n in all_lists]
    cmplx_tup = []
    for n in range(len(node_lists)):
        tup = (set(node_lists[n]), value_fns[n])  # make score the value function of complex
        cmplx_tup.append(tup)

    # postprocessing
    # fileName = "../../humap_network_weighted_edge_lists.txt"
    fileName = args.graph_files
    G = nx.read_weighted_edgelist(fileName, nodetype=str)
    # remove duplicate edges and none
    G.remove_edges_from(nx.selfloop_edges(G))
    for i in list(G.nodes()):
        if i.isnumeric() is False:
            G.remove_node(i)

    # Finding unique complexes
    file = args.train_results + '/value_fn_dens_dict.pkl'
    with open(file, 'rb') as f:
        value_fns_dict = pickle_load(f)
    fin_list_graphs = set([(frozenset(comp), score) for comp, score in cmplx_tup if len(comp) > 2])
    fin_list_graphs_orig = merge_filter_overlapped_score_qi(fin_list_graphs, inputs, G, value_fns_dict)
    fin_list_graphs_orig = [(set(comp), score) for comp, score in fin_list_graphs_orig]
    file = ''
    if inputs['overlap method'] == 'qi':
        file = args.out_dir_name + '/qi_results'
    elif inputs['overlap method'] == '1':  # jaccard coeff
        file = args.out_dir_name + '/jacc_results'

    filename = file + inputs['out_comp_nm']
    with open(filename + '_pred_complexes_pp.pkl', 'wb') as f:
        pickle_dump(fin_list_graphs_orig, f)
    with open(filename + '_pred_complexes_pp.txt', 'w') as f:
        f.writelines([''.join(str(comp)) + '\n' for comp in fin_list_graphs_orig])

    inputs['out_comp_nm'] = '/res_' + inputs['overlap method'] + str(inputs['over_t']) + '/res'
    with open(inputs['out_comp_nm'] + "input_pp.yaml", 'w') as outfile:
        yaml_dump(inputs, outfile, default_flow_style=False)

    # write out protein names
    out_comp_nm = file + inputs['out_comp_nm']
    if inputs['dir_nm'] == "humap":  # humap
        convert2names_wscores(fin_list_graphs_orig, out_comp_nm + '_pred_names.out',
                              G, out_comp_nm + '_pred_edges_names.out')


if __name__ == '__main__':
    main()
