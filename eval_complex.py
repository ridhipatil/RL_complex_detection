import pickle
from numpy import zeros as np_zeros, count_nonzero as np_count_nonzero, sum as np_sum, argmax as np_argmax, sqrt as np_sqrt
from logging import info as logging_info
from matplotlib.pyplot import figure as plt_figure, savefig as plt_savefig, close as plt_close, xlabel as plt_xlabel, title as plt_title, plot as plt_plot,ylabel as plt_ylabel, rc as plt_rc, rcParams as plt_rcParams
from collections import Counter
from pickle import dump as pickle_dump
from pickle import load as pickle_load
from pickle import load as pickle_load
from pickle import dump as pickle_dump
from eval_cmplx import node_comparison_prec_recall
from eval_cmplx import one2one_matches
from eval_cmplx import plot_size_dists
from eval_cmplx import remove_unknown_prots
from eval_cmplx import compute_metrics
from eval_cmplx import eval_complex
from yaml import load as yaml_load, dump as yaml_dump, Loader as yaml_Loader
from argparse import ArgumentParser as argparse_ArgumentParser
import networkx as nx
from postprocess import merge_filter_overlapped_score
from convert_humap_ids2names import convert2names_wscores

with open('predicted_complexes.pkl', 'rb') as f:
    all_lists = pickle_load(f)
with open('predicted_complexes density and value function testing.pkl','rb') as f:
    cmplx_info = pickle_load(f)
value_fns = [v[2] for v in cmplx_info]
cmplx_info = []
for n in range(len(all_lists)):
    tup = (set(all_lists[n]),value_fns[n]) # make score the value function of complex
    cmplx_info.append(tup)

# postprocessing
fileName = "humap_network_weighted_edge_lists.txt"
G = nx.Graph()
G = nx.read_weighted_edgelist(fileName, nodetype=str)
# remove duplicate edges and none
G.remove_edges_from(nx.selfloop_edges(G))
for i in list(G.nodes()):
    if i.isnumeric() is False:
        G.remove_node(i)
# Removing complexes with only two nodes
# Finding unique complexes
parser = argparse_ArgumentParser("Input parameters")
parser.add_argument("--input_file_name", default="input_humap.yaml", help="Input parameters file name")
parser.add_argument("--graph_files_dir", default="", help="Graph files' folder path")
parser.add_argument("--out_dir_name", default="/results", help="Output directory name")
args = parser.parse_args()
with open(args.input_file_name, 'r') as f:
    inputs = yaml_load(f, yaml_Loader)

fin_list_graphs = set([(frozenset(comp), score) for comp, score in cmplx_info if len(comp) > 2])
fin_list_graphs_orig, score_merge = merge_filter_overlapped_score(fin_list_graphs,inputs,G)
out_comp_nm = inputs['dir_nm'] + inputs['out_comp_nm']
if inputs['dir_nm'] == "humap":
    convert2names_wscores(fin_list_graphs, out_comp_nm + '_pred_names.out',
                          G, out_comp_nm + '_pred_edges_names.out')



## Evaluating

with open('testing_CORUM_complexes_node_lists.pkl','rb') as f:
    known_complex_nodes_list = pickle_load(f)
N_test_comp = len(known_complex_nodes_list)
prot_list = [n for sublist in known_complex_nodes_list for n in sublist]
prot_list = set(prot_list)
# Remove all proteins in predicted complexes that are not present in known complex protein list
fin_list_graphs = remove_unknown_prots(fin_list_graphs_orig, prot_list)
N_pred_comp = len(fin_list_graphs)

#
suffix = ''
print(fin_list_graphs)
print(known_complex_nodes_list)
print(out_comp_nm)
print(N_test_comp)
print(N_pred_comp)
print(inputs)
print(suffix)
eval_complex(0, 0,inputs, known_complex_nodes_list, prot_list, fin_list_graphs,suffix="both")
#compute_metrics(known_complex_nodes_list, fin_list_graphs, out_comp_nm, N_test_comp, N_pred_comp, inputs, suffix)
