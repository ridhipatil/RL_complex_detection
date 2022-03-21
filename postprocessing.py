from glob import glob
from pickle import load as pickle_load
from sample import get_score
from postprocess import merge_filter_overlapped_score
from convert_humap_ids2names import convert2names_wscores
from yaml import load as yaml_load, dump as yaml_dump, Loader as yaml_Loader
from argparse import ArgumentParser as argparse_ArgumentParser
import networkx as nx
from eval_cmplx import remove_unknown_prots
from eval_cmplx import eval_complex


with open('predicted_complexes humap.pkl', 'rb') as f:
    pred_comp_list = pickle_load(f)

fileName = "humap_network_weighted_edge_lists.txt"
G = nx.Graph()
G = nx.read_weighted_edgelist(fileName, nodetype=str)
# remove duplicate edges and none
G.remove_edges_from(nx.selfloop_edges(G))
for i in list(G.nodes()):
    if i.isnumeric() is False:
        G.remove_node(i)

#your predictions - [({nodse1,node2,..},val_fun)...]
print(pred_comp_list)
# Removing complexes with only two nodes
# Finding unique complexes
parser = argparse_ArgumentParser("Input parameters")
parser.add_argument("--input_file_name", default="input_toy.yaml", help="Input parameters file name")
parser.add_argument("--graph_files_dir", default="", help="Graph files' folder path")
parser.add_argument("--out_dir_name", default="/results", help="Output directory name")
args = parser.parse_args()
with open(args.input_file_name, 'r') as f:
    inputs = yaml_load(f, yaml_Loader)

fin_list_graphs = set([(frozenset(comp), score) for comp, score in pred_comp_list if len(comp) > 2])
fin_list_graphs_orig, score_merge = merge_filter_overlapped_score(fin_list_graphs,inputs,G)

out_comp_nm = inputs['dir_nm'] + inputs['out_comp_nm']
if inputs['dir_nm'] == "humap":
    convert2names_wscores(fin_list_graphs, out_comp_nm + '_pred_names.out', out_comp_nm + '_pred_edges_names.out')

# evaluation
with open('testing_CORUM_complexes_node_lists.pkl','rb') as f:
    known_complex_nodes_list = pickle_load(f)
N_test_comp = len(known_complex_nodes_list)
prot_list = [n for sublist in known_complex_nodes_list for n in sublist]
prot_list = set(prot_list)
# Remove all proteins in predicted complexes that are not present in known complex protein list
fin_list_graphs = remove_unknown_prots(fin_list_graphs_orig, prot_list)
N_pred_comp = len(fin_list_graphs)
suffix = ''
eval_complex(0, 0,inputs, known_complex_nodes_list, prot_list, fin_list_graphs,suffix="both")