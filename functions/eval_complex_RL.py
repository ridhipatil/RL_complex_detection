from argparse import ArgumentParser as argparse_ArgumentParser, ArgumentParser
from pickle import load as pickle_load
from yaml import load as yaml_load, dump as yaml_dump, Loader as yaml_Loader
from eval_cmplx_sc import eval_complex
from eval_cmplx_sc import remove_unknown_prots
from main6_eval import *

def eval_training(inputs, fin_list_graphs_orig, out_comp_nm, args):
    # TRAINING SET
    with open(out_comp_nm + '_metrics.out', "a") as fid:
        print("\n --- On training set ---", file=fid)

    with open('../../training_CORUM_complexes_node_lists.txt', 'r') as f:
        training = f.read().splitlines()
    for c in range(len(training)):
        training[c] = training[c].split()

    known_complex_nodes_list = training
    N_test_comp = len(known_complex_nodes_list)
    prot_list = [n for sublist in known_complex_nodes_list for n in sublist]
    prot_list = set(prot_list)

    # Remove all proteins in Predicted complexes that are not present in known complex protein list
    fin_list_graphs = remove_unknown_prots(fin_list_graphs_orig, prot_list)
    N_pred_comp = len(fin_list_graphs)

    suffix = ''

    eval_complex(0, 0,inputs, known_complex_nodes_list, prot_list, fin_list_graphs,suffix="_train")

    if args.evaluate_additional_metrics:
        try:
            run_metrics(known_complex_nodes_list, fin_list_graphs, out_comp_nm, "_train")
        except:
            print("Error in running additional metrics for train")

def eval_testing(inputs, fin_list_graphs_orig, out_comp_nm, args):
    # TESTING SET
    with open(out_comp_nm + '_metrics.out', "a") as fid:
        print("\n --- On testing set ---", file=fid)

    with open('../../testing_CORUM_complexes_node_lists.pkl', 'rb') as f:
        testing = pickle_load(f)

    known_complex_nodes_list = testing
    N_test_comp = len(known_complex_nodes_list)
    prot_list = [n for sublist in known_complex_nodes_list for n in sublist]
    prot_list = set(prot_list)
    # Remove all proteins in Predicted complexes that are not present in known complex protein list
    fin_list_graphs = remove_unknown_prots(fin_list_graphs_orig, prot_list)
    N_pred_comp = len(fin_list_graphs)

    suffix = ''
    eval_complex(0, 0,inputs, known_complex_nodes_list, prot_list, fin_list_graphs,suffix="_test")

    if args.evaluate_additional_metrics:
        try:
            run_metrics(known_complex_nodes_list, fin_list_graphs, out_comp_nm, "_test")
        except:
            print("Error in running additional metrics for test")

def eval_both(inputs, fin_list_graphs_orig, out_comp_nm, args):
    # BOTH SETS
    with open(out_comp_nm + '_metrics.out', "a") as fid:
        print("\n --- On both sets ---", file=fid)
    with open('../../testing_CORUM_complexes_node_lists.pkl', 'rb') as f:
        testing = pickle_load(f)
    with open('../../training_CORUM_complexes_node_lists.txt', 'r') as f:
        training = f.read().splitlines()
    for c in range(len(training)):
        training[c] = training[c].split()

    known_complex_nodes_list = testing+training
    N_test_comp = len(known_complex_nodes_list)
    prot_list = [n for sublist in known_complex_nodes_list for n in sublist]
    prot_list = set(prot_list)

    # Remove all proteins in Predicted complexes that are not present in known complex protein list
    fin_list_graphs = remove_unknown_prots(fin_list_graphs_orig, prot_list)
    N_pred_comp = len(fin_list_graphs)
    suffix = ''

    eval_complex(0, 0, inputs, known_complex_nodes_list, prot_list, fin_list_graphs, suffix="_both")

    if args.evaluate_additional_metrics:
        try:
            run_metrics(known_complex_nodes_list, fin_list_graphs, out_comp_nm, "_both")
        except:
            print("Error in running additional metrics for both")


def main():
    # Evaluating
    parser: ArgumentParser = argparse_ArgumentParser("Input parameters")
    parser.add_argument("--input_file_name", default="input_humap.yaml", help="Input parameters file name")
    parser.add_argument("--graph_files_dir", default="", help="Graph files' folder path")
    parser.add_argument("--out_dir_name", default="../results", help="Output directory name")
    parser.add_argument("--evaluate_additional_metrics", default=1, help="complexes file name")
    args = parser.parse_args()
    with open(args.input_file_name, 'r') as f:
        inputs = yaml_load(f, yaml_Loader)
    inputs['out_comp_nm'] = '/res_' + inputs['overlap method'] + str(inputs['over_t']) + '/res'

    # filename = 'humap/results_qi0.350/predicted_postprocess.pkl'
    if inputs['overlap method'] == 'qi':
        file = args.out_dir_name + '/qi_results'
    elif inputs['overlap method'] == '1':  # jaccard coeff
        file = args.out_dir_name + '/jacc_results'
    filename = file + inputs['out_comp_nm'] + "_pred_complexes_pp.pkl"
    with open(filename, 'rb') as f:
        fin_list_graphs_orig = pickle_load(f)

    out_comp_nm = file + inputs['out_comp_nm']
    with open(out_comp_nm + "_input_eval_train.yaml", 'w') as outfile:
        yaml_dump(inputs, outfile, default_flow_style=False)

    # training set
    eval_training(inputs, fin_list_graphs_orig, out_comp_nm, args)
    # testing set
    eval_testing(inputs, fin_list_graphs_orig, out_comp_nm, args)
    # both (training and testing) sets
    eval_both(inputs, fin_list_graphs_orig, out_comp_nm, args)