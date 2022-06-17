from argparse import ArgumentParser as argparse_ArgumentParser, ArgumentParser
from pickle import load as pickle_load
from yaml import load as yaml_load, dump as yaml_dump, Loader as yaml_Loader
from eval_cmplx_sc import eval_complex
from eval_cmplx_sc import remove_unknown_prots
from main6_eval import run_metrics
import os
def main():
    # Evaluating
    parser: ArgumentParser = argparse_ArgumentParser("Input parameters")
    parser.add_argument("--input_file_name", default="", help="Input parameters file name")
    parser.add_argument("--input_training_file", default="", help="Training Graph file path")
    parser.add_argument("--input_testing_file", default="", help="Testing Graph file path")
    parser.add_argument("--out_dir_name", default="", help="Output directory name")
    parser.add_argument("--evaluate_additional_metrics", default=1, help="complexes file name")
    args = parser.parse_args()
    with open(args.input_file_name, 'r') as f:
        inputs = yaml_load(f, yaml_Loader)

    file = ''
    if inputs['overlap_method'] == 'qi':
        file = args.out_dir_name + '/qi_results'
        out_comp_nm = file + '/res'  # inputs['out_comp_nm']
    elif inputs['overlap_method'] == '1':  # jaccard coeff
        file = args.out_dir_name + '/jacc_results'
        out_comp_nm = file + '/res'  # inputs['out_comp_nm']
    with open(out_comp_nm + "_input_eval_train.yaml", 'w') as outfile:
        yaml_dump(inputs, outfile, default_flow_style=False)

    filename = out_comp_nm + "_pred_complexes_pp.pkl"
    with open(filename, 'rb') as f:
        fin_list_graphs_orig = pickle_load(f)

    ## training set
    with open(out_comp_nm + '_metrics.out', "a") as fid:
        print("\n --- On training set ---", file=fid)
    file = args.input_training_file
    with open(file, 'r') as f:
        training = f.read().splitlines()
    for c in range(len(training)):
        training[c] = training[c].split()

    known_complex_nodes_list = training
    prot_list = [n for sublist in known_complex_nodes_list for n in sublist]
    prot_list = set(prot_list)

    # Remove all proteins in Predicted complexes that are not present in known complex protein list
    fin_list_graphs = remove_unknown_prots(fin_list_graphs_orig, prot_list)
    suffix = ''
    eval_complex(0, 0, inputs, known_complex_nodes_list, prot_list, fin_list_graphs, out_comp_nm, suffix="_train")
    if args.evaluate_additional_metrics:
        try:
            run_metrics(known_complex_nodes_list, fin_list_graphs, out_comp_nm, "_train")
        except:
            print("Error in running additional metrics for train")

    ## testing set
    with open(out_comp_nm + '_metrics.out', "a") as fid:
        print("\n --- On testing set ---", file=fid)

    file = args.input_testing_file
    with open(file, 'r') as f:
        testing = f.read().splitlines()
    for c in range(len(testing)):
        testing[c] = testing[c].split() #pickle_load(f)

    known_complex_nodes_list = testing
    prot_list = [n for sublist in known_complex_nodes_list for n in sublist]
    prot_list = set(prot_list)
    # Remove all proteins in Predicted complexes that are not present in known complex protein list
    fin_list_graphs = remove_unknown_prots(fin_list_graphs_orig, prot_list)
    suffix = ''
    eval_complex(0, 0, inputs, known_complex_nodes_list, prot_list, fin_list_graphs, out_comp_nm,suffix="_train")

    if args.evaluate_additional_metrics:
        try:
            run_metrics(known_complex_nodes_list, fin_list_graphs, out_comp_nm, "_test")
        except:
            print("Error in running additional metrics for test")

    ## both (training and testing) sets
    with open(out_comp_nm + '_metrics.out', "a") as fid:
        print("\n --- On both sets ---", file=fid)

    file_test = args.input_testing_file
    with open(file_test, 'r') as f:
        testing = f.read().splitlines()
    for c in range(len(testing)):
        testing[c] = testing[c].split()

    file_train = args.input_training_file
    with open(file_train, 'r') as f:
        training = f.read().splitlines()
    for c in range(len(training)):
        training[c] = training[c].split()

    known_complex_nodes_list = testing + training
    N_test_comp = len(known_complex_nodes_list)
    prot_list = [n for sublist in known_complex_nodes_list for n in sublist]
    prot_list = set(prot_list)

    # Remove all proteins in Predicted complexes that are not present in known complex protein list
    fin_list_graphs = remove_unknown_prots(fin_list_graphs_orig, prot_list)
    N_pred_comp = len(fin_list_graphs)
    suffix = ''

    eval_complex(0, 0, inputs, known_complex_nodes_list, prot_list, fin_list_graphs, out_comp_nm,suffix="_train")

    if args.evaluate_additional_metrics:
        try:
            run_metrics(known_complex_nodes_list, fin_list_graphs, out_comp_nm, "_both")
        except:
            print("Error in running additional metrics for both")

if __name__ == '__main__':
    main()
