# Reinforcement Learning Complex Detection
This reinforcement learning algorithm is a machine learning method for complex detection in networks. Using known communities, it is trained and learns to find new complexes in the network.

# Installation:
Required python3                                  
Requirements installation:                        
`python3 -m pip install -r requirements_py3.txt --user`

# Experiments:
1. For a toy network use input_toy.yaml
2. For hu.MAP - use input file input_humap.yaml


# Instructions:
To run this pipeline on a new network, construct an input file similar to input_toy.yaml specifying where to find the required inputs.
1. Specify input options relating to network: Set options dir_nm (directory containing the network) and netf_nm (file name of the network)
2. Specify input options relating to known communities in network: If you already have sepearated known communities into train and test communitites, specify their paths in the options comf_nm and comf_test_nm (relative to the directory specified in the option:dir_nm) Otherwise, Split complex list into train and test: Set option split_flag = 1 Verify that train test size distributions in figure are the similar. Also check that number of training complexes is not too low by looking at the res_metrics.out file. Set options comf_nm and comf_test_nm with these two files. All the above paths are set relative to the directory specified in the option:dir_nm Make sure to change the option split_flag back to 0 after this step

An example bash script to run the RL pipeline after the above steps is shown below: This is for hu.MAP complexes

`#!/bin/bash
mtype = humap
input_file_name = input_$mtype.yaml
intermed_graph_files_dir = hu.MAP_network/intermediate_data
training_dat = $intermed_graph_files_dir/training_CORUM_complexes_node_lists.txt
testing_dat = $intermed_graph_files_dir/testing_CORUM_complexes_node_lists.txt
out_dir_name = /results
graph_ed_file = hu.MAP_network/input_data/humap_network_weighted_edge_lists.txt
echo Training Algorithm....
python3 functions/main_training.py  --input_training_file $training_dat --graph_file $graph_ed_file --train_results $out_dir_name/train_results
echo Predicting new complexes from known communities...
python3 functions/main_prediction.py --input_file_name $input_file_name --graph_file $graph_ed_file --train_results $out_dir_name/train_results --pred_results $out_dir_name/pred_results
echo Merging similar communities...
python3 functions/postprocessing.py --input_file_name $input_file_name --graph_files $graph_ed_file --train_results $out_dir_name/train_results --pred_results $out_dir_name/pred_results
echo Comparing predicted and known communitites...
python3 functions/eval_complex_RL --input_file_name $input_file_name --graph_files_dir $intermed_graph_files_dir`

# Additional tips:
For each of the scripts, optional arguments can be viewed by running: python3 script_name.py --help
For each command, add the desired argument directly on the terminal.
