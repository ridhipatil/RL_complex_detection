#!/bin/bash

mtype = humap
input_file_name = input_$mtype.yaml
graph_file = hu.MAP_network_experiments/input_data/humap_network_weighted_edge_lists.txt
input_training_file = hu.MAP_network_experiments/intermediate_output_results_data/training_CORUM_complexes_node_lists.txt
input_testing_file = hu.MAP_network_experiments/intermediate_output_results_data/testing_CORUM_complexes_node_lists.txt
out_dir_name = /results_$mtype
train_results = $out_dir_name/train_results
pred_results = $out_dir_name/pred_results
id_map_path = convert_ids/humap_gene_id_name_map.txt

echo Training Algorithm....
python3 functions/main_training.py --input_training_file $input_training_file --graph_file $graph_file --train_results $train_results

echo Predicting new complexes from known communities...
python3 functions/main_prediction.py --graph_file $graph_file --train_results $train_results --out_dir_name $out_dir_name --pred_results $pred_results

echo Merging similar communities...
python3 functions/postprocessing.py --input_file_name $input_file_name --graph_file $graph_file --out_dir_name $out_dir_name --pred_results $pred_results --train_results $train_results --input_training_file $input_training_file --input_testing_file $input_testing_file --id_map_path $id_map_path

echo Comparing predicted and known communitites...
python3 functions/eval_complex_RL --input_file_name $input_file_name  --input_training_file $input_training_file --input_testing_file $input_testing_file --out_dir_name $out_dir_name
