# Reinforcement Learning Complex Detection
This is a reinforcement learning algorithm for community detection in networks. Trained on known communities, it learns to find new communities in a network.

# Installation:
Required python3                                  
Requirements installation:                        
`python3 -m pip install -r requirements_py3.txt --user`

# Experiments:
1. For a toy network use input_toy.yaml
2. For hu.MAP - use input file input_humap.yaml

# Instructions:
To run this pipeline on a new network, construct an input file similar to input_toy.yaml specifying where to find the required inputs.
1. Specify the network input file: Set options dir_nm (directory containing the network) and netf_nm (file name of the network)
2. Specify the paths for train and test communitites, in the options comf_nm and comf_test_nm (relative to the directory specified in the option- dir_nm)

An example bash script to run the RL pipeline after the above steps is shown below: This is for complexes learned on the human PPI network, hu.MAP 1.0:
```
#!/bin/bash

mtype=humap
input_file_name=input_$mtype.yaml
graph_file=hu.MAP_network/input_data/humap_network_weighted_edge_lists.txt
input_training_file=hu.MAP_network/intermediate_data/training_CORUM_complexes_node_lists.txt
input_testing_file=hu.MAP_network/intermediate_data/testing_CORUM_complexes_node_lists.txt
mkdir results_$mtype
out_dir_name=./results_$mtype
train_results=$out_dir_name/train_results
pred_results=$out_dir_name/pred_results
id_map_path=convert_ids/humap_gene_id_name_map.txt

echo Training Algorithm....
python3 main_training.py --input_training_file $input_training_file --graph_file $graph_file --train_results $train_results

echo Predicting new complexes from known communities...
python3 main_prediction.py --graph_file $graph_file --train_results $train_results --out_dir_name $out_dir_name --pred_results $pred_results

echo Merging similar communities...
python3 postprocessing.py --input_file_name $input_file_name --graph_file $graph_file --out_dir_name $out_dir_name --pred_results $pred_results --train_results $train_results --input_training_file $input_training_file --input_testing_file $input_testing_file --id_map_path $id_map_path

echo Comparing predicted and known communitites...
python3 eval_complex_RL --input_file_name $input_file_name  --input_training_file $input_training_file --input_testing_file $input_testing_file --out_dir_name $out_dir_name

```

## Additional tips:
For each of the scripts, optional arguments can be viewed by running: python3 script_name.py --help
For each command, add the desired argument directly on the terminal.

# References:
[Molecular complex detection in protein interaction networks through reinforcement learning](https://doi.org/10.1101/2022.06.20.496772) 

Interactive visualizations of complexes learned by the RL algorithm on two human PPI networks, hu.MAP 1.0 and hu.MAP 2.0 are available here: [https://marcottelab.github.io/RL_humap_prediction/](https://marcottelab.github.io/RL_humap_prediction/)
