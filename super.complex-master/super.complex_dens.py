#!/bin/bash
#
mtype=humap
input_file_name=input_$mtype.yaml
out_dir_name=/results_conn_$mtype
out_dir_name_full=humap$out_dir_name
tpot_tmp_dir=TPOT/tpot_tmp_huamp_$mtype
classifier_file=$out_dir_name_full/res_classifiers.txt
train_dat=$out_dir_name_full/res_train_dat.csv
test_dat=$out_dir_name_full/res_test_dat.csv

#echo Reading network...
#python3 main1_read.py --input_file_name $input_file_name --out_dir_name $out_dir_name

#echo Generating feature matrices for known communities...
#python3 main2_train.py --input_file_name $input_file_name --out_dir_name $out_dir_name --mode gen

#echo Finding the best community fitness function...
#mkdir $tpot_tmp_dir
#python3 TPOT/train_TPOT3.py --training_data $train_dat --testing_data $test_dat --outfile $out_dir_name_full/res_tpot_best_pipeline.py --outfile_classifiers $classifier_file --outfile_fig $out_dir_name_full/res_classifiers_pr.png --generations 50 --population_size 50 --n_jobs 20 --temp_dir $tpot_tmp_dir

echo Training the best community fitness function...
python3 main2_train.py --input_file_name $input_file_name --out_dir_name $out_dir_name --mode non_gen --train_feat_mat $train_dat --test_feat_mat $test_dat --classifier_file $classifier_file

meths=( search_top_neigs metropolis isa cliques )
for meth in "${meths[@]}"
do
out_dir_name_meth=$out_dir_name$meth

echo Partitioning graph nodes across compute nodes...
python3 main3_partition_search_seeds.py --input_file_name $input_file_name --out_dir_name $out_dir_name_meth

echo Growing communities...
python3 main4_sample.py --input_file_name $input_file_name --out_dir_name $out_dir_name_meth --search_method $meth

echo Merging very similar communities...
python3 main5_postprocess.py --input_file_name $input_file_name --out_dir_name $out_dir_name_meth

echo Comparing predicted and known communities...
python3 main6_eval.py --input_file_name $input_file_name --out_dir_name $out_dir_name_meth
done

python3 get_best_f1_score.py