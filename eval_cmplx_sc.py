# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:37:24 2019

@author: Meghana
"""

from seaborn import distplot as sns_distplot
from numpy import zeros as np_zeros, count_nonzero as np_count_nonzero, sum as np_sum, argmax as np_argmax, sqrt as np_sqrt
from logging import info as logging_info
from matplotlib.pyplot import figure as plt_figure, savefig as plt_savefig, close as plt_close, xlabel as plt_xlabel, title as plt_title, plot as plt_plot,ylabel as plt_ylabel, rc as plt_rc, rcParams as plt_rcParams
from convert_humap_ids2names import convert2names_wscores_matches
from collections import Counter
from test_F1_MMR import f1_mmr




def write_best_matches(best_matches_for_known,out_comp_nm,dir_nm,suffix,id_name_map):
       
    sorted_matches = sorted(best_matches_for_known,key=lambda x: x[2],reverse=True)
    if dir_nm == "humap":
        convert2names_wscores_matches(sorted_matches, out_comp_nm + suffix + '_known_pred_matches_names.out',id_name_map)
 
    with open(out_comp_nm + suffix + '_known_pred_matches.out', "w") as fn:
        fn_write = fn.write
        fn_write("Known complex nodes ||| Predicted complex nodes ||| Match F1 score ||| Complex score \n")

        for index in range(len(sorted_matches)):
            known_graph_nodes = sorted_matches[index][0]
            pred_graph_nodes = sorted_matches[index][1]
            match_score = sorted_matches[index][2]
            complex_score = sorted_matches[index][3]
            for node in known_graph_nodes:
                fn_write("%s " % node)
            fn_write(" ||| ")
            for node in pred_graph_nodes:
                fn_write("%s " % node)
            
            fn_write(" ||| ")
            fn_write("%.3f" % match_score)
            fn_write(" ||| ")
            fn_write("%.3f" % float(complex_score))            
            fn_write("\n")


def plot_f1_scores(best_matches,out_comp_nm,suffix,prefix):
    # plot histogram of F1 scores
    max_f1_scores = [match[2] for match in best_matches]
    
    avged_f1_score = sum(max_f1_scores)/float(len(max_f1_scores))
    
    f1_score_counts = Counter()
    
    for score in max_f1_scores:
        f1_score_counts[score] += 1
        
    n_perfect_matches = 0
    if 1 in f1_score_counts:
        n_perfect_matches = f1_score_counts[1]
        
    n_no_matches = 0
    if 0 in f1_score_counts:
        n_no_matches = f1_score_counts[0]    
                
    if len(set(max_f1_scores)) > 1:
        fig = plt_figure(figsize=(12,10))
        plt_rcParams["font.family"] = "Times New Roman"
        plt_rcParams["font.size"] = 16
        sns_distplot(max_f1_scores, hist=True, kde = False)
        plt_xlabel("F1 score")
        plt_ylabel('Frequency')
        plt_title(prefix + "F1 score distribution")
        plt_savefig(out_comp_nm +suffix+ '_f1_scores_histogram.eps',dpi=350,format='eps')
        plt_savefig(out_comp_nm +suffix+ '_f1_scores_histogram.tiff',dpi=350,format='tiff')
        plt_savefig(out_comp_nm +suffix+ '_f1_scores_histogram.jpg',dpi=350,format='jpg')
        
        plt_close(fig)    
        
    with open(out_comp_nm + '_metrics.out', "a") as fid:
        print(prefix, file=fid)
        print("Averaged F1 score = %.3f" % avged_f1_score, file=fid)
        print("No. of perfectly recalled matches = %d" % n_perfect_matches, file=fid)
        print("No. of matches not recalled at all = %d" % n_no_matches, file=fid)     
    return avged_f1_score
            
    
def plot_pr_curve_mmr(Metric,fin_list_graphs,out_comp_nm):
    
    n_divs = 10
    scores_list = [float(pred_complex[1]) for pred_complex in fin_list_graphs]
    #print(scores_list)
    min_score = min(scores_list)
    interval_len = (max(scores_list) - min_score)/float(n_divs)
    thresholds = [min_score + i*interval_len for i in range(n_divs)]
    
    precs = []
    recalls = []
    for thres in thresholds:
        # list of indices with scores greater than the threshold
        col_inds = [j for j,score in enumerate(scores_list) if score >= thres]
        prec_MMR, recall_MMR, f1_MMR, max_matching_edges = f1_mmr(Metric[:,col_inds])
        
        precs.append(prec_MMR)
        recalls.append(recall_MMR)
        
    fig = plt_figure()
    plt_plot(recalls,precs,'.-')
    plt_ylabel("Precision")
    plt_xlabel("Recall")
    plt_title("PR curve for MMR measure")
    plt_savefig(out_comp_nm + '_pr_mmr.png')
    plt_close(fig)    
    

def f1_similarity(P,T):
    C = len(T.intersection(P))
    
    Precision = float(C) / len(P)
    Recall = float(C) / len(T)
    
    if Precision == Recall == 0:
        F1_score = 0
    else:
        F1_score = 2 * Precision * Recall / (Precision + Recall)   
        
    return F1_score, C 


def one2one_matches(known_complex_nodes_list, fin_list_graphs, N_pred_comp, N_test_comp,out_comp_nm,suffix,dir_nm, id_name_map):

    Metric = np_zeros((N_test_comp, N_pred_comp))
    Common_nodes = np_zeros((N_test_comp, N_pred_comp))
    known_comp_lens = np_zeros((N_test_comp, 1))
    pred_comp_lens = np_zeros((1, N_pred_comp))
    
    fl = 1

    for i, test_complex in enumerate(known_complex_nodes_list):
        T = set(test_complex)
        known_comp_lens[i,0] = len(T)
        
        for j, pred_complex in enumerate(fin_list_graphs):
            P = pred_complex[0]
            
            F1_score, C = f1_similarity(P,T)
            Common_nodes[i, j] = C
            
            Metric[i, j] = F1_score
            
            if fl == 1:
                pred_comp_lens[0,j] = len(P)
        fl = 0
        
    max_indices_i_common = np_argmax(Common_nodes, axis=0)
    ppv_list = [ float(Common_nodes[i,j])/pred_comp_lens[0,j] for j,i in enumerate(max_indices_i_common)]
    PPV = sum(ppv_list)/len(ppv_list)
    
    max_indices_j_common = np_argmax(Common_nodes, axis=1)
    sn_list = [ float(Common_nodes[i,j])/known_comp_lens[i,0] for i,j in enumerate(max_indices_j_common)]
    Sn = sum(sn_list)/len(sn_list)
    
    acc_unbiased = np_sqrt(PPV * Sn)

    max_indices_i = np_argmax(Metric, axis=0)
    best_matches_4predicted = [(fin_list_graphs[j][0],known_complex_nodes_list[i],Metric[i,j],fin_list_graphs[j][1]) for j,i in enumerate(max_indices_i)]

    max_indices_j = np_argmax(Metric, axis=1)
    best_matches_4known = [(fin_list_graphs[j][0],known_complex_nodes_list[i],Metric[i,j],fin_list_graphs[j][1]) for i,j in enumerate(max_indices_j)]
    
    avged_f1_score4known = plot_f1_scores(best_matches_4known,out_comp_nm,'_best4known'+suffix,'Best Predicted match for known complexes - ')
    avged_f1_score4pred = plot_f1_scores(best_matches_4predicted,out_comp_nm,'_best4predicted'+suffix,'Best known match for Predicted complexes - ')
    
    avg_f1_score = (avged_f1_score4known + avged_f1_score4pred)/2
    net_f1_score = 2 * avged_f1_score4known * avged_f1_score4pred / (avged_f1_score4known + avged_f1_score4pred)
    
    write_best_matches(best_matches_4known,out_comp_nm,dir_nm,'_best4known' + suffix, id_name_map)
    write_best_matches(best_matches_4predicted,out_comp_nm,dir_nm,'_best4predicted' + suffix, id_name_map)

    prec_MMR, recall_MMR, f1_MMR, max_matching_edges = f1_mmr(Metric)
    
    plot_pr_curve_mmr(Metric,fin_list_graphs,out_comp_nm+suffix)
    
    n_matches = int(len(max_matching_edges)/2)
    
    return avg_f1_score, net_f1_score,PPV,Sn,acc_unbiased,prec_MMR, recall_MMR, f1_MMR, n_matches

def f1_qi(Metric):
    max_i = Metric.max(axis=0)
    prec = sum(max_i)/len(max_i)
    
    max_j = Metric.max(axis=1)
    recall = sum(max_j)/len(max_j)    
    
    return prec,recall


def plot_pr_curve_orig(Metric,fin_list_graphs,out_comp_nm):
    
    n_divs = 10
    scores_list = [float(pred_complex[1]) for pred_complex in fin_list_graphs]
    #print(scores_list)
    min_score = min(scores_list)
    interval_len = (max(scores_list) - min_score)/float(n_divs)
    thresholds = [min_score + i*interval_len for i in range(n_divs)]
    
    precs = []
    recalls = []
    for thres in thresholds:
        # list of indices with scores greater than the threshold
        col_inds = [j for j,score in enumerate(scores_list) if score >= thres]
        
        prec, recall = f1_qi(Metric[:,col_inds])
        
        precs.append(prec)
        recalls.append(recall)
        
    fig = plt_figure()
    plt_plot(recalls,precs,'.-')
    plt_ylabel("Precision")
    plt_xlabel("Recall")
    plt_title("PR curve for Qi et al measure")
    plt_savefig(out_comp_nm + '_pr_qi.png')
    plt_close(fig)    
    
    
def node_comparison_prec_recall(known_complex_nodes_list, fin_list_graphs, N_pred_comp, N_test_comp, p, out_comp_nm):
    N_matches_test = 0

    Metric = np_zeros((N_test_comp, N_pred_comp))

    for i, test_complex in enumerate(known_complex_nodes_list):
        N_match_pred = 0
        for j, pred_complex in enumerate(fin_list_graphs):
            T = set(test_complex)
            P = pred_complex[0]
            C = len(T.intersection(P))
            A = len(P.difference(T))
            B = len(T.difference(P))

            if float(C) / (A + C) > p and float(C) / (B + C) > p:
                Metric[i, j] = 1
                N_match_pred = N_match_pred + 1

        if N_match_pred > 0:
            N_matches_test = N_matches_test + 1
            
    plot_pr_curve_orig(Metric,fin_list_graphs,out_comp_nm)

    Recall = float(N_matches_test) / N_test_comp

    N_matches_pred = np_count_nonzero(np_sum(Metric, axis=0))
    Precision = float(N_matches_pred) / N_pred_comp

    if Precision == Recall == 0:
        F1_score = 0
    else:
        F1_score = 2 * Precision * Recall / (Precision + Recall)

    return Precision, Recall, F1_score

def plot_size_dists(known_complex_nodes_list, fin_list_graphs, sizes_orig, out_comp_nm):
    sizes_known = [len(comp) for comp in known_complex_nodes_list]
    # Size distributions
    sizes_new = [len(comp[0]) for comp in fin_list_graphs]
    fig = plt_figure(figsize=(8,6),dpi=96)
    plt_rc('font', size=14) 

    if len(set(sizes_known)) <= 1:
        return
    sns_distplot(sizes_known, hist=False, label="known")
    if len(set(sizes_orig)) <= 1:
        return
    sns_distplot(sizes_orig, hist=False, label="Predicted")
    if len(set(sizes_new)) <= 1:
        return
    sns_distplot(sizes_new, hist=False, label="predicted_known_prots")
    plt_ylabel("Probability density")    
    plt_xlabel("Complex Size (number of proteins)")
    plt_title("Complex size distributions")
    plt_savefig(out_comp_nm + '_size_dists_known_pred.png')
    plt_close(fig)


def remove_unknown_prots(fin_list_graphs_orig, prot_list):
    # Remove all proteins in Predicted complexes that are not present in known complex protein list
    fin_list_graphs = []
    n_removed = 0
    for comp in fin_list_graphs_orig:
        comp = (comp[0].intersection(prot_list), comp[1])
        n_removed += len(comp[0]-prot_list)
        if len(comp[0]) > 2:  # Removing complexes with only one,two or no nodes
            fin_list_graphs.append(comp)
    print('Number of proteins removed ', n_removed)
    return fin_list_graphs


def compute_metrics(known_complex_nodes_list, fin_list_graphs,out_comp_nm,N_test_comp,N_pred_comp,inputs,suffix, id_name_map):

    if N_test_comp != 0 and N_pred_comp != 0:
        Precision, Recall, F1_score = node_comparison_prec_recall(known_complex_nodes_list,fin_list_graphs, N_pred_comp, N_test_comp, inputs["eval_p"],out_comp_nm+suffix)
        
        avg_f1_score, net_f1_score,PPV,Sn,acc_unbiased,prec_MMR, recall_MMR, f1_MMR,n_matches = one2one_matches(known_complex_nodes_list, fin_list_graphs, N_pred_comp, N_test_comp,out_comp_nm,suffix,inputs['dir_nm'], id_name_map)
        
        with open(out_comp_nm + '_metrics.out', "a") as fid:
            print("No. of matches in MMR = ", n_matches, file=fid)            
            print("FMM Precision = %.3f" % prec_MMR, file=fid)
            print("FMM Recall = %.3f" % recall_MMR, file=fid)
            print("FMM F1 score = %.3f" % f1_MMR, file=fid)               
            print("CMMF = %.3f" % net_f1_score, file=fid)   
            print("Unbiased PPV = %.3f" % PPV, file=fid)
            print("Unbiased Sn = %.3f" % Sn, file=fid)
            print("Unbiased accuracy (UnSPA)= %.3f" % acc_unbiased, file=fid)             
            print("Net Averaged F1 score (Average of Precision and Recall based on F1 score) = %.3f" % avg_f1_score, file=fid)
            print("Qi et al Precision = %.3f" % Precision, file=fid)
            print("Qi et al Recall = %.3f" % Recall, file=fid)
            print("Qi et al F1 score = %.3f" % F1_score, file=fid)    
    
def eval_complex(rf=0, rf_nm=0, inputs={}, known_complex_nodes_list=[], prot_list=[], fin_list_graphs=[], out_comp_nm = '',suffix="both", id_name_map = ""):
    # rf - read flag to read complexes from file
    logging_info("Evaluating complexes..." + suffix)
    if rf == 1:
        if rf_nm == 0:
            rf_nm = out_comp_nm + '_pred.out'
        with open(rf_nm) as fn:
            fin_list_graphs = [(set(line.rstrip('\n').split()),1) for line in fn]  # Space separated text only
            # Just list of list of nodes

    sizes_orig = [len(comp[0]) for comp in fin_list_graphs]

    N_pred_comp = len(fin_list_graphs)
    if N_pred_comp == 0:
        return
    N_test_comp = len(known_complex_nodes_list)

    with open(out_comp_nm + '_metrics.out', "a") as fid:
        print("No. of known complexes = ", N_test_comp, file=fid) 
        print("No. of Predicted complexes = ", N_pred_comp, file=fid)
        print("\n -- Metrics on complexes with all proteins -- ", file=fid)       
    print(out_comp_nm)
    compute_metrics(known_complex_nodes_list, fin_list_graphs, out_comp_nm,N_test_comp,N_pred_comp,inputs,suffix+'_all_prots',id_name_map)            
    
    fin_list_graphs = remove_unknown_prots(fin_list_graphs, prot_list)
    plot_size_dists(known_complex_nodes_list, fin_list_graphs, sizes_orig, out_comp_nm)
    
    N_pred_comp = len(fin_list_graphs)
    with open(out_comp_nm + '_metrics.out', "a") as fid:
        print("No. of Predicted complexes after removing non-gold std proteins = ", N_pred_comp, file=fid)
        print("\n -- Metrics on complexes with only gold std proteins -- ", file=fid)   
    
    compute_metrics(known_complex_nodes_list, fin_list_graphs, out_comp_nm,N_test_comp,N_pred_comp,inputs,suffix+'_gold_std_prots', id_name_map)            
    with open(out_comp_nm + '_metrics.out', "a") as fid:
        print("-- Finished writing main metrics -- \n", file=fid)   

    logging_info("Finished evaluating basic metrics for complexes " + suffix)
