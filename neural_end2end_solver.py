
import math
import torch
import os
import argparse
import numpy as np
import itertools
from tqdm import tqdm

from torch.utils.data import DataLoader
import time
from datetime import timedelta
from clustering_k_means_plusplus_sklearn import k_means_plusplus_clustering_sklearn,cluster_tuning_for_one_node_cluster
from clustering_k_means_sklearn import k_means_clustering_sklearn

from end2end_model.kool2019.eval_zhao import eval_dataset
from end2end_model.kool2019.utils import load_model, move_to
from merge_sub_tsps import merge_sub_tsps

from argparse import Namespace


# tsp_data is a 2d np.array,like [[x,y],[x,y],...],save the coordinates of the TSP nodes.
def solve_one_tsp_by_end2end_model(tsp_data,model,opts,normalize=True):
    tmp_tsp_data=np.copy(tsp_data)

    # if normalize is True, then normalize the tsp data to [0,1]
    if normalize:
        x_min=np.min(tmp_tsp_data[:,0])
        x_max=np.max(tmp_tsp_data[:,0])
        y_min=np.min(tmp_tsp_data[:,1])
        y_max=np.max(tmp_tsp_data[:,1])
        # normalize
        if x_max==x_min:
            tmp_tsp_data[:,0]=0.5
        else:
            tmp_tsp_data[:,0]=(tmp_tsp_data[:,0]-x_min)/(x_max-x_min)
        if y_max==y_min:
            tmp_tsp_data[:,1]=0.5
        else:
            tmp_tsp_data[:,1]=(tmp_tsp_data[:,1]-y_min)/(y_max-y_min)
            
    # tsp is a 2d np.array
    # convert it into a tensor
    tsp_data_tensor=torch.tensor(tmp_tsp_data,dtype=torch.float32)
    tsp_data_set=[]
    tsp_data_set.append(tsp_data_tensor)

    # eval_dataset(dataset_path, width=0, opts.softmax_temperature, opts)
    costs, tours, durations=eval_dataset(model,tsp_data_set, 0, opts.softmax_temperature, opts)

    # create a dict to store the result
    result_dict={}
    result_dict["ori_tsp_data"]=tsp_data

    # sort original_tsp according to the tour 
    sorted_tsp_data=tsp_data[tours]

    result_dict["sorted_tsp_data"]=sorted_tsp_data

    ## dict, contains ori_tsp_data, and the sorted_tsp_data. the type is np.array
    return result_dict


def make_opts(**overrides):
    defaults = dict(
        f=False,
        o=None,
        val_size=1,
        offset=0,
        eval_batch_size=1,
        width=None,  # e.g., [0] to disable, [-1] for infinite, [64] for 64-beam
        decode_strategy='greedy',
        softmax_temperature=1,  # or 1.0 if used as float
        model="end2end_model/kool2019/pretrained/tsp_20/epoch-99.pt",
        no_cuda=True,
        no_progress_bar=False,
        compress_mask=False,
        max_calc_batch_size=1,
        results_dir='results',
        # add any other args you use later, e.g. datasets=None
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def solve_mtsp_by_end2end_model(tsps_list,model_path,model_size,size_scale_factor,cluster_algo,level2_tour_merge_algo="FarestInsertion",normalize=True,max_iterations=100,n_jobs=-1,doLevel2Clustering=True):
    
    opts=make_opts()
    opts.model=model_path

   
    if tsps_list is None or len(tsps_list)==0:
        print("!!! tsps_list is None or len(tsps_list)==0")
        return None
   
    # load model
    model, _ = load_model(opts.model)

    mtsp_subtsp_tour_list=[]
    for tsp in tqdm(tsps_list):   
        if tsp.shape[0]<=model_size*(1+size_scale_factor) or doLevel2Clustering == False:
            ## solve it directly with end2end Model model.
            result_dict=solve_one_tsp_by_end2end_model(tsp,model,opts,normalize)
            mtsp_subtsp_tour_list.append(result_dict["sorted_tsp_data"])
        else:
            k=math.floor(tsp.shape[0]/model_size)
            if tsp.shape[0]-k*model_size>model_size*size_scale_factor:
                k=k+1
            ## cluster tsp into k parts.
            if cluster_algo=="k_means" or cluster_algo=="kmeans":
                clusters=k_means_clustering_sklearn(tsp, k, max_iterations, n_jobs)
                clusters=cluster_tuning_for_one_node_cluster(clusters,False)
            elif cluster_algo=="k_means++" or cluster_algo=="kmeans++" or cluster_algo=="k-means++":
                # use k_means++ to cluster tsp into k parts.
                clusters=k_means_plusplus_clustering_sklearn(tsp, k, max_iterations, n_jobs)
                ### Tuning the clusters, make sure there is no cluster that only contains one point.
                clusters=cluster_tuning_for_one_node_cluster(clusters,False)
            else:
                print("!!! cluster_algo is not supported!")
                return None
            
            # solve each cluster with end2end Model and merge the results
            sub_tsp_result_list=[]
            for cluster in clusters:
                tmp_tsp_data=np.array(cluster['points'])
                ## result_dict, contains ori_tsp_data, and the sorted_tsp_data. the type is np.array
                result_dict=solve_one_tsp_by_end2end_model(tmp_tsp_data,model,opts,normalize)
                result_dict["center"]=cluster['centroid']
                
                sub_tsp_result_list.append(result_dict)
            # merge the results
            merged_tour=merge_sub_tsps(sub_tsp_result_list, strategy=level2_tour_merge_algo)
            if merged_tour is None:
                print("!!! Exception occured during merging tour!")
                return None
            mtsp_subtsp_tour_list.append(merged_tour)

    # a list, each element is a 2d np.array, which is a tour of a sub tsp.
    return mtsp_subtsp_tour_list




if __name__ == "__main__":

    pass

