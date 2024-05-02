import numpy as np
import torch
import os
import random
import statistics
import math
from itertools import combinations


def generate_he_info(n_v, ts, v_simplices):
    full_he_info = {} # for each hyperedge: (set(nodes) and timestamp)
    v_start_idx = 0
    for he_idx, (n_v_i, ts_i) in enumerate(zip(n_v, ts)):
        if n_v_i == 1:
            he_i_node = tuple(v_simplices[v_start_idx : v_start_idx + n_v_i])
            he_i_nodes = he_i_node + he_i_node 
        else:
            he_i_nodes = tuple(v_simplices[v_start_idx : v_start_idx + n_v_i])
                
        
        v_start_idx += n_v_i

        he_i = (he_i_nodes, ts_i)
        full_he_info[he_idx+1] = he_i
    return full_he_info

def generate_hyperedge(n_v, ts, v_simplices):
    full_he_info = {} # for each hyperedge: (set(nodes) and timestamp)
    full_ts = []
    v_start_idx = 0
    he_idx = 0
    for (n_v_i, ts_i) in zip(n_v, ts):
        if n_v_i > 1:
            he_i_nodes = set(v_simplices[v_start_idx : v_start_idx + n_v_i])
            he_i = (he_i_nodes, ts_i)
            full_he_info[he_idx+1] = he_i
            full_ts.append(ts_i)
            he_idx += 1
#         else:
#             he_idx = he_idx -1
            
        v_start_idx += n_v_i

        #he_i = (he_i_nodes, ts_i)
        
    return full_he_info,full_ts

def generate_he_info_for_CE(n_v, ts, v_simplices):
    full_he_info = {} # for each hyperedge: (set(nodes) and timestamp)
    v_start_idx = 0
    s_he_idx = 1
    for he_idx, (n_v_i, ts_i) in enumerate(zip(n_v, ts)):
        he_i_nodes = set(v_simplices[v_start_idx : v_start_idx + n_v_i])
        v_start_idx += n_v_i
        #make simple edges from hyperedge
        vertex_pairs = combinations(he_i_nodes, 2)

        for simple_edge in vertex_pairs:
            s_he_i_nodes = set(simple_edge)
            s_he_i = (s_he_i_nodes, ts_i)
            full_he_info[s_he_idx] = s_he_i
            s_he_idx += 1        

    return full_he_info

def convert_strList_to_intList(l):
    return [int(x) for x in l]

def generate_nc_data_structures(hes, node_labels, label_names):
    hes_list = []
    for he in hes:
        he_nodes = convert_strList_to_intList(he.split(","))
        hes_list += [set(he_nodes)]
    
    node_labels_mapping = {}
    for i, label in enumerate(node_labels):
        node_labels_mapping[i+1] = label
    
    label_name_mapping = {}
    for i, name in enumerate(label_names):
        label_name_mapping[i+1] = name
    
    return hes_list, node_labels_mapping, label_name_mapping

    



def process_sampling_numbers(num_neighbors, num_layers):
    if not type(num_neighbors)==list: # handle default value
        num_neighbors = [num_neighbors]
    num_neighbors = [int(n) for n in num_neighbors]
    if len(num_neighbors) == 1:
        num_neighbors = num_neighbors * num_layers
    else:
        num_layers = len(num_neighbors)
    return num_neighbors, num_layers

def construct_algo_data_given_he_ids(valid_he_ids, he_info):
    src_l, he_offset_l, ts_l = [], [0], []

    prev_he_offset_val = 0
    for he_idx in valid_he_ids:
        he_nodes = he_info[he_idx][0]
        src_l.extend(list(he_nodes))
        prev_he_offset_val += len(he_nodes)
        he_offset_l.append(prev_he_offset_val)
        ts_l.extend([he_info[he_idx][1]]*len(he_nodes))
    
    return src_l, he_offset_l, ts_l 

def nc_transfer_lr_construct_algo_data_given_nodes(batch_nodes, he_info, train_time):
    src_l, he_offset_l, ts_l = [], [0], []

    prev_he_offset_val = 0
    for node in batch_nodes:
        src_l.append(node)
        prev_he_offset_val += 1
        he_offset_l.append(prev_he_offset_val)
        ts_l.append(train_time)
    
    return src_l, he_offset_l, ts_l 

def construct_algo_data_given_nodes(batch_nodes, he_info, sampled_he_per_node):
    
    sampled_he_idxs = []
    for node in batch_nodes:
        #find hes having node in them
        hes_having_node = []
        for i in he_info:
            if(node in he_info[i][0]):
                hes_having_node += [i]
        sampled_hes_for_node = random.choices(hes_having_node, k=sampled_he_per_node)

        sampled_he_idxs.extend(sampled_hes_for_node)
    
    return construct_algo_data_given_he_ids(sampled_he_idxs, he_info)

class RandHyperEdgeSampler(object):
    def __init__(self, nodes):
        nodes =  set().union(*nodes)
        self.nodes_list = np.array(list(nodes))

    def sample(self, src_l, he_offset_l):
        fake_src_l = []
        src_l = np.array(src_l)
        for he_idx in range(len(he_offset_l)-1):
            s_idx, e_idx = he_offset_l[he_idx], he_offset_l[he_idx+1]
            he_nodes = src_l[s_idx:e_idx]
            he_size = e_idx - s_idx
            remained_nodes = np.setdiff1d(self.nodes_list, he_nodes)

            #keep half of the nodes in the source hyperedge and replace the rest
            n_kept_nodes, n_random_nodes = he_size//2, he_size - he_size//2
            kept_nodes = np.random.choice(he_nodes, n_kept_nodes)
            random_nodes = np.random.choice(remained_nodes, n_random_nodes)

            fake_src_l.extend(kept_nodes)
            fake_src_l.extend(random_nodes)

        return fake_src_l


class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-5):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
            
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        self.epoch_count += 1
        return self.num_round >= self.max_round
    

def set_random_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)


def process_sampling_numbers(num_neighbors, num_layers):
  num_neighbors = [int(n) for n in num_neighbors]
  if len(num_neighbors) == 1:
    num_neighbors = num_neighbors * num_layers
  else:
    num_layers = len(num_neighbors)
  return num_neighbors, num_layers

def TecHNet_results(logger, arr, name):
  logger.info(name + " " + str(arr))
  logger.info("Mean " + str(100 * statistics.mean(arr)))
  logger.info("Standard deviation " + str(statistics.pstdev(arr)))
  logger.info("95% " + str(1.96 * 100 * statistics.pstdev(arr) / math.sqrt(len(arr))))
  logger.info("--------")