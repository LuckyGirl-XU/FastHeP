import math
import torch
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from utils import *


def eval_one_epoch(hint, technet, sampler, he_info, bs = 100):
  val_acc, val_ap, val_f1, val_auc, val_neigh_time, val_network_time = [], [], [], [], [], []
  with torch.no_grad():
    technet = technet.eval()
    TEST_BATCH_SIZE = bs
    num_test_instance = len(he_info)

    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
    idx_list = list(he_info.keys())
    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance-1, s_idx + TEST_BATCH_SIZE)
      if s_idx == e_idx:
        continue
      batch_idx = idx_list[s_idx:e_idx]
      src_l_cut, he_offset_l_cut, ts_l_cut = construct_algo_data_given_he_ids(batch_idx, he_info)
      size = len(batch_idx)
     
      bad_l_cut = sampler.sample(src_l_cut, he_offset_l_cut)

      pos_prob, neg_prob, neigh_time, network_time = technet.contrast(src_l_cut, bad_l_cut, he_offset_l_cut, ts_l_cut, test = True)
     
      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      
      pred_label = pred_score > 0.5
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      val_acc.append((pred_label.flatten() == true_label).mean())
      val_ap.append(average_precision_score(true_label, pred_score))
      val_f1.append(f1_score(true_label, pred_label))
      val_auc.append(roc_auc_score(true_label, pred_score))
      val_neigh_time.append(neigh_time)
      val_network_time.append(network_time)
  return np.mean(val_acc), np.mean(val_ap), np.mean(val_f1), np.mean(val_auc), np.sum(val_neigh_time), np.sum(val_network_time)