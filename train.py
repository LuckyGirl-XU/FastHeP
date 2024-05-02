import torch
import numpy as np
from tqdm import tqdm
import math
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from eval import *
import logging
import random
import time
from utils import *
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True

def train_val(train_val_data, model, mode, bs, epochs, criterion, optimizer, early_stopper,  he_infos, rand_samplers, logger, model_dim, n_hop=2):

  train_data, val_data = train_val_data
  train_rand_sampler, val_rand_sampler = rand_samplers
  partial_he_info, full_he_info = he_infos
  if mode == 't':  # transductive
        model.update_he_info(full_he_info)
  elif mode == 'i':  # inductive
        model.update_he_info(partial_he_info)
  device = model.n_feat_th.data.device
  num_instance = len(train_data) 

  num_batch = math.ceil(num_instance / bs)
  logger.info('num of training instances: {}'.format(num_instance))
  logger.info('num of batches per epoch: {}'.format(num_batch))
  idx_list = list(train_data.keys())
  seeds = []
  seed = random.randint(0,100)
  train_time = []
  for epoch in range(epochs):
    model.set_seed(seed)
    set_random_seed(seed)
    seeds.append(seed)
    model.reset_store()#每个epoch
    model.reset_self_rep()#
    acc, ap, f1, auc, m_loss, all_neg_time, all_network_time = [], [], [], [], [], [], []
    logger.info('start {} epoch'.format(epoch))
    train_start = time.time()
    for k in tqdm(range(num_batch)):
      s_idx = k * bs
      e_idx = min(num_instance-1, s_idx + bs)
      
      if s_idx == e_idx:
        continue
      batch_idx = idx_list[s_idx:e_idx] 
      np.random.shuffle(batch_idx)
      src_l_cut, he_offset_l_cut, ts_l_cut = construct_algo_data_given_he_ids(batch_idx, train_data)
      size = len(batch_idx)
      bad_l_cut = train_rand_sampler.sample(src_l_cut, he_offset_l_cut)
      optimizer.zero_grad()
      model.train()
      
      pos_prob, neg_prob, neg_time, network_time = model.contrast(src_l_cut, bad_l_cut, he_offset_l_cut, ts_l_cut)   # the core training code,
      pos_label = torch.ones(size, dtype=torch.float, device=device, requires_grad=False)
      neg_label = torch.zeros(size, dtype=torch.float, device=device, requires_grad=False)
      loss = criterion(pos_prob, pos_label) + criterion(neg_prob, neg_label)
      
      loss.backward()
      optimizer.step()
      
      with torch.no_grad():
        model.eval()
        pred_score = np.concatenate([pos_prob.cpu().detach().numpy(), neg_prob.cpu().detach().numpy()])
        pred_label = pred_score > 0.5
        true_label = np.concatenate([np.ones(size), np.zeros(size)])
        acc.append((pred_label == true_label).mean())
        
        ap.append(average_precision_score(true_label, pred_score))
        f1.append(f1_score(true_label, pred_label))
        m_loss.append(loss.item())
        auc.append(roc_auc_score(true_label, pred_score))
        all_neg_time.append(neg_time)
        all_network_time.append(network_time)
    train_end = time.time()
    train_time.append(train_end - train_start)
    TecHNet_results(logger, train_time, "train_time")
    # validation phase use all information
    val_start = time.time()
    val_acc, val_ap, val_f1, val_auc, _, _ = eval_one_epoch('val for {} nodes'.format(mode), model, val_rand_sampler, val_data)
 
    val_end = time.time()
    logger.info('epoch: {}:'.format(epoch))
    logger.info('epoch mean loss: {}'.format(np.mean(m_loss)))
    logger.info('train acc: {}, val acc: {}'.format(np.mean(acc), val_acc))
    logger.info('train auc: {}, val auc: {}'.format(np.mean(auc), val_auc))
    logger.info('train ap: {}, val ap: {}'.format(np.mean(ap), val_ap))
    logger.info('train ap: {}, val f1: {}'.format(np.mean(f1), val_f1))
    logger.info('train time: {}, val time: {}'.format(train_end - train_start, val_end - val_start))
    logger.info('neighbor time: {}'.format(np.sum(all_neg_time)))
    logger.info('message passing time: {}'.format(np.sum(all_network_time)))
    #exit()
    if epoch == 0:
      
      checkpoint_dir = '/'.join(model.get_checkpoint_path(0).split('/')[:-1])

    # early stop check and checkpoint saving
    if early_stopper.early_stop_check(val_auc):
      logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
      logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
      best_checkpoint_path = model.get_checkpoint_path(early_stopper.best_epoch)
      model.load_state_dict(torch.load(best_checkpoint_path))
      best_ngh_store = []
      model.clear_store()
      for i in range(n_hop + 1):
        best_ngh_store_path = model.get_ngh_store_path(early_stopper.best_epoch, i)
        best_ngh_store.append(torch.load(best_ngh_store_path))
      model.set_neighborhood_store(best_ngh_store)
      best_self_rep_path = model.get_self_rep_path(early_stopper.best_epoch)
      best_prev_raw_path = model.get_prev_raw_path(early_stopper.best_epoch)
      best_self_rep = torch.load(best_self_rep_path)
      best_prev_raw = torch.load(best_prev_raw_path)
      model.set_self_rep(best_self_rep, best_prev_raw)
      model.set_seed(seeds[early_stopper.best_epoch])
      logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
      model.eval()
      break
    else:
      for i in range(n_hop + 1):
        torch.save(model.neighborhood_store[i], model.get_ngh_store_path(epoch, i))
      torch.save(model.state_dict(), model.get_checkpoint_path(epoch))
      torch.save(model.self_rep, model.get_self_rep_path(epoch))
      torch.save(model.prev_raw, model.get_prev_raw_path(epoch))

