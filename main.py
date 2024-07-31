import pandas as pd
from log import *
from parser import *
from eval import *
from utils import *
from train import *
from module import FastHeP
import resource
import torch.nn as nn
import statistics
from load_dataset import *

args, sys_argv = get_args()

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_EPOCH = args.n_epoch
ATTN_NUM_HEADS = args.attn_n_head
DROP_OUT = args.drop_out
DATA = args.data
NUM_HOP = args.n_hop
LEARNING_RATE = args.lr
POS_DIM = args.pos_dim
TOLERANCE = args.tolerance
VERBOSITY = args.verbosity
SEED = args.seed
TIME_DIM = args.time_dim
REPLACE_PROB = args.replace_prob
SELF_DIM = args.self_dim
GPU = args.gpu
NGH_DIM = args.ngh_dim 
REUSE = args.reuse
TEST_REUSE = args.reuse_test
PRETRAINED = args.pretrained
SPLIT = args.split




set_random_seed(SEED)
logger, get_checkpoint_path, get_ngh_store_path, get_self_rep_path, get_prev_raw_path, best_model_path, best_model_ngh_store_path = set_up_logger(args, sys_argv)



n_v, v_simplices, ts, dataset_name =  load_dataset(DATA)


full_he_info = generate_he_info(n_v, ts, v_simplices)

print("full_he_info", len(full_he_info))

total_node_set = np.unique(np.array(v_simplices))


num_total_unique_nodes = len(total_node_set)
max_total = max(total_node_set)
print("max_total", max(total_node_set))
print("num_total_unique_nodes", num_total_unique_nodes)
n_feat = np.zeros((max_total +1, 64))
num_total_hyperedges = len(full_he_info)
print("num_total_hyperedges", num_total_hyperedges)
e_feat = np.zeros((num_total_hyperedges +1, 64))
ts_l = np.array(ts)

if args.data in ["coauth-DBLP", "threads-stack-overflow"]:
    val_time, test_time = list(np.quantile(ts_l, [0.4, 0.6]))
else:
    val_time, test_time = list(np.quantile(ts_l, [SPLIT, SPLIT+0.15]))



transductive_auc = []
transductive_ap = []
inductive_auc = []
inductive_ap = []

test_times = []
early_stoppers = []
total_time = []
for run in range(args.run):
  if args.mode == 't':
    logger.info('Transductive training...')
    valid_train_he_ids = np.where(ts_l <= val_time)[0] + 1 
    valid_val_he_ids = np.where((ts_l > val_time) & (ts_l <= test_time))[0] + 1
    valid_test_he_ids = np.where(ts_l > test_time)[0] + 1

  else:
    assert(args.mode == 'i')
    logger.info('Inductive training...')
    hes_ids_after_val_time = np.where((ts_l > val_time))[0] + 1 
    he_nodes_after_val_time = set().union(*[full_he_info[i][0] for i in hes_ids_after_val_time]) 
    mask_node_set = set(random.sample(he_nodes_after_val_time, int(0.1 * num_total_unique_nodes))) 
    
    he_has_masked_nodes = np.array([len(set(full_he_info[i][0]) & mask_node_set) > 0 for i in range(1, num_total_hyperedges+1)])
    
    valid_train_he_ids = np.where((ts_l <= val_time)  & ~(he_has_masked_nodes))[0]+1# Train edges can not contain any masked nodes
    valid_val_he_ids = np.where((ts_l > val_time) & (ts_l <= test_time) & ~(he_has_masked_nodes))[0]+1# Val edges can not contain any masked nodes
    valid_test_he_ids = np.where((ts_l > test_time) & (he_has_masked_nodes))[0]+1# test edges must contain at least one masked node
    
    he_is_all_masked_nodes = np.array([len(set(full_he_info[i][0]) & mask_node_set) == min(len(full_he_info[i][0]), len(mask_node_set)) for i in range(1, num_total_hyperedges+1)])
    valid_test_all_new_he_ids = np.where((ts_l > test_time) & (he_is_all_masked_nodes))[0]+1
    valid_test_new_old_he_ids = np.setdiff1d(valid_test_he_ids, valid_test_all_new_he_ids)
    
    logger.info('Sampled {} nodes (10 %) which are masked in training and reserved for testing...'.format(len(mask_node_set)))
    
    logger.info('Out of {} test hyperedges, {} are all_new and {} are new_old'.format(len(valid_test_he_ids), len(valid_test_all_new_he_ids), len(valid_test_new_old_he_ids)))
    
  
  train_data = {key: full_he_info[key] for key in valid_train_he_ids}
    
  val_data = {key: full_he_info[key] for key in valid_val_he_ids}
  test_data = {key: full_he_info[key] for key in valid_test_he_ids}
  if args.mode == 'i':
    test_all_new_data = {key: full_he_info[key] for key in valid_test_all_new_he_ids}
    test_new_old_data = {key: full_he_info[key] for key in valid_test_new_old_he_ids}
       
  train_val_data = (train_data, val_data)
  max_idx = max(v_simplices)

  assert(min(v_simplices) > 0)
  train_and_val_he_ids = np.union1d(valid_train_he_ids, valid_val_he_ids)
  partial_he_info = {key: full_he_info[key] for key in train_and_val_he_ids}
  he_infos = partial_he_info, full_he_info

  print("data split details: ", len(train_data), len(val_data), len(test_data))
  

  
  train_nodes = set().union(*[train_data[i][0] for i in train_data])
  val_nodes = set().union(*[val_data[i][0] for i in val_data])
  test_nodes = set().union(*[test_data[i][0] for i in test_data])

  train_rand_sampler = RandHyperEdgeSampler([train_nodes])
  val_rand_sampler = RandHyperEdgeSampler([train_nodes, val_nodes])
  test_rand_sampler = RandHyperEdgeSampler([train_nodes, val_nodes, test_nodes])
  rand_samplers = train_rand_sampler, val_rand_sampler
    

  rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
  resource.setrlimit(resource.RLIMIT_NOFILE, (200*args.bs, rlimit[1]))

  # model initialization
  device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
  device = torch.device(device_string)
  
  feat_dim = n_feat.shape[1]
  e_feat_dim = e_feat.shape[1]
  time_dim = TIME_DIM
  model_dim = feat_dim + e_feat_dim + time_dim
  hidden_dim = e_feat_dim + time_dim
  num_raw = 2 
  memory_dim = NGH_DIM + num_raw 
  num_neighbors = [1]
  for i in range(NUM_HOP):
    num_neighbors.extend([int(NUM_NEIGHBORS[i])])
  

  total_start = time.time()
  technet = TecHNet(n_feat, e_feat, memory_dim, max_idx, time_dim=TIME_DIM, pos_dim=POS_DIM, n_head=ATTN_NUM_HEADS, num_neighbors=num_neighbors, dropout=DROP_OUT,
    linear_out=args.linear_out, get_checkpoint_path=get_checkpoint_path, get_ngh_store_path=get_ngh_store_path, get_self_rep_path=get_self_rep_path, get_prev_raw_path=get_prev_raw_path, verbosity=VERBOSITY,
  n_hops=NUM_HOP, replace_prob=REPLACE_PROB, self_dim=SELF_DIM, ngh_dim=NGH_DIM, device=device)

  if PRETRAINED:
    logger.info('Lodaing pretrained model...')
    pretrained_model_path = "pretrained_models/"+ DATA +"/best-model.pth"
    technet.load_state_dict(torch.load(pretrained_model_path))
  technet.to(device)
  technet.reset_store()#作用是什么

  optimizer = torch.optim.Adam(technet.parameters(), lr=LEARNING_RATE)
  criterion = torch.nn.BCELoss()
  early_stopper = EarlyStopMonitor(tolerance=TOLERANCE)

  # start train and val phases
  if not PRETRAINED:
      train_val(train_val_data, technet, args.mode, BATCH_SIZE, NUM_EPOCH, criterion, optimizer, early_stopper, he_infos, rand_samplers, logger, model_dim, n_hop=NUM_HOP)

  # final testing
  print("_*"*50)
  technet.reset_store()
  technet.reset_self_rep()

  test_start = time.time()
  test_acc, test_ap, test_f1, test_auc, test_neighbr_time, test_network_time = eval_one_epoch('test for {} nodes'.format(args.mode), technet, test_rand_sampler, test_data)
  test_end = time.time()
  logger.info('Test statistics: {} all nodes -- auc: {}, ap: {}, acc: {}, f1: {}, time: {}'.format(args.mode, test_auc, test_ap, test_acc, test_f1, test_end - test_start))
  logger.info("neighbor time: {}".format(test_neighbr_time))
  logger.info('message passing time: {}'.format(test_network_time))
  if args.mode == 'i':
    technet.reset_store()
    technet.reset_self_rep()
    test_new_old_acc, test_new_old_ap, test_new_old_f1, test_new_old_auc, test_new_neighbr_time, test_new_network_time = eval_one_epoch('test for {} nodes'.format(args.mode), technet, test_rand_sampler, test_new_old_data)
    logger.info('Test statistics: {} new_old nodes -- auc: {}, ap: {}, acc: {}, f1: {}'.format(args.mode, test_new_old_auc, test_new_old_ap, test_new_old_acc, test_new_old_f1))
    logger.info("neighbor time: {}".format( test_new_neighbr_time ))
    logger.info('message passing time: {}'.format(test_new_network_time))

  if args.mode == 'i':
    inductive_auc.append(test_new_old_auc)
    inductive_ap.append(test_new_old_ap)
  else:
    transductive_auc.append(test_auc)
    transductive_ap.append(test_ap)
  test_times.append(test_end - test_start)
  early_stoppers.append(early_stopper.best_epoch + 1)
  # save model
  logger.info('Saving TecHNet model ...')
  torch.save(technet.state_dict(), best_model_path)
  logger.info('TecHNet model saved')

  
  total_end = time.time()
  print("TecHNet experiment statistics:")
  if args.mode == "t":
    TecHNet_results(logger, transductive_auc, "transductive_auc")
    TecHNet_results(logger, transductive_ap, "transductive_ap")
  else:
    TecHNet_results(logger, inductive_auc, "inductive_auc")
    TecHNet_results(logger, inductive_ap, "inductive_ap")
  
  TecHNet_results(logger, test_times, "test_times")
  TecHNet_results(logger, early_stoppers, "early_stoppers")
  total_time.append(total_end - total_start)
  TecHNet_results(logger, total_time, "total_time")
