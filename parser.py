import argparse
import sys


def get_args():
  parser = argparse.ArgumentParser('Interface for TecHNet')

  # select dataset and training mode
  parser.add_argument('-d', '--data', type=str, help= "email-Enron")
  parser.add_argument('-m', '--mode', type=str, default='t', choices=['t', 'i'], help='transductive (t) or inductive (i)')

  parser.add_argument('--n_degree', nargs='*', default=['16'],
            help='a list of neighbor sampling numbers for different hops, when only a single element is input n_layer will be activated')
  parser.add_argument('--n_hop', type=int, default=2, help='number of hops the HCNR scheme is used')
  parser.add_argument('--bias', default=0.0, type=float, help='the hyperparameter alpha controlling sampling preference with time closeness, default to 0 which is uniform sampling')
  parser.add_argument('--pos_dim', type=int, default=0, help='dimension of the positional embedding')
  parser.add_argument('--self_dim', type=int, default=100, help='dimension of the self representation')
  parser.add_argument('--ngh_dim', type=int, default=4, help='dimension of the HCNR scheme')
  parser.add_argument('--linear_out', action='store_true', default=False, help="whether to linearly project each node's ")

  parser.add_argument('--attn_n_head', type=int, default=1, help='number of heads used in graph attention')
  parser.add_argument('--time_dim', type=int, default=64, help='dimension of the time embedding')
  parser.add_argument('--n_epoch', type=int, default=30, help='number of epochs')
  parser.add_argument('--bs', type=int, default=64, help='batch_size')
  parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
  parser.add_argument('--drop_out', type=float, default=0.2, help='dropout probability for all dropout layers')
  parser.add_argument('--replace_prob', type=float, default=0.8, help='probability for storing new neighbors to HCNR scheme replacing old ones')
  parser.add_argument('--tolerance', type=float, default=1e-3,
            help='toleratd margainal improvement for early stopper')
  parser.add_argument('--seed', type=int, default=0, help='random seed for all randomized algorithms')
  parser.add_argument('--verbosity', type=int, default=1, help='verbosity of the program output')
  parser.add_argument('--run', type=int, default=1, help='number of model runs')
  parser.add_argument('--reuse', action='store_true', help='reuse historical embeddings')
  parser.add_argument('--reuse_test', action='store_true',help='reuse when testing')
  parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
  parser.add_argument('--pretrained', action='store_true', help='Use pre-trained model')
  parser.add_argument('--split', type=float, default=0.7, help='Data split')


  try:
    args = parser.parse_args()
  except:
    parser.print_help()
    sys.exit(0)
  return args, sys.argv