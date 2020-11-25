"""

"""
import argparse
from os.path import join
import os
import torch
import multiprocessing

parser = argparse.ArgumentParser(description="Go!")
parser.add_argument('--bpr_batch_size', type=int, default=2048, help="the batch size for bpr loss training procedure")
parser.add_argument('--embed_dim', type=int, default=64, help="the embedding size of lightGCN")
parser.add_argument('--n_layers', type=int, default=3, help="the layer num of model")
parser.add_argument('--layer_size', nargs='?', default='[64, 64, 64]', help="sizes of every layer's rep.")
parser.add_argument('--model', type=str, default='ngcf', help='rec-model, support [ngcf, mf, lgn, lgn_ws, lgn_ecc]',
                    choices=['ngcf', 'mf', 'lgn', 'lgn_ws', 'lgn_ecc'])
parser.add_argument('--lr', type=float, default=0.001, help="the learning rate")
parser.add_argument('--reg_lambda', type=float, default=1e-4, help="the lambda coefficient for l2 normalization")
parser.add_argument('--dropout', type=int, default=0, help="using the dropout or not")
parser.add_argument('--drop_rate', type=float, default=0.5, help="the proportion of batch size for training procedure")
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--n_fold', type=int, default=1, help="the fold num used to split large adj matrix, like gowalla")
parser.add_argument('--test_batch', type=int, default=100, help="the batch size of users for testing")
parser.add_argument('--test_every_n_epochs', type=int, default=20, help="test every n epochs")
parser.add_argument('--dataset', type=str, default='gowalla',
                    help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]",
                    choices=['lastfm', 'gowalla', 'yelp2018', 'amazon-book'])
parser.add_argument('--topks', nargs='?', default="[20]", help="@k test list")
parser.add_argument('--tensorboard', type=int, default=0, help="enable tensorboard")
parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
parser.add_argument('--seed', type=int, default=2020, help='random seed')
parser.add_argument('--simutaneously', type=int, default=0, help='whether train embedding and mapping simutaneously')
parser.add_argument('--separate_graph', type=int, default=1, help='whether separate the train graph')
parser.add_argument('--test_graph', type=int, default=0, help='index of test graph [whole, train, val]')
parser.add_argument('--p_val', type=float, default=0.1, help='partition of valset occupying trainset')
parser.add_argument('--interval', type=int, default=1, help='num of epochs to change fixed param')
parser.add_argument('--fine_tune_epochs', type=int, default=0)
parser.add_argument('--ecc', type=int, default=0, help='use 1 ecc or 0 mcc')
parser.add_argument('--ecc_layer', type=int, default=1, help='the layer num of ecc or mcc')
parser.add_argument('--p_dist', type=float, default=0., help='...')

args = parser.parse_args()

# PATH
_ROOT_PATH = "../"
args.CODE_PATH = join(_ROOT_PATH, 'code')
args.DATA_PATH = join(_ROOT_PATH, 'data')
args.BOARD_PATH = join(_ROOT_PATH, 'runs')
args.FILE_PATH = join(_ROOT_PATH, 'checkpoints')
if not os.path.exists(args.FILE_PATH):
    os.makedirs(args.FILE_PATH, exist_ok=True)

# others
args.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
args.n_cores = multiprocessing.cpu_count() // 2
args.layer_size = eval(args.layer_size)
args.topks = eval(args.topks)
