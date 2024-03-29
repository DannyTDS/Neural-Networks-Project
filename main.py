import argparse
import os

from model import *
from dataio import *
from train import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', type=str)
parser.add_argument('--size', help='if not None, resize image', default=None, type=int)

parser.add_argument('--name', help='name of the experiment', default='new_exp', type=str)
parser.add_argument('--log_dir', help='path to keep output log, default is ./log', default='./log', type=str)

parser.add_argument('--start_lr', default=1e-5, type=float)
parser.add_argument('--final_lr', default=1e-6, type=float)
parser.add_argument('--p', default=1.0, help="norm order", type=float)
parser.add_argument('--z_dim', default=128, help="length of encoding vector z", type=int)
parser.add_argument('--clip', default=0.01, type=float)
parser.add_argument('--percep_freq', help='compute clip loss every n-th iterations', default=2, type=int)

parser.add_argument('--W', default=512, help="number of neurons per hidden layer", type=int)
parser.add_argument('--D', default=8, help="number of hidden layers", type=int)
parser.add_argument('--bsize', default=8192, type=int)
parser.add_argument('--iters', default=300000, type=int)
parser.add_argument('--save_freq', default=20000, type=int)
parser.add_argument('--save_dir', default='./out', type=str)
parser.add_argument('--silent', action='store_true')

args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # create save dir if not exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    #init train dataset
    train_dataset = TrainSet(args.data_dir, size=(128, 128))
    #init model
    inter_fn = lerp
    model = Net(n_emb = len(train_dataset), norm_p = args.p, inter_fn=inter_fn, D=args.D, z_dim = args.z_dim, in_feat=2, out_feat=3, W=args.W, with_res=False, with_norm=True)
    model.to(DEVICE)
    #init solver
    solver = Solver(args, train_dataset, model, DEVICE)
    # train the model
    solver.train()


if __name__ == '__main__':
    main()