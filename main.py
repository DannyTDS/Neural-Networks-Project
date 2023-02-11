from model import *
from dataio import *
import argparse
from train import *

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='exp', type=str)
parser.add_argument('--data_dir', default='data', type=str)
parser.add_argument('--dset', default='LF', type=str)
parser.add_argument('--scene', default='knights', type=str)
parser.add_argument('--size', help='if not None, resize image', default=None, type=int)

parser.add_argument('--start_lr', default=1e-5, type=float)
parser.add_argument('--final_lr', default=1e-6, type=float)
parser.add_argument('--p', default=1.0, type=float)
parser.add_argument('--z_dim', default=128, type=int)
parser.add_argument('--clip', default=0.01, type=float)
parser.add_argument('--percep_freq', help='compute clip loss every n-th iterations', default=2, type=int)

parser.add_argument('--W', default=512, type=int)
parser.add_argument('--D', default=8, type=int)
parser.add_argument('--bsize', default=8192, type=int)
parser.add_argument('--iters', default=300000, type=int)
parser.add_argument('--save_freq', default=20000, type=int)
parser.add_argument('--silent', action='store_true')

args = parser.parse_args()

def main():
    #init train dataset
    dataset = TrainSet(args).getTrainDataset()

    #init model
    model = VIINTER()
    #init solver
    solver = Solver(dataset,args,model)
    solver.train()
    pass

def inf():
    #init model
    model = VIINTER()
    #load weight

    #init inf dataset

    #init solver
    solver = Solver(dataset,arge,model)
    solver.inf()
    pass




if "__name__" == "main":
    main()
    # inf()