import argparse
from utils import *

parser = argparse.ArgumentParser() 
parser.add_argument("--task", type=str, default="MF")
parser.add_argument("--model_num", type=int, default=1)
parser.add_argument("--train", action='store_true')
parser.add_argument("--test", action='store_true')
parser.add_argument("--seed", type=int, default=2023)
parser.add_argument("--run_id", type=str, default='test')
parser.add_argument("--num_workers", type=int, default=8)

args = parser.parse_args()

for seed in range(args.seed, args.seed + args.model_num):
    Seed_everything(seed=seed)
    train_and_test(run_id = args.run_id, args=args,seed=seed)


