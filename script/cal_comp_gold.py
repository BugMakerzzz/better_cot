import argparse
import json
from transformers import set_seed
import numpy as np
from pandas import DataFrame
from metrics import draw_line, cal_rouge
import random

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama2_13b')
parser.add_argument('--n_samples', type=int, default=100)
parser.add_argument('--n_examples', type=int, default=3)
parser.add_argument('--dataset', type=str, default='proofwriter_d1')
parser.add_argument('--target', type=str, default='pans')
args = parser.parse_args()
set_seed(17)

model_name = args.model
n_samples = args.n_samples
n_examples = args.n_examples
dataset = args.dataset 
target = args.target

path_file = f'../result/{dataset}/{model_name}/cot_{model_name}_{target}_paths_e3_s100.json'
gold_path_file = f'../result/{dataset}/{model_name}/gold_cot_{model_name}_{target}_paths_e3_s100.json'

with open(path_file, 'r') as f:
    path_data = json.load(f)
    f.close()
with open(gold_path_file, 'r') as f:
    gold_data = json.load(f)
    f.close()  

path_dic = {}

cot_flags = []
nums = []
scores = []

data = {'cot_flag':cot_flags, 'nums':nums, 'score':scores}
data = DataFrame(data)
names = ['nums', 'score', 'cot_flag']
path = f'../result/{dataset}/{model_name}/{target}_overlap_line_fig.pdf'
draw_line(data, path, names)