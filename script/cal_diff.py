import json
import numpy as np
import argparse
import os
from transformers import AutoTokenizer
from config import *
from metrics import draw_box

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama2_13b')
parser.add_argument('--n_samples', type=int, default=100)
parser.add_argument('--n_examples', type=int, default=3)
parser.add_argument('--dataset', type=str, default='proofwriter_d1')
parser.add_argument('--target', type=str, default='cans')
args = parser.parse_args()

model_name = args.model
n_samples = args.n_samples
n_examples = args.n_examples
dataset = args.dataset 
target = args.target

path_file = f'../result/{dataset}/{model_name}/input_{target}_paths_e{n_examples}_s{n_samples}.json'
cot_flags = []
scores = []

with open(path_file, 'r') as f:
    data = json.load(f)

for item in data:
    if not item['cor_flag']:
        continue
    score = 0.0
    cnt = 0
    path = item['path'][-1]
    if path['inp_attr']:
        for tup in path['inp_attr']:
            score += tup['attr']
            cnt += 1
    score /= cnt
    if item['cot_flag'] == 0:
        cot_flags.append(0)
    else:
        cot_flags.append(1)
    # cot_flags.append(item['cot_flag'])
    scores.append(score)

fig_path = f'../result/{dataset}/{model_name}/{target}_fig.pdf'
draw_box(cot_flags, scores, fig_path)