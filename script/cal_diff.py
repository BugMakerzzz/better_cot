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
parser.add_argument('--fig_mode', type=str, default='type')
args = parser.parse_args()

model_name = args.model
n_samples = args.n_samples
n_examples = args.n_examples
dataset = args.dataset 
target = args.target
fig_mode = args.fig_mode

path_file = f'../result/{dataset}/{model_name}/input_{target}_paths_e{n_examples}_s{n_samples}.json'
cot_flags = []
scores = []
samples = []
with open(path_file, 'r') as f:
    data = json.load(f)

idx = -1
for item in data:
    idx += 1
    if target in ['cans', 'pans']:
        path = item['path'][-1]
        if fig_mode == 'type':
            attr = np.mean(np.array([x['attr'] for x in path['inp_attr']]))
        else:
            attr = [x['attr'] for x in path['inp_attr']]
    elif target in ['pcot']:
        ref_ls = []
        for path in item['path']:
            ref_ls.append(path['ref'].strip('.').strip())
        attr = []
        for path in item['path']:
            score = []
            for tup in path['inp_attr'][:3]:
                if tup['inp'].strip('.').strip() in ref_ls:
                    continue
                else:
                    score.append(tup['attr'])
            attr.extend(score)
        attr = np.mean(np.array(attr))
    # elif target in ['pcot']:
    #     path = item['path'][-1]
    #     if fig_mode == 'type':
    #         attr = np.mean(np.array([x['attr'] for x in path['inp_attr']]))
    elif target in ['cot']:
        ans_path = item['path'][-1]
        attr = np.mean(np.array([x['attr'] for x in ans_path['inp_attr']]))
        rule_path = item['path'][-2]
        if rule_path['inp_attr']:
            attr = np.mean(np.array([x['attr'] for x in rule_path['inp_attr']]))
    # elif target in ['diff_pcot']:
    #     if not item['path']:
    #         continue
    #     path = item['path'][-1]
    #     attr = np.max(np.array([x[1] for x in path['diff_inp_attr']]))
    elif target in ['diff_pcot']:
        ref_ls = []
        for path in item['path']:
            ref_ls.append(path['last_ref'].strip('.').strip())
        attr = []
        for path in item['path']:
            score = []
            for tup in path['diff_inp_attr'][:3]:
                if tup[0].strip('.').strip() in ref_ls:
                    continue
                else:
                    score.append(tup[1])
            attr.extend(score)
        attr = np.mean(np.array(attr))
    if fig_mode == 'type':
        samples.append(idx)
        scores.append(attr)
        cot_flags.append(item['cot_flag'])
    else:
        samples.extend([idx]*len(attr))
        scores.extend(attr)
        cot_flags.extend([item['cot_flag']]*len(attr))
        # for inp_path in item['path'][-2]['inp_attr']:
        #     score = inp_path['attr']
        #     samples.append(idx)
        #     scores.append(score)
        #     cot_flags.append(item['cot_flag'])
        # std = np.var(np.array(attrs))
        # score = std

    # cot_flags.append(1)
    
    

fig_path = f'../result/{dataset}/{model_name}/{target}_{fig_mode}_box_fig_{n_samples}.pdf'
draw_box(samples, cot_flags, scores, fig_path, fig_mode)