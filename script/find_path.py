import json
import numpy as np
import argparse
import os
from transformers import AutoTokenizer
from utils.config import *
from utils.model import ModelWrapper

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama3_1_8b_chat')
parser.add_argument('--n_samples', type=int, default=200)
parser.add_argument('--n_examples', type=int, default=3)
parser.add_argument('--dataset', type=str, default='proofwriter_d1')
parser.add_argument('--target', type=str, default='cot')

args = parser.parse_args()

model_name = args.model
n_samples = args.n_samples
n_examples = args.n_examples
dataset = args.dataset 
target = args.target
proxy = args.proxy
method = args.method
proxy = model_name

score_path = f'../result/{dataset}/{model_name}/{target}_info_e{n_examples}_{n_samples}.json'

with open(score_path, 'r') as f:
    score_data = json.load(f)
    f.close()

def find_step_index(tokens, tokenizer):
    step_idx_dic = {}
    start = 0
    end = 0
    num = 0
    ban_token_ls = ['?', '<0x0A>', '\".',] 
    if dataset == 'wino':
        ban_token_ls.extend(['▁but','▁because','_as', '▁,', ','])
    for i in range(len(tokens)):
        token = tokens[i]
        if token == '<0x0A>' and i == 0:
            start = 1
        
        if i == len(tokens)-1 \
            or token in ban_token_ls \
            or token == '.' and (not tokens[i-1].isdigit() or not tokens[i+1].isdigit()):
        
            end = i
            if end - start > 1 or end == len(tokens) - 1:
                if end == len(tokens) - 1 and start == 0:
                    end += 1
                step = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens[start:end]))
                step_idx_dic[start] = {'end':end, 'step':step, 'num':num}
                num += 1
                start = i + 1
    return step_idx_dic


tokenizer = ModelWrapper()

results = []
result_path = f'../result/{dataset}/{model_name}/{target}_path_e{n_examples}_{n_samples}.json'

for item in score_data:
  
    input_tokens = item['input']
    output_tokens = item['out']
    scores = np.array(item['scores'])

    input_step_dic = find_step_index(input_tokens, tokenizer)
    output_step_dic = find_step_index(output_tokens, tokenizer)
    
    step_attr_ls = []
    direct_path = []
    for k1, v1 in output_step_dic.items():
        s1 = k1
        e1 = v1['end']
        n1 = v1['num']
        score_ls = []
        for k2, v2 in input_step_dic.items():
            s2 = k2
            e2 = v2['end']
            n2 = v2['num']
            if target == 'cot' and s2 >= s1-1:
                continue
            attr = scores[s2:e2, s1:e1].mean()
            msg = {'inp_idx':(s2, e2),'inp':v2['step'], 'attr':attr, 'num':n2}
            score_ls.append(msg)
        score_ls = sorted(score_ls, key=lambda x: x['attr'], reverse=True)
        if len(score_ls) > 1:
            n2 = f"{score_ls[0]['num']}&{score_ls[1]['num']}"
        elif len(score_ls) == 1:
            n2 = f"{score_ls[0]['num']}&-1"
        else:
            n2 = "-1&-1"
        step_attr_ls.append({'ref_idx':(s1, e1), 'ref':v1['step'], 'inp_attr':score_ls})
        direct_path.append(f"{n2}->{n1}")
    if 'cot_flag' not in item.keys():
        item['cot_flag'] = None
    result_msg = {'id':item['id'], 'cor_flag':item['cor_flag'], 'cot_flag':item['cot_flag'], 'direct':direct_path, 'path':step_attr_ls}
    
    results.append(result_msg)
    
with open(result_path, 'w') as f:
    json.dump(results, f, indent=4)
