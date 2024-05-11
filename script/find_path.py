import json
import numpy as np
import argparse
import os
from transformers import AutoTokenizer
from config import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama2_13b')
parser.add_argument('--n_samples', type=int, default=100)
parser.add_argument('--n_examples', type=int, default=3)
parser.add_argument('--dataset', type=str, default='proofwriter_d1')
parser.add_argument('--target', type=str, default='cot')
parser.add_argument('--proxy', type=str, default='Llama2_13b')
parser.add_argument('--method', type=str, default='cot')

args = parser.parse_args()

model_name = args.model
n_samples = args.n_samples
n_examples = args.n_examples
dataset = args.dataset 
target = args.target
proxy = args.proxy
method = args.method

score_path = f'../result/{dataset}/{model_name}/{method}_{proxy}_{target}_scores_e{n_examples}_s{n_samples}.json'
if not os.path.exists(score_path):
    if method == 'direct':
        score_path = f'../result/{dataset}/{model_name}/input_{target}_scores_e{n_examples}_s{n_samples}_direct.json'
    else:
        score_path = f'../result/{dataset}/{model_name}/input_{target}_scores_e{n_examples}_s{n_samples}.json'

with open(score_path, 'r') as f:
    score_data = json.load(f)
    f.close()

def get_model_path(model_name):
    if model_name.startswith('Llama'):
        if '7b' in model_name:
            if 'chat' in model_name:
                path = llama2_7b_chat_path
            else:
                path = llama2_7b_path
        else:
            if 'chat' in model_name:
                path = llama2_13b_chat_path
            else:
                path = llama2_13b_path
    elif model_name.startswith('Mistral'):
        path = mistral_7b_path
    else:
        path = None
        pass 
    return path

def find_step_index(tokens, tokenizer):
    step_idx_dic = {}
    start = 0
    end = 0
    for i in range(len(tokens)):
        token = tokens[i]
        if token == '<0x0A>' and i == 0:
            start = 1
        if i == len(tokens)-1 \
            or token in ['?', '<0x0A>', '\".'] \
            or token == '.' and (not tokens[i-1].isdigit() or not tokens[i+1].isdigit()):
            end = i
            if end - start > 1 or end == len(tokens) - 1:
                if end == len(tokens) - 1 and start == 0:
                    end += 1
                step = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens[start:end]))
                step_idx_dic[start] = {'end':end, 'step':step}
                start = i + 1
        
    return step_idx_dic


tokenizer = AutoTokenizer.from_pretrained(get_model_path(proxy))

results = []
result_path = f'../result/{dataset}/{model_name}/{method}_{proxy}_{target}_paths_e{n_examples}_s{n_samples}.json'

for item in score_data:
  
    input_tokens = item['inp']
    output_tokens = item['out']
    scores = np.array(item['scores'])

    input_step_dic = find_step_index(input_tokens, tokenizer)
    output_step_dic = find_step_index(output_tokens, tokenizer)
    
    step_attr_ls = []
    for k1, v1 in output_step_dic.items():
        s1 = k1
        e1 = v1['end']
        score_ls = []
        for k2, v2 in input_step_dic.items():
            s2 = k2
            e2 = v2['end']
            if target == 'cot' and s2 >= s1-1:
                continue
            attr = scores[s2:e2, s1:e1].mean()
            msg = {'inp_idx':(s2, e2),'inp':v2['step'], 'attr':attr}
            score_ls.append(msg)
        score_ls = sorted(score_ls, key=lambda x: x['attr'], reverse=True)
        step_attr_ls.append({'ref_idx':(s1, e1), 'ref':v1['step'], 'inp_attr':score_ls})
    if 'cot_flag' not in item.keys():
        item['cot_flag'] = None
    result_msg = {'id':item['id'], 'cor_flag':item['cor_flag'], 'cot_flag':item['cot_flag'], 'path':step_attr_ls}
    
    results.append(result_msg)
    
with open(result_path, 'w') as f:
    json.dump(results, f, indent=4)
