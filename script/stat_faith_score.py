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
parser.add_argument('--dataset', type=str, default='proofwriter')
parser.add_argument('--target', type=str, default='cot')
parser.add_argument('--score', type=str, default='step')
parser.add_argument('--num', type=str, default=10)
parser.add_argument('--proxy', type=str, default='Llama2_13b')
parser.add_argument('--method', type=str, default='cot')
args = parser.parse_args()

model_name = args.model
n_samples = args.n_samples
n_examples = args.n_examples
dataset = args.dataset 
target = args.target
score = args.score
num = args.num
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

def find_step_index(tokens):
    step_idx_dic = {}
    start = 0
    end = 0
    for i in range(len(tokens)):
        token = tokens[i]
        if token == '<0x0A>' and i == 0:
            continue  
        if i == len(tokens)-1 \
            or token in ['?', '<0x0A>'] \
            or token == '.' and (not tokens[i-1].isdigit() or not tokens[i+1].isdigit()):
            end = i
            if end - start > 1 or end == len(tokens) - 1:
                if end == len(tokens) - 1 and start == 0:
                    end += 1
                step_idx_dic[start] = end
                start = i + 1
        
    return step_idx_dic

if __name__ == '__main__':
    if score == 'step':
        results = []
        result_path = f'../result/{dataset}/{model_name}/input_{target}_step_score_e{n_examples}_s{n_samples}.json'
        bad_tokens = ['<0x0A>', '.', '?']
        for item in score_data[:n_samples]:
            input_tokens = item['inp']
            output_tokens = item['out']
            scores = np.array(item['scores'])
            attrs = []
            if target in ['cans', 'qans']:
                temp_tokens = input_tokens
                input_tokens = output_tokens
                output_tokens = temp_tokens
                scores = np.array(item['scores']).T
            if len(output_tokens) < 20:
                continue
            if dataset in ['gsm8k', 'aqua'] and output_tokens[0] != '<0x0A>':
                continue
            start_idx = 0 
            for i in range(num):
                ref_idx = list(range(start_idx, len(output_tokens) * (i+1) // num))
                start_idx = len(output_tokens) * (i+1) // num
                input_idx = list(range(len(input_tokens)))
                if target == 'cot':    
                    for idx in input_idx[::-1]:
                        if idx >= ref_idx[0]:
                            input_idx.pop(idx)
                for idx in input_idx[::-1]:
                    if input_tokens[idx] in bad_tokens:
                        input_idx.pop(input_idx.index(idx))
                for idx in ref_idx[::-1]:
                    if output_tokens[idx] in bad_tokens:
                        ref_idx.pop(ref_idx.index(idx))
                if not input_idx or not ref_idx:
                    continue
                temp_score = scores[input_idx,:]
                temp_score = temp_score[:,ref_idx]

                # print(ref_idx)
                attr = temp_score.mean()
                attrs.append(attr)
            if 'cot_flag' not in item.keys():
                item['cot_flag'] = None
            result_msg = {'id':item['id'], 'cor_flag':item['cor_flag'], 'cot_flag':item['cot_flag'], 'attr':attrs}
            results.append(result_msg)
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=4)
    else:
        sum_attr = 0
        cnt = 0
        for item in score_data[:n_samples]:
            input_tokens = item['inp']
            output_tokens = item['out']
            scores = np.array(item['scores'])
            
            input_step_dic = find_step_index(input_tokens)
            output_step_dic = find_step_index(output_tokens)
            
            step_attr_ls = []
            for k1, v1 in output_step_dic.items():
                for k2, v2 in input_step_dic.items():
                    if target == 'cot' and k2 >= k1-1:
                        continue
                    attr = scores[k2:v2, k1:v1].mean()
                    step_attr_ls.append(attr)
            
            if not step_attr_ls:
                continue
            attr = np.mean(np.array(step_attr_ls))
            sum_attr += attr
            cnt += 1
         
        print(f"{dataset}\t{target}\t{sum_attr / cnt}")