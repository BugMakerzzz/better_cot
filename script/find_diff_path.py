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
parser.add_argument('--target', type=str, default='pcot')
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


path_file = f'../result/{dataset}/{model_name}/{method}_{proxy}_{target}_paths_e{n_examples}_s{n_samples}.json'

with open(path_file, 'r') as f:
    data = json.load(f)

results = []
for item in data:
    path = item['path']
    step_attr_ls = []
    for idx in range(len(path)):
        tup = path[idx]
        ref = tup['ref']
        last_ref = None
        diff_inp_dic = {x['inp']:x['attr'] for x in tup['inp_attr']} 
        if idx >= 1:
            last_tup = path[idx-1]
            last_ref = last_tup['ref']  
            last_inp_dic = {x['inp']:x['attr'] for x in last_tup['inp_attr']}
            for k in diff_inp_dic.keys():
                if k in last_inp_dic.keys():
                    diff_inp_dic[k] = diff_inp_dic[k] - last_inp_dic[k]
        diff_inp_attr = sorted(diff_inp_dic.items(), key=lambda x:x[1], reverse=True)
        step_attr_ls.append({'last_ref':last_ref, 'ref':ref, 'diff_inp_attr':diff_inp_attr})
    result_msg = {'id':item['id'], 'cor_flag':item['cor_flag'], 'cot_flag':item['cot_flag'], 'path':step_attr_ls}
    results.append(result_msg)
    

result_path = f'../result/{dataset}/{model_name}/{method}_{proxy}_{target}_diff_paths_e{n_examples}_s{n_samples}.json'
with open(result_path, 'w') as f:
    json.dump(results, f, indent=4)

# data_path = f'../result/{dataset}/{model_name}/input_{target}_paths_e{n_examples}_s{n_samples}.json'
# stat_dic = {}
# with open(data_path, 'r') as f:
#     data = json.load(f)
# for item in data:
#     cot_flag = item['cot_flag']
#     path = item['path']
#     flg = 1
#     for tup in path[:-1]:
#         ref = tup['ref'].strip('.').strip()

#         inps = [x['inp'].strip('.').strip() for x in tup['inp_attr'][:2]]
#         # for 
#         # for i in range(len(tup['inp']))
#         # diff_inp = tup['inp_attr'][0]['inp'].strip('.').strip()
#         if ref not in inps:
#             flg = 0  
#     if cot_flag == 3 and flg == 1:
#         print(item['id'])
#     if cot_flag == 0 and flg == 0:
#         print(item['id'])  
#     if cot_flag in stat_dic.keys():
#         stat_dic[cot_flag].append(flg)
#     else:
#         stat_dic[cot_flag] = [flg]

# for k,v in stat_dic.items():
#     stat_dic[k] = np.mean(np.array(v))

# print(stat_dic)
