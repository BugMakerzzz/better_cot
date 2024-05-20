import argparse
import json
from transformers import set_seed
import numpy as np
from pandas import DataFrame
from metrics import draw_box

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama2_13b')
parser.add_argument('--n_samples', type=int, default=100)
parser.add_argument('--n_examples', type=int, default=3)
parser.add_argument('--dataset', type=str, default='proofwriter_d1')
parser.add_argument('--target', type=str, default='pcot')
args = parser.parse_args()
set_seed(17)

model_name = args.model
n_samples = args.n_samples
n_examples = args.n_examples
dataset = args.dataset 
target = args.target

path_file = f'../result/{dataset}/{model_name}/cot_{model_name}_{target}_paths_e3_s100.json'
data_file = f'../result/{dataset}/{model_name}/cot_e3_s100.json'

with open(path_file, 'r') as f:
    path_data = json.load(f)
    f.close()
with open(data_file, 'r') as f:
    data = json.load(f)[:-1]
    f.close()  

cot_dic = {item['id']:item['response'] for item in data}
gold_cot_dic = {item['id']:item['reason'] for item in data}

types = []
scores = []
for item in path_data:
    id = item['id']
    if not item['cor_flag'] or item['cot_flag'] in [0, 1]:
        continue
    path = item['path']
    cot = cot_dic[id]
    gold_cot = gold_cot_dic[id]
    if isinstance(gold_cot, list):
        gold_cot = "".join(gold_cot)
    avg_score = []
    miss_score = []
    hit_score = []
    for tup in path:
        for attr in tup['inp_attr']:
            avg_score.append(attr['attr'])
            if attr['inp'] in cot:
                hit_score.append(attr['attr'])
            else:
                if attr['inp'] in gold_cot:
                    miss_score.append(attr['attr'])
    
    
    miss_score = np.mean(np.array(miss_score))
    types.append('miss')
    scores.append(miss_score)
    hit_score = np.mean(np.array(hit_score))            
    types.append('hit')
    scores.append(hit_score)
    avg_score = np.mean(np.array(avg_score))
    types.append('avg')
    scores.append(avg_score)
    
data = {'type':types, 'AAE':scores}
data = DataFrame(data)
names = ['type','AAE']
path = f'../result/{dataset}/{model_name}/{target}_miss_box_fig.pdf'
draw_box(data, path, names)