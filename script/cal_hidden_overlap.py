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
data_file = f'../result/{dataset}/{model_name}/cot_e3_s100.json'

with open(path_file, 'r') as f:
    path_data = json.load(f)
    f.close()
with open(data_file, 'r') as f:
    data = json.load(f)[:-2]
    f.close()  

cot_flags = []
nums = []
scores = []
for num in range(1, 11):
    path_dic = {}
    rand_path_dic = {}
    for item in path_data:
        if target == 'pcot':
            inp_attr = item['path'][-1]['inp_attr']
        else:
            inp_attr = item['path'][-1]['inp_attr']
        attr_sent = [x['inp'].strip('.').strip() for x in inp_attr[:num]]
        if len(inp_attr) < num:
            k = len(inp_attr)
        else:
            k = num
        rand_attrs = random.sample(item['path'][-1]['inp_attr'], k)
        rand_attr_sent = [x['inp'].strip('.').strip() for x in rand_attrs]
        path_dic[item['id']] = '.'.join(attr_sent)
        rand_path_dic[item['id']] = '.'.join(rand_attr_sent)
    for item in data:
        correct = 0
        if item['id'] not in path_dic.keys():
            continue
        attr_sent = path_dic[item['id']]
        gold_cot = item['reason']
        if isinstance(gold_cot, list):
            r = [cal_rouge(attr_sent, ref, avg=False)['r'] for ref in gold_cot]
            score = np.array(r).max()
        else:
            score = cal_rouge(attr_sent, gold_cot, avg=False)['r']
       
        # score = correct / len(gold_cot.split('.'))
        cot_flags.append(item['cot_flag'])
        nums.append(num)
        scores.append(score)
    items = random.sample(data, 20)
    for item in items:
        correct = 0
        if item['id'] not in rand_path_dic.keys():
            continue
        attr_sent = rand_path_dic[item['id']]
        gold_cot = item['reason']
        if isinstance(gold_cot, list):
            r = [cal_rouge(attr_sent, ref, avg=False)['r'] for ref in gold_cot]
            score = np.array(r).max()
        else:
            score = cal_rouge(attr_sent, gold_cot, avg=False)['r']
       
        cot_flags.append('random')
        nums.append(num)
        scores.append(score)

data = {'cot_flag':cot_flags, 'nums':nums, 'score':scores}
data = DataFrame(data)
names = ['nums', 'score', 'cot_flag']
path = f'../result/{dataset}/{model_name}/{target}_overlap_line_fig.pdf'
draw_line(data, path, names)