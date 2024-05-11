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
args = parser.parse_args()
set_seed(17)

model_name = args.model
n_samples = args.n_samples
n_examples = args.n_examples
dataset = args.dataset 

path_file = f'../result/{dataset}/{model_name}/cot_{model_name}_pans_paths_e3_s100.json'
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
    score_dic = {0:0, 1:0, 2:0, 3:0}
    cnt_dic = {0:0, 1:0, 2:0, 3:0}
    for item in path_data:
        inp_attr = item['path'][-1]['inp_attr']
        attr_sent = [x['inp'].strip('.').strip() for x in inp_attr[:num]]
        if len(inp_attr) < num:
            k = len(inp_attr)
        else:
            k = num
        path_dic[item['id']] = attr_sent
    cnt = 0
    for item in data:
        if item['id'] not in path_dic.keys():
            continue
        cnt_dic[item['cot_flag']] += 1
        attr_sent = path_dic[item['id']]
        gold_cot = item['reason']
        cot = item['response']
        for sent in attr_sent:

            if sent in cot:
                continue
            if isinstance(gold_cot, list):
                for g_cot in gold_cot:
                    if sent in gold_cot:
                        score_dic[item['cot_flag']] += 1
                        break
                # r = [cal_rouge(attr_sent, ref, avg=False)['r'] for ref in gold_cot]
                # score = np.array(r).max()
            else:
                if sent in gold_cot:
                    score_dic[item['cot_flag']] += 1
                # score = cal_rouge(attr_sent, gold_cot, avg=False)['r']
       
        # score = correct / len(gold_cot.split('.'))
    for k, v in score_dic.items():
        cot_flags.append(k)
        nums.append(num)
        scores.append(v/cnt_dic[k])

data = {'cot_flag':cot_flags, 'nums':nums, 'score':scores}
data = DataFrame(data)
names = ['nums', 'score', 'cot_flag']
path = f'../result/{dataset}/{model_name}/sec_overlap_line_fig.pdf'
draw_line(data, path, names)