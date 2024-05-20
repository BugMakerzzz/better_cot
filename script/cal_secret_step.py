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
    data = json.load(f)[:-1]
    f.close()  

def cal_overlap(cot, ref, sents):
    score = 0
    for sent in sents:
        if sent in cot:
            continue
        if isinstance(ref, list):
            for g_cot in ref:
                if sent in ref:
                    score += 1     
        else:
            if sent in gold_cot:
                score += 1
    if isinstance(ref, list):
        score /= len(ref)
    return score

cot_flags = []
nums = []
scores = []
for num in range(1, 11):
    path_dic = {}
    rand_path_dic = {}
    for item in path_data:
        if target == 'pcot':
            inp_attr = item['path'][0]['inp_attr']
        else:
            inp_attr = item['path'][-1]['inp_attr']
        attr_sent = [x['inp'].strip('.').strip() for x in inp_attr[:num]]
        if len(inp_attr) < num:
            k = len(inp_attr)
        else:
            k = num
        rand_sent = random.sample([x['inp'].strip('.').strip() for x in inp_attr], k)
        path_dic[item['id']] = attr_sent
        rand_path_dic[item['id']] = rand_sent
        
    for item in data:
        if item['id'] not in path_dic.keys():
            continue
        attr_sent = path_dic[item['id']]
        gold_cot = item['reason']
        cot = item['response']
        if item['cot_flag'] in [0,1]:
            continue
        score = cal_overlap(cot, gold_cot, attr_sent)
        cot_flags.append('average')
        nums.append(num)
        scores.append(score)
        
        if item['cor_flag']:
            cot_flags.append('unfaithful')   
            nums.append(num)
            scores.append(score)
        # cot_flags.append('golden')   
        # nums.append(num)
        # score = cal_overlap("", cot, attr_sent)
        # scores.append(score)
    # for item in data:
    #     if item['id'] not in rand_path_dic.keys():
    #         continue
    #     rand_sent = rand_path_dic[item['id']]
    #     gold_cot = item['reason']
    #     cot = item['response']
    #     rand_score = cal_overlap(cot, gold_cot, rand_sent)
    #     cot_flags.append('random')
    #     nums.append(num)
    #     scores.append(rand_score)
        # score = correct / len(gold_cot.split('.'))

data = {'cot_flag':cot_flags, 'top k':nums, 'hit count':scores}
data = DataFrame(data)
names = ['top k', 'hit count', 'cot_flag']
path = f'../result/{dataset}/{model_name}/{target}_overlap_line_fig.pdf'
draw_line(data, path, names)