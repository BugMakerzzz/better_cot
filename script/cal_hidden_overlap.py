import argparse
import os
import json
from model import ModelWrapper
from load_data import DataLoader, extract_answer
from transformers import set_seed
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama2_13b')
parser.add_argument('--n_samples', type=int, default=100)
parser.add_argument('--n_examples', type=int, default=3)
parser.add_argument('--dataset', type=str, default='proofwriter_d1')
parser.add_argument('--num', type=int, default=2)
args = parser.parse_args()
set_seed(17)

model_name = args.model
n_samples = args.n_samples
n_examples = args.n_examples
dataset = args.dataset 
num = args.num

path_file = f'../result/{dataset}/{model_name}/input_pans_paths_e3_s1001.json'
data_file = f'../result/{dataset}/{model_name}/cot_e3_s100.json'

with open(path_file, 'r') as f:
    path_data = json.load(f)
    f.close()
path_dic = {}
for item in path_data:
    # if num > len(item['path'][-1]['inp_attr']):
    #     attrs = item['path'][-1]['inp_attr']
    # else:
    attrs = item['path'][-1]['inp_attr'][:num]
    attr_sents = [x['inp'] for x in attrs]
    path_dic[item['id']] = attr_sents

with open(data_file, 'r') as f:
    data = json.load(f)[:-1]
    f.close()  
     
cnt = 0
correct = 0
for item in tqdm(data):
    if item['id'] not in path_dic.keys():
        n_samples -= 1
        continue
    attr_sents = path_dic[item['id']]
    gold_cot = item['reason']
    # if item['cot_flag'] in [0]:
    cnt += 1
    for sent in attr_sents:
        if sent in gold_cot:
            correct += 1
            break     


print(correct / cnt)
