import os
import json
import random
import numpy as np
from metrics import *

random.seed(17)
# models = ['Llama2_13b', 'Mistral_7b']
models = ['Llama2_13b']
methods = ['our', 'wo_aae', 'wo_nli', 'wo_pcot']
# datasets = ['proofwriter_d1', 'proofwriter', 'prontoqa_d2', 'prontoqa']
datasets = ['proofwriter', 'prontoqa']
metrics = ['acc', 'rouge', 'fr']

for model in models:
    for method in methods:
        print('\n')
        avg_fr = []
        for dataset in datasets:
            if dataset.endswith('1') or dataset.endswith('2'):
                n_samples = 100
            else:
                n_samples = 500
            if method not in ['our', 'wo_aae', 'wo_nli', 'wo_pcot', 'add_cans']:
                path = f'../result/{dataset}/{model}/{method}_e3_s{n_samples}.json'
                name_dic = {'gen':'response', 'ref':'reason'}
            else:
                if method == 'our':
                    path = f'../result/{dataset}/{model}/filter_cot_e3_s{n_samples}_nli_pcot.json'
                elif method == 'wo_aae':
                    path = f'../result/{dataset}/{model}/filter_cot_e3_s{n_samples}_nli_pcot_random.json'
                elif method == 'wo_nli':
                    path = f'../result/{dataset}/{model}/filter_cot_e3_s{n_samples}_pcot.json'
                elif method == 'add_cans':
                    path = f'../result/{dataset}/{model}/filter_cot_e3_s{n_samples}_nli_cans_pcot.json'
                else:
                    path = f'../result/{dataset}/{model}/filter_cot_e3_s{n_samples}_nli.json'
                name_dic = {'gen':'f_response', 'ref':'reason'}
            if not os.path.exists(path):
                print(f"{path} no exist!")
                continue
            with open(path, 'r') as f:
                result = json.load(f)
                f.close()
            for item in result[:-1]:
                if method == 'sr':
                    item['response'] = item['response'].split('\n\n')[-1]
                elif method == 'cot_sc':
                    max_answer = max(item['answer'],key=item['answer'].count)
                    cand_responses = []
                    for i in range(len(item['response'])):
                        if item['answer'][i] == max_answer:
                            cand_responses.append(item['response'][i])
                    item['response'] = random.choice(cand_responses)
                elif method == 'ltm':
                    response = ""
                    responses = item['response'].split('\n')
                    for sent in responses:
                        if sent.startswith('A'):
                            response += sent.split(':')[-1].strip()
                    item['response'] = response
            acc = result[-1]['acc']
            rouge = get_rouge(result, name_dic)
            fr = get_fr(result, name_dic)
            avg_fr.append(fr['f'])
            print(f"Model:{model}\tMethod:{method}\tDataset:{dataset}\tAcc:{acc}\tRouge:{rouge['f']}\tfr:{fr['f']}") 
        avg_fr = np.mean(np.array(avg_fr))
        print(f"Model:{model}\tMethod:{method}\tAvg_fr:{avg_fr}")