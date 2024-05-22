import json
import numpy as np
import argparse
import torch
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
from config import *
from metrics import get_bleu, get_rouge

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama2_13b')
parser.add_argument('--n_samples', type=int, default=100)
parser.add_argument('--n_examples', type=int, default=3)
parser.add_argument('--dataset', type=str, default='proofwriter_d1')
parser.add_argument('--proxy', type=str, default='Llama2_13b')
parser.add_argument('--method', type=str, default='attr_cot')
parser.add_argument('--nli_check', action='store_true')
parser.add_argument('--cans_check', action='store_true')
parser.add_argument('--pcot_check', action='store_true')
parser.add_argument('--cot_check', action='store_true')
# parser.add_argument('--weight', type=float, default=0.5)
args = parser.parse_args()
set_seed(17)

model_name = args.model
n_samples = args.n_samples
n_examples = args.n_examples
dataset = args.dataset 
proxy = args.proxy
method = args.method
nli_check = args.nli_check
cans_check = args.cans_check
pcot_check = args.pcot_check
cot_check = args.cot_check

result_file = f'../result/{dataset}/{model_name}/{method}_e{n_examples}_s{n_samples}.json'
with open(result_file, 'r') as f:
    data = json.load(f)[:-1]
    f.close()
    
if nli_check:
    check_model = AutoModelForSequenceClassification.from_pretrained(deberta_path).to('cuda')
    check_tokenizer = AutoTokenizer.from_pretrained(deberta_path)


def check_pcot_paths(dataset):
    def extract_obj(sent):
        pattern = r'\b(\w+)\s+(is|are)\b'
        matches = re.findall(pattern, sent)
        return matches[0][0] 

    pcot_path_file = f'../result/{dataset}/{model_name}/{method}_{proxy}_pcot_diff_paths_e{n_examples}_s{n_samples}.json'
    with open(pcot_path_file, 'r') as f:
        pcot_path_data = json.load(f)
        f.close()

    pcot_scores_dic = {}
    for i in range(len(pcot_path_data)):
        pcot_item = pcot_path_data[i]  
        name = extract_obj(pcot_item['path'][0]['ref'])
        pcot_scores = []
        for j in range(len(pcot_item['path'])-1):
            pcot_path = pcot_item['path'][j]
            if j != 0 and name in pcot_path['ref']:
                pcot_score = 0
            else:
                s = pcot_path['diff_inp_attr']
                if s: 
                    pcot_score = np.mean(np.array([x[1] for x in s[1:2]]))
                else:
                    pcot_score = 0

            pcot_scores.append(pcot_score)
        if pcot_scores:
            pcot_score = np.mean(np.array(pcot_scores))
        else:
            pcot_score = 0    
        
        id = pcot_item['id']
        if id in pcot_scores_dic.keys():
            pcot_scores_dic[id].append(pcot_score)
        else:
            pcot_scores_dic[id] = [pcot_score]
    return pcot_scores_dic

def check_target_paths(target):
    scores = {}
    if target == 'pcot':
        path_file = f'../result/{dataset}/{model_name}/{method}_{proxy}_{target}_diff_paths_e{n_examples}_s{n_samples}.json'
    else:
        path_file = f'../result/{dataset}/{model_name}/{method}_{proxy}_{target}_paths_e{n_examples}_s{n_samples}.json'
    with open(path_file, 'r') as f:
        path_data = json.load(f)
        f.close()
    for item in path_data: 
        if target == 'cans':
            path = item['path'][-1]['inp_attr']
            score = np.array([x['attr'] for x in path]).mean()
        elif target == 'cot':
            path = item['path'][-1]['inp_attr']
            if path:
                score = np.array([x['attr'] for x in path]).min()
            else:
                score = 0
        elif target == 'pcot':
            score = []
            
            for tup in item['path'][:-1]:
                if tup['diff_inp_attr']:
                    inp_attr = tup['diff_inp_attr'][0][1]
                else:
                    inp_attr = 0
                score.append(inp_attr)
            if score:
                score = np.array(score).mean()
            else:
                score = 0
        id = item['id']
        if id in scores.keys():
            scores[id].append(score)
        else:
            scores[id] = [score]
    return scores
 
def check_nli(question, statement, dataset, ):
    input = check_tokenizer(question, statement, return_tensors="pt")
    output = check_model(input["input_ids"].to('cuda')) 
    prediction = torch.softmax(output["logits"][0], -1)
    label_names = ["contradiction", "neutral", "entailment"]
    if label_names[torch.argmax(prediction)] == 'contradiction' and torch.max(prediction) > 0.95:
        if dataset.startswith('proofwriter'):
            return 'B'    
        else:
            return 'False'
    elif label_names[torch.argmax(prediction)] == 'entailment' and torch.max(prediction) > 0.95:
        if dataset.startswith('proofwriter'):
            return 'A'    
        else:
            return 'True'
    else:
        return None
 
results = []
correct = 0
if cans_check:
    cans_scores = check_target_paths(target='cans')
if pcot_check:
    if dataset.startswith('pronto'):
        pcot_scores = check_pcot_paths(dataset)
    else:
        pcot_scores = check_target_paths(target='pcot')
    # 
if cot_check:
    cot_scores = check_target_paths(target='cot')

    
data_dic = {}
for item in data:
    id = item['id'] 
    # if item['label'] == 'C' or item['answer'] == 'C':
    #     continue
    if id in data_dic.keys():
        data_dic[id].append(item)
    else:
        data_dic[id] = [item]

for id, items in tqdm(data_dic.items()):
    item_num = len(items)
    question = items[0]['question']
    if 'reason' not in items[0].keys():
        reason = None
    else:
        reason = items[0]['reason']
    answers = [items[i]['answer'] for i in range(item_num)]
    label = items[0]['label']
    hints = [(item['hint'] if 'hint' in item.keys() else None) for item in items]
    responses = [item['response'] for item in items]
 
    if cans_check and id in cans_scores.keys():
        cans_score = cans_scores[id] 
        for i in range(item_num):
            if answers[i] not in ['True', 'False', 'A', 'B', 'C']:
                cans_score.insert(i, 0)
    else:
        cans_score = item_num * [0]       
    if pcot_check and id in pcot_scores.keys():
        pcot_score = pcot_scores[id] 
        for i in range(item_num):
            if answers[i] not in ['True', 'False', 'A', 'B', 'C']:
                pcot_score.insert(i, 0)
    else:
        pcot_score = item_num * [0]
    if cot_check and id in cot_scores.keys():
        cot_score = cot_scores[id] 
        for i in range(item_num):
            if answers[i] not in ['True', 'False', 'A', 'B', 'C']:
                cot_score.insert(i, 0)
    else:
        cot_score = item_num * [0]
        
    final_answers = []
    scores = []    
    for i in range(item_num):
        if nli_check and answers[i] != 'C':
            response = responses[i] 
            if dataset.startswith('proofwriter'):
                question_stem = question.split('?')[-1].split('.')[0].strip()
                last_step = response.split('\n# Answer:')[0].split('.')[-2].strip()
                question_stem = question_stem.strip() 
                last_step = last_step.lstrip('So').strip()
            else:
                question_stem = question.split(':')[-1].strip().rstrip('.')
                last_step = response.split('\n# Answer:')[0].split('.')[-2].strip()
            pred = check_nli(question=question_stem, statement=last_step, dataset=dataset)
        else:
            pred = answers[i]
        score = cans_score[i] + pcot_score[i] + cot_score[i]
        if pred:
            scores.append(score)
        else:
            scores.append(score - 100)
        final_answers.append(pred)   
 
        
    answer_scores = {}
    for i in range(item_num):
        if final_answers[i] in answer_scores.keys():
            answer_scores[final_answers[i]] += scores[i]
        else:
            answer_scores[final_answers[i]] = scores[i]
    if answer_scores:
        final_answer = max(answer_scores, key=lambda x: answer_scores[x])
        for i in range(item_num):
            if final_answers[i] != final_answer:
                scores[i] -= 100
        final_response = responses[scores.index(max(scores))]
    else:
        final_answer = None
        final_response = None
    if final_answer == label:
        cor_flag = True
        correct += 1
    else:
        cor_flag = False
    msg = {'id':id, 
           'question':question, 
           'hint':hints, 
           'response':responses, 
           'answer':answers, 
           'reason':reason, 
           'label':label, 
           'cans_score':cans_score,
           'pcot_score':pcot_score,
           'cot_score':cot_score,
           'score':scores,
           'f_answers':final_answers,
           'f_response':final_response,
           'f_answer':final_answer,
           'cor_flag':cor_flag}
    results.append(msg)
results.append({'acc': correct / n_samples})
# print(results)
# bleu = get_rouge(results, {'gen':'f_response', 'ref':'reason'})
# print(bleu)
# results.append(bleu)

result_path = f'../result/{dataset}/{model_name}/filter_cot_e{n_examples}_s{n_samples}'
if nli_check:
    result_path += '_nli'
if cans_check:
    result_path += f'_cans'
if pcot_check:
    result_path += f'_pcot'
if cot_check:
    result_path += f'_cot'
if method == 'attr_random_cot':
    result_path += f'_random'
result_path +='.json'
   
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
