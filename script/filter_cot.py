import json
import numpy as np
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
from config import *
from metrics import get_rouge

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama2_13b')
parser.add_argument('--n_samples', type=int, default=100)
parser.add_argument('--n_examples', type=int, default=3)
parser.add_argument('--dataset', type=str, default='proofwriter_d1')
parser.add_argument('--proxy', type=str, default='Llama2_13b')
parser.add_argument('--method', type=str, default='attr_cot')
parser.add_argument('--nli_check', action='store_true')
parser.add_argument('--cans_check', action='store_true')
parser.add_argument('--cot_check', action='store_true')
parser.add_argument('--pcot_check', action='store_true')
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
cot_check = args.cot_check
pcot_check = args.pcot_check

result_file = f'../result/{dataset}/{model_name}/{method}_e{n_examples}_s{n_samples}.json'
with open(result_file, 'r') as f:
    data = json.load(f)[:-1]
    f.close()
    
if nli_check:
    check_model = AutoModelForSequenceClassification.from_pretrained(deberta_path).to('cuda')
    check_tokenizer = AutoTokenizer.from_pretrained(deberta_path)


def check_paths(target):
    scores = {}
    path_file = f'../result/{dataset}/{model_name}/{method}_{proxy}_{target}_paths_e{n_examples}_s{n_samples}.json'
    with open(path_file, 'r') as f:
        path_data = json.load(f)
        f.close()
    for item in path_data: 
        if target == 'cans':
            path = item['path'][-1]['inp_attr']
            score = np.array([x['attr'] for x in path]).mean()
        elif target == 'cot':
            score = []
            for tup in item['path'][1:]:
                inp_attr = tup['inp_attr']
                inp_attr = sorted(inp_attr, key=lambda x: x['inp_idx'][1], reverse=True)
                score.append(inp_attr[0]['attr'])
            score = np.array(score).mean()
        elif target == 'pcot':
            score = []
            for tup in item['path'][:-1]:
                inp_attr = tup['inp_attr'][0]['attr']
                score.append(inp_attr)
            score = np.array(score).min()
        id = item['id']
        if id in scores.keys():
            scores[id].append(score)
        else:
            scores[id] = [score]
    return scores
 
def check_nli(question, statement):
    question = question.strip() + '.'
    statement = statement.strip() + '.'
    input = check_tokenizer(question, statement, return_tensors="pt")
    output = check_model(input["input_ids"].to('cuda')) 
    prediction = torch.softmax(output["logits"][0], -1)
    label_names = ["contradiction", "neutral", "entailment"]
    if label_names[torch.argmax(prediction)] == 'contradiction' and torch.max(prediction) > 0.8:
        return 'B'
    elif label_names[torch.argmax(prediction)] == 'entailment' and torch.max(prediction) > 0.8:
        return 'A'
    else:
        return None
 
results = []
correct = 0
if cans_check:
    cans_scores = check_paths(target='cans')
if cot_check:
    cot_scores = check_paths(target='cot')
if pcot_check:
    pcot_scores = check_paths(target='pcot')

for i in tqdm(range(0, len(data), 3)):
    id = data[i]['id']
    question = data[i]['question']
    hints = [data[j]['hint'] for j in range(i, i+3)]
    answers = [data[j]['answer'] for j in range(i, i+3)]
    reason = data[i]['reason']
    label = data[i]['label']

    question_stem = question.split('?')[-1].split('.')[0].strip() + '.'
    final_answers = []
    scores = []
    responses = []
    if cans_check and id in cans_scores.keys() and len(cans_scores[id]) == 3:
        cans_score = [cans_scores[id][j] for j in range(3)]
    else:
        cans_score = [0, 0, 0]
    if cot_check and id in cot_scores.keys() and len(cot_scores[id]) == 3:
        cot_score = [cot_scores[id][j] for j in range(3)]
    else:
        cot_score = [0, 0, 0]
    if pcot_check and id in pcot_scores.keys() and len(pcot_scores[id]) == 3:
        pcot_score = [pcot_scores[id][j] for j in range(3)]
    else:
        pcot_score = [0, 0, 0]
    for j in range(3):
        response = data[i+j]['response'] 
        if nli_check:
            last_step = response.split('\n# Answer:')[0].split('.')[-2].strip() + '.'
            pred = check_nli(question=question_stem, statement=last_step)
        else:
            pred = answers[j]
        score = cans_score[j] + cot_score[j] + pcot_score[j]
        if pred:
            scores.append(score)
        else:
            scores.append(score - 100)
        responses.append(response)
        final_answers.append(pred)   
 
        
    answer_scores = {}
    for j in range(3):
        if not final_answers[j]:
            continue
        elif final_answers[j] in answer_scores.keys():
            answer_scores[final_answers[j]] += scores[j]
        else:
            answer_scores[final_answers[j]] = scores[j]
    if answer_scores:
        final_answer = max(answer_scores, key=lambda x: answer_scores[x])
        for j in range(3):
            if final_answers[j] != final_answer:
                scores[j] -= 100
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
           'cot_score':cot_score,
           'pcot_score':pcot_score,
           'score':scores,
           'f_answers':final_answers,
           'f_response':final_response,
           'f_answer':final_answer,
           'cor_flag':cor_flag}
    results.append(msg)
results.append({'acc': correct / n_samples})
results.append(get_rouge(results, {'gen':'f_response', 'ref':'reason'}))

result_path = f'../result/{dataset}/{model_name}/filter_cot_e{n_examples}_s{n_samples}'
if nli_check:
    result_path += '_nli'
if cans_check:
    result_path += '_cans'
if cot_check:
    result_path += '_cot'
if pcot_check:
    result_path += '_pcot'
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
