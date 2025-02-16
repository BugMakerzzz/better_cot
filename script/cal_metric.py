import json 
from utils.load_data import load_json_data, write_json_data
from utils.load_data import extract_answer
from rouge import Rouge
import random
import numpy as np
from bert_score import score
random.seed(17)
import sys
sys.setrecursionlimit(2000) 

def cal_acc(data):
    correct = 0
    cnt = 0
    for item in data:
        # if isinstance(item['cor_flag'], list):
        #     if item["cor_flag"].count(True) > len(item["cor_flag"]) // 2:
        #         correct += 1
        # else:
        # print(item)
        if item['cor_flag']:
            correct += 1
        cnt += 1
    return correct / cnt

def cal_bert_score(data, faith):
    candis = []
    refs = []
    for item in data:
        if not item['reason']:
            return 0
        if isinstance(item['response'], list):
            if isinstance(item['response'][0], dict):
                pred = item['pred']
                responses = [item['response'][i]['content'] for i in range(len(item['answer'])) if item['answer'][i] == pred]
                res = random.choice(responses)
                # responses = [item['response'][i] for i in range(len(item['answer'])) if item['answer'][i] == pred]
                # res = max(responses, key=lambda x: x["score"])['content']
            else:
                final_ans = max(item['answer'],key=item['answer'].count)
                # if final_ans != item['label']:
                #     continue
                res = [item['response'][i] for i in range(len(item['answer'])) if item['answer'][i] == final_ans][0]
        else:
            # if not item['cor_flag']:
            #     continue
            res = item['response']
        res = res.split('\n# Answer:')[0].split('# Reasoning:')[-1].strip()
        candis.append(res)
        refs.append(item['reason'])   
    _, _, f1 = score(candis, refs, lang="en", model_type="bert-base-uncased", device="cuda")
    if faith:
        f1 = [f1[i] if data[i]['cor_flag'] else 1 - f1[i] for i in range(len(f1))]
    f1_score = np.mean(np.array(f1)).item()

    return f1_score

def cal_rouge(data, faith=False):
    score = []
    for item in data:
        if not item['reason']:
            return 0
        if faith and not item['cor_flag']:
            continue
        if isinstance(item['response'], list):
            if isinstance(item['response'][0], dict):
                pred = item['pred']
                responses = [item['response'][i] for i in range(len(item['answer'])) if item['answer'][i] == pred]
                res = max(responses, key=lambda x: x["score"])['content']
            else:
                final_ans = max(item['answer'],key=item['answer'].count)
                # if final_ans != item['label']:
                #     continue
                res = [item['response'][i] for i in range(len(item['answer'])) if item['answer'][i] == final_ans][0]
        else:
            # if not item['cor_flag']:
            #     continue
            res = item['response']
        res = res.split('\n# Answer:')[0].split('# Reasoning:')[-1].strip()
        
        # print(res)
        # print(item)
        rouge = Rouge()
        scores = rouge.get_scores(item['reason'],res)
        # print(scores[0]["rouge-l"])
        # if faith and not item['cor_flag']:
        #     score.append(1 - scores[0]["rouge-l"]["f"])
        # else:
        score.append(scores[0]["rouge-l"]["f"])
    return np.mean(np.array(score))




# def cal_rouge(data):
#     score = []
#     for item in data:
#         if not item['reason']:
#             return 0
#         if isinstance(item['answer'], list):
#             final_ans = max(item['answer'],key=item['answer'].count)
#             # if final_ans != item['label']:
#             #     continue
#             res = [item['response'][i] for i in range(len(item['answer'])) if item['answer'][i] == final_ans][0]
#         else:
#             # if not item['cor_flag']:
#             #     continue
#             res = item['response']
#         res = res.split('\n# Answer:')[0].strip()

#         rouge = Rouge()
#         scores = rouge.get_scores(item['reason'],res)
#         score.append(scores[0]["rouge-2"]["p"])
#     return np.mean(np.array(score))



def review_acc(dataset, data):
    correct = 0
    cnt = 0
    for item in data:
        res = item['response']

        pred = extract_answer(dataset,res)
        if pred == item['label']:
            item["cor_flag"] = True
            correct += 1
        else:
            item["cor_flag"] = False
        item['answer'] = pred

        cnt += 1
    return correct / cnt 

# def review_acc(dataset, data):
#     correct = 0
#     cnt = 0
#     for item in data:
#         responses = item['response']
#         answers = []
#         flags = []
#         for res in responses:
#             pred = extract_answer(dataset,res)
#             answers.append(pred)
#             if pred == item['label']:
#                 flags.append(True)
#             else:
#                 flags.append(False)
#         item['answer'] = answers
#         item["cor_flag"] = flags
#         if item["cor_flag"].count(True) > len(item["cor_flag"]) // 2:
#             correct += 1
#         cnt += 1
#     return correct / cnt 

# model_ls = ['Llama3_1_8b_chat', 'Mistral_7b_chat']
# dataset_ls = ['proofwriter', 'prontoqa']
# method_ls = ['cot', 'sc10', 'bridge3']
# # metric_ls = ['acc', 'rouge', 'fr']
# metric_ls = ['acc', 'bert_score']

# for model in model_ls:
#     for dataset in dataset_ls:
#         for method in method_ls:
#             path = f'../result/{dataset}/{model}/{method}_e3_500.json'
#             data = load_json_data(path)[:-1]
#             metric = {}
#             if 'acc' in metric_ls:
#                 acc = cal_acc(data)
#                 metric['acc'] = acc
#             if 'rouge' in metric_ls:
#                 rouge = cal_rouge(data)
#                 metric['rouge'] = rouge
#             if 'bert_score' in metric_ls:
#                 rouge = cal_bert_score(data)
#                 metric['bert_score'] = rouge
#             print(metric)
#             data.append(metric)
#             write_json_data(path, data)