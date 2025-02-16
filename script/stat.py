import argparse
import os
import numpy as np
import pandas as pd
from utils.config import llama3_1_8b_chat_path, mistral_7b_chat_path
from transformers import AutoTokenizer
from utils.metrics import draw_line, draw_scatter, draw_bar, draw_box
from collections import defaultdict
from rouge_score import rouge_scorer
from utils.load_data import load_json_data
from cal_metric import cal_acc, cal_rouge, cal_bert_score

import random 
random.seed(17)

def cal_data_bin_means(data, num_bins=10):
    # 获取每个区间的长度
    bin_size = len(data) // num_bins
    remainder = len(data) % num_bins
    bins = []
    start = 0
    
    # 分配数据到每个区间
    for i in range(num_bins):
        end = start + bin_size + (1 if i < remainder else 0)  # 平均分配余数
        bins.append(data[start:end])
        start = end
    
    mean = [np.mean(bin_data) for bin_data in bins]
    return mean


def get_unfaith_id(dataset, model, n_samples=200, cor_flg=False):
    cot_path = f'../result/{dataset}/{model}/cot_e3_200.json'
    cot_data = load_json_data(cot_path)[:n_samples]
    ids = []
    for item in cot_data:
        if not item['cot_flg']:
            if item['cor_flag'] or cor_flg:
                ids.append(item['id'])
    return ids 

def get_correct_id(dataset, model, n_samples=200):
    cot_path = f'../result/{dataset}/{model}/cot_e3_200.json'
    cot_data = load_json_data(cot_path)[:n_samples]
    ids = []
    for item in cot_data:
        if item['cot_flg'] and item['cor_flag']:
            ids.append(item['id'])
    return ids 

def hit(statement, ref_sentence):
    refs = ref_sentence.split('.')
    for ref in refs:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        score = scorer.score(ref, statement.strip('.'))['rougeL'][2]
        # p, r, f1 = score([ref], [statement.strip('.')], lang='en')
        if score > 0.9:

            return True
    return False

def cal_hit_scores(tokens, scores):
    stop_tokens = ['.\u010a', '.']
    start_idx = [i for i, v in enumerate(tokens) if v == ':\u010a'][0] + 1
    end_idx = [i for i, v in enumerate(tokens) if v == "#"][1]
    tokens = tokens[start_idx:end_idx]   
    scores = scores[start_idx:end_idx] 
    step_scores = {}
    start = 0
    for i in range(len(tokens)):
        token = tokens[i]
        if i == len(tokens)-1 or token in stop_tokens:
            end = i + 1
            step = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens[start:end]))

            step_scores[step.strip()] = np.mean(np.array(scores[start:end]))
            start = i + 1
    return step_scores


def recall_context(tokens, scores):
    stop_tokens = ['.\u010a', '.']
    if model_name.startswith('Llama'):
        start_idx = [i for i, v in enumerate(tokens) if v == ':\u010a'][0] + 1
    else:
        start_idx = [i for i, v in enumerate(tokens) if v == ':'][0] + 1
    end_idx = [i for i, v in enumerate(tokens) if v == "#"][1]
    tokens = tokens[start_idx:end_idx]   
    scores = scores[start_idx:end_idx] 
    step_scores = {}
    start = 0
    for i in range(len(tokens)):
        token = tokens[i]
        if i == len(tokens)-1 or token in stop_tokens:
            end = i + 1
            step = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens[start:end]))
            step_scores[step.strip()] = np.mean(np.array(scores[start:end]))
            start = i + 1
    step_scores = sorted(step_scores.items(), key=lambda item: item[1], reverse=True)
    return step_scores  

def split_difficulty(dataset):
    difficulty_dic = {}
    counts_dic = defaultdict(int)
    models = ['Llama3_1_8b_chat', 'Mistral_7b_chat', 'Gemma2_9b_chat', 'Qwen2_5_14b_chat']
    for model in models:
        sc_path = f'../result/{dataset}/{model}/direct10_e3_200.json'
        sc_result = load_json_data(sc_path)[:-1]
        for item in sc_result:
            id = item['id']
            counts_dic[id] += item['cor_flag'][:10].count(True)
    for id, count in counts_dic.items():
        difficulty = 5 - count // 8
        if difficulty == 0:
            difficulty = 1
        if difficulty in difficulty_dic.keys():
            difficulty_dic[difficulty].append(id)
        else:
            difficulty_dic[difficulty] = [id]
    return difficulty_dic   


def stat_dif_performance(data_dic):
    for dataset, data in data_dic.items():
            
        methods = []
        nums = []
        scores = []
        difficulty_dic = split_difficulty(dataset)
        for difficulty, index in difficulty_dic.items():
            for method, result in data.items():
                result = [item for item in result[:-1] if item['id'] in index]
                nums.append(difficulty)
                methods.append(method)
                score = [int(item['cor_flag']) for item in result]
                scores.append(score.count(1) / len(score))
        
        dir_path = f'../fig/{dataset}/{model_name}/'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        path = dir_path + f'diff_acc.pdf'
        data = {'difficulty':nums, 'accuracy':scores, 'method':methods}
        data = pd.DataFrame(data, columns=list(data.keys()))
        draw_bar(data, path)


def stat_qc_info(data_dic):
    names = []
    # x = []
    scores = []
    for name, data in data_dic.items():
        names += [name] * len(data)
        scores += [item['scores'] for item in data]
        # x += list(range(len(data)))
    dir = f'../fig/{dataset}/{model_name}/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    fig_path = dir + 'qc_fig.pdf'
    data = {'type':names, 'IG':scores}
    data = pd.DataFrame(data, columns=list(data.keys()))
    # print(data)
    draw_box(data, fig_path)      
    

def stat_ca_info(data_dic):
    num_bins = 20
    names = []
    x = []
    aaes = []
    for name, data in data_dic.items():
        for item in data:
            # print(item)
            score = item['scores'][4:-1]
            if not score:
                continue
            scores = cal_data_bin_means(score, num_bins=num_bins)
            x_values = [100 * i / num_bins for i in range(1, num_bins+1)]
            for i in range(len(x_values)):
                x.append(x_values[i])
                aaes.append(scores[i])
                names.append(name)


    data = {'step (%)':x, 'AAE':aaes, 'type':names}
    data = pd.DataFrame(data, columns=list(data.keys()))
    # print(data)
    fig_path = f'../fig/{dataset}/{model_name}/ca_fig.pdf'
    draw_line(data, fig_path, style=True) 


def stat_qa_info(data_dic, reason_dic):
    
    names = []
    x = []
    scores = []
    
    for name, data in data_dic.items():
        for item in data:
            score_dic = recall_context(item['input'], item['scores'])
            reason = reason_dic[item['id']]
            for num in range(1,11):
                num = min(num, len(score_dic)-1)
                x.append(num)
                names.append(name)
                if name == 'random':
                    recall_inputs = [tup[0] for tup in random.sample(score_dic, num)]
                else:
                    recall_inputs = [tup[0] for tup in score_dic[:num]]
                hits = [hit(input, reason) for input in recall_inputs]
                score = hits.count(True)
                scores.append(score)
                # cot_flags.append('golden')   
                # nums.append(num)
                # score = cal_overlap("", cot, attr_sent)
                # scores.append(score)
    fig_path = f'../fig/{dataset}/{model_name}/qa_fig.pdf'
    data = {'Top k':x, 'Hit':scores, 'category':names}
    data = pd.DataFrame(data, columns=list(data.keys()))
    # print(data)
    draw_line(data, fig_path, True)    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Llama3_1_8b_chat')
    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--n_examples', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='proofwriter')
    parser.add_argument('--method', type=str, default='qa')
    args = parser.parse_args()

    model_name = args.model
    n_samples = args.n_samples
    n_examples = args.n_examples
    dataset = args.dataset 
    method = args.method

    
    if model_name.startswith('Llama'):
        tokenizer = AutoTokenizer.from_pretrained(llama3_1_8b_chat_path.format(dir='publiccache'))
    else:
        tokenizer = AutoTokenizer.from_pretrained(mistral_7b_chat_path.format(dir='publiccache'))
    
    if method == 'diff':
        data_dic = defaultdict(dict)
        datasets = [ 'gsm8k', 'wino', ]
        for dataset in datasets:
            data_dic[dataset]['Direct'] = load_json_data(f'../result/{dataset}/{model_name}/direct_e3_200.json')
            data_dic[dataset]['CoT'] = load_json_data(f'../result/{dataset}/{model_name}/cot_e3_200.json')
        stat_dif_performance(data_dic)
    
    elif method == 'result':
        datasets = ['proofwriter', 'prontoqa']
        models = ['Gemma2_9b_chat']
        metrics = ['acc', 'rouge', 'fr']
        methods = ['wo_aae', 'wo_ig']
        # methods = ['bridge']
        for model in models:
            print(model)
            for dataset in datasets:
                print(dataset)
                for method in methods:
                    print(method)
                    if method in ['sc3', 'sr', 'ltm']:
                        path = f'../result/{dataset}/{model}/{method}_e{n_examples}_{n_samples}.json'     
                    elif method == 'cot':
                        path = f'../result/{dataset}/{model}/cot_e3_{n_samples}.json' 
                    elif method == 'bridge':
                        path = f'../result/{dataset}/{model}/bridge_{n_samples}.json'
                    else:
                        path = f'../result/{dataset}/{model}/bridge_{method}_200.json'  
                    if not os.path.exists(path):
                        print(path)
                        continue
                    data = load_json_data(path)[:-1]
                    for metric in metrics:
                        if metric == 'acc':
                            val = cal_acc(data)
                        elif metric == 'rouge':
                            val = cal_bert_score(data, faith=False)
                        else:
                            val = cal_bert_score(data, faith=True)
                        print(f'{metric}:{val}')
    elif method == 'faith':
        datasets = ['prontoqa', 'proofwriter', 'gsm8k', 'aqua', 'wino', 'siqa']
        for dataset in datasets:
            cc_count = 0
            cf_count = 0
            fc_count = 0
            ff_count = 0
            data = load_json_data(f'../result/{dataset}/{model_name}/cot_e3_200.json')[:50]
            for item in data:
                if item['cot_flg']:
                    if item['cor_flag']:
                        cc_count += 1
                    else:
                        cf_count += 1
                else:
                    if item['cor_flag']:
                        fc_count += 1
                    else:
                        ff_count += 1
            print(dataset)
            print(cc_count)
            print(cf_count)
            print(fc_count)
            print(ff_count)
         
    elif method == 'ig':
        data_dic = defaultdict(dict)
        datasets = [ 'proofwriter', 'prontoqa']
        for dataset in datasets:
            data_dic[dataset]['Direct'] = load_json_data(f'../result/{dataset}/{model_name}/direct_e3_200.json')
            data_dic[dataset]['CoT'] = load_json_data(f'../result/{dataset}/{model_name}/cot_e3_200.json')
        stat_dif_performance(data_dic)
    elif method == 'qc':
        data_dic = {}
        index = get_unfaith_id(dataset, model_name, n_samples)
        info_data = load_json_data(f'../result/{dataset}/{model_name}/ig_faith_200.json')
        unfaith_data = [item for item in info_data if item['id'] in index]
        # golden_data = load_json_data(f'../result/{dataset}/{model_name}/ig_faith_200_golden.json')
        index = get_correct_id(dataset, model_name, n_samples)
        correct_data = [item for item in info_data if item['id'] in index]
        data_dic = {'unfaithful':unfaith_data, 'faithful':correct_data, 'average':info_data}
        
        stat_qc_info(data_dic)
    elif method == 'ca':
        data_dic = {}
        index = get_unfaith_id(dataset, model_name, n_samples)
        info_data = load_json_data(f'../result/{dataset}/{model_name}/cans_info_e3_200.json')
        unfaith_data = [item for item in info_data if item['id'] in index]
        # golden_data = load_json_data(f'../result/{dataset}/{model_name}/cans_info_e3_200_golden.json')
        # correct_data = [item for item in golden_data if item['id'] in index]
        index = get_correct_id(dataset, model_name, n_samples)
        correct_data = [item for item in info_data if item['id'] in index]
        data_dic = {'unfaithful':unfaith_data, 'faithful':correct_data}
        
        stat_ca_info(data_dic)
    else:
        info_data = load_json_data(f'../result/{dataset}/{model_name}/qans_info_e3_200.json')
        # golden_data = load_json_data(f'../result/{dataset}/{model_name}/qans_info_e3_200_golden.json')
        index = get_unfaith_id(dataset, model_name)
        unfaith_data = [item for item in info_data if item['id'] in index]
        data_dic = {'unfaithful':unfaith_data, 'average':info_data, 'random':info_data}
        reason_dic = {item['id']:item['reason'] for item in load_json_data(f'../result/{dataset}/{model_name}/cot_e3_200.json')[:-1]}
        
        stat_qa_info(data_dic, reason_dic)  
    #     index = get_unfaith_id(dataset, model_name)
        
    # x = []
    # y = []
    # flgs = []
    # if dataset == 'prontoqa':
    #     key_token = ':'
    # else:
    #     key_token = '?'
    
    # for item in tqdm(info_data):
    #     if item['id'] not in results.keys():
    #         continue
    #     flg = results[item['id']]['cot_flg']

    #     golden_cot = results[item['id']]['reason']

    #     input = item['input']

    #     target_idx = [i for i, v in enumerate(input) if v == key_token][-1] + 2   
    #     target_scores = cal_hit_scores(input, item['scores'])
    #     target_score = []
    #     for k,v in target_scores.items():
    #         if hit(k, golden_cot) and not hit(k, results[item['id']]['response']):
    #             target_score.append(v)
    #     target_score = np.mean(np.array(target_score))
        
    #     if flg == False and results[item['id']]['cor_flag']:
    #         type = 'unfaithful'
    #     else:
    #         type = 'faithful'

    #     for k in range(1, 10):
    #         top_k_items = sorted(target_scores.items(), key=lambda item: item[1], reverse=True)[:k]
    #         top_k_keys = [item[0] for item in top_k_items] 
    #         cnt = 0
    #         for sent in top_k_keys:
    #             if hit(sent, golden_cot):
    #                 cnt += 1
                
    #     golden_scores = cal_hit_scores(input, golden_info_dic[item['id']]['scores'])
    #     # golden_score = []
        # for k,v in golden_scores.items():
        #     if k in golden_cot:
        #         print(v)
        #         golden_score.append(v)
        # golden_score = np.mean(np.array(golden_score))
        # print(target_score)
        # print(golden_score)
        # if len(target_score) == 1:
        #     continue

                # x.append(k)
                # y.append(cnt)
                # flgs.append(type) 
            # x.append(True)
            # y.append(golden_score)
            # if np.max(target_score) >= 0.2:
            #     print(item['id'])
            #     print(flg)
            #     print(target_score)
            #     print('------------')
     
