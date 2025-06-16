import argparse
import os
import json
import numpy as np
from utils.model import ModelWrapper
from utils.load_data import DataLoader, extract_answer, load_json_data
from transformers import set_seed
from tqdm import tqdm
from utils.math_eval import is_equiv
import random
from bridge_reason import bridge_reason, cal_ig_score
import time

def sc_reason(model, inputs, nums=10, cache=None):
    if cache:
        response = cache['response']
        answer = cache['answer']
        corrects = cache['corrects'] if 'corrects' in cache.keys() else None
        tups = [{'question':inputs.copy(), 'response':res} for res in response]
        scores = [cal_ig_score(model, tup) for tup in tups]
        response = [{'content': res, 'score':score} for res, score in zip(response, scores)]
        coef = {}
        for i in range(len(scores)):
            if not answer[i]:
                continue
            if answer[i] not in coef.keys():
                coef[answer[i]] = scores[i]
            else:
                coef[answer[i]] += scores[i]
        if not coef:
            pred = None 
        else:
            pred = max(coef, key=lambda x: coef[x])
    else:
        response = model.generate(inputs, sample_cnt=nums)
        answer = []
        corrects = []
        for res in response:
            ans = extract_answer(dataset, res)
            answer.append(ans)
            if ans and ans.lower() == item['answer'].lower():
                flag = True
            else:
                flag = False
            corrects.append(flag)
        pred = max(answer, key=answer.count)
    if not pred:
        cor_flag = False
    else:
        if pred.lower() == item['answer'].lower():
            cor_flag = True
        else:
            cor_flag = False         
    return response, answer, cor_flag, pred, corrects

def format_question(question, split_str):
    sessions = question.split('####')

    if model.is_mistral or model.is_gemma or model.is_o1:
        inputs = []
    else:
        inputs = [{"role": "system", "content": sessions[0]}]
    for session in sessions[1:]:
        user_content, assistant_content = session.split(split_str)
        assistant_content = split_str + assistant_content
        inputs += [{"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content}]
    inputs = inputs[:-1]
    if model.is_mistral or model.is_gemma or model.is_o1:
        inputs[0]['content'] = sessions[0] + '\n' + inputs[0]['content']
    return inputs 


def sr_reason(model, item, old_cot):
    question = item['question'].rstrip() + f"\n{old_cot}\n# Answer:"
    inputs = format_question(question, split_str='# Answer:')
    response = model.generate(inputs)
    item['reasoning'] = old_cot
    item['feedback'] = response.split('# Answer:')[-1].strip()
    question = dataloader.reformat_question(item, method='sr_refine',n_examples=3)
    inputs = format_question(question, split_str='# Reasoning:')
    response = model.generate(inputs)
    answer = extract_answer(dataset, response)
    cor_flag = is_equiv(answer, item['answer'])
    return response, answer, cor_flag


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Llama3_1_8b_chat')
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--n_examples', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='proofwriter')
    parser.add_argument('--method', type=str, default='cot')
    parser.add_argument('--lora',  action='store_true')
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--sc_num', type=int, default=None)
    parser.add_argument('--remote', action='store_true')
    parser.add_argument('--ig', action='store_true')
    parser.add_argument('--weighted', action='store_true')
    args = parser.parse_args()
    set_seed(17)
    random.seed(17)
    
    model_name = args.model
    n_samples = args.n_samples
    n_examples = args.n_examples
    dataset = args.dataset 
    method = args.method
    lora = args.lora
    topk = args.topk
    sc_num = args.sc_num
    remote = args.remote
    ig = args.ig
    weighted = args.weighted

    dataloader = DataLoader(dataset=dataset, n_samples=n_samples)
    if method in ['sc', 'bridge']:
        data = dataloader.load_data(method='cot', n_examples=n_examples)
    else:
        data = dataloader.load_data(method=method, n_examples=n_examples)
    lora_path = f'/mnt/userdata/ljc/code/faithful_cot/LLaMA-Factory/saves/{model_name}/lora/dpo'

    model = ModelWrapper(model_name, remote)
    if lora:    
        model.merge_lora(lora_path)
    result = []
    correct = 0
    if method in ['cot', 'bridge', 'sc', 'reward_sc', 'sr', 'ltm']:
        split_str = '# Reasoning:'
    else:
        split_str = '# Answer'

    if method == 'sr':
        cot_path = f'../result/{dataset}/{model_name}/cot_e3_{n_samples}.json'
        # with open(cot_path, 'r') as f:
        cot_dic = {item['id']:item['response'] for item in load_json_data(cot_path)[:-1]}
    # elif method == 'bridge':
    #     bridge_path =  f'../result/{dataset}/{model_name}/bridge{topk}_sc{sc_num}_w0_e{n_examples}_{n_samples}.json'
    #     if os.path.exists(bridge_path):
    #         bridge_dic = {item['id']:item for item in load_json_data(bridge_path)[:-1]}
    #     else:
    #         bridge_dic = None 
    elif method == 'sc' and ig:
        sc_path =  f'../result/{dataset}/{model_name}/{method}{sc_num}_e{n_examples}_{n_samples}.json'
        if os.path.exists(sc_path):
            sc_dic = {item['id']:item for item in load_json_data(sc_path)[:-1]}
        else:
            sc_dic = None 
    cnt = 0
    start_time = time.time()
    for item in tqdm(data):
        inputs = format_question(item['question'], split_str=split_str)
        # print(inputs)
        pred = None 
        corrects = None
        hints = None 
        if method == 'bridge':
            # if item['id'] in info_dic.keys():
            #     item['input'] = info_dic[item['id']]['input']
            #     item['scores'] = info_dic[item['id']]['scores']
            # else:
            #     item['input'] = None 
            #     item['scores'] = None 
            item['question'] = item['raw_question']
            if ig:
                # cache_item = bridge_dic[item['id']]
            # if not os.path.exists(raw_result_path):
                # 
                response, answer, corrects, cor_flag, hints, pred = bridge_reason(model, inputs, item, dataset, topk=topk, sc=sc_num, random_sample=True, weighted=weighted, cache=False)
            else:
                # if ig:
                response, answer, corrects, cor_flag, hints, pred = bridge_reason(model, inputs, item, dataset, topk=topk, sc=sc_num, random_sample=False, weighted=weighted, cache=False)
            correct += int(cor_flag)
        elif sc_num:
            if ig:
                response, answer, cor_flag, pred, corrects = sc_reason(model, inputs, nums=sc_num, cache=sc_dic[item['id']])
            else:
                 response, answer, cor_flag, pred, corrects = sc_reason(model, inputs, nums=sc_num)
            correct += int(cor_flag) 
          
        elif method == 'sr':
            cot = cot_dic[item['id']].split('# Answer:')[0].strip().split('# Reasoning:')[-1].strip()
            response, answer, cor_flag = sr_reason(model, item, cot)
            correct += int(cor_flag)
        else:
            response = model.generate(inputs)
            # print(response)
            if method == 'explain':
                answer = extract_answer(dataset, response, method='explain')
            else:
                answer = extract_answer(dataset, response)
            if dataset == 'math':
                if is_equiv(answer, item['answer']):
                    cor_flag = True 
                    correct += 1
                else:
                    cor_flag = False 
            else:
                if answer and answer.lower() == item['answer'].lower():
                    cor_flag = True
                    correct += 1
                else:
                    cor_flag = False
        if 'reason' not in item.keys():
            item['reason'] = None 
        msg = {'id':item['id'], 
               'question':item['raw_question'], 
               'response':response, 
               'answer':answer, 
               'pred': pred,
               'label':item['answer'], 
               'cor_flag':cor_flag, 
               'reason':item['reason'],
               'corrects':corrects,
               'hints':hints
               }
        result.append(msg)
        cnt += 1
    result.append({'acc': correct / cnt})
    end_time = time.time()  
    result_dir = f'../result/{dataset}/{model_name}/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if lora:
        result_path = os.path.join(result_dir, f'{method}_lora_e{n_examples}_{n_samples}.json')
    elif method == 'sc':
        if ig:
            result_path = os.path.join(result_dir, f'bridge_wo_aae_{n_samples}.json')
        else:
            result_path = os.path.join(result_dir, f'sc{sc_num}_e{n_examples}_{n_samples}.json')
    elif method == 'bridge':
        if ig:
            result_path = os.path.join(result_dir, f'bridge{topk}_sc{sc_num}_wo_aae_{n_samples}.json')
        else:
            result_path = os.path.join(result_dir, f'bridge{topk}_sc{sc_num}_w{str(int(weighted))}_e{n_examples}_{n_samples}.json')  
    elif sc_num:
        result_path = os.path.join(result_dir, f'{method}{sc_num}_e{n_examples}_{n_samples}.json')
    else:   
        result_path = os.path.join(result_dir, f'{method}_e{n_examples}_{n_samples}.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=4)
        f.close()
    print(f"代码运行时间: {end_time - start_time} 秒")