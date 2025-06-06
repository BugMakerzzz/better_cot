import argparse
import torch
import json
import numpy as np
from model import ModelWrapper
from utils.load_data import DataLoader
from transformers import set_seed
from tqdm import tqdm

def cal_attr(expl, L=10, b=5, p=2, eps=1e-7):

    zeros = np.zeros_like(expl)
    expl = np.abs(expl)

    expls = expl / (expl.max(axis=0, keepdims=True) + eps)
    expls = np.ceil(expls * L)
    expls = np.where(expls <= b, zeros, expls)
    l1 = expls.sum(axis=-1)
    lp = (expls ** p).sum(axis=-1) ** (1. / p) + eps
    input_attrs = (l1 / lp)
    
    expls = expl / (expl.max(axis=-1, keepdims=True) + eps)
    expls = np.ceil(expls * L)
    expls = np.where(expls <= b, zeros, expls)
    l1 = expls.sum(axis=0)
    lp = (expls ** p).sum(axis=0) ** (1. / p) + eps
    output_attrs = (l1 / lp)

    return input_attrs, output_attrs

def prepare_idx(dataset, target, method, inps):
    if target == 'cans':
        if dataset in ['gsm8k', 'aqua', 'gsmic']:
            start_idx = [i for i, v in enumerate(inps) if v == "#"][-2] + 5
        else:
            start_idx = [i for i, v in enumerate(inps) if v == ":"][-3] + 1      
        end_idx = [i for i, v in enumerate(inps) if v == "#"][-1] - 1    
    elif target == 'qans':
        if dataset in ['gsm8k']:
            if method == 'direct':
                start_idx = [i for i, v in enumerate(inps) if v == "#"][-2] + 3 
            else:
                start_idx = [i for i, v in enumerate(inps) if v == "#"][-3] + 3 
        elif dataset in ['aqua']:
            if method == 'direct':
                start_idx = [i for i, v in enumerate(inps) if v == "#"][-3] + 3
            else:
                start_idx = [i for i, v in enumerate(inps) if v == "#"][-4] + 3
        else:
            if method == 'direct':
                start_idx = [i for i, v in enumerate(inps) if v == ":"][-4] + 1
            else:
                start_idx = [i for i, v in enumerate(inps) if v == ":"][-6] + 1
        end_idx = [i for i, v in enumerate(inps) if v == "#"][-2] - 1
    elif target == 'pans':
        if dataset in ['aqua']:
            if method == 'direct':
                start_idx = [i for i, v in enumerate(inps) if v == "#"][-3] + 3 
                end_idx = [i for i, v in enumerate(inps) if v == "#"][-2] - 1
            else:
                start_idx = [i for i, v in enumerate(inps) if v == "#"][-4] + 3 
                end_idx = [i for i, v in enumerate(inps) if v == "#"][-3] - 1
        elif dataset in ['gsm8k']:
            if method == 'direct':
                start_idx = [i for i, v in enumerate(inps) if v == "#"][-3] + 3 
                end_idx = [i for i, v in enumerate(inps) if v == "#"][-2] - 1
            else:
                start_idx = [i for i, v in enumerate(inps) if v == "#"][-3] + 3 
                end_idx = [i for i, v in enumerate(inps) if v == "#"][-2] - 1
        elif dataset in ['addition', 'lastletter', 'coinflip', 'wino', 'siqa']:
            if method == 'direct':
                start_idx = [i for i, v in enumerate(inps) if v == ":"][-4] + 1
                end_idx = [i for i, v in enumerate(inps) if v == "#"][-2] - 1
            else:
                start_idx = [i for i, v in enumerate(inps) if v == ":"][-5] + 1
                end_idx = [i for i, v in enumerate(inps) if v == "#"][-3] - 1
        elif dataset in ['prontoqa', 'prontoqa_d2']:
            if method == 'direct':
                start_idx = [i for i, v in enumerate(inps) if v == ":"][-5] + 1
                end_idx = [i for i, v in enumerate(inps) if v == "#"][-2] - 1
            else:
                start_idx = [i for i, v in enumerate(inps) if v == ":"][-6] + 1
                end_idx = [i for i, v in enumerate(inps) if v == "#"][-3] - 1
        elif dataset in ['logiqa', 'folio']:
            start_idx = [i for i, v in enumerate(inps) if v == "#"][-5] + 3
            end_idx = [i for i, v in enumerate(inps) if v == "#"][-4] - 1
        else:
            if method == 'direct':
                start_idx = [i for i, v in enumerate(inps) if v == ":"][-5] + 1
                end_idx = [i for i, v in enumerate(inps) if v == "#"][-3] - 1
            else:
                start_idx = [i for i, v in enumerate(inps) if v == ":"][-6] + 1
                end_idx = [i for i, v in enumerate(inps) if v == "#"][-4] - 1
             
                print(start_idx)
                print(end_idx)
    elif target == 'cot':
        if dataset in ['gsm8k', 'aqua']:
            start_idx = [i for i, v in enumerate(inps) if v == '#'][-1] + 5
        else:
            start_idx = [i for i, v in enumerate(inps) if v == ":"][-1] + 1
        end_idx = len(inps)
    elif target == 'pcot':
        if dataset in ['aqua']:
            start_idx = [i for i, v in enumerate(inps) if v == "#"][-3] + 3
            end_idx = [i for i, v in enumerate(inps) if v == "#"][-2] - 1
        elif dataset in ['gsm8k']:
            start_idx = [i for i, v in enumerate(inps) if v == "#"][-2] + 3
            end_idx = [i for i, v in enumerate(inps) if v == "#"][-1] - 1
        elif dataset in ['addition', 'lastletter', 'coinflip', 'wino', 'siqa']:
            start_idx = [i for i, v in enumerate(inps) if v == ":"][-3] + 1
            end_idx = [i for i, v in enumerate(inps) if v == "#"][-2] - 1
        elif dataset in ['prontoqa', 'prontoqa_d2']:
            start_idx = [i for i, v in enumerate(inps) if v == ":"][-4] + 1
            end_idx = [i for i, v in enumerate(inps) if v == "#"][-2] - 1
        elif dataset in ['logiqa', 'folio']:
            start_idx = [i for i, v in enumerate(inps) if v == "#"][-4] + 3
            end_idx = [i for i, v in enumerate(inps) if v == "#"][-3] - 1
        else:
            start_idx = [i for i, v in enumerate(inps) if v == ":"][-4] + 1
            end_idx = [i for i, v in enumerate(inps) if v == "#"][-3] - 1
    else:
        if dataset in ['gsm8k', 'aqua', 'gsmic']:
            start_idx = [i for i, v in enumerate(inps) if v == "#"][-2] + 3
        elif dataset in ['prontoqa', 'prontoqa_d2']:
            start_idx = [i for i, v in enumerate(inps) if v == ":"][-4] + 1
            end_idx = [i for i, v in enumerate(inps) if v == "#"][-2] - 1
        else:
            start_idx = [i for i, v in enumerate(inps) if v == ":"][-4] + 1
        end_idx = [i for i, v in enumerate(inps) if v == "#"][-1] - 1
        
    return start_idx, end_idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Llama3_1_8b_chat')
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--n_examples', type=int, default=3)
    parser.add_argument('--method', type=str, default='cot')
    parser.add_argument('--dataset', type=str, default='prontoqa')
    parser.add_argument('--target', type=str, default='pcot')
    args = parser.parse_args()
    set_seed(17)

    model_name = args.model
    n_samples = args.n_samples
    n_examples = args.n_examples
    dataset = args.dataset 
    target = args.target
    method = args.method

    
    result_path = f'../result/{dataset}/{model_name}/{method}_e{n_examples}_{n_samples}.json'
    if method == 'gold_cot':
        result_path = f'../result/{dataset}/{model_name}/cot_e{n_examples}_{n_samples}.json'
    with open(result_path, 'r') as f:
        results = json.load(f)[:-1]
    
    
    model = ModelWrapper(model_name)
    dataloader = DataLoader(dataset=dataset, n_samples=n_samples)
    # if method in ['attr_cot', 'gold_cot', 'attr_random_cot']:
    data = dataloader.load_data(method='cot', n_examples=3)
    # elif method == 'inter_cot':
    #     data = results
    # else:
    #     data = dataloader.load_data(method=method, n_examples=3)
    question_dic = {item['id']:item['question'] for item in data}
    
    score_results = []  
    for item in tqdm(results):
        if item['answer'] == 'None':
            continue
        question = question_dic[item['id']]
        cot, answer = item['response'].split('\n# Answer:\n')
        cot = cot.strip()
        
        if model.is_chat:
            sessions = question.split('####')
            if model.is_mistral:
                input = []
            else:
                input = [{"role": "system", "content": sessions[0]}]
            for session in sessions[1:]:
                user_content, assistant_content = session.split('# Reasoning:')
                assistant_content = '# Reasoning:' + assistant_content
                input += [{"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content}]
            input = input[:-1]
            if model.is_mistral:
                input[0]['content'] = sessions[0] + '\n' + input[0]['content']
            if target in ['cot', 'qcot', 'pcot']:
                ref = {"role": "assistant", "content": cot}
                input = model.tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=False)
                print(input)
            else:
                prefix, pred = answer.split(': ')
                input = input + {"role": "assistant", "content": cot + '\n# Answer:\n' + prefix + ': '}
                ref = pred.strip().rstrip('.')
            
           

        
        else:
            input = item['question']
            if target in ['cot', 'qcot', 'pcot']:
                ref = cot
            else:
                prefix, pred = answer.split(': ')
                input = input + cot + '\n# Answer:\n' + prefix + ': '
                ref = pred.strip().rstrip('.')
    
        
        
        inps, refs, scores = model.input_explain(input, ref)
        start_idx, end_idx = prepare_idx(dataset, target, method, inps)
        inps = inps[start_idx:end_idx]
        scores = scores[start_idx:end_idx]
        assert scores.shape[0] == len(inps), "Inps Shape Not Align!!! " + str(scores.shape[0]) + " | " + str(len(inps))
        # assert scores.shape[2] == len(refs), "Refs Shape Not Align!!! " + str(scores.shape[2]) + " | " + str(len(refs))
        
        if 'cot_flag' in item.keys():
            cot_flag = item['cot_flag']
        else:
            cot_flag = None
        score_tup = {'id':item['id'],
                    'inp':inps, 
                    'out':refs,
                    'cor_flag':item['cor_flag'], 
                    'cot_flag':cot_flag,
                    'scores':scores.tolist()}
        score_results.append(score_tup)
    

    score_path = f'../result/{dataset}/{model_name}/{method}_{target}_scores_e{n_examples}_s{n_samples}.json' 


    with open(score_path, 'w') as f:
        json.dump(score_results, f, indent=4)