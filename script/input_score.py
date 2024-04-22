import argparse
import torch
import json
import numpy as np
from model import ModelWrapper
from load_data import DataLoader
from transformers import set_seed
from tqdm import tqdm
from IPython.core.display import HTML

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama2_13b')
parser.add_argument('--n_samples', type=int, default=500)
parser.add_argument('--n_examples', type=int, default=3)
parser.add_argument('--method', type=str, default='cot')
parser.add_argument('--dataset', type=str, default='proofwriter')
parser.add_argument('--target', type=str, default='cans')
args = parser.parse_args()
set_seed(17)

model_name = args.model
n_samples = args.n_samples
n_examples = args.n_examples
dataset = args.dataset 
target = args.target
method = args.method


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


def main():
    model = ModelWrapper(model_name)
    dataloader = DataLoader(dataset=dataset, n_samples=500)
    data = dataloader.load_data(method=method, n_examples=3)
    wrap_question_dic = {item['id']:item['question'] for item in data}
    result_path = f'../result/{dataset}/{model_name}/{method}_e{n_examples}_s500.json'
    with open(result_path, 'r') as f:
        results = json.load(f)[:n_samples]
    
    score_results = []  
    for item in tqdm(results):
        if item['answer'] == 'None':
            continue
        input = wrap_question_dic[item['id']]  
        
        try:
            cot, answer = item['response'].split('\n# Answer:\n')
            prefix, pred = answer.split(': ')
        except:
            continue
        if target in ['cot', 'qcot', 'pcot']:
            input = input + cot
            ref = item['response'].split('\n# Answer')[0]
        else:
            input = input + cot + '\n# Answer:\n' + prefix + ': '
            ref = pred.strip().rstrip('.')   
      
        inps, refs, scores = model.input_explain(input, ref)
        
        if target == 'cans':
            start_idx = [i for i, v in enumerate(inps) if v == ":"][-3] + 1
            end_idx = [i for i, v in enumerate(inps) if v == "#"][-1] - 1  
        elif target == 'qans':
            start_idx = [i for i, v in enumerate(inps) if v == ":"][-5] + 1
            end_idx = [i for i, v in enumerate(inps) if v == "#"][-2] - 1
        elif target == 'pans':
            start_idx = [i for i, v in enumerate(inps) if v == ":"][-6] + 1
            end_idx = [i for i, v in enumerate(inps) if v == "#"][-3] - 1
        elif target == 'cot':
            start_idx = [i for i, v in enumerate(inps) if v == ":"][-1] + 1
            end_idx = len(inps)
        elif target == 'pcot':
            start_idx = [i for i, v in enumerate(inps) if v == ":"][-4] + 1
            end_idx = [i for i, v in enumerate(inps) if v == "#"][-3] - 1
        else:
            start_idx = [i for i, v in enumerate(inps) if v == ":"][-3] + 1
            end_idx = [i for i, v in enumerate(inps) if v == "#"][-1] - 1
    
        inps = inps[start_idx:end_idx]
        scores = scores[start_idx:end_idx]
        assert scores.shape[0] == len(inps), "Inps Shape Not Align!!! " + str(scores.shape[0]) + " | " + str(len(inps))
        # assert scores.shape[2] == len(refs), "Refs Shape Not Align!!! " + str(scores.shape[2]) + " | " + str(len(refs))
        
        
        score_tup = {'id':item['id'],
                    'inp':inps, 
                    'out':refs,
                    'cor_flag':item['cor_flag'], 
                    'scores':scores.tolist()}
        score_results.append(score_tup)
        
    score_path = f'../result/{dataset}/{model_name}/input_{target}_scores_e{n_examples}_s{n_samples}.json'

    with open(score_path, 'w') as f:
        json.dump(score_results, f, indent=4)


if __name__ == '__main__':
    main()
