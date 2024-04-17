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
parser.add_argument('--dataset', type=str, default='proofwriter')
parser.add_argument('--golden', action='store_true')
args = parser.parse_args()
set_seed(17)

model_name = args.model
n_samples = args.n_samples
n_examples = args.n_examples
dataset = args.dataset 
golden = args.golden


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
    data = dataloader.load_data(method='cot', n_examples=3)
    wrap_question_dic = {item['id']:item['question'] for item in data}
    result_path = f'../result/{dataset}/{model_name}/cot_e{n_examples}_s500.json'
    with open(result_path, 'r') as f:
        results = json.load(f)[:n_samples]
    
    score_results = []  
    for item in tqdm(results):
        if golden:
            cot = item['gold_cot'].strip()
        else:
            cot = item['response'].split('# Answer')[0].strip()
        if dataset == 'gsm8k':
            reason_steps = cot.split('\n')
            answer = f"\n# Answer:\nThe answer is: {item['answer']}"
        else:
            reason_steps = cot.split('.')[:-1]
            answer = f"\n# Answer:\nThe correct option is: {item['answer']}"
        reason_steps.append(answer)
    
        input = wrap_question_dic[item['id']]
        prompt = ('####').join(input.split('####')[:-1])
        prompt_len = len(model.tokenize(prompt))
        scores = []
        for step in reason_steps:
            ref = step
            inps, refs, expls = model.input_explain(input, ref)
            expls = expls[prompt_len:,:]
            input_attr, output_attr = cal_attr(expls)
            score_tup = {'inp':inps[prompt_len:], 
                         'out':refs, 
                         'scores':expls.tolist(), 
                         'input_attrs':input_attr.tolist(),
                         'output_attrs':output_attr.tolist()}
            scores.append(score_tup)
            input += step

        score_item = {'id':item['id'],
                    'question':item['question'],  
                    'label':item['label'], 
                    'cor_flag':item['cor_flag'],
                    'scores':scores}
        score_results.append(score_item)
        
    score_path = f'../result/{dataset}/{model_name}/faith_scores_e{n_examples}_s{n_samples}_g{golden}.json'

    with open(score_path, 'w') as f:
        json.dump(score_results, f, indent=4)


if __name__ == '__main__':
    main()
