import argparse
import torch
import json
import numpy as np
from model import ModelWrapper
from load_data import DataLoader
from transformers import set_seed
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama2_13b')
parser.add_argument('--n_samples', type=int, default=500)
parser.add_argument('--n_examples', type=int, default=3)
parser.add_argument('--method', type=str, default='cot')
parser.add_argument('--dataset', type=str, default='proofwriter')
parser.add_argument('--target', type=str, default='cot')
args = parser.parse_args()
set_seed(17)

model_name = args.model
n_samples = args.n_samples
n_examples = args.n_examples
dataset = args.dataset 
target = args.target
method = args.method

def get_mask_idx(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    cache_tokens = []
    cache_idx = []
    mask_idx = []
    for i, token in enumerate(tokens):
        if token == '.':
            step = tokenizer.decode(tokenizer.convert_tokens_to_ids(cache_tokens))
            mask_idx.append([step, cache_idx])
            cache_tokens = []
            cache_idx = []
        else:
            cache_tokens.append(token)
            cache_idx.append(i)
    return mask_idx

def main():
    model_wrapper = ModelWrapper(model_name)
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer
    
    dataloader = DataLoader(dataset=dataset, n_samples=500)
    data = dataloader.load_data(method=method, n_examples=3)
    wrap_question_dic = {item['id']:item['question'] for item in data}
    result_path = f'../result/{dataset}/{model_name}/{method}_e{n_examples}_s500.json'
    with open(result_path, 'r') as f:
        results = json.load(f)[:-1][:n_samples]
    
    score_results = []  
    for item in tqdm(results):
        if item['answer'] == 'None':
            continue
        
        input = wrap_question_dic[item['id']].strip() + '\n'  
        prompt = ('####').join(input.split('####')[:-1]) + '####\n# Context:'
        context = input.split('####')[-1].split('\n# Question:')[0].lstrip('\n# Context:')
        context_start_idx = len(tokenizer.tokenize(prompt))
        cot, answer = item['response'].split('\n# Answer:\n')
        cot = cot.strip()
        answer = answer.rstrip().rstrip('.')
        cot_start_idx = len(tokenizer.tokenize(input))
        text = input + cot + '\n# Answer:\n' + answer
        ids = tokenizer(text, return_tensors="pt")['input_ids'].to(model.device)
        if target == 'context':
            mask_idx = get_mask_idx(context, tokenizer)
            for tup in mask_idx:
                tup[1] = [x + context_start_idx for x in tup[1]]
        else:
            mask_idx = get_mask_idx(cot, tokenizer)
            for tup in mask_idx:
                tup[1] = [x + cot_start_idx for x in tup[1]]
        step_fider = []
        base_attn_masks = torch.ones_like(ids)
        base_probs = torch.softmax(model(input_ids=ids, attention_mask=base_attn_masks)["logits"], -1)
        base_score = base_probs[0, -2, ids[0,-1]]
        with torch.no_grad():
            for tup in mask_idx:
                step, mask = tup
                attn_mask = base_attn_masks.clone()
                attn_mask[:, mask] = 0
                probs = torch.softmax(model(input_ids=ids, attention_mask=attn_mask)["logits"], -1)
                score = probs[0, -2, ids[0,-1]]
                fider = (base_score - score).item()
                step_fider.append([step, fider])
            
        if 'cot_flag' in item.keys():
            cot_flag = item['cot_flag']
        else:
            cot_flag = None
        score_tup = {'id':item['id'],
                    'cor_flag':item['cor_flag'], 
                    'cot_flag':cot_flag,
                    'fider':step_fider}
        score_results.append(score_tup)
    

    score_path = f'../result/{dataset}/{model_name}/fider_{target}_scores_e{n_examples}_s{n_samples}.json'

    with open(score_path, 'w') as f:
        json.dump(score_results, f, indent=4)


if __name__ == '__main__':
    main()
