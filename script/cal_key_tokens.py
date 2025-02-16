import json 
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Llama3_1_8b_chat')
    parser.add_argument('--dataset', type=str, default='gsm8k')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--method', type=str, default='cans')
    args = parser.parse_args()
    
    model_name = args.model
    dataset = args.dataset 
    topk = args.topk
    method = args.method

    data_path =  f'../result/{dataset}/{model_name}/{method}_info_e3_200.json' 
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    results = {}
    for item in data:
        input = item['input']
        scores = [score for score in item['scores']]
        topk_indices = np.argsort(scores)[-topk:][::-1]
        topk_tokens = [input[i] for i in topk_indices]
        topk_scores = [scores[i] for i in topk_indices]
        results[item['id']] = {'cot_tokens':topk_tokens, 'cot_scores':topk_scores, 'faith_scores':-1}
    
    data_path =  f'../result/{dataset}/{model_name}/mail_faith_200.json' 
    with open(data_path, 'r') as f:
        data = json.load(f)
    for item in data:
        if item['id'] in results.keys():
            results[item['id']]['faith_scores'] = item['scores']
    
    result = [{'id':k}|v for k,v in results.items()]
    result = sorted(result, key=lambda x: x["faith_scores"], reverse=True)
    result_path =  f'../result/{dataset}/{model_name}/{method}_{topk}_tokens_e3_200.json' 
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=4)
    