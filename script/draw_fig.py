import json
import numpy as np
import argparse
from metrics import draw_heat, draw_plot

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama2_13b')
parser.add_argument('--n_samples', type=int, default=10)
parser.add_argument('--dataset', type=str, default='proofwriter')
parser.add_argument('--mode', type=str, default='answer')
parser.add_argument('--golden', action='store_true')
args = parser.parse_args()

model_name = args.model
n_samples = args.n_samples
dataset = args.dataset 
mode = args.mode
golden = args.golden


score_path = f'../result/{dataset}/{model_name}/input_score_e3_s100_m{mode}.json'

def plt_input_explain(path):
    with open(score_path, 'r') as f:
        data = json.load(f) 
    
    for item in data[:n_samples]:
        input_tokens = item['input_tokens']
        input_tokens = [token.strip('\u2581') for token in input_tokens]
        response_tokens = item['response_tokens']
        response_tokens = [token.strip('\u2581') for token in response_tokens]
        scores = item['scores']
        idx = 0
        i = 0
        for token in input_tokens:
            if token == '####':
                idx = i
            i += 1
        
        input_tokens = input_tokens[idx:]
        scores = np.array(scores)[idx:,:]
        norm_scores = item['norm_scores'][idx:]
        input_scores = item['input_attrs'][idx:]
        output_scores = item['output_attrs']
        
        if mode == 'answer':
            i = 0
            for token in input_tokens:
                if token == '#':
                    idx = i
                i += 1   
            input_tokens = input_tokens[idx:]
            scores = scores[idx:,:]
            norm_scores = norm_scores[idx:]
            input_scores = input_scores[idx:]
        
        draw_heat(response_tokens, 
                  input_tokens, 
                  scores,
                  path=path + f'{mode}_heat_{data.index(item)}.pdf')
        draw_plot(input_tokens, 
                  input_scores, 
                  path=path + f'{mode}_input_{data.index(item)}.pdf')
        draw_plot(response_tokens, 
                  output_scores, 
                  path=path + f'{mode}_output_{data.index(item)}.pdf')


result_path = f'../result/{dataset}/{model_name}/fig/'
plt_input_explain(result_path)