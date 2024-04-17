import json
import numpy as np
import argparse
import os
from metrics import draw_heat, draw_plot

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama2_13b')
parser.add_argument('--n_samples', type=int, default=1)
parser.add_argument('--dataset', type=str, default='proofwriter')
parser.add_argument('--golden', action='store_true')
args = parser.parse_args()

model_name = args.model
n_samples = args.n_samples
dataset = args.dataset 
golden = args.golden


score_path = f'../result/{dataset}/{model_name}/alti_scores_e3_s{n_samples}.json'



def plt_input_explain(result_path):
    with open(score_path, 'r') as f:
        data = json.load(f) 
    
    for item in data:
        id = item['id']
        dir_path = os.path.join(result_path, id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        for tup in item['scores']:
            input_tokens = tup['inp']
            output_tokens = tup['out']
            input_tokens = [token.strip('\u2581') for token in input_tokens]
            output_tokens = [token.strip('\u2581') for token in output_tokens]
            scores = tup['scores']
            
            input_scores = tup['input_attrs']
            output_scores = tup['output_attrs']

            draw_heat(output_tokens, 
                    input_tokens, 
                    scores,
                    path=os.path.join(dir_path, f"heat_{item['scores'].index(tup)}.pdf"))
            draw_plot(input_tokens, 
                    input_scores, 
                    path=os.path.join(dir_path, f"input_{item['scores'].index(tup)}.pdf"))
            draw_plot(output_tokens, 
                    output_scores, 
                    path=os.path.join(dir_path, f"output_{item['scores'].index(tup)}.pdf"))
            

def plt_alti(result_path):
    with open(score_path, 'r') as f:
        data = json.load(f) 
    
    for item in data:
        id = item['id']
        dir_path = os.path.join(result_path, id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        input_tokens = item['inp']
        output_tokens = item['out']
        print(len(input_tokens))
       
        input_tokens = [token.strip('\u2581') for token in input_tokens]
        output_tokens = [token.strip('\u2581') for token in output_tokens]
        scores = item['scores']
        print(len(scores[0]))

        draw_heat(input_tokens, 
                output_tokens, 
                scores,
                path=os.path.join(dir_path, f"alti_heat.pdf"))


result_path = f'../result/{dataset}/{model_name}/fig/'
plt_alti(result_path)