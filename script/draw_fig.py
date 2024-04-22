import json
import numpy as np
import argparse
import os
from metrics import draw_heat, draw_plot

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama2_13b')
parser.add_argument('--n_samples', type=int, default=1)
parser.add_argument('--n_examples', type=int, default=1)
parser.add_argument('--dataset', type=str, default='proofwriter')
parser.add_argument('--target', type=str, default='cans')
parser.add_argument('--score', type=str, default='input')
args = parser.parse_args()

model_name = args.model
n_samples = args.n_samples
dataset = args.dataset 
n_examples = args.n_examples
target = args.target
score = args.score


score_path = f'../result/{dataset}/{model_name}/{score}_{target}_scores_e{n_examples}_s{n_samples}.json'

def plt_score():
    result_path = f'../result/{dataset}/{model_name}/fig/'
    with open(score_path, 'r') as f:
        data = json.load(f) 
    
    for item in data:
        id = item['id']
        dir_path = os.path.join(result_path, f"{id}_{item['cor_flag']}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        input_tokens = item['inp']
        output_tokens = item['out']
        input_tokens = [token.strip('\u2581') for token in input_tokens]
        output_tokens = [token.strip('\u2581') for token in output_tokens]
        scores = item['scores']

        if score == 'alti':
            draw_heat(input_tokens, 
                    output_tokens, 
                    scores,
                    path=os.path.join(dir_path, f"alti_{target}_score.pdf"))
        else:
            draw_heat(output_tokens, 
                    input_tokens, 
                    scores,
                    path=os.path.join(dir_path, f"input_{target}_score.pdf"))
            
            
# def plt_score():
#     result_path = f'../result/{dataset}/{model_name}/'
#     with open(score_path, 'r') as f:
#         data = json.load(f) 
    
#     for item in data:
#         id = item['id']
#         dir_path = os.path.join(result_path, f"{item['cor_flag']}_fig")
#         if not os.path.exists(dir_path):
#             os.makedirs(dir_path)
#         input_tokens = item['inp']
#         output_tokens = item['out']
       
#         input_tokens = [token.strip('\u2581') for token in input_tokens]
#         output_tokens = [token.strip('\u2581') for token in output_tokens]
#         scores = item['dif_scores']

#         # draw_heat(input_tokens, 
#         #         output_tokens, 
#         #         scores,
#         #         path=os.path.join(dir_path, f"{id}_alti_heat.pdf"))
#         draw_plot(input_tokens[:-len(item['gol'])], 
#                 scores[:-1],
#                 path=os.path.join(dir_path, f"{id}_{score}.pdf"))

plt_score()