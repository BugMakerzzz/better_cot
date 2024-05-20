import json
import numpy as np
import argparse
import os
from metrics import draw_heat, draw_line
from pandas import DataFrame

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama2_13b')
parser.add_argument('--n_samples', type=int, default=100)
parser.add_argument('--n_examples', type=int, default=3)
parser.add_argument('--dataset', type=str, default='proofwriter')
parser.add_argument('--target', type=str, default='cans')
parser.add_argument('--score', type=str, default='step')
args = parser.parse_args()

model_name = args.model
n_samples = args.n_samples
dataset = args.dataset 
n_examples = args.n_examples
target = args.target
score = args.score


def plt_score():
    score_path = f'../result/{dataset}/{model_name}/input_{target}_scores_e{n_examples}_s{n_samples}.json'
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

        draw_heat(output_tokens, 
            input_tokens, 
            scores,
            path=os.path.join(dir_path, f"input_{target}_score.pdf"))
            
def plot_step_line():
    # dataset_ls = ['gsm8k', 'aqua', 'proofwriter', 'folio', 'wino', 'siqa']
    # target_ls = ['qcot', 'cot', 'cans', 'qans']
    dataset_ls = ['aqua','gsm8k','siqa']
    target_ls = ['cans']
    
    name_map = {'aqua':'aqua', 'gsm8k':'gsm8k', 'wino':'winogrande', 'siqa':'socialiqa', 'proofwriter':'proofwriter', 'prontoqa':'prontoqa'}
    
    for target in target_ls:
        datasets = []
        x = []
        scores = []
        for dataset in dataset_ls:
            path = f'../result/{dataset}/{model_name}/cot_{target}_{score}_score_e3_s100.json'
            if not os.path.exists(path):
                path = f'../result/{dataset}/{model_name}/input_{target}_{score}_score_e3_s100.json'
            with open(path, 'r') as f:
                data = json.load(f)
            for item in data:
                attr = item['attr']
                if len(attr) == 9:
                    attr = [0] + attr
                elif len(attr) < 9:
                    print(item['id'])
                for i in range(10):
                    x.append( 10 * (i+1))
                    scores.append(attr[i])
                    datasets.append(name_map[dataset])
        data = {'dataset':datasets, 'step (%)':x, 'AAE':scores}
        data = DataFrame(data)
        path = f'./{model_name}_{target}_{score}_line_fig.pdf'
        names = ['step (%)', 'AAE', 'dataset']
        draw_line(data, path, names)
        
        
def plot_type_line():
    path = f'../result/{dataset}/{model_name}/cot_{target}_{score}_score_e3_s100.json'
    if not os.path.exists(path):
        path = f'../result/{dataset}/{model_name}/input_{target}_{score}_score_e3_s100.json'
    gold_path = f'../result/{dataset}/{model_name}/gold_cot_{target}_{score}_score_e3_s100.json'
    with open(path, 'r') as f:
        data = json.load(f)
        f.close()
    with open(gold_path, 'r') as f:
        gold_data = json.load(f)
        f.close()
    gold_attrs = {item['id']:item['attr'] for item in gold_data}
    for flag in [0,3]:
        x = []
        scores = []
        cot_flags = []
        for item in data:
            if item['cot_flag'] != flag:
                continue
            attr = item['attr']
            if len(attr) == 9:
                attr = [0] + attr
            elif len(attr) < 9:
                print(item['id'])
            for i in range(10):
                x.append( 10 * (i+1))
                scores.append(attr[i])
                cot_flags.append('generate')
            if item['id'] in gold_attrs.keys():
                gold_attr = gold_attrs[item['id']]
            for i in range(10):
                x.append(10 * (i+1))
                scores.append(gold_attr[i])
                cot_flags.append('golden')
        result = {'type':cot_flags, 'step (%)':x, 'AAE':scores}
        result = DataFrame(result)
        path = f'../result/{dataset}/{model_name}/{target}_{score}_{flag}_line_fig.pdf'
        names = ['step (%)', 'AAE', 'type']
        draw_line(result, path, names)
        
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

# plot_step_line()
plot_type_line()