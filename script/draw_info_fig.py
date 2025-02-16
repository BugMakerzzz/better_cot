import json
import numpy as np
import argparse
from utils.metrics import draw_line, draw_bar, draw_box
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr
from utils.config import datasets_with_options


def cal_avg_length(dataset_ls, model_name):
    length_ls = []
    for dataset in dataset_ls:
        path = f'../result/{dataset}/{model_name}/cans_info_e3_200.json'
        with open(path, 'r') as f:
            data = json.load(f)
        lengths = []
        for item in data:
            score = item['scores']
            # print(score)
            lengths.append(len(score))
        length = np.mean(np.array(lengths))
        length_ls.append(length)
    return length_ls


def cal_data_bin_means(data, num_bins=10):
    # 获取每个区间的长度
    bin_size = len(data) // num_bins
    remainder = len(data) % num_bins
    bins = []
    start = 0
    
    # 分配数据到每个区间
    for i in range(num_bins):
        end = start + bin_size + (1 if i < remainder else 0)  # 平均分配余数
        bins.append(data[start:end])
        start = end
    
    mean = [np.mean(bin_data) for bin_data in bins]
    return mean


def draw_info_line(dataset_ls, model_name, task, num_bins=20):
    datasets = []
    x = []
    aaes = []
    for dataset in dataset_ls:
        path = f'../result/{dataset}/{model_name}/{task}_info_e3_200.json'
        with open(path, 'r') as f:
            data = json.load(f)
        for item in data:
            score = item['scores'][4:-1]
            if not score:
                continue
            # scores = [s[0] for s in score]
            scores = cal_data_bin_means(score, num_bins=num_bins)
            # sigma = 1  # 控制平滑度的参数，值越大平滑程度越高
            # scores = gaussian_filter1d(scores, sigma=sigma)
            x_values = [100 * i / num_bins for i in range(1, num_bins+1)]
            for i in range(len(x_values)):
                x.append(x_values[i])
                aaes.append(scores[i])
                datasets.append(dataset)
    data = {'step (%)':x, 'AAE':aaes, 'dataset':datasets,}
    data = pd.DataFrame(data, columns=['step (%)', 'AAE', 'dataset'])
    # print(data)
    path = f'../fig/{model_name}_{task}_info_fig.pdf'
    draw_line(data, path, style=True)
    
    
def draw_cot_info_bar(dataset_ls, model_ls, task, num_bins=10):
    datasets = []
    scores = []
    models = []
    map = {'Mistral_7b_chat':'Mistral-7B', 'Gemma2_9b_chat':'Gemma2-9B', 'Llama3_1_8b_chat':'Llama3.1-8B', 'Qwen2_5_14b_chat':'Qwen2.5-14B'}
    for model_name in model_ls:
        for dataset in dataset_ls:
            path = f'../result/{dataset}/{model_name}/{task}_info_e3_200.json'
            with open(path, 'r') as f:
                data = json.load(f)
            for item in data:
                if not item['scores']:
                    continue
                score = []
                for s in item['scores'][1:]:
                    score.append(np.mean(np.array(s)))
                # score = np.mean(np.array(score[1:]), axis=-1)
                # score = [s for s in score]
                # score = cal_data_bin_means(score, num_bins=num_bins)
                score, _ = spearmanr(score, range(len(score)))
                # diff_score = np.diff(score)
                # avg_score = (np.array(score[1:]) + np.array(score[:-1]) ) / 2
                # score = np.mean(diff_score)
                datasets.append(dataset)
                scores.append(score)
                models.append(map[model_name])
    data = {'dataset':datasets, 'score':scores, 'model':models}
    data = pd.DataFrame(data, columns=['dataset', 'score', 'model'])
    # print(data)
    path = f'../fig/{task}_grad_info_fig.pdf'
    draw_bar(data, path, long_x=True)
        
    
    
def draw_info_bar(dataset_ls, model_ls, task, num_bins=20):
    datasets = []
    scores = []
    models = []
    map = {'Mistral_7b_chat':'Mistral-7B', 'Gemma2_9b_chat':'Gemma2-9B', 'Llama3_1_8b_chat':'Llama3.1-8B', 'Qwen2_5_14b_chat':'Qwen2.5-14B'}
 
    for model_name in model_ls:
        for dataset in dataset_ls:
            path = f'../result/{dataset}/{model_name}/{task}_info_e3_200.json'
            with open(path, 'r') as f:
                data = json.load(f)
            for item in data:
                score = item['scores'][4:-1]
                if not score:
                    continue
                # score = [s for s in score]

                score = cal_data_bin_means(score, num_bins=num_bins)
                score, _ = spearmanr(score, range(len(score)))
                # diff_score = np.diff(score)
                # avg_score = (np.array(score[1:]) + np.array(score[:-1]) ) / 2
                # score = np.mean(diff_score)
                datasets.append(dataset)
                scores.append(score)
                models.append(map[model_name])
    data = {'dataset':datasets, 'score':scores, 'model':models}
    data = pd.DataFrame(data, columns=['dataset', 'score', 'model'])
    # print(data)
    path = f'../fig/{task}_grad_info_fig.pdf'
    draw_bar(data, path, long_x=True)
    

def draw_cot_info_line(dataset_ls, model_name, task, cot_num=-1):
    datasets = []
    x = []
    aaes = []
    for dataset in dataset_ls:
        path = f'../result/{dataset}/{model_name}/{task}_info_e3_200.json'
        with open(path, 'r') as f:
            data = json.load(f)
        for item in data:
            score = item['scores'][1:]
            if not score:
                continue
            # for i in range(len(score)):
            # x.append(i)
            aaes.append(np.mean(np.array(score[cot_num])))
            datasets.append(dataset)
            # scores = cal_data_bin_means(score, num_bins=num_bins)
           
            # sigma = 1  # 控制平滑度的参数，值越大平滑程度越高
            # scores = gaussian_filter1d(scores, sigma=sigma)
            # x_values = [100 * i / num_bins for i in range(1, num_bins+1)]
            # for i in range(len(x_values)):
            #     x.append(x_values[i])
            #     aaes.append(scores[i])
                
  
    path = f'./fig/{model_name}_{task}_cn{cot_num}_info_fig.pdf'
    data = {'AAE':aaes, 'dataset':datasets}
    data = pd.DataFrame(data, columns=['dataset', 'AAE'])
    draw_box(data, path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Llama3_1_8b_chat')
    parser.add_argument('--task', type=str, default='cans')
    parser.add_argument('--split', type=int, default=20)
    parser.add_argument('--grad', action='store_true')
    parser.add_argument('--cot_num', type=int, default=-1)
    parser.add_argument('--bridge', action='store_true')
    args = parser.parse_args()

    model_name = args.model
    task = args.task
    num_bins = args.split
    grad = args.grad
    cot_num = args.cot_num


    model_ls = [ 'Mistral_7b_chat', 'Gemma2_9b_chat', 'Llama3_1_8b_chat', 'Qwen2_5_14b_chat']
    dataset_ls = ['gsm8k', 'folio', 'ecqa']
    # dataset_ls = ['gsmic', 'gsm8k', 'aqua', 'proofwriter', 'folio', 'prontoqa', 'wino', 'siqa','ecqa']
    if task in ['ccot', 'qcot']:
        if grad:
            draw_cot_info_bar(dataset_ls, model_ls, task, num_bins)
        else:
            draw_cot_info_line(dataset_ls, model_name, task, cot_num)
    else:
        if grad: 
            draw_info_bar(dataset_ls, model_ls, task, num_bins)
        else:
            draw_info_line(dataset_ls, model_name, task, num_bins)
    # print(cal_avg_length(dataset_ls, model_name))