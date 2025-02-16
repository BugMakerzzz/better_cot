import argparse
import json 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from rouge_score import rouge_scorer
from utils.config import llama3_1_8b_chat_path
from transformers import AutoTokenizer

def hit(statement, ref_sentence):
    refs = ref_sentence.split('.')
    for ref in refs:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        score = scorer.score(ref, statement.strip('.'))['rougeL'][2]
        # p, r, f1 = score([ref], [statement.strip('.')], lang='en')
        if score > 0.8:

            return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Llama3_1_8b_chat')
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--n_examples', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='proofwriter')
    parser.add_argument('--method', type=str, default='mail')
    args = parser.parse_args()

    model_name = args.model
    n_samples = args.n_samples
    n_examples = args.n_examples
    dataset = args.dataset 
    method = args.method

    # tokenizer = AutoTokenizer.from_pretrained(llama3_1_8b_chat_path)

    result_path = f'../result/{dataset}/{model_name}/cot_e{n_examples}_200.json'
    with open(result_path, 'r') as f:
        results = json.load(f)[:-1]
    
    cot_flgs = {item['id']:item['cot_flg'] for item in results if len(item['reason'].split('.')) > 3}
    with open(f'../result/{dataset}/{model_name}/qcot_info_e3_200.json') as f:
        info_data = json.load(f)
    
    # with open(f'../result/{dataset}/{model_name}/qcot_info_e3_200_golden.json') as f:
    #     golden_info_dic = {item['id']:item for item in json.load(f)}
    
    x = []
    y = []
    if dataset == 'prontoqa':
        key_token = ':'
    else:
        key_token = '?'
    
    for item in info_data:
        if item['id'] not in cot_flgs.keys():
            continue
        flg = cot_flgs[item['id']]
        if flg != False:
            continue
        input = item['input']
        score = item['scores']
        golden_score = golden_info_dic[item['id']]['scores']
        if not input:
            continue

        target_idx = [i for i, v in enumerate(input[1]) if v == key_token][-1] + 1


        target_score = np.array(score[1:-1])[target_idx:]
        # golden_score = np.array(golden_score[1:-1])
        # if len(target_score) == 1:
        #     continue
        x.append(False)
        y.append(np.mean(target_score))
        x.append(True)
        y.append(np.mean(golden_score))
        # if np.max(target_score) >= 0.2:
        #     print(item['id'])
        #     print(flg)
        #     print(target_score)
        #     print('------------')
            
    data = pd.DataFrame({
        'metric': y,
        'category': x
    })

    # 绘制密度图
    plt.figure(figsize=(8, 6))
    # sns.kdeplot(data=data, x='metric', hue='category', fill=True, common_norm=False)
    sns.boxplot(y="metric", x="category", data=data)
    plt.title('Density Plot of Metric for Two Categories')
    plt.xlabel('Metric')
    plt.ylabel('Density')
    # plt.show()
    plt.savefig('test.png')
