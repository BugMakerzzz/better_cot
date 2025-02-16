import argparse
import json 
import numpy as np
import pandas as pd
from utils.metrics import draw_bar, figure_colors
from tqdm import tqdm 

def hit(statement, ref_sentence):
    refs = ref_sentence.split('.')
    for ref in refs:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        score = scorer.score(ref, statement.strip('.'))['rougeL'][2]
        # p, r, f1 = score([ref], [statement.strip('.')], lang='en')
        if score > 0.8:

            return True
    return False

def extract_context_scores(model, tokens, scores, target=False):
    if model.startswith('Llama'):
        start_idx = [i for i, v in enumerate(tokens) if v == ':\u010a'][0] + 1
    else:
        start_idx = [i for i, v in enumerate(tokens) if v == ':'][0] + 1
    end_idx = [i for i, v in enumerate(tokens) if v == "#"][1]
    if target:
        score = np.mean(np.array(scores)[:,end_idx:])
    else:
        score = np.mean(np.array(scores)[:,start_idx:end_idx])
    return score


def cal_hit_scores(tokens, scores):
    stop_tokens = ['.\u010a', '.']
    start_idx = [i for i, v in enumerate(tokens) if v == ':\u010a'][0] + 1
    end_idx = [i for i, v in enumerate(tokens) if v == "#"][1]
    tokens = tokens[start_idx:end_idx]   
    scores = scores[start_idx:end_idx] 
    step_scores = {}
    start = 0
    for i in range(len(tokens)):
        token = tokens[i]
        if i == len(tokens)-1 or token in stop_tokens:
            end = i + 1
            step = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens[start:end]))

            step_scores[step.strip()] = np.mean(np.array(scores[start:end]))
            start = i + 1
    return step_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Llama3_1_8b_chat')
    parser.add_argument('--dataset', type=str, default='proofwriter')
    parser.add_argument('--method', type=str, default='ig')
    parser.add_argument('--golden', action='store_true')
    args = parser.parse_args()

    model_name = args.model
    dataset = args.dataset 
    method = args.method
    golden = args.golden

    result_path = f'../result/{dataset}/{model_name}/cot_e3_200.json'
    with open(result_path, 'r') as f:
        results = json.load(f)[:-1]
        results = {item['id']:item for item in results}
 
    if method in ['mail', 'early', 'ig']:
        with open(f'../result/{dataset}/{model_name}/{method}_faith_200.json') as f:
            info_data = json.load(f)
        if golden:
            with open(f'../result/{dataset}/{model_name}/{method}_faith_200_golden.json') as f:
                golden_info_dic = {item['id']:item for item in json.load(f)}
    elif method == 'direct':
        with open(f'../result/{dataset}/{model_name}/direct_e3_200.json') as f:
            info_data = json.load(f)[:-1]
    else:
        with open(f'../result/{dataset}/{model_name}/{method}_info_e3_200.json') as f:
            info_data = json.load(f)
        if golden:
            with open(f'../result/{dataset}/{model_name}/{method}_info_e3_200_golden.json') as f:
                golden_info_dic = {item['id']:item for item in json.load(f)}
        # tokenizer = AutoTokenizer.from_pretrained(llama3_1_8b_chat_path)
    
    x = []
    y = []
    datasets = []    
    for dataset in ['prontoqa', 'proofwriter']:
        if dataset == 'prontoqa':
            key_token = ':'
        else:
            key_token = '?'

        for item in tqdm(info_data):
            if item['id'] not in results.keys():
                continue
            cot_flg = results[item['id']]['cot_flg']
            golden_cot = results[item['id']]['reason']
            response = results[item['id']]['response']
            cor_flg = results[item['id']]['cor_flag']
            
            if golden:
                if cot_flg:
                    continue
            if cot_flg == None:
                cot_flg = False
                
            
            if method in ['mail', 'early', 'ig']:
                score = item['scores']
                if golden:
                    if item['id'] not in golden_info_dic.keys():
                        continue
                    x.append(False)
                    y.append(score)
                    x.append(True)
                    y.append(golden_info_dic[item['id']]['scores'])
                    datasets.append(dataset)
                else:
                    x.append(cot_flg)
                    y.append(score)
            else:
                if len(item['scores']) < 2:
                    continue
                score = extract_context_scores(model_name, item['input'][1], item['scores'][1:-1], target=True)
                if golden:
                    if item['id'] not in golden_info_dic.keys() or len(golden_info_dic[item['id']]['scores']) < 3:
                        continue
                    x.append(False)
                    y.append(score)
                    x.append(True)
                    y.append(extract_context_scores(model_name, golden_info_dic[item['id']]['input'][1], golden_info_dic[item['id']]['scores'][1:-1], target=True))
                    datasets.append(dataset)
                else:
                    x.append(cot_flg)
                    y.append(score)
            datasets.append(dataset)

    data = pd.DataFrame({
        'dataset': datasets,
        'score': y,
        'correct': x,
    })
    draw_bar(data, f'fig/{method}_g{golden}_corr.pdf', ['salmon', figure_colors[1]])
    # 绘制密度图
    # plt.figure(figsize=(8, 6))
    # sns.kdeplot(data=data, x='metric', hue='category', fill=True)
    # # sns.boxplot(x="category", y="metric", data=data)
    # plt.title('Density Plot of Metric for Two Categories')
    # plt.xlabel('Metric')
    # plt.ylabel('Density')
        # plt.show()
    # else:
    #     percentages_1 = (np.array(values_1) / total_values) * 100
    #     percentages_2 = (np.array(values_2) / total_values) * 100

    #     # Plot the pie chart
    #     fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    #     # Pie chart for Values 1
    #     ax[0].pie(percentages_1, labels=categories, autopct='%1.1f%%', startangle=90)
    #     ax[0].set_title('Values 1 Proportion')
    
      # score = item['scores']
        # if not input:
        #     continue
        # target_idx = [i for i, v in enumerate(input) if v == key_token][-1] + 2   
        # target_scores = cal_hit_scores(input, item['scores'])
        # target_score = []
        # for k,v in target_scores.items():
        #     if hit(k, golden_cot) and not hit(k, results[item['id']]['response']):
        #         target_score.append(v)
        # target_score = np.mean(np.array(target_score))
        
        # if flg == False and results[item['id']]['cor_flag']:
        #     type = 'unfaithful'
        # else:
        #     type = 'faithful'

        # for k in range(1, 10):
        #     top_k_items = sorted(target_scores.items(), key=lambda item: item[1], reverse=True)[:k]
        #     top_k_keys = [item[0] for item in top_k_items] 
        #     cnt = 0
        #     for sent in top_k_keys:
        #         if hit(sent, golden_cot):
        #             cnt += 1
                
        # golden_scores = cal_hit_scores(input, golden_info_dic[item['id']]['scores'])
        # golden_score = []
        # for k,v in golden_scores.items():
        #     if k in golden_cot:
        #         print(v)
        #         golden_score.append(v)
        # golden_score = np.mean(np.array(golden_score))
        # print(target_score)
        # print(golden_score)
        # if len(target_score) == 1:
        #     continue

                # x.append(k)
                # y.append(cnt)
                # flgs.append(type) 
            # x.append(True)
            # y.append(golden_score)
            # if np.max(target_score) >= 0.2:
            #     print(item['id'])
            #     print(flg)
            #     print(target_score)
            #     print('------------')
 
    # data = {'topk':x, 'hits':y, 'category':flgs,}
    # data = pd.DataFrame(data, columns=['topk', 'hits', 'category'])
    # # print(data)
    # path = f'test1.png'
    # draw_line(data, path)
     