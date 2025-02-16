import pandas as pd
import json
import argparse
from utils.metrics import draw_bar
from bert_score import score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default='Llama3_1_8b_chat')
    parser.add_argument('--method', type=str, default='early')
    args = parser.parse_args()

    # model_name = args.model
    method = args.method 
    datasets = ['gsmic', 'gsm8k', 'aqua', 'proofwriter', 'folio', 'prontoqa', 'wino', 'siqa','ecqa']
    # models = ['Llama3_1_8b_chat', 'Mistral_7b_chat', 'Gemma2_9b_chat']
    eval_model = 'gpt-4o'
    models = ['Gemma2_9b_chat', 'Llama3_1_8b_chat']
    # datasets = ['aqua', 'ecqa', 'gsm8k',  'lastletter', 'prontoqa', 'proofwriter']
    data_list = []
    map = {'Mistral_7b_chat':'Mistral-7B', 'Gemma2_9b_chat':'Gemma2-9B', 'Llama3_1_8b_chat':'Llama3.1-8B', 'Qwen2_5_14b_chat':'Qwen2.5-14B'}
    # map = {'gsmic':'GIC', 'gsm8k':'GSM', 'auqa':'AUQ', 'proofwriter':'PRW', 'folio':'FOL', 'prontoqa'}
    for model in models:
        for dataset in datasets: 
            if method == 'perform':
                with open(f'../result/{dataset}/{model}/cot_e3_200.json', 'r') as f:
                    cot_acc = json.load(f)[-1]['acc']
                    f.close()
                with open(f'../result/{dataset}/{model}/direct_e3_200.json', 'r') as f:
                    direct_acc = json.load(f)[-1]['acc']
                    f.close()
                # dataset = map[dataset]
                data_list.append([dataset, cot_acc-direct_acc, map[model]])
            else:
                if method == 'gpt':
                    path = f'../result/{dataset}/{model}/{eval_model}_eval_cot_200.json'
                elif method == 'bert':
                    path = f'../result/{dataset}/{model}/cot_e3_200.json'
                else:
                    path = f'../result/{dataset}/{model}/{method}_faith_200.json'
                with open(path, 'r') as f:
                    data = json.load(f)
                for item in data:
                    if method == 'gpt':
                        if item['cot'] and item['score']:
                            data_list.append([dataset, min(1,item['score']), model])
                    elif method == 'bert':
                        if item['reason']:
                            cot = item['response'].split('# Answer')[0].split('# Reasoning:')[-1].strip()
                            bert_p, bert_r, bert_f1 = score([cot], [item['reason']], lang='en')
                            data_list.append([dataset, bert_f1, model])
                    else:
                        data_list.append([dataset, item['scores'], map[model]])
    # 转换为DataFrame
    data = pd.DataFrame(data_list, columns=['dataset', 'IG', 'model'])
    path = f'../fig/{method}_fig.pdf'
    draw_bar(data, path, long_x=True)