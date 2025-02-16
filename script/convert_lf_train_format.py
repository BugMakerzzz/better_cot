import json 
import argparse
import numpy as np
from utils.load_data import load_prompt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Llama3_1_8b_chat')
    parser.add_argument('--datalength', type=int, default=5000)
    parser.add_argument('--dataset', type=str, default='proofwriter')
    args = parser.parse_args()
    
    model_name = args.model
    datalength = args.datalength 
    dataset = args.dataset 
    
    raw_data_path = f'../data/{dataset}/{model_name}_train_{datalength}.json'
    
    with open(raw_data_path, 'r') as f:
        data = json.load(f)
    
    instruction = load_prompt(dataset, 'cot', 3).split('####')[0].strip()
    lf_data = []
    for item in data:
        idx = []
        for i in range(len(item['cor_flgs'])):
            if item['cor_flgs'][i]:
                idx.append(i)
        if len(idx) <= 1:
            continue
        scores = [item['sp_scores'][i] for i in idx]
        if all(x == scores[0] for x in scores):
            continue
        max_idx = np.argmax(np.array(scores))
        min_idx = np.argmin(np.array(scores))
        good_res = item['responses'][max_idx]
        bad_res = item['responses'][min_idx]
        msg = {
            "conversations":[
                {
                    "from":"system",
                    "value":instruction
                },
                {
                    "from":"human",
                    "value":item['question']
                }
            ],
            "chosen": {
                "from":"gpt",
                "value":good_res
            },
            "rejected": {
                "from":"gpt",
                "value":bad_res
            }
        }
        lf_data.append(msg)

    lf_data_path = f'../data/{dataset}/{model_name}_lf_sp_data.json'
    with open(lf_data_path, 'w') as f:
        json.dump(lf_data, f, indent=4)
    
    data_info_path = '/mnt/userdata/ljc/code/faithful_cot/LLaMA-Factory/data/dataset_info.json'
    with open(data_info_path, 'r') as f:
        data = json.load(f)
    data[f'{model_name}_{dataset}'] = {
            "file_name": f"/mnt/userdata/ljc/code/faithful_cot/data/{dataset}/{model_name}_lf_sp_data.json",
            "ranking": True,
            "formatting": "sharegpt",
            "columns": {
            "messages": "conversations",
            "chosen": "chosen",
            "rejected": "rejected"
                }
    }
    with open(data_info_path, 'w') as f:
        json.dump(data, f, indent=4)
        