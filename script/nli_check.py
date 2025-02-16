from utils.config import deberta_path
import argparse
import json 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import re
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Llama3_1_8b_chat')
    parser.add_argument('--dataset', type=str, default='proofwriter')
    args = parser.parse_args()

    model_name = args.model
    dataset = args.dataset 

    model = AutoModelForSequenceClassification.from_pretrained(deberta_path).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(deberta_path)
    
    result_path = f'../result/{dataset}/{model_name}/cot_e3_200.json'
    with open(result_path, 'r') as f:
        results = json.load(f)
    for item in tqdm(results[:-1]):        
        if '# Answer:' in item['response']:
            cot = item['response'].split('# Answer:')[0]
        else:
            cot = ('\n\n').join(item['response'].split('\n\n')[:-1])
        cots = re.split(r'[\.|\n]', cot)
        cot_chunks = [chunk.strip() for chunk in cots if len(chunk) >= 3]
        if not cot_chunks:
            continue
        
        premise = item['question'].split('?')[-1].strip()
        statement = cot_chunks[-1]
        
        input = tokenizer(premise, statement, return_tensors="pt")
        output = model(input["input_ids"].to('cuda'))  # device = "cuda:0" or "cpu"
        prediction = torch.softmax(output["logits"][0], -1).tolist()
        if item['answer'] == 'B':
            if np.argmax(prediction) == 0:
                item['nli_flg'] = True 
            else:
                item['nli_flg'] = False 
        else:
            if np.argmax(prediction) == 2:
                item['nli_flg'] = True 
            else:
                item['nli_flg'] = False
        # print(prediction)
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)