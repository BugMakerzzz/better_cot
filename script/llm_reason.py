import argparse
import os
import json
from model import ModelWrapper
from load_data import DataLoader, extract_answer
from transformers import set_seed
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama2_13b')
parser.add_argument('--n_samples', type=int, default=500)
parser.add_argument('--n_examples', type=int, default=5)
parser.add_argument('--dataset', type=str, default='proofwriter')
parser.add_argument('--method', type=str, default='cot')
args = parser.parse_args()
set_seed(17)

model_name = args.model
n_samples = args.n_samples
n_examples = args.n_examples
dataset = args.dataset 
method = args.method 

dataloader = DataLoader(dataset=dataset, n_samples=n_samples)
model_wrapper = ModelWrapper(model_name)
model = model_wrapper.model
tokenizer = model_wrapper.tokenizer


data = dataloader.load_data(method=method, n_examples=n_examples)
result = []
correct = 0
    
for item in data:
    inputs = tokenizer(item['question'], return_tensors="pt")
    input_ids = inputs["input_ids"].to(model_wrapper.device)
    outputs = model.generate(input_ids, max_new_tokens=500)
    response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True).split('####')[0]
    answer = extract_answer(dataset, response)
    
    cor_flag = False
    if answer == item['answer']:
        cor_flag = True
        correct += 1
    if 'reason' not in item.keys():
        item['reason'] = None
    
    msg = {'id':item['id'], 'question':item['raw_question'], 'gold_cot':item['reason'], 'response':response, 'answer':answer, 'label':item['answer'], 'cor_flag':cor_flag}
    result.append(msg)

result.append({'acc': correct / n_samples})
result_dir = f'../result/{dataset}/{model_name}/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
result_path = os.path.join(result_dir, f'{method}_e{n_examples}_s{n_samples}.json')
with open(result_path, 'w') as f:
    json.dump(result, f, indent=4)
    f.close()