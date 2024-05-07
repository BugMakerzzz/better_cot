import argparse
import os
import json
from model import ModelWrapper
from load_data import DataLoader, extract_answer
from transformers import set_seed
from tqdm import tqdm
from gpt_reason import gpt_reason

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama3_8b')
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
data = dataloader.load_data(method=method, n_examples=n_examples)

if model_name.startswith('gpt'):
    result = gpt_reason(data, model_name, method, dataset)
else:
    model_wrapper = ModelWrapper(model_name)
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer
    result = []
    correct = 0
    max_new_tokens = 200
    split_token = '####'

    for item in tqdm(data):
        inputs = tokenizer(item['question'], return_tensors="pt")
        input_ids = inputs["input_ids"].to(model_wrapper.device)
        if 'sc' in method:
            top_k = 40
            temperature = 0.7
            num_beams = 20
            num_beam_groups = 20
            diversity_penalty = 0.1
            if model_name.startswith('Llama3'):
                stop_token = "<|eot_id|>"
                stop_token_id = tokenizer.encode(stop_token)[0]
                outputs = model.generate(input_ids, 
                                        max_new_tokens=max_new_tokens, 
                                        do_sample=False, 
                                        eos_token_id=stop_token_id,
                                        top_k=top_k, 
                                        temperature=temperature, 
                                        num_beams=num_beams, 
                                        num_return_sequences=num_beams,
                                        num_beam_groups=num_beam_groups,
                                        diversity_penalty=diversity_penalty)
            else:
                outputs = model.generate(input_ids, 
                                        max_new_tokens=max_new_tokens, 
                                        do_sample=False, 
                                        top_k=top_k, 
                                        temperature=temperature, 
                                        num_beams=num_beams, 
                                        num_return_sequences=num_beams,
                                        num_beam_groups=num_beam_groups,
                                        diversity_penalty=diversity_penalty)
            response = []
            answer = []
            for i in range(num_beams):
                res = tokenizer.decode(outputs[i][len(input_ids[0]):], skip_special_tokens=True).split(split_token)[0]
                ans = extract_answer(dataset, res)
                response.append(res)
                answer.append(ans)
            max_answer = max(answer,key=answer.count)
            cor_flag = False
            if max_answer == item['answer']:
                cor_flag = True
                correct += 1       
        else:
            if model_name.startswith('Llama3'):
                stop_token = "<|eot_id|>"
                stop_token_id = tokenizer.encode(stop_token)[0]
                outputs = model.generate(input_ids, max_new_tokens=max_new_tokens, eos_token_id=stop_token_id, pad_token_id=128002)
                response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True).split(split_token)[0]
                response = ('\n').join(response.split('\n')[1:])
            else:
                outputs = model.generate(input_ids, max_new_tokens=max_new_tokens)
                response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True).split(split_token)[0]
            if method == 'gold_cot':
                response = item['reason'] + response
            if method == 'decom':
                answer = []
                response_ls = response.split('#')
                for re in response_ls:
                    if ':' in re:
                        answer.append(re.split(':')[1].strip())
                cor_flag = None
            else:
                answer = extract_answer(dataset, response)
                cor_flag = False
                if answer == item['answer']:
                    cor_flag = True
                    correct += 1
        if 'reason' in item.keys():
            reason = item['reason']
        else:
            reason = None 
        msg = {'id':item['id'], 'question':item['raw_question'], 'response':response, 'answer':answer, 'reason':reason, 'label':item['answer'], 'cor_flag':cor_flag}
        result.append(msg)
    result.append({'acc': correct / n_samples})
    
result_dir = f'../result/{dataset}/{model_name}/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
result_path = os.path.join(result_dir, f'{method}_e{n_examples}_s{n_samples}.json')
with open(result_path, 'w') as f:
    json.dump(result, f, indent=4)
    f.close()