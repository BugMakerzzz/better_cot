import argparse
import os
import json
from model import ModelWrapper
from load_data import DataLoader, extract_answer
from transformers import set_seed
from tqdm import tqdm
from gpt_reason import gpt_reason

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Llama2_13b')
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--n_examples', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='proofwriter_d1')
    parser.add_argument('--proxy', type=str, default='Llama2_13b')
    parser.add_argument('--method', type=str, default='cot')
    parser.add_argument('--target', type=str, default='pans')
    # parser.add_argument('--cand_num', type=int, default=3)

    args = parser.parse_args()
    set_seed(17)

    model_name = args.model
    n_samples = args.n_samples
    n_examples = args.n_examples
    dataset = args.dataset 
    proxy = args.proxy
    method = args.method
    # num = args.cand_num
    
    dataloader = DataLoader(dataset=dataset, n_samples=n_samples)
    data = dataloader.load_data(method='attr_sr', n_examples=n_examples)
    if dataset == 'coinflip':
        path_file = f'../result/{dataset}/{model_name}/{method}_{proxy}_pcot_paths_e{n_examples}_s100.json'
    else:
        path_file = f'../result/{dataset}/{model_name}/{method}_{proxy}_pans_paths_e{n_examples}_s100.json'
    cot_file = f'../result/{dataset}/{model_name}/cot_e{n_examples}_s100.json'
    # if not os.path.exists(path_file):
    #     if method == 'direct':
    #         path_file = f'../result/{dataset}/{model_name}/input_pans_paths_e{n_examples}_s{n_samples}_direct.json'
    #     else:
    #         path_file = f'../result/{dataset}/{model_name}/input_pans_paths_e{n_examples}_s{n_samples}.json'
    
    with open(path_file, 'r') as f:
        path_data = json.load(f)
        f.close()
    with open(cot_file, 'r') as f:
        cot_data = json.load(f)[:-1]
        f.close()
        
    path_dic = {}
    for item in path_data:
        # if num > len(item['path'][-1]['inp_attr']):
        #     attrs = item['path'][-1]['inp_attr']
        # else:
        if dataset == 'coinflip':
            attrs = item['path'][0]['inp_attr'][:3]
        else:
            attrs = item['path'][-1]['inp_attr'][:3]
        attr_sents = [x['inp'] for x in attrs]
        # attr_sent = '. '.join(attr_sents)
        path_dic[item['id']] = attr_sents
    cot_dic = {}
    for item in cot_data:
        cot = item['response'].split('\n# Answer:')[0].strip()
        cot_dic[item['id']] = cot    


    model_wrapper = ModelWrapper(model_name)
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer
    max_new_tokens = 200
    split_token = '####'
    result = []
    for item in tqdm(data):
        id = item['id']
        if id not in path_dic.keys() or id not in cot_dic.keys():
            continue
        hints = path_dic[id]
        cot = cot_dic[id]
        for hint in hints:
            if dataset.startswith('prontoqa'):
                input = item['question'] + f"\n# Hint:\nThe reasoning should contains the statement: {hint}\n# Reasoning: {cot}\n# Answer:\n"
            else:
                input = item['question'] + f"\n# Hint:\nYou should focus on: {hint}\n# Reasoning: {cot}\n# Answer:\n"
            inputs = tokenizer(input, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model_wrapper.device)  
            outputs = model.generate(input_ids, max_new_tokens=max_new_tokens)
            response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True).split(split_token)[0]
            answer = cot + response
            if 'reason' in item.keys():
                reason = item['reason']
            else:
                reason = None 
            msg = {'id':item['id'], 'question':item['raw_question'], 'hint':hint, 'response':response, 'answer':answer, 'reason':reason, 'label':item['answer']}
            result.append(msg)
    result_dir = f'../result/{dataset}/{model_name}/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_path = os.path.join(result_dir, f'attr_sr_e{n_examples}_s{n_samples}.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=4)
        f.close()