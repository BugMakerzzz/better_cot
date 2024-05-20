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
    data = dataloader.load_data(method='cot', n_examples=n_examples)
    
    pans_path_file = f'../result/{dataset}/{model_name}/{method}_{proxy}_pans_paths_e{n_examples}_s{n_samples}.json'

    # if not os.path.exists(path_file):
    #     if method == 'direct':
    #         path_file = f'../result/{dataset}/{model_name}/input_pans_paths_e{n_examples}_s{n_samples}_direct.json'
    #     else:
    #         path_file = f'../result/{dataset}/{model_name}/input_pans_paths_e{n_examples}_s{n_samples}.json'
    
    with open(pans_path_file, 'r') as f:
        pans_path_data = json.load(f)
        f.close()

    path_dic = {}
    for item in pans_path_data:
        attrs = item['path'][-1]['inp_attr'][:3]
        sents = [x['inp'] for x in attrs]
        path_dic[item['id']] = sents

    if model_name.startswith('gpt'):
        new_data = []
        for item in data:
            id = item['id']
            if id not in path_dic.keys():
                continue
            hints = path_dic[id]
            question = item['question'].copy()
            for hint in hints:
                input = question + f"\n# Hint:\nYou should focus on: {hint}\n# Reasoning:\n"
                item['question'] = input 
                item['hint'] = hint
                new_data.append(item)
        result = gpt_reason(data, model_name, 'attr_cot', dataset)
    else:
        model_wrapper = ModelWrapper(model_name)
        model = model_wrapper.model
        tokenizer = model_wrapper.tokenizer
        max_new_tokens = 200
        split_token = '####'
        result = []
        correct = 0
        cnt = 0
        for item in tqdm(data):
            id = item['id']
            if id not in path_dic.keys():
                continue
            hints = path_dic[id]
            for hint in hints:
            # for sent in sents:
                # if sent in cot:
                    # continue
                # input = item['question'] + sent
                input = item['question'] + f"\n# Hint:\nYou should focus on: {hint}.\n# Reasoning:\n"
                inputs = tokenizer(input, return_tensors="pt")
                input_ids = inputs["input_ids"].to(model_wrapper.device)  
                outputs = model.generate(input_ids, max_new_tokens=max_new_tokens)
                response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True).split(split_token)[0]
                # response = sent + response
                answer = extract_answer(dataset, response)
                cor_flag = False
                if answer == item['answer']:
                    cor_flag = True
                    correct += 1
                if 'reason' in item.keys():
                    reason = item['reason']
                else:
                    reason = None 
                msg = {'id':item['id'], 'question':item['raw_question'], 'hint':hint, 'response':response, 'answer':answer, 'reason':reason, 'label':item['answer'], 'cor_flag':cor_flag}
                # msg = {'id':item['id'], 'question':item['raw_question'], 'response':response, 'answer':answer, 'reason':reason, 'label':item['answer'], 'cor_flag':cor_flag}
                result.append(msg)
                cnt += 1
        result.append({'acc': correct / n_samples})
    result_dir = f'../result/{dataset}/{model_name}/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_path = os.path.join(result_dir, f'attr_cot_e{n_examples}_s{n_samples}.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=4)
        f.close()