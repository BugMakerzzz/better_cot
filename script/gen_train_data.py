import argparse
import os
import json
from utils.model import ModelWrapper
from utils.load_data import DataLoader, extract_answer
from transformers import set_seed
from tqdm import tqdm
from measure_info import prepare_idx
from scipy.stats import spearmanr
from draw_info_fig import cal_data_bin_means
import re
from utils.config import datasets_with_options

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Llama3_1_8b_chat')
    parser.add_argument('--datalength', type=int, default=5000)
    parser.add_argument('--dataset', type=str, default='prontoqa')
    args = parser.parse_args()
    set_seed(17)

    model_name = args.model
    datalength = args.datalength 
    dataset = args.dataset 

    result_path = f'../data/{dataset}/{model_name}_train_{datalength}.json'
    model = ModelWrapper(model_name)
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            result = json.load(f)
            f.close()
        for item in tqdm(result):
            responses = item['responses']
            input = item['input']
    
            res_scores = []
            for i in range(3):
                print(item['answers'][i])
                if item['answers'][i] not in ['True', 'False', 'A', 'B', 'C']:
                    res_scores.append([])
                    continue
                res = responses[i]
                if '# Answer:' in res:
                    cot = res.split('# Answer:')[0]
                elif '\n\n' in res:
                    cot = ('\n\n').join(res.split('\n\n')[:-1])
                else:
                    cot = res
                cots = re.split(r'[\.|\n]', cot)
                cot_chunks = [chunk.strip() for chunk in cots if len(chunk) >= 3]
                cot = ""
                scores = []
                for chunk in cot_chunks:
                    if cot:
                        content = input + [{"role": "assistant", "content": cot}]
                    else:
                        content = input 
                    inputs = model.tokenizer.apply_chat_template(content, tokenize=False, add_generation_prompt=False)
                    target = chunk
            
                    score = model.input_explain(inputs, target).mean(-1)
                    inp = model.tokenizer.tokenize(inputs)
                    
                    if dataset in datasets_with_options:
                        start_idx, end_idx = prepare_idx(model, 'qcot', inp, option_flg=True)
                    else:
                        start_idx, end_idx = prepare_idx(model, 'qcot', inp)
                    
                    scores.append(score[start_idx+1:end_idx+1].tolist())
                    cot += chunk + '. '
                res_scores.append(scores)
                print(res_scores)
            item['qcot_info'] = res_scores
    else:
        dataloader = DataLoader(dataset=dataset, n_samples=datalength)
        data = dataloader.load_data(method='cot', n_examples=3, mode='train')

        result = []
        split_str = '# Reasoning:'
            
        for item in tqdm(data):
            if model.is_chat:
                sessions = item['question'].split('####')
                if model.is_mistral or model.is_gemma:
                    inputs = []
                else:
                    inputs = [{"role": "system", "content": sessions[0]}]
                for session in sessions[1:]:
                    user_content, assistant_content = session.split(split_str)
                    assistant_content = split_str + assistant_content
                    inputs += [{"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content}]
                inputs = inputs[:-1]
                if model.is_mistral or model.is_gemma:
                    inputs[0]['content'] = sessions[0] + '\n' + inputs[0]['content']
            else:
                inputs = item['question']
            # print(inputs)
            responses = model.generate(inputs, sample_cnt=3)
            sp_scores = []
            answers = []
            cor_flgs = []
            for res in responses:
                answer = extract_answer(dataset, res)
                if not answer:
                    continue
                if '# Answer:' in res:
                    cot = res.split('# Answer:')[0]
                elif '\n\n' in res:
                    cot = ('\n\n').join(res.split('\n\n')[:-1])

                prefix = inputs + [{"role": "assistant", "content": cot}]
                prefix = model.tokenizer.apply_chat_template(prefix, tokenize=False, add_generation_prompt=False)
                prefix = '<'.join(prefix.split('<')[:-1]) + '\n# Answer:\nThe answer is: '
                target = answer 
                if_score = model.input_explain(prefix, target).mean(-1)
                inps = model.tokenizer.tokenize(prefix)
                start_idx, end_idx = prepare_idx(model, 'cans', inps)
                if_score = if_score[start_idx+1:end_idx+1].tolist()
                if_score = cal_data_bin_means(if_score, num_bins=20)
                sp_score, _ = spearmanr(if_score, range(len(if_score)))
                
                answers.append(answer)
                sp_scores.append(sp_score)
                cor_flgs.append(answer == item['answer'])
            
            msg = {
                'id':item['id'],
                'question':item['raw_question'],
                'input':inputs,
                'responses':responses,
                'answers':answers,
                'label':item['answer'],
                'cor_flgs':cor_flgs,
                'sp_scores':sp_scores,
            }
            result.append(msg)
                
    
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=4)
        f.close()