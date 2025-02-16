import argparse
import torch
import json
import numpy as np
import re
from utils.model import ModelWrapper
from utils.load_data import DataLoader
from transformers import set_seed
from tqdm import tqdm
from utils.load_data import extract_answer
from measure_faith import cal_pred_logits, calculate_aoc
from measure_info import prepare_idx
from draw_info_fig import cal_data_bin_means
from scipy.stats import spearmanr

def cal_early_score(res):
    answer = extract_answer(dataset, res)
    if '# Answer:' in res:
        cot = res.split('# Answer:')[0]
    else:
        cot = ('\n\n').join(res.split('\n\n')[:-1])
     
    cots = re.split(r'[\.|\n]', cot)
    cot_chunks = [chunk.strip() for chunk in cots if len(chunk) >= 3]

    logits = []
    cot = ""
    for chunk in cot_chunks:
        cot += chunk + '\n'
        logit = cal_pred_logits(model, question, cot, answer)
        logits.append(logit)
    x_ticks = np.linspace(0, 1, len(cot_chunks))
    score = calculate_aoc(x_ticks, logits)
    print(logits)
    return score 

def cal_mail_score(prefix, res):
    answer = extract_answer(dataset, res)
    if '# Answer:' in res:
        cot = res.split('# Answer:')[0]
    elif '\n\n' in res:
        cot = ('\n\n').join(res.split('\n\n')[:-1])
    else:
        cot = res
    pred = answer
    input = prefix
    
    input += [{"role": "assistant", "content": cot}]
    input = model.tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=False)
    input = '<'.join(input.split('<')[:-1]) + '\n# Answer:\nThe answer is: '
    target = pred
        
    scores = model.input_explain(input, target).mean(-1)
    inps = model.tokenizer.tokenize(input)
    

    start_idx, end_idx = prepare_idx(model, 'cans', inps)
    
    
    inps = inps[start_idx:end_idx]
    score = scores[start_idx+1:end_idx+1].tolist()
    score = cal_data_bin_means(score, num_bins=len(score)//3)

    score, _ = spearmanr(score, range(len(score)))
    return score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Llama3_1_8b_chat')
    parser.add_argument('--dataset', type=str, default='proofwriter')
    args = parser.parse_args()
    set_seed(17)

    model_name = args.model
    dataset = args.dataset 

    
    model = ModelWrapper(model_name)
    dataloader = DataLoader(dataset=dataset, n_samples=200)
    data = dataloader.load_data(method='cot', n_examples=3)
    question_dic = {item['id']:item['question'] for item in data}
     
    id = "Pronto_QT61"

    query = "# Question:\nJompuses are shumpuses. Jompuses are zumpuses. Each zumpus is sour. Each jompus is not floral. Sterpuses are rainy. Every sterpus is a wumpus. Sally is a jompus. Sally is a sterpus. True or false: Sally is not floral.",
        # "response": "# Reasoning:\nSally is a jompus. Each jompus is not floral. Sally is not floral.\n# Answer:\nThe answer is: False.",
    # cot = "# Reasoning:\nFirst, we need to convert the length from yards to feet.\n"
    # label = "98"
    question = question_dic[id]
    if model.is_chat:
        sessions = question.split('####')
        if model.is_mistral or model.is_gemma:
            input = []
        else:
            input = [{"role": "system", "content": sessions[0]}]
        for session in sessions[1:]:
            user_content, assistant_content = session.split('# Reasoning:')
            assistant_content = '# Reasoning:' + assistant_content
            input += [{"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content}]
        input = input[:-1]
        input[-1] = {"role": "user", "content": query}
        # input += [{"role": "assistant", "content": cot}]
        if model.is_mistral or model.is_gemma:
            input[0]['content'] = sessions[0] + '\n' + input[0]['content']
        # input = model.tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
        # text = '<'.join(input.split('<')[:-1]).rstrip() + '\n# Answer:\nThe answer is: '
        # text = model.tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)

        # input = model.tokenizer(text, return_tensors="pt")['input_ids'].to(model.device)
        res = model.generate(input)
        
        
        # res = model.tokenizer.decode(output[0], skip_special_tokens=True)
        # self.model.generate(inputs, max_new_tokens=500, do_sample=False)
        print(res)
        # score = cal_early_score(res)
        score = cal_mail_score(input, res)
        print(score)
