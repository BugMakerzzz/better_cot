import random
import numpy as np
import re 
random.seed(17)
from utils.config import datasets_with_options
from utils.load_data import extract_answer, load_prompt, format_prompt
from draw_info_fig import cal_data_bin_means
from scipy.stats import spearmanr


def prepare_idx(model, task, inps, option_flg=False):
    split_token = model.split_token
    start_bias = model.start_bias
    end_bias = model.end_bias
    if task == 'cans':
        start_idx = [i for i, v in enumerate(inps) if v == split_token][-1] + start_bias 
        end_idx = [i for i, v in enumerate(inps) if v == "#"][-1]    
    elif task in ['qans', 'qcot']:
        start_idx = [i for i, v in enumerate(inps) if v == split_token][-2] + start_bias 
        end_idx = [i for i, v in enumerate(inps) if v == split_token][-1] - end_bias
        if option_flg:
            end_idx = [i for i, v in enumerate(inps[:end_idx]) if v == "#"][-1]
    else:
        start_idx = [i for i, v in enumerate(inps) if v == split_token][-1] + start_bias 
        end_idx = len(inps)

    return start_idx, end_idx



def recall_key_statements(model, item, dataset, topk):
    input = item['question']
    
    if '# Answer:' in item['response']:
        cot = item['response'].split('# Answer:')[0]
    elif 'answer is:' in item['response']:
        cot = item['response'].split('# answer is:')[0]
    elif '\n\n' in item['response']:
        cot = ('\n\n').join(item['response'].split('\n\n')[:-1])
    else:
        cot = item['response']
    pred = item['answer']
    input += [{"role": "assistant", "content": cot}]
    input = model.tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=False)
    input = '<'.join(input.split('<')[:-1]) + '\n# Answer:\nThe answer is: '
    target = pred
    scores = model.input_explain(input, target).mean(-1)
    inps = model.tokenizer.tokenize(input)
    
    if dataset in datasets_with_options:
        start_idx, end_idx = prepare_idx(model, 'qans', inps, option_flg=True)
    else:
        start_idx, end_idx = prepare_idx(model, 'qans', inps)
    
    
    tokens = inps[start_idx:end_idx]
    scores = scores[start_idx+1:end_idx+1].tolist()
    stop_tokens = ['.\u010a', '.']
    # print(tokens)
    if model.is_llama:
        start_idx = [i for i, v in enumerate(tokens) if v == ':\u010a'][0] + 1
    elif model.is_qwen:
        start_idx = [i for i, v in enumerate(tokens) if v == ':Ċ'][0] + 1
    else:
        start_idx = [i for i, v in enumerate(tokens) if v == ':'][0] + 1
    end_idx = [i for i, v in enumerate(tokens) if v == "#"][1]
    tokens = tokens[start_idx:end_idx]   
    scores = scores[start_idx:end_idx] 
    step_scores = {}
    start = 0
    for i in range(len(tokens)):
        token = tokens[i]
        if i == len(tokens)-1 or token in stop_tokens:
            end = i + 1
            step = model.tokenizer.decode(model.tokenizer.convert_tokens_to_ids(tokens[start:end]))
            step_scores[step.strip()] = np.mean(np.array(scores[start:end]))
            start = i + 1
    step_scores = sorted(step_scores.items(), key=lambda item: item[1], reverse=True)[:topk]
    key_statements = [item[0] for item in step_scores if len(item[0]) > 5] 
    return key_statements


def cal_mail_score(model, item, dataset):
    input = item['question']
    if '# Answer:' in item['response']:
        cot = item['response'].split('# Answer:')[0]
    elif '\n\n' in item['response']:
        cot = ('\n\n').join(item['response'].split('\n\n')[:-1])
    else:
        cot = item['response']
    
    pred = item['answer']
    input += [{"role": "assistant", "content": cot}]
    input = model.tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=False)
    input = '<'.join(input.split('<')[:-1]) + '\n# Answer:\nThe answer is: '
    target = pred
        
    scores = model.input_explain(input, target).mean(-1)
    inps = model.tokenizer.tokenize(input)
    
    if dataset in datasets_with_options:
        start_idx, end_idx = prepare_idx(model, 'cans', inps, option_flg=True)
    else:
        start_idx, end_idx = prepare_idx(model, 'cans', inps)
    
    
    inps = inps[start_idx:end_idx]
    scores = scores[start_idx+1:end_idx+1].tolist()
    if inps[0] == '#':
        if model.is_llama or model.is_gemma:  
            scores = scores[4:]
        else:
            scores = scores[6:]
        
    scores = cal_data_bin_means(scores, num_bins=len(scores)//3)
    score, _ = spearmanr(scores, range(len(scores)))

    return score 



def cal_ig_score(model, item):
    input = item['question']
    if '# Answer:' in item['response']:
        cot = item['response'].split('# Answer:')[0]
    else:
        cot = ('\n\n').join(item['response'].split('\n\n')[:-1])
    cot_input = input + [{"role": "assistant", "content": cot}]
    cot_input = model.tokenizer.apply_chat_template(cot_input, tokenize=False, add_generation_prompt=False)
    cond_ent = model.cal_cond_entropy(cot_input, cot)
    cot_ent = model.cal_entropy(cot)
    return cot_ent - cond_ent


def bridge_reason(model, inputs, data, dataset, topk=3, sc=3, random_sample=False, weighted=False, cache=False):
    if not cache:
        responses = model.generate(inputs, sample_cnt=sc)
        answers = [extract_answer(dataset, res) for res in responses]
        answer = [ans for ans in answers if ans]
        while not answer:
            responses = model.generate(inputs, sample_cnt=sc)
            answers = [extract_answer(dataset, res) for res in responses]
            answer = [ans for ans in answers if ans]
        answer = max(answer, key=answer.count)
        responses = [responses[i] for i in range(sc) if answers[i] == answer]
        # items = [{'question':inputs.copy(), 'response':res, 'answer':answer} for res in responses]
        item = {'question':inputs.copy(), 'response':responses[0], 'answer':answer} 
        responses = []
        hints = []
        prompt = load_prompt(dataset, 'bridge', 3)
        # item = 
        # for item in items:
        if random_sample:
            context = data['raw_question'].split('.')
            context = [inp for inp in context if len(inp) > 1][:-1]
            key_statements = random.sample(context, topk)
        else:
            key_statements = recall_key_statements(model, item, dataset, topk)

        input = format_prompt(prompt, data)
        sessions = input.split('####')
        if model.is_mistral or model.is_gemma or model.is_o1:
            input = []
        else:
            input = [{"role": "system", "content": sessions[0]}]
        for session in sessions[1:]:
            user_content, assistant_content = session.split('# Reasoning:')
            assistant_content = '# Reasoning:' + assistant_content
            input += [{"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content}]
        input = input[:-1]
        if model.is_mistral or model.is_gemma or model.is_o1:
            input[0]['content'] = sessions[0] + '\n' + input[0]['content']
        question_prompt = input[-1]['content']
        for hint in key_statements:
            hint = f'You should focus on: {hint}'
            question = question_prompt.format(hint=hint)
            # print(question)
            input[-1]['content'] = question 
            response = model.generate(input, sample_cnt=1)
            # responses.append(response)
            responses += response
            hints.append(hint)  
        answers = [extract_answer(dataset, res) for res in responses]
        corrects = [answer and answer.lower() == data['answer'].lower() for answer in answers]
        items = [{'question':inputs.copy(), 'response':responses[i], 'answer':answers[i]} for i in range(len(responses))]
        label = data['answer']
    else:
        items = [{'question':inputs.copy(), 'response':data['response'][i], 'answer':data['answer'][i]} for i in range(len(data['response']))]
        responses = data['response']
        answers = data['answer']
        corrects = data['corrects'] if 'corrects' in data else None 
        hints = data['hints'] if 'hints' in data else None 
        label = data['label']
        # items['question'] = inputs.copy()
        # print(items)
    if weighted:
        scores = [cal_ig_score(model, item) for item in items]
        responses = [{'content': res, 'score':score} for res, score in zip(responses, scores)]
        coef = {}
        for i in range(len(scores)):
            if not answers[i]:
                continue
            if answers[i] not in coef.keys():
                coef[answers[i]] = scores[i]
            else:
                coef[answers[i]] += scores[i]
        if not coef:
            pred = None 
        else:
            pred = max(coef, key=lambda x: coef[x])
    else:
        pred = max(answers, key=answers.count)
    if not pred:
        cor_flag = False
    else:
        if pred.lower() == label.lower():
            cor_flag = True
        else:
            cor_flag = False         
    return responses, answers, corrects, cor_flag, hints, pred