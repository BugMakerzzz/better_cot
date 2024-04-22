import re
import json 
import random
from config import *


def add_bias_sentence(prompt, bias_sentence):
    pattern = r"(#|##) Reasoning"

    matches = list(re.finditer(pattern, prompt))

    if not matches:
        return prompt+bias_sentence+'\n'

    # 获取最后一个匹配项的位置
    last_match = matches[-1].start()
    return prompt[:last_match] + bias_sentence+ '\n' + prompt[last_match:]

def load_prompt(dataset, method, n_examples):
    # load prompt file
    prompt_file = f'../prompts/{method}/{dataset}_{n_examples}.txt'
    with open(prompt_file, 'r') as fin:
        lines = [line.strip() for line in fin.readlines()]
    full_prompt = '\n'.join(lines)


    return full_prompt

def format_prompt(full_prompt, item):
    fields = re.findall('\{\{\w+\}\}', full_prompt)
    for field in fields:
        if field[2:-2] not in item.keys():
            continue
        value = item[field[2:-2]]
        
        if type(value) == list:
            value = '\n'.join(value)
        
        full_prompt = full_prompt.replace(field, value)
         
    return full_prompt


def load_dataset(dataset, nsamples, mode='dev'):
    data_file = f'../data/{dataset}/{mode}.jsonl'
    with open(data_file, 'r') as fin:
        items = [json.loads(line) for line in fin]
    for idx, item in enumerate(items):
        if dataset in ['gsm8k', 'gsmic']:
            question = item['question']
            parts = item['answer'].split('####')
            item.clear()
            item['id'] = f'{dataset.upper()}_Q{idx}'
            item['question'] = question
            item['reason'] = parts[0].strip()
            item['answer'] = str(int(parts[1].strip().replace(',', '')))
        elif dataset == 'csqa':
            question = item['question']['stem']
            label = str(ord(item['answerKey']) - ord("A") + 1)
            options = []
            for i in range(len(item['question']['choices'])):
                tup = item['question']['choices'][i]
                options.append(tup['text'])
            item['answer'] = label
            item['options'] = options
        elif dataset == 'wino':
            question = item['sentence']
            label = item['answer']
            options = [item['option1'], item['option2']]      
            item['answer'] = label
            item['options'] = options
    random.shuffle(items)
    return items[:nsamples] if nsamples > 0 else items


def extract_logic(answer):
    pattern1 = r"correct \w+ is:?\s*([A-D])"
    pattern2 = r"correct option is: (true|false|unknown)"
    pattern3 = r"([A-C])\)\s*(True|False|Unknown)"
    pattern4 = r"([A-D])\) "
    pattern5 = r"^[A-D]\.?$"

    match = re.search(pattern1, answer)
    option = None
    # extract pattern
    if match:
        option = match.group(1)
    
    if not option:
        match = re.search(pattern2, answer, re.IGNORECASE)
        if match:
            word_to_option = {"true": "A", "false": "B", "unknown": "C"}
            option = word_to_option.get(match.group(1).lower())

    if not option:
        match = re.search(pattern3, answer, re.IGNORECASE)
        if match:
            option = match.group(1)
    if not option and len(answer)<16:
        if 'true' in answer.lower():
            option = 'A'
        elif 'false' in answer.lower():
            option = 'B'
        elif 'unknown' in answer.lower():
            option = 'C'
    if not option:
        match = re.match(pattern4, answer)
        if match:
            option = match.group(1)
    if not option:
        match = re.match(pattern5, answer)
        if match:
            option = match.group(0) 
    if not option:
        option = None
        # wrong_data.append(d)
    return option

def extract_answer(dataset, output):

    if dataset in ['gsm8k', 'addition', 'product', 'gsmic']:
        answer = output.replace(',', '')  # remove middle ',' from numbers like '1,234'
        answer = re.findall('\d+', answer)
        if len(answer) == 0:
            return 'None'
        answer = answer[-1]
        answer = answer.strip()
        return str(int(answer))  # expect integer only
    elif dataset == 'proofwriter':
        answer = extract_logic(output)
        return str(answer)
    elif dataset == 'logiqa':
        answer = extract_logic(output)
        return str(answer)
    elif dataset == 'folio':
        answer = extract_logic(output)
        return str(answer)
    else:
        return output 

class DataLoader(object):
    def __init__(self, dataset, n_samples) -> None:
        self.dataset = dataset
        self.n_samples = n_samples
        
    def load_data(self, method, n_examples):
        data = load_dataset(self.dataset, self.n_samples)
        prompt = load_prompt(self.dataset, method, n_examples)
        for item in data:
            if 'question' in item.keys():
                item['raw_question'] = item['question']
            else:
                item['raw_question'] = f"{item['number1']} {item['number2']}"
            if 'context' in item.keys():
                item['raw_question'] = item['context'] + ' ' + item['raw_question']
            if 'options' in item.keys():
                item['raw_question'] = item['raw_question'] + ' ' + ' '.join(item['options'])
            item['question'] = format_prompt(prompt, item)
        return data 
    
    