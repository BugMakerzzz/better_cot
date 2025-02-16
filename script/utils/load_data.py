import re
import json 
import random
from .config import *


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

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval



def load_dataset(dataset, nsamples, mode='dev'):
    data_file = f'../data/{dataset}/{mode}.jsonl'
    with open(data_file, 'r') as fin:
        items = [json.loads(line) for line in fin]
    for idx, item in enumerate(items):
        if 'id' not in item.keys():
            item['id'] = f'{dataset.upper()}_Q{idx}'
        if dataset in ['gsm8k', 'gsmic', 'gsmp']:
            question = item['question']
            parts = item['answer'].split('####')
            item.clear()
            item['id'] = f'{dataset.upper()}_Q{idx}'
            item['question'] = question
            item['reason'] = parts[0].strip()
            item['answer'] = str(eval(parts[1].strip().replace(',', '')))
        elif dataset == 'aqua':
            item['answer'] = item['correct']
            item['reason'] = item['rationale'] 
        elif dataset == 'math':
            item['question'] = item['problem']
            item['reason'] = item['solution']
            item['answer'] = remove_boxed(last_boxed_only_string(item['solution']))
        elif dataset == 'csqa':
            question = item['question']['stem']
            label = item['answerKey']
            options = [f"({tup['label']}) {tup['text']}" for tup in item['question']['choices']]
            item['question'] = question  
            item['id'] = f'{dataset.upper()}_Q{idx}'
            item['answer'] = label
            item['options'] = options
        elif dataset == 'wino':
            question = item['sentence']
            label_dic = {'1':'A', '2':'B'}
            label = label_dic[item['answer']]
            options = [f"(A) {item['option1']}", f"(B) {item['option2']}"]  
            item['question'] = question  
            item['answer'] = label
            item['options'] = options
        elif dataset == 'siqa':
            question = item['context'] + ' ' + item['question']
            item['question'] = question 
    random.shuffle(items)
    return items[:nsamples] if nsamples > 0 else items




def extract_logic(answer):
    pattern1 = r"The answer is:?\s*([A-E])"
    pattern2 = r"The answer is:?\s*(True|False|Unknown)"
    # pattern3 = r"([A-E])"
    pattern4 = r"(True|False|Unknown)"

    option = None 
    match = re.search(pattern1, answer, re.IGNORECASE)
    if match:
        option = match.group(1)
    
    if not option:
        match = re.search(pattern2, answer, re.IGNORECASE)
        if match:
            option = match.group(1)

    # if not option:
    #     match = re.findall(pattern3, answer)
    #     if match:
    #         option = match[-1]
    
            
    if not option:
        match = re.search(pattern4, answer, re.IGNORECASE)
        if match:
            option = match.group(1)
            
        # wrong_data.append(d)
    # print(option)
    return option

def extract_answer(dataset, output, method=None):
    if dataset in ['gsm8k', 'addition', 'product', 'gsmic', 'gsmp']:
        answer = output.replace(',', '')  # remove middle ',' from numbers like '1,234'
        answer = re.findall(r'\d*\.?\d+', answer)
        if len(answer) == 0:
            return 'None'
        if method:
            answer = answer[0]
        else:
            answer = answer[-1]
        answer = answer.strip()
        return str(eval(answer)) 
    elif dataset in ['lastletter', 'coinflip']:
        answer = output.split(':')[-1]
        matches = re.findall(r'[a-zA-Z]+', answer)
        if not matches:
            return None 
        else:
            answer = matches[0]
        return answer
    elif dataset == 'math':
        if '# Answer:' in output:
            answer = output.split(':')[-1]
            answer = answer.strip().strip('.').strip()
        else:
            match_string = last_boxed_only_string(output)
            if match_string:
                answer = remove_boxed(match_string)
            else:
                answer = None 
                
        return answer
    else:
        answer = extract_logic(output)
        return answer



class DataLoader(object):
    def __init__(self, dataset, n_samples) -> None:
        self.dataset = dataset
        self.n_samples = n_samples
        
    def load_data(self, method, n_examples, mode='dev'):
        data = load_dataset(self.dataset, self.n_samples, mode=mode)
        prompt = load_prompt(self.dataset, method, n_examples)
        for item in data:
            item['raw_question'] = item['question']
            item['question_target'] = item['question']
            if 'context' in item.keys():
                item['raw_question'] = item['context'] + ' ' + item['raw_question']
            if 'options' in item.keys():
                item['raw_question'] = item['raw_question'] + ' ' + ' '.join(item['options'])
                item['question_target'] = item['question_target'] + ' ' + ' '.join(item['options'])
            item['question'] = format_prompt(prompt, item)
        return data 
    
    def reformat_question(self, item, method, n_examples):
        prompt = load_prompt(self.dataset, method, n_examples)
        item['question'] = item['question_target']
        question = format_prompt(prompt, item)
        return question


 
class InterventionData():
    def __init__(self, msg, tokenizer) -> None:
        self.question = None 
        self.cot = None 
        self.pred = None 
        self.load_data(msg)
        
        self.question_end = None
        self.cot_end = None 
        self.cot_input_ids = None 
        self.pred_ids = None
        self.tokenize_data(tokenizer)
        
        self.cot_intervention_idx = {}
        self.get_intervention_idx()

        return 
    
    def load_data(self, msg):
        self.question = msg['question']
        self.cot = msg['response'].split('\n# Answer:')[0]
        self.pred = msg['answer']

        return 
    
    
    def tokenize_data(self, tokenizer):

        cot_input = self.question + self.cot + f'\n# Answer:\nThe answer is: {self.pred}'
        self.question_end = len(tokenizer(self.question, return_tensors="pt").input_ids[0])
        self.cot_end = len(tokenizer(self.question + self.cot, return_tensors="pt").input_ids[0])
        self.cot_input_ids = tokenizer(cot_input, return_tensors="pt").input_ids
        pred_len = len(tokenizer(self.pred, return_tensors="pt").input_ids[0])
        self.pred_ids = self.cot_input_ids[:,-pred_len]
            
        return 
        
        
    def get_intervention_idx(self):
        interval_length = self.cot_end - self.question_end
        if interval_length == 0:
            interval_length += 1
   
        start = self.question_end
        for cnt in range(1, 11):
            end = interval_length * cnt // 10 + self.question_end
            if end == start:
                self.cot_intervention_idx[cnt] = list(range(start, start + 1)) 
            else:
                self.cot_intervention_idx[cnt] = list(range(start, end))
            start = end
        return 
    
def load_json_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
        f.close()
    return data

def write_json_data(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
        f.close()
    return 