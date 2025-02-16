import argparse
import json
import re 
from utils.model import ModelWrapper
from utils.load_data import DataLoader
from transformers import set_seed
from tqdm import tqdm
from utils.config import datasets_with_options

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

def cal_info(task, model, item):
    if item['answer'] == 'None' or not item['answer'] or len(item['answer']) >= 20:
        return None, None 
    question = question_dic[item['id']]
    if golden:
        cot = '# Reasoning:\n' + item['reason']
        pred = item['label']
    else:
        if '# Answer:' in item['response']:
            cot = item['response'].split('# Answer:')[0]
        elif '\n\n' in item['response']:
            cot = ('\n\n').join(item['response'].split('\n\n')[:-1])
        else:
            cot = item['response']
        pred = item['answer']
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
    if model.is_mistral or model.is_gemma:
        input[0]['content'] = sessions[0] + '\n' + input[0]['content']
    
    if task in ['qcot', 'ccot']:
        cots = re.split(r'[\.|\n]', cot)
        cot_chunks = [chunk.strip() for chunk in cots if len(chunk) >= 3]
        cot = ""
        inps = []
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
                start_idx, end_idx = prepare_idx(model, task, inp, option_flg=True)
            else:
                start_idx, end_idx = prepare_idx(model, task, inp)
            
            inps.append(inp[start_idx:end_idx])
            # print(score[start_idx+1:end_idx+1])
            scores.append(score[start_idx+1:end_idx+1].tolist())
            cot += chunk + '. '
    else:
        input += [{"role": "assistant", "content": cot}]
        input = model.tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=False)
        input = '<'.join(input.split('<')[:-1]) + '\n# Answer:\nThe answer is: '
        target = pred
            
        scores = model.input_explain(input, target).mean(-1)
        inps = model.tokenizer.tokenize(input)
        
        if dataset in datasets_with_options:
            start_idx, end_idx = prepare_idx(model, task, inps, option_flg=True)
        else:
            start_idx, end_idx = prepare_idx(model, task, inps)
        
        
        inps = inps[start_idx:end_idx]
        scores = scores[start_idx+1:end_idx+1].tolist()

    return inps, scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Llama3_1_8b_chat')
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--n_examples', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='prontoqa')
    parser.add_argument('--task', type=str, default='qcot')
    parser.add_argument('--golden', action='store_true')
    parser.add_argument('--bridge', action='store_true')
    args = parser.parse_args()
    set_seed(17)

    model_name = args.model
    n_samples = args.n_samples
    n_examples = args.n_examples
    dataset = args.dataset 
    task = args.task
    golden = args.golden
    bridge = args.bridge
    
    result_path = f'../result/{dataset}/{model_name}/cot_e{n_examples}_{n_samples}.json'
    if bridge:
        result_path = f'../result/{dataset}/{model_name}/bridge3_e{n_examples}_500.json'
    with open(result_path, 'r') as f:
        results = json.load(f)[:-1]
    
    
    model = ModelWrapper(model_name)
    dataloader = DataLoader(dataset=dataset, n_samples=500)
    data = dataloader.load_data(method='cot', n_examples=3)
    question_dic = {item['id']:item['question'] for item in data}
    
    score_results = []  
    for item in tqdm(results[:n_samples]):
        if bridge:
            inps = {}
            scores = {}
            for i in range(len(item['response'])):
                tup = {}
                for k, v in item.items():
                    if isinstance(v, list):
                        tup[k] = v[i]
                    else:
                        tup[k] = v
                inp, score = cal_info(task, model, tup)
                inps[i] = inp
                scores[i] = score          
        else:
            inps, scores = cal_info(task, model, item)         
      
        if not inps or not scores:
            continue  
         
        msg = {'id':item['id'],
            'question':item['question'],
            'input':inps,
            'cor_flag':item['cor_flag'], 
            'scores':scores}
        score_results.append(msg)
    
    if golden:
        score_path = f'../result/{dataset}/{model_name}/{task}_info_e{n_examples}_{n_samples}_golden.json' 
    elif bridge:
        score_path = f'../result/{dataset}/{model_name}/{task}_info_e{n_examples}_{n_samples}_bridge3.json'
    else:
        score_path = f'../result/{dataset}/{model_name}/{task}_info_e{n_examples}_{n_samples}.json' 
   
    
    with open(score_path, 'w') as f:
        json.dump(score_results, f, indent=4)