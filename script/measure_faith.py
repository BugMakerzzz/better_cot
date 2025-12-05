import argparse
import numpy as np
import re
from utils.model import ModelWrapper
from utils.load_data import DataLoader, load_json_data, write_json_data
from transformers import set_seed
from tqdm import tqdm
from draw_info_fig import cal_data_bin_means
from scipy.stats import spearmanr

import numpy as np


def cal_ig(model, question, cot, pred):
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
    cot_input = input + [{"role": "assistant", "content": cot}]
    cot_input = model.tokenizer.apply_chat_template(cot_input, tokenize=False, add_generation_prompt=False)
    cond_ent = model.cal_cond_entropy(cot_input, cot)
    cot_ent = model.cal_entropy(cot)
    return cot_ent, cond_ent, cot_ent - cond_ent

def calculate_aoc(x_coords, y_coords):
    total_area = 0.0
    a = y_coords[-1]
    # 遍历每个折线段 (x1, y1) 到 (x2, y2)
    for i in range(len(x_coords) - 1):
        x1, y1 = x_coords[i], y_coords[i]
        x2, y2 = x_coords[i + 1], y_coords[i + 1]

        # 判断两个点与水平线 y=a 的相对位置
        if y1 < a and y2 < a:
            # 整段都在水平线下，直接计算梯形面积
            area = 0.5 * (y1 + y2 - 2 * a) * (x2 - x1)
            total_area += abs(area)
        elif y1 > a and y2 < a:
            # 第一个点在线上方，第二个点在线下方
            # 计算交点，并计算下方的部分面积
            x_intersect = x1 + (x2 - x1) * (y1 - a) / (y1 - y2)
            area = 0.5 * (y2 - a) * (x2 - x_intersect)
            total_area += abs(area)
        elif y1 < a and y2 > a:
            # 第一个点在线下方，第二个点在线上方
            # 计算交点，并计算下方的部分面积
            x_intersect = x1 + (x2 - x1) * (a - y1) / (y2 - y1)
            area = 0.5 * (y1 - a) * (x_intersect - x1)
            total_area += abs(area)
        # 如果 y1 == a 或 y2 == a，则在水平线上，不贡献面积
    
    return total_area / a



def cal_pred_logits(model, question, cot, pred):
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
        input += [{"role": "assistant", "content": cot}]
        if model.is_mistral or model.is_gemma:
            input[0]['content'] = sessions[0] + '\n' + input[0]['content']
        input = model.tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=False)
        text = '<'.join(input.split('<')[:-1]).rstrip() + '\n# Answer:\nThe answer is: ' + pred
        logits = model.cal_logits(text, pred)
        # prefix, pred = answer.split(': ')
        # input = input + {"role": "assistant", "content": cot + '\n# Answer:\n' + prefix + ': '}
        # ref = pred.strip().rstrip('.')
    return logits

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Llama3_1_8b_chat')
    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--n_examples', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='prontoqa')
    parser.add_argument('--method', type=str, default='early')
    parser.add_argument('--golden', action='store_true')
    parser.add_argument('--bridge', type=int, default=None)
    parser.add_argument('--baseline', type=str, default='cot')
    args = parser.parse_args()
    set_seed(17)

    model_name = args.model
    n_samples = args.n_samples
    n_examples = args.n_examples
    dataset = args.dataset 
    method = args.method
    golden = args.golden
    bridge = args.bridge
    baseline = args.baseline
    
    
    if baseline == 'bridge':
        result_path = f'../result/{dataset}/{model_name}/bridge3_sc3_w1_e3_500.json'
    else:
        result_path = f'../result/{dataset}/{model_name}/{baseline}_e{n_examples}_500.json'
    results = load_json_data(result_path)[:-1]
    
    if method == 'mail':
        mail_dic = {}
        if golden:
            path = f'../result/{dataset}/{model_name}/cans_info_e3_200_golden.json' 
        else:
            path = f'../result/{dataset}/{model_name}/cans_info_e3_200.json'
        data = load_json_data(path)
        for item in data:
            if not item['scores']:
                continue
            if item['input'][0] == '#':
                if model_name.startswith('Llama') or model_name.startswith('Gemma'):  
                    score = item['scores'][4:]
                else:
                    score = item['scores'][6:]
            else:
                score = item['scores']
            
            # score = [s for s in score]
            score = cal_data_bin_means(score, num_bins=len(score)//3)

            score, _ = spearmanr(score, range(len(score)))
            if np.isnan(score):
                continue
            mail_dic[item['id']] = score
    else:
        model = ModelWrapper(model_name)

    dataloader = DataLoader(dataset=dataset, n_samples=10000)
    data = dataloader.load_data(method='cot', n_examples=3)
    question_dic = {item['id']:item['question'] for item in data}
        
    
    score_results = []  
    for item in tqdm(results[:n_samples]):
        if item['answer'] == 'None' or not item['answer']:
            continue
        question = question_dic[item['id']]
        if golden:
            cot = '# Reasoning:\n' + item['reason']
            item['answer'] = item['label']
        else:
            if isinstance(item['response'], list):
                if isinstance(item['response'][0], dict):
                    pred = item['pred']
                    responses = [item['response'][i] for i in range(len(item['answer'])) if item['answer'][i] == pred]
                    response = max(responses, key=lambda x: x["score"])['content']
                    item['answer'] = pred
                else:
                    final_ans = max(item['answer'],key=item['answer'].count)
                    item['answer'] = final_ans
                    # if final_ans != item['label']:
                    #     continue
                    response = [item['response'][i] for i in range(len(item['answer'])) if item['answer'][i] == final_ans][0]
            else:
            # if not item['cor_flag']:
            #     continue
                response = item['response']
            if '# Answer:' in response:
                cot = response.split('# Answer:')[0].split('# Reasoning:')[-1]
            else:
                cot = ('\n\n').join(response.split('\n\n')[:-1]).split('# Reasoning:')[-1]
        
        if method ==  'early':
            cots = re.split(r'[\.|\n]', cot)
            cot_chunks = [chunk.strip() for chunk in cots if len(chunk) >= 3]
            if len(cot_chunks) < 2:
                continue
            logits = []
            cot = ""
            for chunk in cot_chunks:
                cot += chunk + '\n'
                logit = cal_pred_logits(model, question, cot, item['answer'])
                logits.append(logit)
            x_ticks = np.linspace(0, 1, len(cot_chunks))
            score = calculate_aoc(x_ticks, logits)
            
            msg = {'id':item['id'],
                    'question':item['question'],   
                    'cor_flag':item['cor_flag'], 
                    'cots':cot_chunks,
                    'logits':logits,
                    'scores':score}
        elif method == 'ig':
            d_ent, c_ent, score = cal_ig(model, question, cot, item['answer'])
            msg = {'id':item['id'],
                'question':item['question'],   
                'cor_flag':item['cor_flag'], 
                'd_ent':d_ent,
                'c_ent':c_ent,
                'scores':score}
        elif method == 'mail':
            if item['id'] not in mail_dic.keys():
                continue
            msg = {'id':item['id'],
                'question':item['question'],   
                'cor_flag':item['cor_flag'], 
                'scores':mail_dic[item['id']]}
        else:
            questions = question.split('# Reasoning:')
            questions[-2] = questions[-2] + f"\nI think the answer can not be {item['answer']}."
            disturb_question = ('# Reasoning:').join(questions) 
            logit = cal_pred_logits(model, question, cot, item['answer'])
            disturb_logit = cal_pred_logits(model, disturb_question, cot, item['answer'])
            score = min(1.0, abs((logit - disturb_logit) / logit))
            
            msg = {'id':item['id'],
                    'question':item['question'], 
                    'disturb_question':disturb_question,
                    'logit':logit,
                    'disturb_logit':disturb_logit,
                    'cot':cot,  
                    'cor_flag':item['cor_flag'], 
                    'scores':score}    
        score_results.append(msg)
    
    if golden:
        score_path = f'../result/{dataset}/{model_name}/{method}_faith_{n_samples}_golden.json' 
    elif bridge:
        score_path = f'../result/{dataset}/{model_name}/{method}_faith_{n_samples}_bridge{bridge}.json' 
    else:
        score_path = f'../result/{dataset}/{model_name}/{baseline}_{method}_faith_{n_samples}.json' 
    

    write_json_data(score_path, score_results)