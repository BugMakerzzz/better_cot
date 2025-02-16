import argparse
import json 
import re
from utils.model import ModelWrapper
from tqdm import tqdm

instruction = """You are tasked with evaluating the semantic similarity between two given sentences. Please analyze how closely the meanings of these sentences align.
For the two sentences:
Sentence 1: [First sentence]
Sentence 2: [Second sentence]
Please provide your analysis in this structure:
Similarity Score: [0-1]
Justification: [Your explanation]
Key Differences (if any): [List main differences]
Note: Focus on meaning rather than surface-level lexical similarities. Two sentences can use different words but convey the same meaning, or use similar words but convey different meanings.
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Llama3_1_8b_chat')
    parser.add_argument('--dataset', type=str, default='proofwriter')
    parser.add_argument('--eval_model', type=str, default='gpt-4o')
    parser.add_argument('--n_samples', type=int, default=200)
    args = parser.parse_args()

    model_name = args.model 
    dataset = args.dataset 
    eval_model = args.eval_model
    n_samples = args.n_samples
    
    model = ModelWrapper(eval_model)
    result_path = f'../result/{dataset}/{model_name}/cot_e3_200.json'
    with open(result_path, 'r') as f:
        data = json.load(f)[:-1]
    
    results = []
    for item in tqdm(data[:n_samples]):
        if '# Answer:' in item['response']:
            cot = item['response'].split('# Answer:')[0]
        elif '\n\n' in item['response']:
            cot = ('\n\n').join(item['response'].split('\n\n')[:-1])
        else:
            cot = item['response']
        cot = ':'.join(cot.split(':')[1:]).strip()
        input = f"Sentence 1: {cot}\nSentence 2: {item['reason']}"
        inputs = [{"role": "system", "content": instruction}, {"role": "user", "content": input}]
        response = model.generate(inputs)
        pattern = r'\d+\.?\d*'
        match = re.findall(pattern,response.split('\n')[0])
        if match:
            score = float(match[0]) 
        else:
            score = None 
        msg = {
            'id':item['id'],
            'cor_flag':item['cor_flag'],
            'question':item['question'],
            'cot':cot,
            'gold_cot': item['reason'],
            'response':response,
            'score':score
        }
        results.append(msg)
    eval_path = f'../result/{dataset}/{model_name}/{eval_model}_eval_cot_{n_samples}.json'
    with open(eval_path, 'w') as f:
        json.dump(results, f, indent=4)