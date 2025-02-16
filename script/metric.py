from utils.load_data import load_json_data
import numpy as np

dataset = 'prontoqa'
method = 'early'
model_name = 'Llama3_1_8b_chat'
n_samples = 500
for baseline in ['cot', 'bridge']:
    score_path = f'../result/{dataset}/{model_name}/{baseline}_{method}_faith_{n_samples}.json'
    scores = load_json_data(score_path)

    score = np.mean(np.array([item['scores'] for item in scores]))
    print(score)