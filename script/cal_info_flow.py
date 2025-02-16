from utils.load_data import load_json_data
import argparse
import numpy as np

def extract_context_scores(model, tokens, scores, target=False):
    if model.is_llama:
        start_idx = [i for i, v in enumerate(tokens) if v == ':\u010a'][0] + 1
    else:
        start_idx = [i for i, v in enumerate(tokens) if v == ':'][0] + 1
    end_idx = [i for i, v in enumerate(tokens) if v == "#"][1]
    if target:
        score = np.mean(np.array(scores)[:,end_idx:])
    else:
        score = np.mean(np.array(scores)[:,start_idx:end_idx])
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Llama3_8b_chat')
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--n_examples', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='proofwriter')
    parser.add_argument('--target', type=str, default='qans')
    args = parser.parse_args()
    
    model_name = args.model
    n_samples = args.n_samples
    n_examples = args.n_examples
    dataset = args.dataset 
    method = args.target
    
    info_path = f'../result/{dataset}/{model_name}/{method}_info_e{n_examples}_{n_samples}.json'
    cot_data_path = f'../result/{dataset}/{model_name}/cot_e{n_examples}_{n_samples}.json'
    
    