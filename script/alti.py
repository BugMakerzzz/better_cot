import torch.nn.functional as F
from contributions import ModelWrapper
from config import *
import torch
import json
import random
import numpy as np
import argparse
from collections import defaultdict
from load_data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from tqdm import tqdm
set_seed(17)


def prepare4explain(tokenizer, inp, ref, device):
    text = inp + " " + ref
    inp = tokenizer.tokenize(inp)
    ref = tokenizer.tokenize(ref)
    txt = tokenizer.tokenize(text)
    assert len(txt) - (len(inp) + len(ref)) <= 1, str(txt) + " | " + str(inp) + " | " + str(ref)
    # the insert blank may be splitted into multiple tokens
    ref = txt[len(inp):]
    ref_idx = range(len(inp), len(txt))
    text = tokenizer(text, return_tensors="pt").to(device)
    return txt, ref, ref_idx, text


def normalize_contributions(model_contributions,scaling='minmax',resultant_norm=None):
    """Normalization of the matrix of contributions/weights extracted from the model."""

    normalized_model_contributions = torch.zeros(model_contributions.size())
    for l in range(0,model_contributions.size(0)):

        if scaling == 'min_max':
            ## Min-max normalization
            min_importance_matrix = model_contributions[l].min(-1, keepdim=True)[0]
            max_importance_matrix = model_contributions[l].max(-1, keepdim=True)[0]
            normalized_model_contributions[l] = (model_contributions[l] - min_importance_matrix) / (max_importance_matrix - min_importance_matrix)
            normalized_model_contributions[l] = normalized_model_contributions[l] / normalized_model_contributions[l].sum(dim=-1,keepdim=True)

        elif scaling == 'sum_one':
            normalized_model_contributions[l] = model_contributions[l] / model_contributions[l].sum(dim=-1,keepdim=True)
            #normalized_model_contributions[l] = normalized_model_contributions[l].clamp(min=0)

        # For l1 distance between resultant and transformer vectors we apply min_sum
        elif scaling == 'min_sum':
            if resultant_norm == None:
                min_importance_matrix = model_contributions[l].min(-1, keepdim=True)[0]
                normalized_model_contributions[l] = model_contributions[l] + torch.abs(min_importance_matrix)
                normalized_model_contributions[l] = normalized_model_contributions[l] / normalized_model_contributions[l].sum(dim=-1,keepdim=True)
            else:
                # print('resultant_norm[l]', resultant_norm[l].size())
                # print('model_contributions[l]', model_contributions[l])
                # print('normalized_model_contributions[l].sum(dim=-1,keepdim=True)', model_contributions[l].sum(dim=-1,keepdim=True))
                normalized_model_contributions[l] = model_contributions[l] + torch.abs(resultant_norm[l].unsqueeze(1))
                normalized_model_contributions[l] = torch.clip(normalized_model_contributions[l],min=0)
                normalized_model_contributions[l] = normalized_model_contributions[l] / normalized_model_contributions[l].sum(dim=-1,keepdim=True)
        elif scaling == 'softmax':
            normalized_model_contributions[l] = F.softmax(model_contributions[l], dim=-1)
        elif scaling == 't':
            model_contributions[l] = 1/(1 + model_contributions[l])
            normalized_model_contributions[l] =  model_contributions[l]/ model_contributions[l].sum(dim=-1,keepdim=True)
        else:
            print('No normalization selected!')
    return normalized_model_contributions


def compute_joint_attention(att_mat):
    """ Compute attention rollout given contributions or attn weights + residual."""

    joint_attentions = torch.zeros(att_mat.size()).to(att_mat.device)

    layers = joint_attentions.shape[0]

    joint_attentions = att_mat[0].unsqueeze(0)

    for i in range(1,layers):

        C_roll_new = torch.matmul(att_mat[i],joint_attentions[i-1])

        joint_attentions = torch.cat([joint_attentions, C_roll_new.unsqueeze(0)], dim=0)
        
    return joint_attentions


def contribution(model_wrapped, tokenizer, input, response, device):
    """Contribution rollout-based relevancies and blank-out relevancies in SVA."""
    
    relevancies = defaultdict(list)
    layer = -1

    #for i in args_rel.random_samples_list:
    inp, ref, target_idx, text = prepare4explain(tokenizer, input, response, device)
    
    
    contributions_data = model_wrapped(text)

    # Our method relevances
    resultant_norm = torch.norm(torch.squeeze(contributions_data['resultants']),p=1,dim=-1)
    normalized_contributions = normalize_contributions(contributions_data['contributions'],scaling='min_sum',resultant_norm=resultant_norm)
    contributions_mix = compute_joint_attention(normalized_contributions)
    contributions_mix_relevances = contributions_mix[layer][target_idx]
    relevancies = np.asarray(contributions_mix_relevances)
    del contributions_data, resultant_norm, normalized_contributions, contributions_mix, contributions_mix_relevances
    torch.cuda.empty_cache()
    return inp, ref, relevancies



def main(args):
    model_name = args.model
    dataset = args.dataset
    n_samples = args.n_samples
    
    if model_name.startswith('Llama'):
        if '7b' in model_name:
            if 'chat' in model_name:
                path = llama2_7b_chat_path
            else:
                path = llama2_7b_path
        else:
            if 'chat' in model_name:
                path = llama2_13b_chat_path
            else:
                path = llama2_13b_path
    elif model_name.startswith('Mistral'):
        path = mistral_7b_path
    else:
        path = None
        pass 
    
    tokenizer = AutoTokenizer.from_pretrained(path, torch_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map='auto')
    model_wrapped = ModelWrapper(model)
    dataloader = DataLoader(dataset=dataset, n_samples=n_samples)
    data = dataloader.load_data(method='cot', n_examples=3)
    wrap_question_dic = {item['id']:item['question'] for item in data}
    result_path = f'../result/{dataset}/{model_name}/cot_e3_s500.json'
    with open(result_path, 'r') as f:
        results = json.load(f)[:n_samples]
    
    score_results = []  
    for item in tqdm(results):
        input = wrap_question_dic[item['id']]
        question = input.split('####')[-1]       
        response = item['response']

        inps, refs, relevancies = contribution(model_wrapped, tokenizer, question, response, model.device)
            
        score_tup = {'id':item['id'],
                    'inp':inps, 
                    'out':refs, 
                    'scores':relevancies.tolist()}
        score_results.append(score_tup)

    # Save sva rank relevancies to compute correlations between BERT and DistilBERT
    score_path = f'../result/{dataset}/{model_name}/alti_scores_e3_s{n_samples}.json'
    with open(score_path, 'w') as f:
        json.dump(score_results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Llama2_13b')
    parser.add_argument('--n_samples', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='proofwriter')
    args=parser.parse_args()

    main(args)