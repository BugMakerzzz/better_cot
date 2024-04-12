import argparse
import torch
import json
import numpy as np
from model import ModelWrapper
from load_data import DataLoader
from transformers import set_seed
from tqdm import tqdm
from IPython.core.display import HTML

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama2_13b')
parser.add_argument('--n_samples', type=int, default=500)
parser.add_argument('--n_examples', type=int, default=5)
parser.add_argument('--dataset', type=str, default='proofwriter')
parser.add_argument('--mode', type=str, default='answer')
parser.add_argument('--golden', action='store_true')
args = parser.parse_args()
set_seed(17)

model_name = args.model
n_samples = args.n_samples
n_examples = args.n_examples
dataset = args.dataset 
mode = args.mode
golden = args.golden

# dataloader = DataLoader(dataset=dataset, n_samples=n_samples)
# model = ModelWrapper(model_name)



# def cal_attr_score(value, grad, steps=20):
#     grad_int = torch.zeros_like(value*grad)
#     for i in range(steps):
#         k = (i+1) / steps
#         grad_int += k * grad
#     scores = 1 / steps * value * grad_int
#     return torch.abs(scores) 
    
def split_input(question, response, mode='answer'):
    sp_question = question.split('####\n')
    prompt = ('####\n').join(sp_question[:-1])
    question = '####\n' + sp_question[-1]
    output_ls = response.split('# Answer:')
    if len(output_ls) == 1:
        reason = 'None'
        answer = 'None'
    else:
        reason = output_ls[0]
        pred = output_ls[1]
        pred = '# Answer:' + pred
        answer = pred.strip()
    if mode == 'answer':
        input = prompt + question + reason
        ref = answer 
    else:
        input = prompt + question
        ref = reason 
    # msg = {'prompt':prompt, 'question':question, 'reason':reason, 'answer':answer}
    return input, ref 
    

# def cal_attn_attr(item, mode='answer'):
#     prompt = item['prompt']
#     question = item['question']
#     reason = item['reason']
#     answer = item['answer']
#     input_text = prompt + question + reason + answer
#     prompt_len = len(tokenizer(prompt, return_tensors="pt").input_ids[0])
#     question_len = len(tokenizer(question, return_tensors="pt").input_ids[0])
#     reason_len = len(tokenizer(reason, return_tensors="pt").input_ids[0])
#     answer_len = len(tokenizer(answer, return_tensors="pt").input_ids[0])
#     input_ids = tokenizer(input_text, return_tensors="pt").input_ids
#     labels = torch.full_like(input_ids, -100)
#     if mode == 'answer':
#         labels[:, -1] = input_ids[:, -1]
#     else:
#         labels[:, prompt_len+question_len:prompt_len+question_len+reason_len-4] = input_ids[:, prompt_len+question_len:prompt_len+question_len+reason_len-4]

#     scores = {}
    
#     outputs = model(
#         input_ids=input_ids.to(model_wrapper.device),
#         labels=labels.to(model_wrapper.device),
#         return_dict=True,
#         output_attentions=True,
#         output_hidden_states=False,
#     )
#     del input_ids, labels
#     loss = outputs['loss']
#     for layer in range(model_wrapper.n_layers): 
#         attn_values = outputs['attentions'][layer]
#         attn_values.retain_grad()
        
#     loss.backward()
#     for layer in range(model_wrapper.n_layers): 
#         attn_values = outputs['attentions'][layer]
#         attn_grad = attn_values.grad
#         grad = attn_grad.detach().cpu()
#         del attn_grad
#         attn_values = torch.squeeze(attn_values).detach().cpu()
#         attn_grad = torch.squeeze(grad)
#         attn_scores = torch.zeros_like(attn_values[0,:,:])
#         for i in range(40):
#             attn_scores += cal_attr_score(attn_values[i,:,:], attn_grad[i,:,:])
#         if mode == 'answer':
#             prompt_attn_scores = attn_scores[-2, :prompt_len].sum().detach().cpu().numpy().tolist()
#             question_attn_scores = attn_scores[-2, prompt_len:prompt_len + question_len].sum().detach().cpu().numpy().tolist()
#             reason_attn_scores = attn_scores[-2, prompt_len+question_len:prompt_len+question_len+reason_len].sum().detach().cpu().numpy().tolist()
#             score = {'prompt':prompt_attn_scores, 'question':question_attn_scores, 'cot':reason_attn_scores}
#         else:
#             prompt_attn_scores = attn_scores[prompt_len+question_len:prompt_len+question_len+reason_len-4, :prompt_len].sum().detach().cpu().numpy().tolist()
#             question_attn_scores = attn_scores[prompt_len+question_len:prompt_len+question_len+reason_len-4, prompt_len:prompt_len + question_len].sum().detach().cpu().numpy().tolist()
#             score = {'prompt':prompt_attn_scores, 'question':question_attn_scores}
#         scores[layer] = score
        
        
#         del attn_values, attn_grad, attn_scores
#         model.zero_grad() 
#         torch.cuda.empty_cache()
            
#     del loss, outputs
#     torch.cuda.empty_cache()
    
#     return scores

def cal_attr(expl, L=10, b=7, p=4, eps=1e-7):

    expl = np.abs(np.array(expl))
    # sparsify and normalize
    expl = expl / (expl.max(axis=0, keepdims=True) + eps)
    expl = np.ceil(expl * L)
    zeros = np.zeros_like(expl)
    expl = np.where(expl <= b, zeros, expl)
    
    
    l1 = expl.sum(axis=-1)
    lp = (expl ** p).sum(axis=-1) ** (1. / p) + eps
    input_attrs = (l1 / lp).tolist()
    
    l1 = expl.sum(axis=0)
    lp = (expl ** p).sum(axis=0) ** (1. / p) + eps
    output_attrs = (l1 / lp).tolist()

    return input_attrs, output_attrs



# result_path = f'../result/{dataset}/{model_name}/cot_e{n_examples}_s500.json'
# data = dataloader.load_data(method='cot', n_examples=n_examples)
# with open(result_path, 'r') as f:
#     result = json.load(f)
# idx = 0
# scores = []
# for item in tqdm(data):
#     r_item = result[idx]
#     idx += 1
#     if golden:
#         response = r_item['gold_cot'] + f"\n# Answer:\nThe answer is: {r_item['label']}\n" 
#     else:
#         response = r_item['response']
#     input, ref = split_input(question=item['question'], response=response, mode=mode)
#     if r_item['answer'] == 'None' or ref == 'None':
#         continue
    
#     # cot_scores = cal_attn_attr(msg, mode='cot')
#     inps, refs, expls, attrs, confs = model.input_explain(input, ref)
#     # answer_scores = cal_attn_attr(msg, mode='answer')
#     score_item = {'input_tokens':inps, 
#                   'response_tokens':refs, 
#                   'scores':expls,
#                   'norm_scores':attrs,
#                   'probs':confs,
#                   'answer':r_item['answer'], 
#                   'label':r_item['label'], 
#                   'cor_flag':r_item['cor_flag']}
   
#     scores.append(score_item)
      
# score_path = f'../result/{dataset}/{model_name}/input_score_e{n_examples}_s{n_samples}_m{mode}_g{golden}.json'

# with open(score_path, 'w') as f:
#     json.dump(scores, f, indent=4)

score_path = f'../result/{dataset}/{model_name}/input_score_e{n_examples}_s{n_samples}_m{mode}.json'
with open(score_path, 'r') as f:
    scores = json.load(f)
    
for item in tqdm(scores):
    expls = item['scores']
    input_attrs, output_attrs = cal_attr(expls)
    item['input_attrs'] = input_attrs
    item['output_attrs'] = output_attrs
    
with open(score_path, 'w') as f:
    json.dump(scores, f, indent=4)
