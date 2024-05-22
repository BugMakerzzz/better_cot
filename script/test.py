import argparse
import json
from load_data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama2_13b')
parser.add_argument('--n_samples', type=int, default=100)
parser.add_argument('--n_examples', type=int, default=3)
parser.add_argument('--dataset', type=str, default='proofwriter_d1')
parser.add_argument('--method', type=str, default='cot')
args = parser.parse_args()

model_name = args.model
n_samples = args.n_samples
n_examples = args.n_examples
dataset = args.dataset 
method = args.method

path = f'../result/{dataset}/{model_name}/filter_cot_e3_s{n_samples}_nli_pcot.json'
with open(path, 'r') as f:
    data = json.load(f)[:-1]    
    f.close()

result_path = './human.json'
results = []
for item in data:
    msg = {'question':item['question'], 'label':item['label'],'reason':item['reason'],'cot1':item['f_response'], 'ans1':item['f_answer'], 'cot2':item['response'][-1], 'ans2':item['answer'][-1]}
    results.append(msg)
with open(result_path, 'w') as f:
    json.dump(results, f, indent=4)
    f.close()
 
# dataloader = DataLoader(dataset=dataset, n_samples=1000)
# data = dataloader.load_data(method=method, n_examples=n_examples)
# reason_dic = {item['id']:item['reason'] for item in data}

# path = f"../result/{dataset}/{model_name}/{method}_e{n_examples}_s{n_samples}.json"
# with open(path, 'r') as f:
#     result_data = json.load(f)
#     f.close()
# i = 0
# for item in result_data[:-1]:
    
#     if 'reason' not in item.keys():
#         i += 1
#         item['reason'] = reason_dic[item['id']]
#     if 'gold_cot' in item.keys():
#         del item['gold_cot']
# result_path = f"../result/{dataset}/{model_name}/{method}_e{n_examples}_s{n_samples}.json"
# with open(result_path, 'w') as f:
#     json.dump(result_data, f, indent=4)
#     f.close()


# path = f"../result/{dataset}/{model_name}/{method}_e{n_examples}_s{n_samples}.json"
# # with open(path, 'r') as f:
# #     data = json.load(f)[:-1]
# #     f.close()
# # cot_flag_dic = {item['id']:item['cot_flag'] for item in data}

# # path = "../result/prontoqa_d2/Mistral_7b/cot_Mistral_7b_pcot_paths_e3_s100.json"
# # with open(path, 'r') as f:
# #     result_data = json.load(f)[:-1]
# #     f.close() 

# # i = 0
# # for item in result_data[:-1]:
# #     if not item['cot_flag']:
# #         item['cot_flag'] = cot_flag_dic[item['id']]

# # with open(path, 'w') as f:
# #     json.dump(result_data, f, indent=4)
# #     f.close()

# for model in ['Llama2_13b', 'Mistral_7b']:
#     for dataset in ['proofwriter_d1', 'prontoqa_d2']:
#         path = f"../result/{dataset}/{model}/cot_e3_s100.json"
#         with open(path, 'r') as f:
#             data = json.load(f)[:-1]
#             f.close()
#         dic = {'Val->C':0, 'Ival->W':0, 'Val->W':0, 'Ival->C':0}
#         for item in data:
#             if item['cor_flag'] and item['cot_flag'] in [0,1]:
#                 dic['Val->C'] += 1
#             elif item['cor_flag'] and item['cot_flag'] in [2,3]:
#                 dic['Ival->C'] += 1
#             elif not item['cor_flag'] and item['cot_flag'] in [0,1]:
#                 dic['Val->W'] += 1
#             else:
#                 dic['Ival->W'] += 1
#         print(f'{model}\t{dataset}:')
#         for k, v in dic.items():
#             print(f"{k}:{v}")