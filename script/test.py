# from utils.load_data import load_json_data, write_json_data, extract_answer

# def is_equiv(str1, str2):
#     if not str1 or not str2:
#         return False
#     return str1 and str2 and str1.lower() == str2.lower()


# model = 'Mistral_7b_chat'
# # datasets = ['prontoqa', 'proofwriter']
# datasets = ['proofwriter','prontoqa', ]
# for dataset in datasets:
#     # print(dataset)
#     sc_path = f'../result/{dataset}/{model}/sc10_e3_500.json'
#     # direct_path = f'../result/{dataset}/Llama3_1_8b_chat/direct_e3_200.json'
    
#     # cot_data = load_json_data(cot_path)[:-1]
#     # direct_data = load_json_data(direct_path)[:-1]
    
#     # direct_dic = {item['id']:item['cor_flag'] for item in direct_data}
#     # result_dic = {False:{False:0, True:0}, True:{False:0, True:0}}
#     # # result_dic = {False:{False:{False:0, True:0, None:0}, True:{False:0, True:0, None:0}}, True:{False:{False:0, True:0, None:0}, True:{False:0, True:0, None:0}}}
#     # for item in cot_data:
#     #     # result_dic[direct_dic[item['id']]][item['cor_flag']][item['cot_flg']] += 1
#     #     result_dic[direct_dic[item['id']]][item['cor_flag']] += 1
#     # print(result_dic)
#     sc_data = load_json_data(sc_path)[:-1]
#     cnt = 0
#     correct = 0
#     results = []
#     for item in sc_data:
#         answer = []
#         for res in item['response']:
#             ans = extract_answer(dataset, res)
#             answer.append(ans)
#         pred = max(answer, key=answer.count)
#         corrects = [is_equiv(ans, item['label']) for ans in answer]
#         cor_flag = is_equiv(pred, item['label'])
#         item = {'id':item['id'], 'question':item['question'], 'reason':item['reason'], 'response':item['response'], 'answer':answer, 'pred':pred, 'label':item['label'], 'corrects':corrects, 'cor_flag':cor_flag}
#         correct += int(cor_flag)
#         results.append(item)
#         cnt += 1
#     results.append({'acc':correct/cnt})

# from utils.load_data import extract_answer, load_json_data, write_json_data
# path = '/mnt/userdata/ljc/code/faithful_cot/result/prontoqa/Gemma2_9b_chat/cot_e3_200.json'
# # def create()

# # result_dir =  f'./result/{dataset}/'
# # all_paths = list_all_files(result_dir)
# # for path in all_paths:
# #     if 'eval' in path:
# #         continue
# #     data = load_json_data(path)
# data = load_json_data(path)
# cor = 0
# cnt = 0
# for item in data:
#     if 'acc' in item.keys():
#         item['acc'] = cor / cnt 
#         break
    
#     answer = extract_answer('prontoqa', item['response'])
#     if answer and answer.lower() == item['label'].lower():
#         cor_flag = True
#         cor += 1
#     else:
#         cor_flag = False
#     item['cor_flag'] = cor_flag
#     item['answer'] = answer 
#     cnt += 1
# write_json_data(path, data)


from utils.load_data import load_json_data, extract_answer, write_json_data
# from utils.eval import is_equiv
import os
import numpy as np 
import random 
random.seed(17)

def list_all_files(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


def get_acc(result:list[dict], n_samples:int, method:str) -> list[dict]:
    cor_flag = 0
    if method == 'bridge':
        ind1 = random.choice([0,1,2])
        ind2 = random.choice([3,4,5])
        ind3 = random.choice([6,7,8])
        index = [ind1, ind2, ind3]
    elif method == 'sc':
        index = random.sample(range(10), n_samples)
    else:
        index = range(0,n_samples)
    new_results = []
    for item in result:
        if method == 'cot':
            index = random.sample(range(3), n_samples)
        # print(item['response'])
        # if len(item['response']) == 6:
        responses = item['response']
        answers = item['answer']
        corrects = item['corrects']
        # else:
        # responses = [item['response'][idx] for idx in index]
        # answers = [item['answer'][idx] for idx in index]
        # corrects = [item['corrects'][idx] for idx in index]
        if method == 'cot':
            responses = responses[0]['content']
            answers = answers[0]
            corrects = corrects[0]
            cor = corrects
            cor_flag += int(cor)
            pred = answers
        elif method == 'sc' or method == 'bridge_ig':
            best_idx = answers.index(max(answers, key=answers.count))
            pred = answers[best_idx]
            cor = corrects[best_idx]
            if cor:
                cor_flag += int(cor)
            else:
                cor = False
                cor_flag += 0
        elif method == 'bridge':
            scores = [item['score'] for item in responses]
            coef = {}
            for i in range(len(scores)):
                if not answers[i]:
                    continue
                if answers[i] not in coef.keys():
                    coef[answers[i]] = scores[i]
                else:
                    coef[answers[i]] += scores[i]
            if not coef:
                pred = None
                cor = False 
            else:
                pred = max(coef, key=lambda x: coef[x])
                cor = pred.lower() == item['label'].lower()
                cor_flag += int(cor)
            
            # elif method == 'mcts':
            #     best_idx = max(enumerate(responses), key=lambda x: x[1]['reward'])[0]
            #     cor_flag += int(corrects[best_idx])
        new_results.append({'id':item['id'], 'response':responses, 'answer':answers, 'corrects':corrects, 'cor_flag':cor, 'label':item['label'], 'reason':item['reason'], 'pred':pred})
    new_results.append({'acc':cor_flag / len(result)})
    return new_results
# 'bridge_aae', 'bridge_ig'
# 示例用法
methods = ['bridge_ig']
datasets = ['prontoqa', 'proofwriter']
models = ['Gemma2_9b_chat']
for dataset in datasets:
    for model in models:
        for method in methods:
            if method == 'cot':
                new_path = f'../result/{dataset}/{model}/cot_500.json'
                # if os.path.exists(new_path):
                #     continue
                old_path = f'../result/{dataset}/{model}/sc3_e3_500.json'
                new_result = get_acc(load_json_data(old_path)[:-1], 1, 'cot')
                write_json_data(new_path, new_result)
            elif method == 'sc':
                new_path = f'../result/{dataset}/{model}/sc3_e3_200.json'
                # if os.path.exists(new_path):
                #     continue
                old_path = f'../result/{dataset}/{model}/sc10_e3_200.json'
                new_result = get_acc(load_json_data(old_path)[:-1], 3, 'sc')
                write_json_data(new_path, new_result)
            elif method == 'bridge':
                new_path = f'../result/{dataset}/{model}/bridge_200.json'
                old_path = f'../result/{dataset}/{model}/bridge3_sc3_w1_e3_200.json'
                new_result = get_acc(load_json_data(old_path)[:-1], 9, 'bridge')
                write_json_data(new_path, new_result)
            elif method == 'bridge_ig':
                new_path = f'../result/{dataset}/{model}/bridge_wo_ig_200.json'
                old_path = f'../result/{dataset}/{model}/bridge_200.json'
                new_result = get_acc(load_json_data(old_path)[:-1], 3, 'bridge_ig')
                write_json_data(new_path, new_result)