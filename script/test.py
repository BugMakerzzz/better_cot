import json
from metrics import cal_rouge

data_path = '/netdisk/ljc/code/faithful_cot/result/prontoqa_d2/Llama2_13b/filter_cot_e3_s100_nli_pcot.json'
# result_path = '/mnt/userdata/ljc/code/faithful_cot/result/folio/Mistral_7b/cot_e3_s100.json'
with open(data_path, 'r') as f:
    data = json.load(f)
    f.close()
for item in data[:-2]:
    if item['f_response'] != item['response'][-1]:
        cot1 = item['f_response']
        rouge1 = cal_rouge(cot1, item['reason'], avg=False)['r']
        cot2 = item['response'][-1]
        rouge2 = cal_rouge(cot2, item['reason'], avg=False)['r']
        if rouge1 < rouge2:
            print(item['id'])
# with open(data_path, 'r') as f:
#     data = json.load(f)[:-1]
#     f.close()

# results = []
# cnt = 0
# cor = 0
# for item in data:
#     if item['answer'] != 'None':
#         if item['cor_flag']:
#             cor += 1
#         cnt += 1
#         results.append(item)

# results.append({'acc': cor / cnt})
# with open(result_path, 'w') as f:
#     json.dump(results, f, indent=4)
#     f.close()