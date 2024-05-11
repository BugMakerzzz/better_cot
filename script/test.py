import json

data_path = '/mnt/userdata/ljc/code/faithful_cot/result/folio/Mistral_7b/cot_e3_s500.json'
result_path = '/mnt/userdata/ljc/code/faithful_cot/result/folio/Mistral_7b/cot_e3_s100.json'

with open(data_path, 'r') as f:
    data = json.load(f)[:-1]
    f.close()

results = []
cnt = 0
cor = 0
for item in data:
    if item['answer'] != 'None':
        if item['cor_flag']:
            cor += 1
        cnt += 1
        results.append(item)

results.append({'acc': cor / cnt})
with open(result_path, 'w') as f:
    json.dump(results, f, indent=4)
    f.close()