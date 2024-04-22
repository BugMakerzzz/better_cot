import json

data_path = '../data/gsmic/GSM-IC_mstep.json'
result_path = '../data/gsmic/dev.jsonl'
train_data_path = '../data/gsm8k/train.jsonl'
dev_data_path = '../data/gsm8k/dev.jsonl'

reason_dic = {}

with open(dev_data_path, 'r') as f:
    data = [json.loads(line) for line in f]
    for item in data:
        reason_dic[item['question']] = item['answer']
    f.close()

with open(train_data_path, 'r') as f:
    data = [json.loads(line) for line in f]
    for item in data:
        reason_dic[item['question']] = item['answer']
    f.close()   


with open(data_path, 'r') as f:
    data = json.load(f)
    f.close()

results = []
for item in data:
    origin_question = item['original_question']
    if origin_question in reason_dic.keys():
        answer = reason_dic[origin_question]
        msg = {'question':item['new_question'], 'answer':answer}
        results.append(msg)
    
with open(result_path, 'w') as f:
    for item in results:
        json.dump(item, f)
        f.write('\n')
