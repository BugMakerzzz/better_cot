import json

result_path = '../result/gsm8k/Llama2_13b/cot_e3_s500.json'
data_path = '../data/gsm8k/dev.jsonl'

with open(data_path, 'r') as f:
    data = [json.loads(line) for line in f]
    f.close()
    
with open(result_path, 'r') as f:
    result_data = json.load(f)
    f.close()

# data_dic = {}
# for item in data:
#     data_dic[item['id']] = item

# for item in result_data:
#     if 'id' in item.keys():
#         data_item = data_dic[item['id']]
#         item['context'] = data_item['context']
#         item['question'] = data_item['question']
#         item['gold_cot'] = data_item['reason']

# with open(result_path, 'w') as f:
#     json.dump(result_data, f, indent=4)
#     f.close()

result_path = '../data/addition/dev.jsonl'
data_path = '../data/addition/dev.json'

with open(data_path, 'r') as f:
    data = json.load(f)
    f.close()
    
with open(result_path, 'w') as f:
    for item in data:
        json.dump(item, f)
        f.write('\n')
