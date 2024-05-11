import json
import jsonlines
import re
from metrics import get_bleu

def transfer_proofwriter(item):
    def prepare_reason(proof, index_dic):
        reason = ""
        cache = ""
        for str in proof:
            if str in ['(', '-', '>', '[', ']']:
                continue
            elif str in [')', ' ']:
                if cache:
                    statement = index_dic[cache]['text']
                    reason += statement + ' '
                cache = ""
            elif str == '%':
                reason += 'So '
            elif str == 'O':
                break
            else:
                cache += str
        return reason[:-1]
    
    results = []
    id = item['id']
    context = item['theory']
    stat_dic = item['triples']
    stat_dic.update(item['rules'])
    options = [
      "A) True",
      "B) False"]
    for qid, tup in item['questions'].items():
        if tup['QDep'] != depth:
            continue
        qid = id + f'_{qid}'
        question = f"Based on the above information, is the following statement true, false, or unknown? {tup['question']}"
        answer = tup['answer']
        if answer == 'Unknown':
            continue
        elif answer:
            answer = "A"
        else:
            answer = "B"
        proofs = tup['proofsWithIntermediates']
        reasons = []
        for proof in proofs:
            stat_dic.update(proof['intermediates'])
            reason = prepare_reason(proof['representation'], stat_dic)
            reasons.append(reason)
        if len(reasons) == 1:
            reason = reasons[0]
        else:
            reason = reasons
        msg = {'id':qid, 'context':context, 'question':question, 'options':options, 'answer':answer, 'reason':reason}
        results.append(msg)
    return results


def trans_siqa():
    label_path = '/netdisk/ljc/code/faithful_cot/data/siqa/dev-labels.lst'
    data_path = '/netdisk/ljc/code/faithful_cot/data/siqa/dev.jsonl'
    labels = []
    label_dic = {'1':'A', '2':'B', '3':'C'}
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            labels.append(label_dic[line[:-1]])
    with open(data_path, 'r') as fin:
        items = [json.loads(line) for line in fin]
    
    for i in range(len(items)):
        item = items[i]
        answerA = f"(A) {item['answerA']}"
        answerB = f"(B) {item['answerB']}"
        answerC = f"(C) {item['answerC']}"
        options = [answerA, answerB, answerC]
        item['options'] = options
        item['answer'] = labels[i]
        
    with jsonlines.open(data_path, 'w') as f:
        f.write_all(items)

def extract_answer():
    def extract_logic(answer):
        pattern1 = r"correct \w+ is:?\s*([A-E])"
        pattern2 = r"correct option is: (true|false|unknown)"
        pattern3 = r"([A-C])\)\s*(True|False|Unknown)"
        pattern4 = r"([A-E])\) "
        pattern5 = r"^[A-E]\.?$"

        match = re.search(pattern1, answer)
        option = None
        # extract pattern
        if match:
            option = match.group(1)
        
        if not option:
            match = re.search(pattern2, answer, re.IGNORECASE)
            if match:
                word_to_option = {"true": "A", "false": "B", "unknown": "C"}
                option = word_to_option.get(match.group(1).lower())

        if not option:
            match = re.search(pattern3, answer, re.IGNORECASE)
            if match:
                option = match.group(1)
        if not option and len(answer)<16:
            if 'true' in answer.lower():
                option = 'A'
            elif 'false' in answer.lower():
                option = 'B'
            elif 'unknown' in answer.lower():
                option = 'C'
        if not option:
            match = re.match(pattern4, answer)
            if match:
                option = match.group(1)
        if not option:
            match = re.match(pattern5, answer)
            if match:
                option = match.group(0) 
        if not option:
            option = None
            # wrong_data.append(d)
        return option

    result_path = '/netdisk/ljc/code/faithful_cot/result/wino/Mistral_7b/cot_e3_s100.json'
    with open(result_path, 'r') as f:
        result = json.load(f)
    correct = 0
    for item in result[:-1]:
        # answer = extract_logic(item['answer'])
        # item['answer'] = answer
        if item['label'] == '1':
            item['label'] = 'A'
        elif item['label'] == '2':
            item['label'] = 'B'
        if item['answer'] == item['label']:
            item['cor_flag'] = True
            correct += 1
    result[-1]['acc'] = correct / 100
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=4)


def cal_cot_bleu():
    path = '../result/coinflip/Llama2_13b/cot_e3_s100.json'
    with open(path, 'r') as f:
        result = json.load(f)
        f.close()
    bleu = get_bleu(result, {'gen':'response', 'ref':'reason'})
    result.append(bleu)
    with open(path, 'w') as f:
        json.dump(result, f, indent=4)
        f.close()
    return 

def make_proof_dataset():
    depth = 1
    split = 'dev'
    src_path = f'/mnt/userdata/ljc/dataset/proofwriter-dataset-V2020.12.3/OWA/depth-{depth}/meta-{split}.jsonl'
    with open(src_path, 'r') as fin:
        items = [json.loads(line) for line in fin]
        fin.close()

    results = []
    for item in items:
        msgs = transfer_proofwriter(item)
        results.extend(msgs)
        
    result_path = f'/mnt/userdata/ljc/code/faithful_cot/data/proofwriter_d{depth}/{split}.jsonl'
    with jsonlines.open(result_path, 'w') as f:
        f.write_all(results)
        
    result_path = f'/mnt/userdata/ljc/code/faithful_cot/data/proofwriter_d{depth}/{split}.json'    
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    cal_cot_bleu() 
    
  