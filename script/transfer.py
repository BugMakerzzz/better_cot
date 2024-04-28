import json
import jsonlines

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

if __name__ == '__main__':
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