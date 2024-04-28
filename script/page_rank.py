import argparse
import json
import numpy as np
from transformers import set_seed

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama2_13b')
parser.add_argument('--n_samples', type=int, default=500)
parser.add_argument('--n_examples', type=int, default=3)
parser.add_argument('--method', type=str, default='cot')
parser.add_argument('--dataset', type=str, default='proofwriter')
args = parser.parse_args()
set_seed(17)


def pagerank_algorithm(init_page_rank, adjacency_matrix, damping_factor=0.85, max_iterations=10, convergence_threshold=0.0001):
    n = len(init_page_rank)
    
    # 构建转移矩阵
    transition_matrix = adjacency_matrix / np.sum(adjacency_matrix, axis=0, keepdims=True)
    
    # 初始化PageRank向量
    pagerank = init_page_rank
    
    # 开始迭代
    for i in range(max_iterations):
        old_pagerank = np.copy(pagerank)
        # print(np.dot(transition_matrix, old_pagerank))
        # 计算新的PageRank向量
        pagerank = damping_factor * np.dot(transition_matrix, old_pagerank) + (1 - damping_factor) / n
        # print(pagerank)
        # 判断是否收敛
        if np.sum(np.abs(pagerank - old_pagerank)) < convergence_threshold:
            break
    
    return pagerank

if __name__ == '__main__':
    model_name = args.model
    n_samples = args.n_samples
    n_examples = args.n_examples
    dataset = args.dataset 
    method = args.method
    
    pcot_path_file = f'../result/{dataset}/{model_name}/input_pcot_paths_e{n_examples}_s500.json'
    cot_path_file = f'../result/{dataset}/{model_name}/input_cot_paths_e{n_examples}_s500.json'
    cans_path_file = f'../result/{dataset}/{model_name}/input_cans_paths_e{n_examples}_s500.json'
    
    with open(pcot_path_file, 'r') as f:
        pcot_data = json.load(f)
        f.close()
    with open(cot_path_file, 'r') as f:
        cot_data = json.load(f)
        f.close()
    with open(cans_path_file, 'r') as f:
        cans_data = json.load(f)
        f.close()
        
        
    idx = 0
    
    for item in cans_data[:n_samples]:
        path = item['path'][0]['inp_attr']
        ref_list = [x['inp'].strip(':').strip('.').strip() for x in path]
        init_page_rank = [x['attr'] for x in path]
        init_page_rank = np.array(init_page_rank)
        init_page_rank = init_page_rank / init_page_rank.sum()
        n = init_page_rank.shape[0]
        adj_matrix = np.zeros((n,n))
        cot_paths = cot_data[idx]['path']
        for path in cot_paths:
            ref_idx = ref_list.index(path['ref'].strip(':').strip('.').strip())
            if path['inp_attr'] == []:
                for i in range(n):
                    adj_matrix[ref_idx, i] = 1/n
            for tup in path['inp_attr']:
                inp_idx = ref_list.index(tup['inp'].strip(':').strip('.').strip())
                adj_matrix[ref_idx, inp_idx] = tup['attr']
        page_rank = pagerank_algorithm(init_page_rank, adj_matrix)
        pcot_paths = pcot_data[idx]['path']
        dic = {}
        for path in pcot_paths:
            ref_idx = ref_list.index(path['ref'].strip(':').strip('.').strip())
            for tup in path['inp_attr']:
                inp = tup['inp'].strip(':').strip('.').strip()
                if inp in dic.keys():
                    dic[inp] += page_rank[ref_idx] * tup['attr']
                else:
                    dic[inp] = page_rank[ref_idx] * tup['attr']
        dic = sorted(dic.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
        print(dic)
        idx += 1        
        
        