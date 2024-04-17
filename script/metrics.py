import json 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import rcParams



def cal_coef():
    score_path = '../result/gsm8k/Llama2_13b/score_e3_s100.json'
    with open(score_path, 'r') as f:
        data = json.load(f)    

    tf_ls = []
    pc_ls = []
    qc_ls = []
    pa_ls = []
    qa_ls = []
    ca_ls = []

    for item in data:
        if item['answer'] == 'A':
            continue
        if item['answer'] == 'A':
            tf_ls.append(2)
        elif item['answer'] == 'B':
            tf_ls.append(1)
        else:
            tf_ls.append(0)
        pc_score = 0
        qc_score = 0
        pa_score = 0
        qa_score = 0
        ca_score = 0
        cot_scores = item['cot_scores']
        for k, v in cot_scores.items():
            pc_score += v['prompt']
            qc_score += v['question']
        answer_scores = item['answer_scores']
        for k, v in answer_scores.items():
            pa_score += v['prompt']
            qa_score += v['question']
            ca_score += v['cot']
        pc_ls.append(pc_score)
        qc_ls.append(qc_score)
        pa_ls.append(pa_score)
        qa_ls.append(qa_score)
        ca_ls.append(ca_score)

    corr = np.corrcoef(tf_ls, pc_ls)[0,1]
    print(f'Prompt to CoT:{corr}')
    corr = np.corrcoef(tf_ls, qc_ls)[0,1]
    print(f'Question to CoT:{corr}')
    corr = np.corrcoef(tf_ls, pa_ls)[0,1]
    print(f'Prompt to Answer:{corr}')
    corr = np.corrcoef(tf_ls, qa_ls)[0,1]
    print(f'Question to Answer:{corr}')
    corr = np.corrcoef(tf_ls, ca_ls)[0,1]
    print(f'CoT to Answer:{corr}')
    

def draw_heat(x_labels, y_labels, scores, path):

    ax=sns.heatmap(scores, cmap="RdBu_r", center=0)

    x_ticks = np.arange(0.5, len(x_labels)+0.5)
    y_ticks = np.arange(0.5, len(y_labels)+0.5)

    
    plt.ylabel('Output', fontdict={'family' : 'Times New Roman', 'size':2})
    plt.xlabel('Input', fontdict={'family' : 'Times New Roman', 'size':4})
    plt.yticks(ticks=y_ticks, labels=y_labels, fontsize=2, rotation=45)
    plt.xticks(ticks=x_ticks, labels=x_labels, fontsize=2, rotation=45)
    plt.subplots_adjust(left=0.15, right=0.99, top=0.98, bottom=0.15)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    cbarlabels = cbar.ax.get_yticklabels() 
    [label.set_fontname('Times New Roman') for label in cbarlabels]
    plt.savefig(path)
    plt.close()
    
def draw_plot(tokens, scores, path):

    # 创建图形
    plt.figure(figsize=(10, 6))

    
    x_ticks = range(len(tokens))
    plt.bar(x_ticks, scores)
    
    # 设置图形标题和坐标轴标签
    plt.xlabel('Tokens', fontdict={'family' : 'Times New Roman'})
    plt.ylabel('Scores', fontdict={'family' : 'Times New Roman'})
    plt.xticks(ticks=x_ticks, labels=tokens, fontproperties = 'Times New Roman', fontsize=2, rotation=45)
    
    plt.savefig(path)
    plt.close()



