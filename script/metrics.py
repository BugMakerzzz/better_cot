import json 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import rcParams
from pandas import DataFrame
import palettable
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu

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
    
    
def draw_line(data, path, names):
    
    sns.set_theme(style="whitegrid",font='Times New Roman')
    # custom_colors = ['#7976A2', '#4A5E65', '#E29957', '#86B5A1', '#B95A58', '#4292C6']
    custom_colors = ['#7976A2', '#4A5E65', '#E29957', '#86B5A1', '#B95A58', '#4292C6'][::-1]
    ax = sns.lineplot(x = names[0], 
                      y = names[1], 
                      hue = names[2],
                    #   palette=sns.color_palette("hls", n_colors=6, desat=.6), 
                      marker = 'o',
                      palette=sns.color_palette(custom_colors),
                    #   ci=None,
                      data=data)
    # ax.lines[2].set_linestyle("--")
    ax.set_xlabel(ax.get_xlabel(), fontsize=22)  # X轴标签
    ax.set_ylabel(ax.get_ylabel(), fontsize=22)  # Y轴标签
    ax.legend(fontsize='18', loc='upper left')  # 图例

    # 调整刻度文字大小
    ax.tick_params(axis='both', which='major', labelsize=20)

    # 调整图像边距
    plt.subplots_adjust(left=0.15, right=0.99, top=0.99, bottom=0.16)
    plt.savefig(path)
    plt.close()    
    
def draw_box(data, path, names):
    sns.set_theme(style="whitegrid",font='Times New Roman')
    custom_colors = ['#7976A2', '#4A5E65', '#E29957', '#86B5A1', '#B95A58', '#4292C6']
    ax = sns.boxplot(x=names[0], 
                    y=names[1], 
                    #  hue='cot_flag',
                    data=data,
                    flierprops={'marker':'d', 'markerfacecolor':'black', 'markersize':1},
                    # palette=sns.color_palette("hls", 4, desat=.6)
                    palette=sns.color_palette(custom_colors)
                    )
    ax.set_xlabel(ax.get_xlabel(), fontsize=22)  # X轴标签
    ax.set_ylabel(ax.get_ylabel(), fontsize=22)  # Y轴标签
    # ax.legend(fontsize='18', loc='upper left')  # 图例

    # 调整刻度文字大小
    ax.tick_params(axis='both', which='major', labelsize=20)

    # 调整图像边距
    plt.subplots_adjust(left=0.17, right=0.99, top=0.98, bottom=0.16)
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
    
def get_rouge(results, name_dic):
    generate_sents = []
    ref_sents = []
    for item in results[:-1]:
        if not item[name_dic['gen']]:
            continue
        cot = item[name_dic['gen']].split('\n# Answer:')[0]
        generate_sents.append(cot)
        if isinstance(item[name_dic['ref']], list):
            f = 0
            for ref in item[name_dic['ref']]:
                rouge = cal_rouge(cot, ref, avg=False)
                if rouge['f'] > f:
                    reason = ref
                    f = rouge['f']
            ref_sents.append(reason)
        else:
            ref_sents.append(item[name_dic['ref']])
    rouge = cal_rouge(generate_sents, ref_sents)
    return rouge
            
            
def cal_rouge(generate_sents, ref_sents, avg=True):
    rouge = Rouge()
    rouge_score = rouge.get_scores(hyps=generate_sents, refs=ref_sents, avg=avg)
    if avg:
        return rouge_score['rouge-l']
    else:
        return rouge_score[0]['rouge-l']


def get_bleu(results, name_dic):
    bleu = {'bleu1':[], 'bleu2':[], 'bleu3':[], 'bleu4':[]}
    for item in results[:-1]:
        if not item[name_dic['gen']]:
            continue
        candidate = item[name_dic['gen']].split('\n# Answer:')[0].split()
        if isinstance(item[name_dic['ref']], list):
            reference = [sent.split() for sent in item[name_dic['ref']]]
        else:
            reference = [item[name_dic['ref']].split()]
        score1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
        score2 = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0))
        score3 = sentence_bleu(reference, candidate, weights=(0, 0, 1, 0))
        score4 = sentence_bleu(reference, candidate, weights=(0, 0, 0, 1))
        bleu['bleu1'].append(score1)
        bleu['bleu2'].append(score2)
        bleu['bleu3'].append(score3)
        bleu['bleu4'].append(score4)
    for k,v in bleu.items():
        v = np.array(v).mean()
        bleu[k] = v
    return bleu