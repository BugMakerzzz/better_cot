from utils.load_data import load_json_data
from utils.config import figure_colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def draw_cot_performance():
    model = 'Llama3_1_8b_chat'
    for dataset in ['gsmic', 'gsm8k', 'aqua', 'coinflip', 'lastletter', 'proofwriter', 'folio', 'prontoqa', 'logiqa', 'wino', 'siqa','ecqa', 'csqa']:
        d_path = f'../result/{dataset}/{model}/direct10_e3_200.json'
        data = load_json_data(d_path)[:-1]
        dc_dic = {}
        for item in data:
            difficulty = 5 - item['cor_flag'].count(True) // 2
            dc_dic[item['id']] = [difficulty,  item['cor_flag'].count(True) / 10]  
        c_path = f'../result/{dataset}/{model}/sc10_e3_200.json'
        data = load_json_data(c_path)[:-1]
        direct_data = [0] * 6
        cot_data = [0] * 6
        cnts = [0] * 6
        for item in data:
            difficulty = dc_dic[item['id']][0]
            cnts[difficulty] += 1
            direct_data[difficulty] += dc_dic[item['id']][1]
            cot_data[difficulty] += item['cor_flag'].count(True) / 10
        
        categories = []
        direct_score = []
        cot_score = []
        for i in range(6):
            if cnts[i] == 0:
                continue
            categories.append(f'Difficulty {i}')
            direct_score.append(direct_data[i] / cnts[i])
            cot_score.append(cot_data[i] / cnts[i]) 
        x = np.arange(len(categories))
        # 绘制柱状图
        fig, ax = plt.subplots(figsize=(10, 6))
        width = 0.35
        # 画分段柱状图
        bars1 = ax.bar(x - width / 2, direct_score, color='salmon', width=width, hatch='//', edgecolor='white', label='Direct')
        bars2 = ax.bar(x + width / 2, cot_score, color=figure_colors[1], width=width, hatch='//', edgecolor='white', label='CoT')

        # 图例
        ax.legend(loc='upper left', fontsize=10)
        plt.xticks(x, categories)
        # 设置轴标签和标题
        ax.set_ylabel('Score')
        # ax.set_title('Bar Chart with Residual and Solved-by-All')

        # 优化图表布局
        plt.tight_layout()
        plt.savefig(f'fig/{dataset}_test.png')



model = 'Gemma2_9b_chat'
difficulty_scores = []
datasets = ['gsmic', 'gsm8k', 'aqua', 'proofwriter', 'folio',  'prontoqa', 'wino', 'siqa','ecqa']
for dataset in datasets:
    d_path = f'../result/{dataset}/{model}/direct10_e3_200.json'
    data = load_json_data(d_path)[:-1]
    # dc_dic = {}
    scores = []
    for item in data:
        difficulty = 5 - item['cor_flag'].count(True) // 2
        if difficulty == 0:
            difficulty = 1
        # dc_dic[item['id']] = [difficulty,  item['cor_flag'].count(True) / 10]  
        scores.append(difficulty)
    difficulty_scores.append(scores)
    
flattened_data = {
    'difficulty': [item for sublist in difficulty_scores for item in sublist],
    'dataset': [datasets[i] for i, sublist in enumerate(difficulty_scores) for _ in sublist]
}

df = pd.DataFrame(flattened_data)

# 绘制小提琴图
plt.figure(figsize=(10, 6))
ax = sns.violinplot(x='dataset', y='difficulty', data=df, palette=figure_colors)
ax.set_xlabel(ax.get_xlabel(), fontsize=22)  # X轴标签
ax.set_ylabel(ax.get_ylabel(), fontsize=22)  # Y轴标签
    # ax.legend(fontsize='18', loc='upper left', fancybox=True, framealpha=0.5)  # 图例

    # 调整刻度文字大小
ax.tick_params(axis='both', which='major', labelsize=16)

# 调整图像边距
plt.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.13)

# plt.title('Difficulty Level Distribution')
plt.savefig(f'../fig/{model}_dif_static.pdf')