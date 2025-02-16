import json
import argparse
import plotly.express as px
import pandas as pd
import plotly.io as pio    
import matplotlib.pyplot as plt
import numpy as np
import re 
import plotly.graph_objects as go

def draw_token_bar(tokens, importance_scores, path):


    df = pd.DataFrame({'Token': tokens, 'Importance Score': importance_scores})

# 创建条形图，保持原始顺序
    unique_tokens = [f"{token}_{i}" for i, token in enumerate(tokens)]

# 创建 DataFrame，确保包含所有 token 和对应的分数
    df = pd.DataFrame({'Token': unique_tokens, 'Original Token': tokens, 'Importance Score': importance_scores})

    # 创建条形图，保持原始顺序
    fig = px.bar(df, x='Importance Score', y='Token', orientation='h',
                title='Token Importance Visualization', 
                labels={'Importance Score': 'Importance Score', 'Token': 'Token'},
                category_orders={'Token': unique_tokens})  # 保持原始顺序



    # 添加一个空白的 y 轴，以确保 0 分数的 token 可以显示
    fig.update_layout(yaxis=dict(tickmode='linear', dtick=1))

    # 显示图形
    pio.write_html(fig,path)


def draw_token_heatmap(x_tokens, y_tokens, importance_scores, path):
    # 创建 DataFrame，包含 Token 和对应的分数
    unique_x_tokens = [f"{token}_{i}" for i, token in enumerate(x_tokens)]
    unique_y_tokens = [f"{token}_{i}" for i, token in enumerate(y_tokens)]
    
    # 生成二维热力图
    fig = go.Figure(data=go.Heatmap(
        z=importance_scores,      # 二维列表，表示热力值
        x=unique_x_tokens,        # 横轴唯一 token 编号
        y=unique_y_tokens,        # 纵轴唯一 token 编号
        colorscale='Viridis',     # 色系
        colorbar=dict(title="Importance Score")
    ))

    # 设置标题和坐标轴标签
    fig.update_layout(
        title="Token Pair Importance Heatmap",
        xaxis=dict(title="X Tokens"),
        yaxis=dict(title="Y Tokens")
    )

    # 保存为 HTML 文件
    pio.write_html(fig, path)
    # 保存为 HTML 文件
    pio.write_html(fig, path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Llama3_1_8b_chat')
    parser.add_argument('--task', type=str, default='cans')
    parser.add_argument('--dataset', type=str, default='prontoqa')
    parser.add_argument('--id', type=str)
    parser.add_argument('--golden', action='store_true')
    args = parser.parse_args()

    model_name = args.model
    task = args.task
    dataset = args.dataset
    id = args.id
    golden = args.golden
    
    if golden:
        data_path =  f'../result/{dataset}/{model_name}/{task}_info_e3_200_golden.json' 
    else:
        data_path =  f'../result/{dataset}/{model_name}/{task}_info_e3_200.json' 
    with open(data_path, 'r') as f:
        data = json.load(f)
    results = {}
    with open(f'../result/{dataset}/{model_name}/cot_e3_200.json', 'r') as f:
        cot_data = json.load(f)
    
        res_dic = {item['id']:item for item in cot_data[:-1]}

    for item in data:
        if item['id'] != id:
            continue
        if golden:
            path = f"../result/{dataset}/{model_name}/token_fig/{task}_{id}_golden.html"
        else:
            path = f"../result/{dataset}/{model_name}/token_fig/{task}_{id}.html"
        if task in ['cans', 'qans']:
            draw_token_bar(item['input'], item['scores'], path)
        else:
            # if task == 'ccot':
            # scores = []
            # for score in item['scores'][1:]:
            #     scores.append(np.mean(np.array(score)))
            # score = list(np.mean(np.array(item['scores'][1:]),-1))
            # x = len(scores)
            if golden:
                res = res_dic[id]['reason']
            else:
                res = res_dic[id]['response']
            if '# Answer:' in res:
                cot = res.split('# Answer:')[0]
            elif '\n\n' in res:
                cot = ('\n\n').join(res.split('\n\n')[:-1])
            else:
                cot = res
            cots = re.split(r'[\.|\n]', cot)
            cot_chunks = [chunk.strip() for chunk in cots if len(chunk) >= 3]
            # input = [f"{item['input'][1][i]}_{i}" for i in range(len(item['input'][1]))]

            draw_token_heatmap(item['input'][1], cot_chunks[1:], item['scores'][1:], path)    
                
 
    