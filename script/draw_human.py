import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Prepare the data
data = {
    'Case': ['L@1', 'L@1', 
                'L@1', 'L@2', 'L@2', 
                'L@2',
                'M@1', 'M@1','M@1',
                'M@2','M@2','M@2'],
    'Type': ['Win', 'Tie', 'Lose', 'Win', 'Tie', 'Lose','Win', 'Tie', 'Lose', 'Win', 'Tie', 'Lose'],
    'Votes': [29, 56, 15, 24, 58, 18, 26, 58, 16, 42, 48, 10]
}

# data = {
#     'Dataset': ['AQA.', 'GSM.', 'SIQA', 'WG.', 'PW', 'PQA.'],
#     'CCR':[-0.40, -0.54, -0.08, 0.07, -0.002, 0.02]
# }
 


# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Create the bar plot
plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid",font='Times New Roman')
custom_colors = ['#7976A2', '#4A5E65', '#E29957', '#86B5A1', '#B95A58', '#4292C6']
# ax = sns.barplot(x='Dataset', y='CCR', data=df, palette=sns.color_palette(custom_colors),width=0.5)
ax = sns.barplot(x='Case', y='Votes', hue='Type', data=df, palette=sns.color_palette(custom_colors),width=0.5)
# ax.set_xlabel('ax.get_xlabel()', fontsize=22)  # X轴标签
plt.xlabel('')
ax.set_ylabel(ax.get_ylabel(), fontsize=24)  # Y轴标签
# ax.legend(fontsize='18', loc='upper left')  # 图例
ax.legend(fontsize='22', loc='upper left') 
# 调整刻度文字大小
ax.tick_params(axis='both', which='major', labelsize=22)

# 调整图像边距
plt.subplots_adjust(left=0.07, right=0.99, top=0.98, bottom=0.10)

plt.savefig('./human.pdf')
plt.close()

# Additional plot formatting