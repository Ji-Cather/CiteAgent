import matplotlib.pyplot as plt



import os
from typing import List
# from evaluate.visualize.article import plot_gini
import networkx as nx
import matplotlib.dates as mdates
import pandas as pd
from LLMGraph.utils.io import readinfo
import numpy as np

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import copy
def generate_monthly_dates(start_date, months):
    # 转换为 datetime 对象
    start = datetime.strptime(start_date, '%Y-%m')  
    date_list = []

    for i in range(months):
        date_list.append(start + relativedelta(months=i))

    return date_list

import matplotlib.patches as patches
def add_arrowed_spines(ax,x = True):
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_position('zero')
        ax.spines[spine].set_path_effects([])
    
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    for direction in ['left', 'bottom']:
        xy = (1, 0) if direction == 'bottom' else (0, 1)  # arrow at the end
        style = patches.ArrowStyle.CurveFilledB(head_length=0.15, head_width=0.08)
        arrow = patches.FancyArrowPatch((0, 0), xy, transform=ax.transAxes,
                                        fc='k', lw=0, mutation_scale=15, arrowstyle=style)
        ax.add_patch(arrow)
    
    if x:
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_tick_params(direction='out')
    else:
        ax.yaxis.set_ticks_position('left')
        ax.yaxis.set_tick_params(direction='out')




# 新增函数：绘制core和core+ph数据的beta值对比图
def plot_core_data(save_dir: str = "evaluate/Graph/graph4"):
    """
    从data_paths中读取两个group数据，绘制beta随year变化的折线图
    
    参数:
    save_dir: 图片保存路径
    """
    # 设置默认字体
    # 设置默认字体为Times New Roman
    from matplotlib.font_manager import FontProperties
    import matplotlib as mpl

    
    data_paths = {
        "Core": "evaluate/Graph/graph4/core.csv",
        "Core + Periphery": "evaluate/Graph/graph4/core+ph.csv"
    }
    
    # 读取数据
    data_dict = {}
    for group_name, file_path in data_paths.items():
        if not os.path.exists(file_path):
            print(f"文件 {file_path} 不存在")
            return
        
        # 读取CSV数据
        data = pd.read_csv(file_path, header=None, names=['beta', 'year'])
        data_dict[group_name] = data
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 绘制两条折线
    colors = ['#E6846D', '#8DCDD5']
    markers = ['o', 's']
    
    for idx, (group_name, data) in enumerate(data_dict.items()):
        # 按年份排序
        data = data.sort_values('year')
        ax.plot(data['year'], data['beta'], 
                marker=markers[idx], 
                color=colors[idx], 
                label=group_name, 
                linewidth=2, 
                markersize=6)
    
    # 设置图形属性
    ax.set_ylabel('$\\beta$ coefficient', fontsize=22)

    
    # 设置x轴每10年显示一个标签
    all_years = sorted(set().union(*[data['year'] for data in data_dict.values()]))
    min_year = min(all_years)
    max_year = max(all_years)
    
    # 计算刻度，每10年一个点
    xticks = list(range(min_year, max_year + 1, 10))
        
     # y轴添加箭头
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.annotate('', xy=(0, 1), xytext=(0, 0),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle="->", color='black'))
    
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, rotation=0)
    yticks = ax.get_yticks()
    ax.set_yticklabels(yticks)  
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "beta_comparison.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图形已保存到: {save_path}")
    
    # 显示图形
    plt.show()
   







if __name__ == "__main__":
    
    # 调用新函数绘制core数据对比图
    plot_core_data()
    