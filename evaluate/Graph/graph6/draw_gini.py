

import networkx as nx
import os
from collections import defaultdict
import numpy as np
# import pygraphviz as pgv
import matplotlib.pyplot as plt
# 设置默认字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']  # 选择你想要的字体
import networkx as nx
import json
import pandas as pd
import powerlaw
import numpy as np
from scipy.stats import skew
from matplotlib.ticker import FormatStrFormatter


def readinfo(data_dir):
    assert os.path.exists(data_dir),"no such file path: {}".format(data_dir)
    with open(data_dir,'r',encoding = 'utf-8') as f:
        data_list = json.load(f)
    return data_list



def lorenz_curve(values):
    sorted_values = np.sort(values)
    cum_values = np.cumsum(sorted_values)
    cum_percentage = cum_values / cum_values[-1]
    lorenz_curve = np.insert(cum_percentage, 0, 0)
    return lorenz_curve

# 计算基尼系数
def gini_coefficient(x):
    n = len(x)
    x = np.array(x)
    x_sum = np.sum(x)
    x = np.sort(x)
    index = np.arange(1, n + 1)
    return 2 * np.sum((2 * index - n - 1) * x) / (n * x_sum) - (n + 1) / n





def draw_gini_2kinds():
    llms_path_map ={
        "GPT-3.5":"evaluate/Graph/graph6/reason_analysis_nosocial/gpt3.5/",
        "GPT-4o-mini": "evaluate/Graph/graph6/reason_analysis_nosocial/gpt4-mini/",
        "LLAMA-3-70B": "evaluate/Graph/graph6/reason_analysis_nosocial/vllm/",
        
        # "QWEN2.":"evaluate/Graph/graph6/reason_analysis_nosocial/qwen2"
    }
    llm_name_map ={
       "gpt3.5":"GPT-3.5", 
        "gpt4-mini":"GPT-4o-mini",
        "vllm":"LLAMA-3-70B",
        # "qwen2":"QWEN2."
    }

    compare_configs_map = {
        
        # "search_shuffle_no_country_{llm}":"PB.",
        "search_shuffle_anonymous_{llm}":"Anonymous",
        # "search_shuffle_base_nosocial_{llm}":"Single Author",
        "search_shuffle_base_{llm}":"Public",
    }
    col_map ={
        "gini_cite":"Citation",
        "gini_num":"Paper Number",
    }
    # gini = pd.read_csv("evaluate/Graphgraph7/gini_filtered.csv",index_col=0)
    gini = pd.read_csv("evaluate/Graph/graph6/gini.csv",index_col=0)
    gini_delta = {}
    for llm in llm_name_map.keys():
        gini_llm = gini.loc[llm]
        gini_delta[llm] = (
            gini_llm[gini_llm["config"]==f"search_shuffle_base_{llm}"],
            gini_llm[gini_llm["config"]==f"search_shuffle_anonymous_{llm}"]
        )

    
    from matplotlib.ticker import FuncFormatter
    # 设置 y 轴格式为两位小数
    def format_func(value, tick_number):
        return f'{value:.2f}'
    import matplotlib.cm as cm
    # color_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k','orange']
    # ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    import seaborn as sns
    colors_map_a = sns.color_palette("rocket", as_cmap=True)
    colors_map_a = sns.color_palette('viridis', as_cmap=True)
    colors_map_a = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
    colors_map_a = sns.color_palette("Set2", as_cmap=True)
    # colors_map_a = sns.color_palette("Paired",as_cmap=True)

    colors = ["#F5AB5E","#5F97C6"]
    # colors_map_a = sns.color_palette("hls", 8, as_cmap=True)
    
    # plt.figure(figsize=(18, 6))
    fig, axs = plt.subplots(2, figsize=(8, 10))
    width = 0.3  # the width of the bars
    x = np.arange(len(llm_name_map))
    # axs[0].axhline(y=0.3, color="gray", linestyle='--', label="co-Gini = 0.3")
    labels = []
    for col,ax in zip (col_map.keys(),axs):
        idx = 0
        for setting, id in zip(["Anonymous","Public"],
                                [1,0]):
            # values = [gini_delta[llm][idx][col].values[0] for llm in llm_name_map.keys()]
            values = [np.average(gini_delta[llm][id][col].values) for llm in llm_name_map.keys()]
            stds = [np.std(gini_delta[llm][id][col].values) for llm in llm_name_map.keys()]
            rects1 = ax.bar(x+ id*width, values, width, label=setting,
                            color = colors[id],yerr=stds)
            labels.append(rects1.get_label())

        # ax2 = ax.twiny()  # 创建第二个x轴
        # ax2.set_ylim(ax.get_ylim())  # 使第二个x轴与第一个x轴对齐
        # ax2.set_ylabel(col_map[col])
        # ax2.xaxis.set_visible(False)
        ax.set_title(col_map[col],fontsize=22)
        ax.legend(loc='best',fontsize=22,ncol=1, frameon=False)
        idx+=1
        

    ax.set_xticks(x+ 0.5*width)
    
    ax.set_xticklabels(llms_path_map.keys(),fontsize=18)
    
   
    for ax in axs:
        # ax.tick_params(axis='both', which='major', labelsize=16)  # 更改主要刻度标记的字体大小
        # ax.tick_params(axis='both', which='minor', labelsize=18)  # 更改次要刻度标记的字
        ax.yaxis.set_tick_params(labelsize=18) 
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # plt.subplots_adjust(top=0.8, bottom=0.15, left=0.13, hspace=0.1)
    # handles, labels = axs[0].get_legend_handles_labels()
    # # handles_2,labels_2 = axs[1].get_legend_handles_labels()
    # fig.legend(handles=handles, labels=labels,loc='lower center',ncol=3, fontsize=20)
    axs[1].set_ylim(0.5,1)
    axs[0].set_ylim(0,0.6)
    axs[0].yaxis.set_major_formatter(FuncFormatter(format_func))
    axs[0].xaxis.set_visible(False)
    
    fig.text(0.01, 0.5, "Gini Coefficient", va='center', rotation='vertical', fontsize=22)
    # fig.text(0.45, 0.07, "Country", va='center', fontsize=22)
    axs[1].yaxis.set_major_formatter(FuncFormatter(format_func))
    # 显示图形
    plt.savefig(f"evaluate/Graph/graph6/ex_figures/6_fits.pdf")
    plt.clf()

# llms = ["gpt3.5","gpt4", "qwen2", "vllm"]
# for llm in llms:
#     draw_lorenz_curve(llm)
# draw_fit_impact()
if __name__ == "__main__":
   draw_gini_2kinds()