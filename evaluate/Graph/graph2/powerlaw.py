import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter
# 设置 y 轴格式为两位小数
def format_func(value, tick_number):
    return f'{value:.2f}'

def plot_D(root_dir ="evaluate/Graph/graph2/powerlaw_reason"):
    dfs = {
        "$D$":pd.read_csv(f"{root_dir}/D.csv",index_col=0),
        
    }
    llm_name_map ={
       "gpt3.5":"GPT-3.5",
        "gpt4-mini":"GPT-4o-mini",
        # "llama8b":"LLAMA8B",
        "vllm":"LLAMA-3-70B",
    }
    column_map = {
        "base":"Base",
        "recall100":"R.S",
        "noinfo":"C.B"
    }
    
    for df in dfs.values():
        df.rename(columns=column_map, inplace=True)
   
    # 创建柱状图
    fig, axs = plt.subplots(len(dfs), figsize=(8, 5))
    if len(dfs) == 1:
        axs = [axs]
    width = 0.3  # the width of the bars
    colors = ["#F5AB5E","#5F97C6","#F09496"]
    x = np.arange(len(llm_name_map))
    labels = []
    for fig_title,ax in zip (list(dfs.keys()),axs):
        df = dfs[fig_title]
        for idx,col in enumerate(column_map.values()):
            # values = [gini_delta[llm][idx][col].values[0] for llm in llm_name_map.keys()]
            values = [np.average(df.loc[llm,col].values) for llm in llm_name_map.keys()]
            stds = [np.std(df.loc[llm,col].values) for llm in llm_name_map.keys()]
            rects1 = ax.bar(x+ idx*width, values, width, label=col,
                            color = colors[idx],yerr=stds)
            labels.append(rects1.get_label())
    
        ax.set_title(fig_title,fontsize=22)
        ax.legend(loc='best',fontsize=22,ncol=2, frameon=False)
        
    ax.set_xticks(x+ width)
    ax.set_xticklabels(llm_name_map.values(),fontsize=18)
        
    for ax in axs:
        # ax.tick_params(axis='both', which='major', labelsize=16)  # 更改主要刻度标记的字体大小
        # ax.tick_params(axis='both', which='minor', labelsize=18)  # 更改次要刻度标记的字
        ax.yaxis.set_tick_params(labelsize=18) 
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


    axs[0].set_ylim(0.1,0.35)
    # axs[0].yaxis.set_major_formatter(FuncFormatter(format_func))
    # axs[0].xaxis.set_visible(False)
    axs[-1].xaxis.set_visible(True)
    
    # axs[1].yaxis.set_major_formatter(FuncFormatter(format_func))
    # 显示图形
    plt.savefig(f"evaluate/Graph/graph2/reason_figures/14_powerlaw_reason_d.pdf")
    plt.clf()

def plot_maxk(root_dir ="evaluate/Graph/graph2/powerlaw_reason"):
    dfs = {
        "$\max(k)$":pd.read_csv(f"{root_dir}/largest_indegree.csv",index_col=0)
    }
    llm_name_map ={
       "gpt3.5":"GPT-3.5",
        "gpt4-mini":"GPT-4o-mini",
        # "llama8b":"LLAMA8B",
        "vllm":"LLAMA-3-70B",
    }
    column_map = {
        "base":"Base",
        "recall100":"R.S",
        "noinfo":"C.B"
    }
    
    for df in dfs.values():
        df.rename(columns=column_map, inplace=True)
   
    # 创建柱状图
    fig, axs = plt.subplots(len(dfs), figsize=(8, 5))
    if len(dfs) == 1:
        axs = [axs]
    width = 0.3  # the width of the bars
    colors = ["#F5AB5E","#5F97C6","#F09496"]
    x = np.arange(len(llm_name_map))
    labels = []
    for fig_title,ax in zip (list(dfs.keys()),axs):
        df = dfs[fig_title]
        for idx,col in enumerate(column_map.values()):
            # values = [gini_delta[llm][idx][col].values[0] for llm in llm_name_map.keys()]
            values = [np.average(df.loc[llm,col].values) for llm in llm_name_map.keys()]
            stds = [np.std(df.loc[llm,col].values) for llm in llm_name_map.keys()]
            rects1 = ax.bar(x+ idx*width, values, width, label=col,
                            color = colors[idx],yerr=stds)
            labels.append(rects1.get_label())
    
        ax.set_title(fig_title,fontsize=22)
        ax.legend(loc='best',fontsize=22,ncol=2, frameon=False)
        
    ax.set_xticks(x+ width)
    ax.set_xticklabels(llm_name_map.values(),fontsize=18)
        
    for ax in axs:
        # ax.tick_params(axis='both', which='major', labelsize=16)  # 更改主要刻度标记的字体大小
        # ax.tick_params(axis='both', which='minor', labelsize=18)  # 更改次要刻度标记的字
        ax.yaxis.set_tick_params(labelsize=18) 
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    axs[-1].set_ylim(0,120)
    
    axs[-1].xaxis.set_visible(True)
    
    # axs[1].yaxis.set_major_formatter(FuncFormatter(format_func))
    # 显示图形
    plt.savefig(f"evaluate/Graph/graph2/reason_figures/14_powerlaw_reason_mi.pdf")
    plt.clf()

if __name__ == "__main__":
    plot_D()
    plot_maxk()
