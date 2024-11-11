import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
import pandas as pd
import os
import shutil
import json
import yaml
import time
import openai
import matplotlib.dates as mdates


def readinfo(data_dir):
    assert os.path.exists(data_dir),"no such file path: {}".format(data_dir)
    with open(data_dir,'r',encoding = 'utf-8') as f:
        data_list = json.load(f)
    return data_list





def draw_number_split():

    llm_name_map = {
        "gpt3.5": "GPT-3.5",
        "gpt4-mini": "GPT-4o-mini",
        "vllm": "LLAMA-3-70B",
        # "qwen2": "QWEN2."
    }
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Times New Roman'

    label_style_map = {
        "Single Author": "--",
        "Multiple Authors": "-"
    }
    label_config_map = {
        "Single Author": "search_shuffle_base_nosocial_{llm}",
        "Multiple Authors": "search_shuffle_base_{llm}"
    }
    # 设置日期格式
    fig, axs = plt.subplots(1, len(llm_name_map), figsize=(10, 5), sharey=True)

    num_count = readinfo("evaluate/Graph/graph6/num_count.json")
    idx = 0
    labels = []
    colors = ["#E6846D","#8DCDD5"]
    # 绘制累计增长分布图
    for llm_name, llm_name_plot in llm_name_map.items():
        ax = axs[idx]
        idy = 0
        for label,style in label_style_map.items():
            config = label_config_map[label].format(llm = llm_name)
            data = num_count[config] 
            # 将数据转换为DataFrame
            df = pd.DataFrame(data).T  # 转置DataFrame，使得title作为行索引
            
            df['time'] = pd.to_datetime(df['time'], format="%Y-%m")

            # 计算每个时间点的累计数量
            df_filtered = df[df['time'] > '2023-04-01']
            df_filtered = df_filtered[df_filtered['time'] < '2025-03-01']

            # 计算每个时间点的累计数量
            df_cumulative = df_filtered.groupby('time').size().cumsum().reset_index(name='cumulative_count')
                
            lines = ax.plot(df_cumulative['time'], df_cumulative['cumulative_count'], marker='o', linestyle=style,
                    label=label,color = colors[idy])
            # plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            # plt.gcf().autofmt_xdate()
            ax.tick_params(axis='x', rotation=30,labelsize=14)
            for line in lines:
                labels.append(line.get_label())
            idy += 1
        ax.set_title(llm_name_plot,fontsize=18)
        if idx ==0:
            ax.set_ylabel('Cumulative Article Count',fontsize=18)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # 
        plt.gcf().autofmt_xdate()
        
        idx+=1
    
    # plt.xlabel('Time')
    plt.subplots_adjust(top=0.8, bottom=0.23,wspace=0)
    # plt.title('Cumulative Growth of Articles Over Time')
    # plt.grid(True)
    fig.legend(labels=labels[:2],loc='lower center',ncol=4, fontsize=16)
    plt.savefig(f"evaluate/Graph/graph6/ex_figures/6_article_number.pdf")
    plt.clf()


if __name__ == "__main__":
    draw_number_split()