import numpy as np
import matplotlib.pyplot as plt
# 设置默认字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']  # 选择你想要的字体
import pandas as pd
import os
from matplotlib.patches import Patch
import seaborn as sns
import json
import os
import torch
import pandas as pd

def readinfo(data_dir):
    file_type = os.path.basename(data_dir).split('.')[1]
    # assert os.path.exists(data_dir),"no such file path: {}".format(data_dir)
    try:
        if file_type == "pt":
            return torch.load(data_dir)
    except Exception as e:
        data_dir = os.path.join(os.path.dirname(data_dir),f"{os.path.basename(data_dir).split('.')[0]}.json")
    try:
        with open(data_dir,'r',encoding = 'utf-8') as f:
            data_list = json.load(f)
            return data_list
    except:
        raise ValueError("file type not supported")

def draw_cross_llm_explain():
    # 示例数据

    categories = [
        "power_law",
        "lognormal",
        "stretched_exponential",
    ]

    index_map ={
        "power_law": "Power Law",
        "lognormal": "Log-Normal",
        "stretched_exponential": "Stretched-Eponential",
    }
    columns_map = {
        'content': "Paper Content",
        'cite': "Paper Citation",
        'paper_time': "Paper Timeliness",
        'topic': "Paper Topic",
        'author': 'Author Name',
        'author_cite': "Author Citation",
        'country': "Author Country",
        # 'author_topic':"Author Expertise",
        # 'social': "Social",
    }

    save_root = "evaluate/Graph/graph3/pl_reason_figures/"

    llms_df_map ={
        "GPT-3.5":pd.read_csv("evaluate/Graph/graph3/llm_agent_reasons/gpt3.5/impact_explan_ver1.csv",index_col=0),
        "GPT-4o.": pd.read_csv("evaluate/Graph/graph3/llm_agent_reasons/gpt4-mini/impact_explan_ver1.csv",index_col=0),
        "LLAMA-3.": pd.read_csv("evaluate/Graph/graph3/llm_agent_reasons/vllm/impact_explan_ver1.csv",index_col=0),
        "QWEN2.": pd.read_csv("evaluate/Graph/graph3/llm_agent_reasons/qwen2/impact_explan_ver1.csv",index_col=0),
    }

    for k, df in llms_df_map.items():
        df = df.loc[categories]
        df.index = [index_map.get(id) for id in df.index]
        df.columns = [columns_map.get(id) for id in df.columns]
        df = df[columns_map.values()]
        llms_df_map[k] = df

    color_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k','orange']

    # 创建图表


    bar_width = 0.5

    fig, axs = plt.subplots(len(llms_df_map)//2, 2,
                            figsize=(10, 8), sharey=True)


    for id, llm_df in enumerate(llms_df_map.values()):
        df = llm_df
        bottoms = np.zeros(len(df.index))
        for id_c, column in enumerate(df.columns):
            axs[id//2][id%2].bar(df.index.to_list(), df[column].to_list(),
                width = bar_width,
                bottom = bottoms, 
                label=column, color=color_map[id_c])
            bottoms += df[column].to_list()
            axs[id//2][id%2].set_xticks([])  # 旋转x轴标签
        axs[id//2][id%2].set_ylabel('Relative explanation (%)')
        
        # # 设置两个x轴标签
        # ax2 = axs[id].twiny()  # 创建第二个x轴
        # # ax2.set_xlim(axs[id].get_xlim())  # 使第二个x轴与第一个x轴对齐
        # ax2.set_xlabel(list(llms_df_map.keys())[id])
        axs[id//2][id%2].set_title(f"{list(llms_df_map.keys())[id]}")
        # axs[id].tick_params(axis='x', which='both', 
        #                     bottom=False, top=False, 
        #                     labelbottom=False)


    # fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5))
    # plt.legend( labels, loc='lower left', bbox_to_anchor=(1, 1))
    # 创建统一的Legend
    handles, labels = axs[0][-1].get_legend_handles_labels()
    plt.legend( labels, loc='center left', bbox_to_anchor=(1, 1),
            handletextpad=2, columnspacing=2, labelspacing=2, )
    
    # 布局调整
    # plt.tight_layout()
    plt.subplots_adjust(right=0.7)  # 调整图表以给图例留出空间
    plt.savefig("evaluate/Graph/graph3/pl_reason_figures/3_llm_agent.pdf")



def draw_KS_cross_llm():
    llms_df_map ={
        "GPT-3.5":pd.read_csv("evaluate/Graph/graph3/llm_agent_reasons/gpt3.5/article_citation_all_power_law.csv",index_col=0),
        "GPT-4o.": pd.read_csv("evaluate/Graph/graph3/llm_agent_reasons/gpt4-mini/article_citation_all_power_law.csv",index_col=0),
        "LLAMA-3.": pd.read_csv("evaluate/Graph/graph3/llm_agent_reasons/vllm/article_citation_all_power_law.csv",index_col=0),
        "QWEN2.": pd.read_csv("evaluate/Graph/graph3/llm_agent_reasons/qwen2/article_citation_all_power_law.csv",index_col=0),
    }
    llm_name_map ={
       "gpt3.5":"GPT-3.5",
        "vllm":"LLAMA-3.",
        "gpt4-mini":"GPT-4o.",
        "qwen2":"QWEN2."
    }
    ks_list =[]
    for llm_name, llm_plt_name in llm_name_map.items():
        config_name = f"search_shuffle_base_{llm_name}"
        df = llms_df_map[llm_plt_name]
        # df = df[df["ex_name"]==config_name]
        ks_list.append((df.loc["power_law", "KS"].mean(),df.loc["power_law", "KS"].std()))

    plt.figure(figsize=(4, 2))
    # 创建柱状图
    fig, ax = plt.subplots()
    # bars = ax.bar(list(llm_name_map.values()), [x[0] for x in ks_list], yerr=[x[1] for x in ks_list],  capsize=10, color='blue', alpha=0.7, 
    #               error_kw={'elinewidth': 2, 'ecolor': 'red', 'capthick': 2},label='Power-Law Distribution')
    bars = ax.bar(list(llm_name_map.values()), [x[0] for x in ks_list], capsize=10, color='#87CEFA', alpha=0.7)
     # 绘制带误差棒的图表
    plt.errorbar(list(llm_name_map.values()), [x[0] for x in ks_list], yerr=[x[1] for x in ks_list], 
                 fmt='-o', ecolor='#696969', capsize=5, label='KS: Power-Law Distribution') #Kolmogorov-Smirnov
    # plt.xlabel('X Axis')
    plt.ylabel('Log-Normal Positive($D$)')
    # plt.yticks(np.arange(0, 0.55, 0.05))
    plt.axhline(y=0.1, color='#696969', linestyle='--',label='$D$ threshold = 0.1', alpha=0.7)
    # plt.title('Plot with Error Bars')
    plt.legend()
    plt.grid(True)
    plt.savefig("evaluate/Graph/graph3/pl_reason_figures/3_cross_llm_ks.pdf")


def draw_llm_distribution():
    model_map = {
        "TPL": "truncated_power_law",
        "LN": "lognormal",
        "LNP": "lognormal_positive",
        "SE": "stretched_exponential",
        "EXP": "exponential",
        "PL": "power_law"
    }
    llms_path_map = {
        "GPT-3.5": "evaluate/Graph/graph3/llm_agent_reasons/gpt3.5/",
        "GPT-4o.": "evaluate/Graph/graph3/llm_agent_reasons/gpt4-mini/",
        "LLAMA-3.": "evaluate/Graph/graph3/llm_agent_reasons/vllm/",
        "QWEN2.": "evaluate/Graph/graph3/llm_agent_reasons/qwen2/"
    }
    llm_name_map = {
        "gpt3.5": "GPT-3.5",
        "vllm": "LLAMA-3.",
        "gpt4-mini": "GPT-4o.",
        "qwen2": "QWEN2."
    }
    columns_map = {
        'content': "Paper Content",
        'cite': "Paper Citation",
        'paper_time': "Paper Timeliness",
        'topic': "Paper Topic",
        'author': 'Author Name',
        'author_cite': "Author Citation",
        'country': "Author Country",
        'author_topic': "Author Expertise",
    }

    # Use a lighter, high saturation academic color palette
    colors_a = sns.color_palette("muted", len(model_map))
    colors_b = sns.color_palette("pastel", len(columns_map))

    for llm_name, llm_plt_name in llm_name_map.items():
        config_name = f"search_shuffle_base_{llm_name}"
        llm_path_pl = os.path.join(llms_path_map[llm_plt_name], "article_citation_all_power_law.csv")
        df = pd.read_csv(llm_path_pl, index_col=0)
        ks_list = []
        for model_name, model_id in model_map.items():
            ks_list.append((df.loc[model_id, "KS"].mean(), df.loc[model_id, "KS"].std()))

        # Create plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))

        # Bar plot
        bars = ax1.bar(list(model_map.keys()), [x[0] for x in ks_list], yerr=[x[1] for x in ks_list], capsize=16, color=colors_a, alpha=0.8)
        ax1.set_xticks(range(len(model_map)))
        ax1.set_xticklabels(list(model_map.keys()), fontsize=16)
        ax1.axhline(y=0.1, color='#696969', linestyle='--', label='$D$ threshold = 0.1', alpha=0.7)
        ax1.set_ylabel('K-S Distance ($D$)', fontsize=16)
        plt.yticks(fontsize=16)
        ax1.legend(fontsize=16)

        explain_path = os.path.join(llms_path_map[llm_plt_name], "impact_explan_ver1.csv")
        df = pd.read_csv(explain_path, index_col=0)
        values = df.loc["power_law", columns_map.keys()].to_list()
        categories = list(columns_map.values())

        def custom_autopct(pct, threshold=3):
            return ('%.1f%%' % pct) if pct >= threshold else ""

        # Pie chart
        wedges, texts, autotexts = ax2.pie(values, autopct=lambda pct: custom_autopct(pct), startangle=140, colors=colors_b,
                                           textprops={'fontsize': 14})

        # Save the figure
        plt.savefig(f"evaluate/Graph/graph3/pl_reason_figures/3_{llm_name}_distribution.pdf")

def draw_llm_distribution_all_ks():
    model_map = {
        "LN": "lognormal",
        "LNP": "lognormal_positive",
        "SE": "stretched_exponential",
        "EXP": "exponential",
        "TPL": "truncated_power_law",
        "PL": "power_law"
    }
    
    model_legend_map = {
        "LN": "Log-Normal",
        "LNP": "Log-Normal Positive",
        "EXP": "Exponential",
        "SE": "Stretched Exponential",
        "TPL": "Truncated Power-Law",
        "PL": "Power-Law"
    }

    root_data = "evaluate/Graph/graph3/citeseer_1"
    llm_name_map = {
        "gpt3.5": "GPT-3.5",
        "gpt4-mini": "GPT-4o-mini",
        "vllm": "LLAMA-3-70B",
    }
    columns_map = {
        'content': "Paper Content",
        'cite': "Paper Citation",
        'paper_time': "Paper Timeliness",
        'topic': "Paper Topic",
        'author': 'Author Name',
        'author_cite': "Author Citation",
        'country': "Author Country",
        'author_topic': "Author Expertise",
    }

    # Use a lighter, high saturation academic color palette
    colors_a = sns.color_palette("muted", len(model_map))
    fig, ax = plt.subplots(1, 3, figsize=(15, 6), sharey=False)

    labels = []
    idx = 0
    llm_path_pl = os.path.join(root_data, "article_citation_in_power_law.csv")
    df = pd.read_csv(llm_path_pl, index_col=0)
    patterns = ['/', '//', '|', '-', '+', 'x', 'o']
    for llm_name, llm_plt_name in llm_name_map.items():
        config_name = f"search_shuffle_base_{llm_name}"
        config_names = [f"fast_{llm_name}_2", f"fast_{llm_name}"]
        ks_list = []
        for model_name, model_id in model_map.items():
            # df = df[df["ex_name"]==config_name]
            ks_list.append((df[df["ex_name"].isin(config_names)].loc[model_id, "KS"].mean(), df[df["ex_name"].isin(config_names)].loc[model_id, "KS"].std()))
        ax1 = ax[idx]
        # Bar plot
        bars = ax1.bar(list(model_map.keys()), [x[0] for x in ks_list], yerr=[x[1] for x in ks_list], capsize=12, color=colors_a, alpha=0.8)
        id_p =0
        for bar, label in zip(bars, model_map.keys()):
            bar.set_label(model_legend_map[label])
            # bar.set_hatch(patterns[id_p])
            id_p+=1


        labels.extend([bar.get_label() for bar in bars])

        ax1.set_xticks(range(len(model_map)))
        ax1.set_xticklabels(list(model_map.keys()), fontsize=14)
        label = ax1.axhline(y=0.1, color='#696969', linestyle='--', label='$D$ threshold = 0.1', alpha=0.7)
        labels.append(label.get_label())
        # ax1.legend(fontsize=16)
        ax1.set_title(llm_plt_name, fontsize=16)
        idx+=1

    for ax_ in ax:
        y = ax_.get_yticks()
        y = ["{:.2f}".format(i) for i in y]
        ax_.set_yticklabels(y, fontsize=14)
    ax[0].set_ylabel('K-S Distance ($D$)', fontsize=16)
    
    plt.subplots_adjust(top=0.8, bottom=0.2, left=0.1, right=0.9, hspace=0.4)
    
    fig.legend(labels=labels[:7], loc='lower center', ncol=4, fontsize=16)
    # Save the figure
    plt.savefig(f"evaluate/Graph/graph3/pl_reason_figures/3_ks_all_distribution_3.pdf")
    plt.clf()

def draw_llm_change():
    
    llms_path_map = {
        "GPT-3.5": "evaluate/Graph/graph3/llm_agent_reasons/gpt3.5/",
        "GPT-4o.": "evaluate/Graph/graph3/llm_agent_reasons/gpt4-mini/",
        "LLAMA-3.": "evaluate/Graph/graph3/llm_agent_reasons/vllm/",
        "QWEN2.": "evaluate/Graph/graph3/llm_agent_reasons/qwen2/"
    }
    llm_name_map = {
        "gpt3.5": "GPT-3.5",
        "vllm": "LLAMA-3.",
        "gpt4-mini": "GPT-4o.",
        "qwen2": "QWEN2."
    }
    columns_map = {
        'content': "Paper Content",
        'cite': "Paper Citation",
        'paper_time': "Paper Timeliness",
        'topic': "Paper Topic",
        'author': 'Author Name',
        'author_cite': "Author Citation",
        'country': "Author Country",
        # 'author_topic': "Author Expertise",
    }
    columns_config_map = [
        "search_shuffle_no_content_{llm}",
        "search_shuffle_nocite_{llm}",
        "search_shuffle_nopapertime_{llm}",
        'search_shuffle_notopic_{llm}',
        "search_shuffle_noauthor_{llm}", 
        'search_shuffle_base_noauthorcite_{llm}', 
        "search_shuffle_no_country_{llm}",
        # 'search_shuffle_noauthortopic_{llm}', 
    ]

    # Use a lighter, high saturation academic color palette
    colors_b = sns.color_palette("pastel", len(columns_map))
    fig, ax = plt.subplots(1, 4, figsize=(15, 6), sharey=False)

    labels = []
    bar_width = 0.15
    fig, ax = plt.subplots(figsize=(15, 6))
    index = np.arange(len(columns_map))  
    idx = 0
    for llm_name, llm_plt_name in llm_name_map.items():
        df_fit = pd.read_csv(os.path.join(llms_path_map[llm_plt_name], "article_citation_all_power_law.csv"), index_col=0)
        # explain_path = os.path.join(llms_path_map[llm_plt_name], "impact_relative_change_ver1.csv")
        # df = pd.read_csv(explain_path, index_col=0)
        # values = df.loc["power_law", columns_map.keys()].to_list()
        values_base = df_fit[df_fit["ex_name"]==f"search_shuffle_base_{llm_name}"].loc["power_law", "KS"]
        mean_values = df_fit.groupby('ex_name')
        values = []
        for config in columns_config_map:
            sub_df = mean_values.get_group(config.format(llm=llm_name)).loc["power_law", "KS"]
            values.append(sub_df.values[-1]-values_base.values[-1])
            # values.append(abs(sub_df.values[0]-values_base.values[0]))
            # values.append(abs(sub_df.mean()-values_base.mean()))
            # values.append(sub_df.mean()-values_base.mean())

        bar = ax.bar(index+idx*bar_width, values, width=bar_width, label=llm_plt_name, color=colors_b[idx%4])
        labels.append(bar.get_label())
        
        idx+=1

    ax.set_xticks(index + 1.5 * bar_width)
    ax.set_xticklabels(columns_map.values(),fontsize = 14,rotation=25, ha='right')
    # ax.set_ylabel('$//Delta //theta$(%)', fontsize=16)
    ax.set_ylabel('$//Delta D$', fontsize=16)
    
    plt.subplots_adjust(top=0.9, bottom=0.2, left=0.1, right=0.9, hspace=0.1)
    
    fig.legend(labels=labels, loc='lower center', ncol=4, fontsize=16)
    # Save the figure
    plt.savefig(f"evaluate/Graph/graph3/pl_reason_figures/3_ks_all_change.pdf")

def draw_llm_reason():
    
    llms_path_map = {
        "GPT-3.5": "evaluate/Graph/graph3/llm_agent_reasons/gpt3.5/",
        "GPT-4o-mini": "evaluate/Graph/graph3/llm_agent_reasons/gpt4-mini/",
        "LLAMA-3-70B": "evaluate/Graph/graph3/llm_agent_reasons/vllm/",
        # "QWEN2.": "evaluate/Graph/graph3/llm_agent_reasons/qwen2/"
    }
    llm_name_map = {
        "gpt3.5": "GPT-3.5",
        "gpt4-mini": "GPT-4o-mini",
        "vllm": "LLAMA-3-70B",
        # "qwen2": "QWEN2."
    }
    columns_map = {
        'content': "Paper Content",
        'cite': "Paper Citation",
        'paper_time': "Paper Timeliness",
        'topic': "Paper Topic",
        'author': 'Author Name',
        'author_cite': "Author Citation",
        'country': "Author Country",
    }
    
    cols = [
        "Paper Content",
        "Paper Citation",
        "Paper Timeliness",
        "Paper Topic",
        'Author Name',
        "Author Citation",
        "Author Country",
    ]
    
    # Use a lighter, high saturation academic color palette
    colors_b = sns.color_palette("pastel", len(columns_map))
    fig, ax = plt.subplots(1, len(llm_name_map), figsize=(15, 5), sharey=False)
    ax = [ax]
    reasons_llm_counts = [
        readinfo(os.path.join("evaluate/Graph/graph3/reason/1000nodes", file)) for file in os.listdir("evaluate/Graph/graph3/reason/1000nodes")
    ]
    
    labels = []
    idx = 0
    
    for llm_name, llm_plt_name in llm_name_map.items():
        
        # explain_path = os.path.join(llms_path_map[llm_plt_name], "impact_explan_ver1.csv")
        # df = pd.read_csv(explain_path, index_col=0)
        # reasons_map = df.loc["power_law", columns_map.keys()].to_dict()
        # reasons_map = {columns_map[k]:reasons_map[k] for k in reasons_map.keys()}
        # reasons_map = parse_reason_dict_if(reasons_map)
        reasons_llm_alls = [reasons_llm_count[llm_name]["all"] for reasons_llm_count in \
        reasons_llm_counts]

        reasons_llm_all = {col:np.average([reasons_llm_all.get(col,0)
                                           for reasons_llm_all in reasons_llm_alls
                                           ]
                                          ) for col in columns_map.values()}
        reasons_llm_all = parse_reason_dict_if(reasons_llm_all) 
        reasons_llm_all ={k:v*100 for k,v in reasons_llm_all.items()}
        def custom_autopct(pct, threshold=3):
            return ('%.1f%%' % pct) if pct >= threshold else ""
         
        ax1 = ax[0][idx%4]
        wedges, texts, autotexts = ax1.pie([reasons_llm_all[c] for c in cols], autopct=lambda pct: custom_autopct(pct), startangle=140, colors=colors_b,
        textprops={'fontsize': 14}
        , radius=1.25)
        for wedge,label in zip(wedges,cols):
            wedge.set_label(label)
            labels.append(label)
       
        # ax2 = ax[1][idx%4]
        # # Pie chart
        # wedges, texts, autotexts = ax2.pie([reasons_map[c] for c in cols], autopct=lambda pct: custom_autopct(pct), startangle=140, colors=colors_b,
        #                                    textprops={'fontsize': 14}
        #                                    , radius=1.25)
        ax1.set_title(llm_plt_name, fontsize=18)
        if llm_name == "gpt3.5":
            # ax2.set_ylabel('LLM-CIA', fontsize=16)
            ax1.set_ylabel('LLM-SE', fontsize=18)
        for wedge,label in zip(wedges,cols):
            wedge.set_label(label)
            labels.append(label)
        
        idx+=1

    # bar_width = 0.15
    # fig, ax = plt.subplots(figsize=(15, 6))
    # index = np.arange(len(columns_map))  
    # idx = 0
    # for llm_name, llm_plt_name in llm_name_map.items():
    #     df_fit = pd.read_csv(os.path.join(llms_path_map[llm_plt_name], "article_citation_all_power_law.csv"), index_col=0)
    #     explain_path = os.path.join(llms_path_map[llm_plt_name], "impact_relative_change_ver1.csv")
    #     df = pd.read_csv(explain_path, index_col=0)
    #     values = df.loc["power_law", columns_map.keys()].to_list()

    #     ax.bar(index+idx*bar_width, values, width=bar_width, label=llm_plt_name, color=colors_b[idx%4])
    #     ax.set_title(llm_plt_name, fontsize=16)
    #     ax.set_ylabel('Relative Change', fontsize=16)
    #     idx+=1

    # ax.set_xticks(index + 1.5 * bar_width)
    # ax.set_xticklabels(columns_map.values(),fontsize = 14,rotation=30, ha='right')
    # ax[0].set_ylabel('K-S Distance ($D$)', fontsize=16)
    
    plt.subplots_adjust(top=0.9, bottom=0.2, left=0.1, right=0.9, hspace=0.1)
    
    fig.legend(labels=cols, loc='lower center', ncol=4, fontsize=18)
    # Save the figure
    plt.savefig(f"evaluate/Graph/graph3/pl_reason_figures/3_ks_all_reason_4.pdf")


def parse_reason_dict_if(impact_factor_dict):
    reason_map={
        "Paper Content":"Content",
        "Paper Citation":'Citation',
        "Paper Timeliness":'Timeliness',
        "Paper Topic":'Content',
        'Author Name':'Author',
        "Author Citation":'Citation',
        "Author Country":'Country',
        # 'author_topic': "Author Expertise",
    }
    reason_map = {k:k for k,v in reason_map.items()}
    transfered_dict = {}
    for k,v in impact_factor_dict.items():
        if reason_map[k] not in transfered_dict.keys():
            transfered_dict[reason_map[k]] = v
        else:
            transfered_dict[reason_map[k]] += v
        # transfered_dict[k] = v

    return transfered_dict

if __name__ == "__main__":
    # draw_llm_distribution_all_ks()
    draw_llm_reason()
