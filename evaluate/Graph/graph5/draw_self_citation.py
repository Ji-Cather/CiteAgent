import matplotlib.pyplot as plt

import numpy as np
import os

import json
import os
import torch
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

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
    

def plot_gt_preference_gini():
    
    sc_path ="evaluate/article/preference/gt/self_citation.json"
    sc= readinfo(sc_path)
    countrys = {"united states":"US",
                "united kingdom":"UK",
                "canada":"CA",
                "australia":"AU",
                "germany":"DE",
                "netherlands":"NL",
                "india":"IN",
                "france":"FR",
                "china":"CN",
                }
    rps_df = pd.read_csv("evaluate/article/preference/gt/preference_RPS.csv",index_col=0)

    gridspec_kw = {'width_ratios': [8, 2],}
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharey=False)
    labels = []
    
    
    bar_width = 0.3
    idx = id
    
    # 定义每组柱子的x轴
    index = np.arange(len(countrys))
    import matplotlib.cm as cm
    # color_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k','orange']
    # ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    import seaborn as sns
    colors_map_a = sns.color_palette("rocket", as_cmap=True)
    colors_map_a = sns.color_palette('viridis', as_cmap=True)
    colors_map_a = sns.color_palette("Paired",as_cmap=True)
    # colors_map_a = sns.color_palette("hls", 8, as_cmap=True)
    colors_a = colors_map_a(np.linspace(0, 1, 2) )
    # colors_ = colors_map_a(np.linspace(0, 1, 2) )
   
    # 设置正方向的误差（下边的误差为0）
    ax_1 = axs[0]

    ax_1.bar(index, [sc["citeseer_1"][c] for c in countrys.keys()], bar_width, label="Citeseer",
            color= colors_a[0])
    ax_1.bar(index+bar_width, [sc["cora_1"][c] for c in countrys.keys()], bar_width, label="Cora",
            color= colors_a[1])
    ax_1.set_ylabel("Self-Citation Rate",fontsize = 16)

    ax_2 = axs[1]
    bar1= ax_2.bar(index, rps_df.loc["GPT-3.5",countrys.keys()], bar_width, label="Citeseer",
            color= colors_a[0])
    bar2 = ax_2.bar(index+bar_width, rps_df.loc["LLAMA-3-70B",countrys.keys()], bar_width,       label="Cora",
            color= colors_a[1])
    ax_2.axhline(y=1, color='black', linestyle='--', label=f'RPS = 1', alpha=0.7)
    ax_2.set_ylabel(f"RPS",fontsize = 16)
    ax_2.tick_params(axis='y', labelsize=14)
    ax_2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax_2.set_xticks(index + 0.5 * bar_width)
    ax_2.set_xticklabels(list(countrys.values()),fontsize = 16,ha='right')
    ax_1.set_xticks(index + 0.5 * bar_width)
    ax_1.get_xaxis().set_visible(False)
    ax_1.tick_params(axis='y', labelsize=14)
    # plt.xticks(rotation=30, ha='right')  # 将国家名称标签旋转45度以减少重叠
    plt.subplots_adjust(top=0.8, bottom=0.13,hspace=0.1)
    # labels =[*labels[1:],labels[0]]
    handles, labels = axs.flat[0].get_legend_handles_labels() 
    handles_2, labels_2 = axs.flat[1].get_legend_handles_labels()
    handles =[handles_2[0],*handles]
    fig.legend(handles=handles, labels=[labels_2[0],*labels],loc='lower center',ncol=6, fontsize=16)
    # 显示图形
    plt.savefig(f"evaluate/visualize/for_paper/preference_all.pdf")


def plot_distortion_examine():
    # plot top行 四个图 
    src_datas = get_src_data()
    llms = ["GPT-3.5",
            "GPT-4o-mini",
            "LLAMA-3-70B"]
    dfs = [pd.DataFrame({k:src_datas[k]}) for k in list(src_datas.keys())[:1]]
    for llm in llms:
        src_datas_sub = src_datas[llm]
        for src_data_sub in src_datas_sub:
            dfs.append(pd.DataFrame({llm:src_data_sub}))
    src_datas = pd.concat(dfs,axis=1).T

    src_datas_equal = get_src_data_equal()
    dfs_equal = [pd.DataFrame({k:src_datas_equal[k]}) for k in list(src_datas_equal.keys())[:1]]
    for llm in llms:
        src_datas_sub = src_datas_equal[llm]
        for src_data_sub in src_datas_sub:
            dfs_equal.append(pd.DataFrame({llm:src_data_sub}))
    src_datas_equal = pd.concat(dfs_equal,axis=1).T

    gridspec_kw = {'height_ratios': [5,5,5]}
    fig, axs = plt.subplots(3, 1, figsize=(8, 15), sharey=False,gridspec_kw=gridspec_kw)
    x = np.arange(src_datas.shape[1])
    import matplotlib.cm as cm
    # color_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k','orange']
    # ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    color_map_a = cm.get_cmap('plasma', 6)  # 获取color map
    import seaborn as sns
    colors_map_a = sns.color_palette("rocket", as_cmap=True)
    colors_map_a = sns.color_palette('viridis', as_cmap=True)
    colors_map_a = sns.color_palette("Paired",as_cmap=True)
    # colors_map_a = sns.color_palette("hls", 8, as_cmap=True)
    colors_a = colors_map_a(np.linspace(0, 1, 6))
    colors_a = ["#6bbc46","#b4dea2","#d3270B","#f29091","#3b549d","#9eaad1"]
    labels_sr = []

    gt = ["Scopus"]
    width = 0.3
    ax_sr_gt = axs[0]
    for i in range(len(gt)):
        bar = ax_sr_gt.bar(x+i*width, src_datas.loc[gt[i],:], width, label=gt[i],color = colors_a[i])
        labels_sr.append(bar.get_label())
    ax_sr_gt.get_xaxis().set_visible(False)

    llms = ["GPT-3.5","GPT-4o-mini","LLAMA-3-70B"]
    ax_sr_llm = axs[1]
    width = 0.2
    
    for i in range(len(llms)):
        bar = ax_sr_llm.bar(x+i*width, 
                            src_datas.loc[llms[i],:].mean(axis=0), width, label=llms[i],color = colors_a[2+i],
                           
                            )
        labels_sr.append(bar.get_label())
    ax_sr_llm.get_xaxis().set_visible(False)

    ax_equal_llm = axs[2]
    for i in range(len(llms)):
        bar = ax_equal_llm.bar(x+i*width, 
                            # src_datas_equal.loc[llms[i],:].mean(axis=0), 
                            src_datas_equal.loc[llms[i],src_datas.columns].mean(axis=0), 
                            width, 
                            label=llms[i],
                            color = colors_a[2+i],
                            )
        labels_sr.append(bar.get_label())
        
    for ax in axs:
        # ax.tick_params(axis='both', which='major', labelsize=16)  # 更改主要刻度标记的字体大小
        # ax.tick_params(axis='both', which='minor', labelsize=18)  # 更改次要刻度标记的字
        ax.yaxis.set_tick_params(labelsize=19) 
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.legend(loc='best',ncol=1, fontsize=22,frameon=False)
    axs[0].set_title('Real Citation Graph',fontsize = 22)
    axs[1].set_title('Generated Graph (Base)',fontsize = 22)
    axs[2].set_title('Generated Graph (Equal Author)',fontsize = 22)
    axs[2].set_ylim(0, 0.6)
    axs[-1].set_xticks(list(v+width for v in range(len(src_datas.columns)))) 
    axs[-1].set_xticklabels(src_datas.columns,fontsize = 19)
    fig.text(0.02, 0.5, "Self Citation Rate (SCR)", va='center', rotation='vertical', fontsize=22)
    fig.text(0.45, 0.07, "Country", va='center', fontsize=22)
    # plt.subplots_adjust(top=0.9, bottom=0.14,left=0.05, hspace=0,wspace=0)
    # plt.subplots_adjust(top=0.8, bottom=0.15, left=0.13, hspace=0.1)
    # fig.legend(labels=labels_sr,loc='lower center',ncol=3, fontsize=22)
    plt.savefig(f"evaluate/Graph/graph5/distortion_figures/4_self_citation_all.pdf")

def get_src_data():
    plt_key = "self_citation"
    json_key = "self_citation_rate"
    # plt_key = "article_num"
    # json_key = "country_pub_nums"
    llms_path_map = { 
          "GPT-3.5":f"evaluate/Graph/graph5/self_citation/gpt3.5",
            "GPT-4o-mini": f"evaluate/Graph/graph5/self_citation/gpt4",
            "LLAMA-3-70B": f"evaluate/Graph/graph5/self_citation/vllm",
            }

    sr_rates = readinfo(f"evaluate/Graph/graph5/self_citation/gt/self_citation.json")

    # for llm_name, path in llms_path_map.items():
    #     data_path = os.path.join(path,f"{json_key}.json")
    #     sr_rate = readinfo(data_path)
    #     sr_rates[llm_name] = sr_rate

    # countrys_in_common = set(sr_rates["GPT-3.5"].keys()) & \
    #     set(sr_rates["GPT-4o-mini"].keys()) \
    #     & set(sr_rates["LLAMA-3-70B"].keys())
    #https://www.sciencedirect.com/science/article/pii/S2405650221000253
    sr_rates={"Scopus":{
        "United States":0.47,
        "United Kingdom":0.205,
        "Germany":0.223,
        "China":0.355,
        "Canada":0.154,
        "India":0.216,
        "Israel":0.09,
        "Netherlands":0.135,
        "Australia":0.162,
        "Austria":0.086,
        "France":0.213,
        "Japan":0.217,
        "South Korea":0.15,
        "Spain":0.23,
        "Singapore":0.092,
        "Saudi Arabia":0.082,
        "Brazil":0.22,
        "Switzerland":0.107,
        "Hong Kong":0.104,
        "New Zealand":0.076
    }}
    
    countrys = {
        "United States":"US",
        "United Kingdom":"UK",
        "Germany":"DE",
        "China":"CN",
        "Canada":"CA",
        "India":"IN",
        "Israel":"IL",
        "Netherlands":"NLD",
        "Australia":"AU",
        "Austria":"AUT",
        "France":"FR",
        "Japan":"JP",
        "South Korea":"KR",
        "Spain":"ES",
        "Singapore":"SG",
        "Saudi Arabia":"SA",
        "Brazil":"BRA",
        "Switzerland":"SUI",
        "Hong Kong":"HK",
        "New Zealand":"NT"
        }
    
    countrys = {k.lower():v for k,v in countrys.items()}
    countrys_in_common = [
        "United States",
        "United Kingdom",
        "Germany",
        "China",
        "Canada",
        "Israel",
        "Australia",
        "France",
        "Japan",
        "South Korea",
        "Singapore",
        "Saudi Arabia",
        "Switzerland",
    ]
    # countrys_in_common = list(filter(lambda x: x in countrys.keys(), countrys_in_common))
    countrys_in_common = list(v.lower() for v in countrys_in_common)
    sr_rates = {data_name:{countrys[k.lower()]:v for k,v in sr_rate_dict.items() if k.lower() in countrys_in_common} 
                for data_name, sr_rate_dict in sr_rates.items()}

    for llm_name in llms_path_map.keys():
        files = ["self_citation_rate_1.json","self_citation_rate_2.json","self_citation_rate_3.json"]
        sr_rates[llm_name] = []
        path = llms_path_map[llm_name]
        for file in files:
            data_path = os.path.join(path,file)
            sr_rate = readinfo(data_path)
            sr_rate = dict(filter(lambda x: x[0] in countrys_in_common, sr_rate.items()))
            sr_rate = {countrys[k]:v for k,v in sr_rate.items()}
            sr_rates[llm_name].append(sr_rate)
    return sr_rates


def get_src_data_equal():
    json_key = "self_citation_rate"
    # plt_key = "article_num"
    # json_key = "country_pub_nums"
   
    llms_path_map = {
        "GPT-3.5":f"evaluate/Graph/graph5/self_citation_equal/gpt3.5",
        "GPT-4o-mini": f"evaluate/Graph/graph5/self_citation_equal/gpt4",
        "LLAMA-3-70B": f"evaluate/Graph/graph5/self_citation_equal/vllm",
    }

    sr_rates = readinfo(f"evaluate/Graph/graph5/self_citation/gt/self_citation.json")


    #https://www.sciencedirect.com/science/article/pii/S2405650221000253
    sr_rates={"Scopus":{
        "United States":0.47,
        "United Kingdom":0.205,
        "Germany":0.223,
        "China":0.355,
        "Canada":0.154,
        "India":0.216,
        "Israel":0.09,
        "Netherlands":0.135,
        "Australia":0.162,
        "Austria":0.086,
        "France":0.213,
        "Japan":0.217,
        "South Korea":0.15,
        "Spain":0.23,
        "Singapore":0.092,
        "Saudi Arabia":0.082,
        "Brazil":0.22,
        "Switzerland":0.107,
        "Hong Kong":0.104,
        "New Zealand":0.076
    }}
    
    countrys = {
        "United States":"US",
        "United Kingdom":"UK",
        "Germany":"DE",
        "China":"CN",
        "Canada":"CA",
        "India":"IN",
        "Israel":"IL",
        "Netherlands":"NLD",
        "Australia":"AU",
        "Austria":"AUT",
        "France":"FR",
        "Japan":"JP",
        "South Korea":"KR",
        "Spain":"ES",
        "Singapore":"SG",
        "Saudi Arabia":"SA",
        "Brazil":"BRA",
        "Switzerland":"SUI",
        "Hong Kong":"HK",
        "New Zealand":"NT"
        }
    countrys_in_common = [
        "United States",
        "United Kingdom",
        "Germany",
        "China",
        "Canada",
        "Israel",
        "Australia",
        "France",
        "Japan",
        "South Korea",
        "Singapore",
        "Saudi Arabia",
        "Switzerland",
    ]
    countrys = {k.lower():v for k,v in countrys.items()}
    countrys_in_common = list([v.lower() for v in countrys_in_common])
    sr_rates = {data_name:{countrys[k.lower()]:v for k,v in sr_rate_dict.items() if k.lower() in countrys_in_common} 
                for data_name, sr_rate_dict in sr_rates.items()}

    for llm_name in llms_path_map.keys():
        files = ["self_citation_rate_1.json","self_citation_rate_2.json","self_citation_rate_3.json"]
        sr_rates[llm_name] = []
        path = llms_path_map[llm_name]
        for file in files:
            data_path = os.path.join(path,file)
            sr_rate = readinfo(data_path)
            sr_rate = dict(filter(lambda x: x[0] in countrys_in_common, sr_rate.items()))
            sr_rate = {countrys[k]:v for k,v in sr_rate.items()}
            sr_rates[llm_name].append(sr_rate)
    return sr_rates

def plot_llm_src_gini():

    plt_key = "self_citation"
    json_key = "self_citation_rate"
    # plt_key = "article_num"
    # json_key = "country_pub_nums"
    llms_path_map = { 
            "GPT-3.5":f"evaluate/Graph/graph5/self_citation/gpt3.5",
            "GPT-4o-mini": f"evaluate/Graph/graph5/self_citation/gpt4",
            "LLAMA-3-70B": f"evaluate/Graph/graph5/self_citation/vllm",
            # "QWEN2.":f"evaluate/Graph/graph5/self_citation/qwen2"
            }


    sr_rates = {}
    files = ["self_citation_rate_1.json","self_citation_rate_2.json","self_citation_rate_3.json"]
    for llm_name, path in llms_path_map.items():
        sr_rates[llm_name] = []
        for file in files:
            data_path = os.path.join(path,file)
            sr_rate = readinfo(data_path)
            sr_rates[llm_name].append(sr_rate)
        

    countrys_in_common = set(sr_rates["GPT-3.5"].keys()) & \
        set(sr_rates["GPT-4o-mini"].keys()) \
        & set(sr_rates["LLAMA-3-70B"].keys()) \

    forbidden_countrys = ["israel", "netherlands","france"] # all None
    countrys_in_common = list(filter(lambda x: x not in forbidden_countrys, countrys_in_common))

    countrys_in_common = [
        "united states",
        "china",
        "united kingdom",
        "switzerland",
        "japan",
        "germany",
        "saudi arabia",
        "india",
        "australia",
        "singapore",
        "canada",
        "spain",
        "new zealand",
        "south korea"
    ]

    for llm_name in sr_rates.keys():
        sr_rates[llm_name] = {k:v for k,v in sr_rates[llm_name].items() if k in countrys_in_common}

    # 获取所有的键
    keys = countrys_in_common

    # 定义柱的宽度
    bar_width = 0.2

    # 定义每组柱子的x轴
    index = np.arange(len(keys))

    # 绘制柱状图
    fig, ax = plt.subplots()
    import matplotlib.cm as cm
    # color_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k','orange']
    # ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    color_map_a = cm.get_cmap('plasma', len(sr_rates))  # 获取color map
    colors_a = color_map_a(np.linspace(0, 1, len(sr_rates)))

    for idx, k in enumerate(sr_rates.keys()):
        values = [np.average([sr_rate_[c] for sr_rate_ in sr_rates[k]]) for c in countrys_in_common]
        stds = [np.std([sr_rate_[c] for sr_rate_ in sr_rates[k]]) for c in countrys_in_common]
        ax.bar(index+idx*bar_width, values, bar_width, label=f"{k}",
            color= colors_a[idx],)

    # 添加标签、标题和图例
    # ax.set_xlabel('')

    ax.set_ylabel('Self-Citation Rate by Country',fontsize = 16)
    # ax.set_ylabel('Article Number by Country')
    # ax.set_title('Bar graph of four dictionaries')
    ax.set_xticks(index + 1.5 * bar_width)
    ax.set_xticklabels([key.capitalize() for key in keys],fontsize = 16)
    ax.legend(fontsize = 16)

    plt.xticks(rotation=45, ha='right')  # 将国家名称标签旋转45度以减少重叠
    plt.tight_layout()
    # 显示图形
    plt.savefig(f"evaluate/Graph/graph5/distortion_figures/4_{plt_key}.pdf")

if __name__ == "__main__":
    plot_distortion_examine()