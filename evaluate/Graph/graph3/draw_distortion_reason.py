import seaborn as sns
import matplotlib.pyplot as plt
# 设置默认字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']  # 选择你想要的字体
import pandas as pd
import os
import json
import os
import torch
import pandas as pd
import numpy as np
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
    


# 添加数据标签（显示在柱子上方中央）
def autolabel(ax, rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() - rect.get_width()/2., 1.08*height,
                '{:.0f}'.format(height),
                ha='center', va='bottom')


def plot_reason_core_phei_country():
    legend_map ={
        "country_all":"Core + Periphery",
        "country_core":"Core",
        "country_used":"Core + Periphery"
    }
    reasons_all = [

    ]
    llm_name_map = {
        "gpt3.5": "GPT-3.5",
        "gpt4-mini": "GPT-4o-mini",
        "vllm": "LLAMA-3-70B",
        # "qwen2": "QWEN2."
    }
    plt.figure(figsize=(16, 6))
    fig, ax = plt.subplots()
    cols = [
        "Paper Content",
        "Paper Citation",
        "Paper Timeliness",
        "Paper Topic",
        'Author Name',
        "Author Citation",
        "Author Country",
    ]
    
    x = np.arange(len(llm_name_map))
    idx =0
    width = 0.23
    lables = []
    import matplotlib.cm as cm
    colors_map_a = sns.color_palette("Paired",as_cmap=True)
    colors_a = colors_map_a(np.linspace(0, 1,len(llm_name_map)) )
    colors_a = ["#F9BEBB","#89C9C8"]
    colors_a = ["#299D8F" ,"#E9C46A" ,"#D87659"]
    error_config = {'ecolor': '0.3'}
    reasons_llm_counts = [
        readinfo(os.path.join("evaluate/Graph/graph3/reason/1000nodes", file)) for file in os.listdir("evaluate/Graph/graph3/reason/1000nodes")
    ]
    for country_type, plt_lable in {"core":"Core",
                                    "non_core":"Periphery",
                                    "all":"All"
                                        }.items():
           
        # values = {
        #     llm_name:reasons[llm_name][country_type]["Author Country"]/sum(reasons[llm_name][country_type].get(k,0) for k in cols)
        #     for llm_name in reasons.keys()
        # }
        values = {
            llm_name:[reasons[llm_name][country_type]["Author Country"]
                      for reasons in reasons_llm_counts
                      ]
            for llm_name in llm_name_map.keys()
        }
        means = list(np.mean(v) for v in values.values())
        stds = np.array(list(np.std(v) for v in values.values()))
        stds[stds == 0] = 0.0001
        
        rects1 = ax.bar(x+ idx*width, means, width, label=plt_lable,color = colors_a[idx],
                        yerr = stds,error_kw=error_config)
        lables.append(rects1.get_label())
    
        idx+=1

    # autolabel(ax, rects1)
    ax.set_xticks(x+width)
    ax.set_title('Citation Selection based on Country',fontsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xticklabels(llm_name_map.values(),fontsize=16,)
    ax.tick_params(axis='x', labelsize=18)
    plt.subplots_adjust(top=0.9, bottom=0.2, left=0.2, right=0.9, hspace=0.1)
    plt.ylim(0.0,0.2)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=18)
    plt.ylabel('$\operatorname{R_\\text{country}}$',fontsize=20)
    # fig.legend(labels=lables[:2], loc='lower center', ncol=4, fontsize=18)
    # ax.legend(loc='best',ncol=2,frameon=False,fontsize=20)
    ax.legend(loc='best',ncol=2,frameon=False,fontsize=20)
    plt.savefig("evaluate/Graph/graph3/pl_reason_figures/4_reason_country.pdf")

if __name__ == "__main__":
    plot_reason_core_phei_country()