import networkx as nx
import os
from collections import defaultdict
import numpy as np
# import pygraphviz as pgv
import matplotlib.pyplot as plt

import networkx as nx
import json
import pandas as pd

def readinfo(data_dir):
    assert os.path.exists(data_dir),"no such file path: {}".format(data_dir)
    with open(data_dir,'r',encoding = 'utf-8') as f:
        data_list = json.load(f)
    return data_list

def writeinfo(data_dir,info):
    with open(data_dir,'w',encoding = 'utf-8') as f:
            json.dump(info, f, indent=4,separators=(',', ':'),ensure_ascii=False)




def segment_data(data):
    # 计算每篇文章的平均重要性
    average_importance = {paper: np.mean(attributes['importance'])
                        for paper, attributes in data.items()}

    # 按平均重要性，从高到低排序文章
    sorted_papers_by_importance = sorted(average_importance.items(), key=lambda item: -item[1])

    # 分割列表到5等分，每段大约包含total/5篇文章
    total_papers = len(sorted_papers_by_importance)
    papers_per_segment = total_papers // 5
    segments = [sorted_papers_by_importance[i * papers_per_segment: (i + 1) * papers_per_segment]
                for i in range(5)]

    # 如果不是5的整数倍，我们需要将剩余的文章分配到已有的分段
    remaining_papers = sorted_papers_by_importance[papers_per_segment * 5:]
    for index, paper in enumerate(remaining_papers):
        segments[index].append(paper)

    # 打印分好段的文章 (只打印文章标题）
    return segments

def plot_reason_visualize(save_dir:str):
    # llms = ["gpt3.5","vllm", "gpt4","qwen2"]
    llms = ["gpt3.5", "gpt4","vllm"]
    dfs = []
    for llm in llms:
        reason_info_path = f"evaluate/Graph/graph10/new_reason/{llm}/reason_info.json"
        os.makedirs(save_dir,exist_ok=True)
        reason_info = readinfo(reason_info_path)
        segments = segment_data(reason_info)

        df_r = plot_reason(segments,reason_info,save_dir)
        df_s = plot_section(segments,reason_info,save_dir)
        df_p = plot_part(segments,reason_info,save_dir)
        df = pd.concat([df_r,df_s, df_p],axis=1)
        dfs.append(df)
        plot_reason_importance_one(df,llm,save_dir)
    
    # plot_reason_bar(dfs,llms,save_dir)
    # plot_reason_bar_accumulative_section(dfs,llms,save_dir)
    plot_reason_bar_accumulative(dfs,llms,save_dir)
    
    # draw_heatmap(df,save_dir)
    # cite_importance_visualize(list(reason_info.keys()),reason_info,save_dir)


def plot_part_pie(dfs,llms,save_dir):
    fig, ax = plt.subplots(1, 4, figsize=(15, 6), sharey=False)
    idx = 0
    colors_b = sns.color_palette("Set2")
    labels = []
    for df, llm_name in zip(dfs,llms):
        section_labels = df.columns[7:]
        section_frequency = df.iloc[:,7:].sum().values

        def custom_autopct(pct, threshold=3):
            return ('%.1f%%' % pct) if pct >= threshold else ""
        ax2 = ax[idx%4]
        # Pie chart
        wedges, texts, autotexts = ax2.pie(section_labels, autopct=lambda pct: custom_autopct(pct), startangle=140, colors=colors_b,
                                           textprops={'fontsize': 14}
                                           , radius=1.25)
        ax2.set_title(llm_name, fontsize=16)
        if llm_name == "gpt3.5":
            ax2.set_ylabel('$\\tilde{{\\theta}}$', fontsize=16)
        for wedge,label in zip(wedges,section_labels):
            wedge.set_label(label)
            labels.append(label)
        
        idx+=1
   
    plt.subplots_adjust(top=0.9, bottom=0.05, left=0.1, right=0.9, hspace=0.1)
    
    fig.legend(labels=labels[:8], loc='lower center', ncol=4, fontsize=16)
    # Save the figure
    plt.savefig(f"evaluate\Graph\graph10/plt_figures/5_all_reason.pdf")
    pass


def draw_heatmap(data:pd.DataFrame,
                 save_dir):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    # 设置默认字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']  # 选择你想要的字体
    import pandas as pd

    # 检查seaborn版本
    import seaborn as sns
    print(f"Seaborn version: {sns.__version__}")

    data.sort_index(inplace=True)
    data.index =[
        f"{index:.2f}".format(index = index)
        for index in data.index
    ]
    
    # 使用Seaborn库绘制热力图
    plt.figure(figsize=(16, 8))  # 设置图表大小
    
    # 创建顶轴用于放置大括号
    

    plt.subplots_adjust(top=0.85, bottom=0.2)  # 向下移动0.1的空间
    
    ax = sns.heatmap(data, annot=True, cmap='coolwarm',
                cbar_kws={'label': 'Frequency'})  # 选择颜色地图样式等
    # 设置 x 轴标签的旋转角度
    plt.xticks(rotation=30)
    
    plt.ylabel('Importance')

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks([])
    ax_top.xaxis.set_visible(False)
    # 在顶部添加大括号
    def add_bracket(ax, x0, x1, y, text):
        bracket_height = 0.2  # 大括号的高度
        text_offset = 0.02  # 文字标签的偏移量
        ax.plot([x0, x0, x1, x1], [y, y - bracket_height, y - bracket_height, y], 
                lw=3, c='black')
        ax.text((x0 + x1) * 0.5, y - bracket_height - text_offset, text, 
                ha='center', va='top', fontsize=15)
    # 添加大括号和标签 
    # todo :这个大括号不显示，需要debug
    add_bracket(ax_top, 0, 4, 0, 'Motive')
    add_bracket(ax_top, 4, 8, 0, 'Section')
    # 显示热力图
    plt.savefig(os.path.join(save_dir,"5_importance_synthetic.pdf"))


def plot_reason(segments,
                reason_info,
                save_dir:str):
    df = pd.DataFrame()
    for i, segment in enumerate(segments):
        print(f"\nSegment {i+1}:")
        segment_keys = [item[0] for item in segment]
        reason_distribution_update, avg_importance = cite_plot_reason_visualize(segment_keys,
                              reason_info=reason_info,
                              save_dir=os.path.join(save_dir, f"segment_{i+1}"),
                              save = False
                              )
    
        for k, v in reason_distribution_update.items():
            df.loc[avg_importance,k] = v
    return df
    
def plot_section(segments,
                reason_info,
                save_dir:str):
    df = pd.DataFrame()
    for i, segment in enumerate(segments):
        print(f"\nSegment {i+1}:")
        segment_keys = [item[0] for item in segment]
        section_distribution_update, avg_importance = cite_section_visualize(segment_keys,
                              reason_info=reason_info,
                              save_dir=os.path.join(save_dir, f"segment_{i+1}"),
                              save = False
                              )
        for k, v in section_distribution_update.items():
            df.loc[avg_importance,k] = v
    return df

def plot_part(segments,
                reason_info,
                save_dir:str):
    df = pd.DataFrame()
    for i, segment in enumerate(segments):
        print(f"\nSegment {i+1}:")
        segment_keys = [item[0] for item in segment]
        section_distribution_update, avg_importance = cite_part_visualize(segment_keys,
                              reason_info=reason_info,
                              save_dir=os.path.join(save_dir, f"segment_{i+1}"),
                              save = False
                              )
        for k, v in section_distribution_update.items():
            df.loc[avg_importance,k] = v
    return df
    
def cite_plot_reason_visualize(segment_keys,
                          reason_info:dict ,
                          save_dir:str,
                          save = False):
    # reason_map ={
    #     "1":"Background",
    #     "2":"Fundamental idea",
    #     "3":"Technical basis",
    #     "4":"Comparison"
    # }
    # reason_map ={
    #     "1":"Background",
    #     "2":"Background",
    #     "3":"Method",
    #     "4":"Comparison"
    # }
    reason_map ={
        "1":"B-I",
        "2":"R-C",
        "3":"F-M"
    }
    # Initialize the aggregators
    reason_info = dict(filter(lambda item: item[0] in segment_keys,reason_info.items()))
    reason_distribution = defaultdict(int)
    importance_distribution = defaultdict(int)

    # Process the data to fill the aggregators
    for paper, attributes in reason_info.items():
        try:
            for reason, count in attributes['motive_reason'].items():
                reason_distribution[reason] += count
            for importance in attributes['importance']:
                importance_distribution[importance] += 1
        except:
            continue

    reason_distribution = dict(sorted(reason_distribution.items(), key=lambda item: item[0]))
    
    
    # Now we can print distributions
    avg_importance = sum([k*v for k,v in importance_distribution.items()])/sum(importance_distribution.values())
    print("Reason Distribution:", dict(reason_distribution))
    
    total_reason_count = sum(reason_distribution.values())
    reason_distribution = {reason: count / total_reason_count for reason, count in reason_distribution.items()}
        # And plot the importance distribution

    reason_distribution_update = {
        reason_map.get(reason): reason_distribution[reason]
        for reason in reason_distribution.keys()
    }
    
    return reason_distribution_update, avg_importance
    


def cite_section_visualize(segment_keys,
                          reason_info:dict ,
                          save_dir:str,
                          save = False):
    section_map ={
        "1": "Introduction",
        "2": "Related Work",
        "3": "Methodology",
        "4": "Results"
    }

    section_map ={
        "1": "Intro",
        "2": "Backgr",
        "3": "Discuss",
        "4": "Results",
        "5": "Meth",
        
    }

    # Initialize the aggregators
    reason_info = dict(filter(lambda item: item[0] in segment_keys,reason_info.items()))
    section_distribution = defaultdict(int)
    importance_distribution = defaultdict(int)

    # Process the data to fill the aggregators
    for paper, attributes in reason_info.items():
        for reason, count in attributes['section'].items():
            section_distribution[reason] += count
        for importance in attributes['importance']:
            importance_distribution[importance] += 1

    section_distribution = dict(sorted(section_distribution.items(), key=lambda item: item[0]))
    
    # Now we can print distributions
    avg_importance = sum([k*v for k,v in importance_distribution.items()])/sum(importance_distribution.values())
    print("Section Distribution:", dict(section_distribution))
    # And plot the importance distribution
    
    section_distribution = dict(filter(lambda item: item[0] in section_map.keys(), section_distribution.items()))
    total_section_count = sum(section_distribution.values())
    section_distribution = {reason: count / total_section_count for reason, count in section_distribution.items()}


    section_distribution_update = {
        section_map.get(section): section_distribution[section]
        for section in section_distribution.keys()
    }
    
    return section_distribution_update, avg_importance

def cite_part_visualize(segment_keys,
                          reason_info:dict ,
                          save_dir:str,
                          save = False):
    reason_map =["Paper Content",
                 "Paper Citation",
                 "Paper Timeliness", 
                 "Author Citation",
                 "Author Country",
                 "Paper Topic",
                 'Author Name']
    reason_map = {idx: reson for idx, reson in enumerate(reason_map)}
    # Initialize the aggregators
    reason_info = dict(filter(lambda item: item[0] in segment_keys,reason_info.items()))
    section_distribution = defaultdict(int)
    importance_distribution = defaultdict(int)

    # Process the data to fill the aggregators
    for paper, attributes in reason_info.items():
        for reason, count in attributes['section'].items():
            section_distribution[reason] += count
        for importance in attributes['importance']:
            importance_distribution[importance] += 1

    section_distribution = dict(sorted(section_distribution.items(), key=lambda item: item[0]))
    
    # Now we can print distributions
    avg_importance = sum([k*v for k,v in importance_distribution.items()])/sum(importance_distribution.values())
    print("Section Distribution:", dict(section_distribution))
    # And plot the importance distribution
    total_section_count = sum(section_distribution.values())
    section_distribution = {reason: count / total_section_count for reason, count in section_distribution.items()}

    section_distribution_update = {
        reason_map[int(section)]: section_distribution[section]
        for section in section_distribution.keys()
    }
    
    return section_distribution_update, avg_importance

def cite_importance_visualize(segment_keys,
                          reason_info:dict ,
                          save_dir:str,
                          save = False):
    
    # Initialize the aggregators
    reason_info = dict(filter(lambda item: item[0] in segment_keys,reason_info.items()))
    importance_distribution = defaultdict(int)

    # Process the data to fill the aggregators
    for paper, attributes in reason_info.items():
        for importance in attributes['importance']:
            importance_distribution[importance] += 1

    importance_distribution = dict(sorted(importance_distribution.items(), key=lambda item: item[0]))

    
    # Now we can print distributions
    avg_importance = sum([k*v for k,v in importance_distribution.items()])/sum(importance_distribution.values())
    
    plt.axvline(x=avg_importance, color='r', linestyle='--', label=f'Avg Importance: {avg_importance:.2f}')
    # And plot the importance distribution
    
    total_importance_count = sum(importance_distribution.values())
    importance_distribution = {reason: count / total_importance_count for reason, count in 
                               importance_distribution.items()}
   
    
    plt.plot(list(importance_distribution.keys()), 
             list(importance_distribution.values()), 
             marker='o')
    
    plt.title('Importance Distribution')
    plt.xlabel('Importance score')
    plt.ylabel('Frequency')
    
    save_path = os.path.join(save_dir, "importance_distribution.pdf")
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.legend()
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.savefig(save_path)
    


# def plot_reason_bar(df,llm,save_dir):
#     # 数据
#     motive_labels = df.columns[:4]
#     section_labels = df.columns[4:]
#     motive_importance = df.iloc[:,:4].mean().values
#     section_importance = df.iloc[:,4:].mean().values

#     # 创建绘图
#     fig, ax = plt.subplots(figsize=(10, 8))

#     # 定义柱的位置
#     y_pos_motive = np.arange(len(motive_labels))
#     y_pos_section = np.arange(len(section_labels)) + len(motive_labels) + 1  # 偏移

#     # 绘制柱状图
#     ax.barh(y_pos_motive, motive_importance, color='skyblue', label='Motive')
#     ax.barh(y_pos_section, section_importance, color='lightgreen', label='Section')

#     # 添加标签
#     ax.set_yticks(np.concatenate([y_pos_motive, y_pos_section]))
#     ax.set_yticklabels([*motive_labels, *section_labels],fontsize=18)
#     ax.set_xlabel('Importance',fontsize=18)
#     # ax.set_title('Importance of Motives and Sections')

#     # 显示图例
#     ax.legend(fontsize=18)
#     # 设置x轴和y轴的刻度标记及其字体大小
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)

    
#     plt.tight_layout()
#     # 显示图形
#     plt.savefig(os.path.join(save_dir,f"5_{llm}_importance.pdf"))

import seaborn as sns
def plot_reason_bar(dfs,llms,save_dir):
    # plot frequency
    fig, ax = plt.subplots(2,1,figsize=(15, 6))
    idx = 0
    bar_width = 0.15
    labels = []
    colors_b = sns.color_palette("Set2")
    colors_s = sns.color_palette("Set2")
    

    llm_name_map = {
        "gpt3.5": "GPT-3.5",
        "vllm": "LLAMA-3.",
        "gpt4": "GPT-4o.",
        "qwen2": "QWEN2."
    }
     # plot real data
    # 绘制柱状图
    # https://arxiv.org/pdf/1903.07547.pdf   
    section_dis = [281358,292766,46527,83583]
    section_dis = section_dis/np.sum(section_dis)
    bar = ax[1].bar(np.arange(4)+idx*bar_width, section_dis,bar_width, color=colors_s[idx],  label="Real Data")

    # https://arxiv.org/pdf/1904.01608.pdf 
    bar = ax[0].bar(np.arange(3)+idx*bar_width, [0.58, 0.29,0.13], bar_width,color=colors_b[idx], label="Real Data")
    labels.append(bar.get_label())
    idx +=1

    for df,llm in zip(dfs,llms):
                # 数据
        motive_labels = df.columns[:3]
        motive_frequency = df.iloc[:,:3].sum().values
        motive_frequency = motive_frequency/np.sum(motive_frequency)
        section_labels = df.columns[3:7]
        section_frequency = df.iloc[:,3:7].sum().values
        section_frequency = section_frequency/np.sum(section_frequency)
        # 定义柱的位置
        y_pos_motive = np.arange(len(motive_labels))
       
        # 绘制柱状图
        bar = ax[0].bar(y_pos_motive+idx*bar_width, motive_frequency, bar_width,color=colors_b[idx], label=llm_name_map[llm])
        # ax[0].xaxis.set_visible(False)
        y_pos_section = np.arange(len(section_labels)) 
        bar = ax[1].bar(y_pos_section+idx*bar_width, section_frequency,bar_width, color=colors_s[idx], label=llm_name_map[llm])
        labels.append(bar.get_label())

        # 添加标签
        ax[1].set_xticks(y_pos_section+1.5*bar_width)
        ax[1].set_xticklabels(section_labels,fontsize=18)
        ax[0].set_xticks(y_pos_motive+1.5*bar_width)
        ax[0].set_xticklabels(motive_labels,fontsize=18)
        ax[0].yaxis.set_tick_params(labelsize=16) 
        ax[1].yaxis.set_tick_params(labelsize=16) 
        ax2 = ax[0].twinx()
        ax2.set_yticks([])
        ax2.set_ylabel('Motive',fontsize=18)
        # ax2.yaxis.set_visible(False)
        ax2 = ax[1].twinx()
        ax2.set_yticks([])
        ax2.set_ylabel('Section',fontsize=18)
        # ax2.yaxis.set_visible(False)
        # ax.set_ylabel('Importance',fontsize=18)
        idx+=1
        # ax.set_title('Importance of Motives and Sections')

   

    # plt.ylabel('Importance',fontsize=18)
    fig.text(0.08, 0.55, r'Frequency', va='center', rotation='vertical', fontsize=18)
    # plt.xticks(rotation=30,fontsize=16)
    plt.subplots_adjust(top=0.9, bottom=0.24, left=0.14, right=0.9, 
    hspace=0.3)
    fig.legend(labels=labels[-5:], loc='lower center', ncol=4, fontsize=18)

    plt.savefig(os.path.join(save_dir,f"5_importance.pdf"))


def plot_reason_bar_accumulative(dfs,llms,save_dir):
    # plot frequency
    fig, ax = plt.subplots(1,1,figsize=(11, 6))
    idx = 0
    bar_width = 0.1
    labels = []
    colors_b = sns.color_palette("Set2")
    colors_b = ["#43978F","#9EC4BE","#ABD0F1",
        "#DCE9F4","#E56F5E","#F19685","#F6C957",
        "#FFB77F"
        ]
    llm_name_map = {
        "gpt3.5": "GPT-3.5",
        "vllm": "LLAMA-3.",
        "gpt4": "GPT-4o.",
        # "qwen2": "QWEN2."
    }
     # plot real data
    # 绘制柱状图
    # https://arxiv.org/pdf/1903.07547.pdf   
    section_dis = [281358,292766,433625,83583,46527,]
    section_dis = section_dis/np.sum(section_dis)
    # https://arxiv.org/pdf/1904.01608.pdf 
    motive_dis = [0.58, 0.29,0.13]

    section_labels = dfs[0].columns[3:8]
    motive_labels = dfs[0].columns[:3]
    datas ={
        "Motivation":{"Real Data":motive_dis},
        "Section":{"Real Data":section_dis}
    }
    # 3+6+7
    for df,llm in zip(dfs,llms):
        motive_frequency = df.iloc[:,:3].sum().values
        motive_frequency = motive_frequency/np.sum(motive_frequency)
        section_frequency = df.iloc[:,3:8].sum().values
        section_frequency = section_frequency/np.sum(section_frequency)
        datas["Motivation"][llm_name_map[llm]]=motive_frequency
        datas["Section"][llm_name_map[llm]]=section_frequency

    x = np.arange(len(["Real Data",*llm_name_map.values()]))
    bar_width = 0.25
    legends =[]
    for data_type,labels in zip(["Motivation","Section"],
                                [motive_labels,section_labels]):
        
        for idx, category in enumerate(["Real Data",*llm_name_map.values()]):
            frequency = datas[data_type][category]
            bottoms = 0
            for idy,label in enumerate(labels):
                if label in section_labels: 
                    color = colors_b[idy+3]
                    loc = x[idx]+0.5*bar_width
                else: 
                    color = colors_b[idy]
                    loc = x[idx]-0.5*bar_width
                bar = ax.bar(loc, frequency[idy], bar_width, bottom = bottoms, color=color, label=label)
                bottoms += frequency[idy]
                if idx ==0:
                    legends.append(bar.get_label())

    # 添加标签
    ax.set_xticks(x)
    ax.set_xticklabels(["Real Data",*llm_name_map.values()],fontsize=18)
    ax.yaxis.set_tick_params(labelsize=16) 
    

    

    # 添加主题文本
    # fig.text(0.9, 0.95, 'Motivation', fontsize=20, ha='center')
    # fig.text(0.9, 0.95, 'Section', fontsize=20, ha='center')
    handles, labels = ax.get_legend_handles_labels()
    # handles = [handles[:3],handles[-4:]]
    # labels = [labels[:3],labels[-4:]]
    # 设置图例
    # fig.legend(labels=labels[:3], handles=handles[:3], loc='upper right', ncol=1, fontsize=18, bbox_to_anchor=(0.1,0.8))
    # fig.legend(labels=labels[-4:], handles=handles[-4:], loc='upper right', ncol=1, fontsize=18, bbox_to_anchor=(0.1,0.8))
    # plt.text(0.98, 0.95, 'Motivation', transform=ax.transAxes, 
    #      fontsize=20, ha='right', va='top')
    # plt.subplots_adjust(top=0.9, bottom=0.2, left=0.1, right=0.6, 
    # hspace=0.3)
    # fig.legend(labels=[*labels[-4:],*labels[:3]], 
    #            handles=[*handles[-4:],*handles[:3]], loc='upper right', ncol=1, fontsize=18, bbox_to_anchor=(0.9,0.9))
    plt.subplots_adjust(top=0.9, bottom=0.23, left=0.13, right=0.9, 
    hspace=0.3)
    labels = [*labels[-5:],*labels[:3]]
    handles = [*handles[-5:],*handles[:3]]

    # fig.legend(labels=[lables[0],lables[4],lables[1],lables[5],lables[2],lables[6],lables[3]]   , 
    #            handles=[handles[0],handles[4],handles[1],handles[5],handles[2],handles[6],handles[3]], loc='lower center', ncol = 5, fontsize=18)
    # fig.legend(labels=[lables[0],lables[4],lables[1],lables[5],lables[2],lables[6],lables[3]] , 
    #            handles=[handles[0],handles[4],handles[1],handles[5],handles[2],handles[6],handles[3]], loc='lower center', ncol = 5, fontsize=18)
    fig.legend(labels=[labels[0],labels[5],labels[1],labels[6],labels[2],labels[7],labels[3],labels[4]], 
               handles=[handles[0],handles[5],handles[1],handles[6],handles[2],handles[7],handles[3],handles[4]], loc='lower center', ncol = 5, fontsize=17)
    
    fig.text(0.02, 0.115, r'Section', fontsize=18,va='center')
    fig.text(0.02, 0.055, r'Motive', fontsize=18,va='center')

    fig.text(0.04, 0.55, r'Frequency', va='center', rotation='vertical', fontsize=18)

    plt.savefig(os.path.join(save_dir,f"5_importance_new.pdf"))


def plot_reason_bar_accumulative_section(dfs,llms,save_dir):
    # plot frequency
    fig, ax = plt.subplots(1,1,figsize=(15, 6))
    idx = 0
    bar_width = 0.15
    labels = []
    colors_b = sns.color_palette("Set2")
    colors_b = ["#43978F","#9EC4BE","#ABD0F1",
        "#DCE9F4","#E56F5E","#F19685"]
    llm_name_map = {
        "gpt3.5": "GPT-3.5",
        "gpt4": "GPT-4o-mini",
        "vllm": "LLAMA-3-70B",
        
    }
     # plot real data
    # 绘制柱状图
    # https://arxiv.org/pdf/1903.07547.pdf   
    # section_dis = [281358,292766,46527,83583,2827,433625]
    section_dis = [281358,292766,433625,83583,46527,]
    section_dis = section_dis/np.sum(section_dis)

    section_labels = dfs[0].columns[3:8]
    datas ={
        "Section":{"Real Data":section_dis}
    }
    # 3+6+7
    for df,llm in zip(dfs,llms):
        section_frequency = df.iloc[:,3:8].sum().values
        section_frequency = section_frequency/np.sum(section_frequency)
        datas["Section"][llm_name_map[llm]]=section_frequency

    x = np.arange(len(["Real Data",*llm_name_map.values()]))
    bar_width = 0.3
    legends =[]
    for data_type,labels in zip(["Section"],
                                [section_labels]):
        
        for idx, category in enumerate(["Real Data",*llm_name_map.values()]):
            frequency = datas[data_type][category]
            bottoms = 0
            for idy,label in enumerate(labels):
                if label in section_labels: 
                    color = colors_b[idy]
                    loc = x[idx]+0.5*bar_width
                else: 
                    color = colors_b[idy]
                    loc = x[idx]-0.5*bar_width
                bar = ax.bar(loc, frequency[idy], bar_width, bottom = bottoms, color=color, label=label)
                bottoms += frequency[idy]
                if idx ==0:
                    legends.append(bar.get_label())

    # 添加标签
    ax.set_xticks(x)
    ax.set_xticklabels(["Real Data",*llm_name_map.values()],fontsize=18)
    ax.yaxis.set_tick_params(labelsize=16) 
    
    # 添加主题文本
    handles, labels = ax.get_legend_handles_labels()

    plt.subplots_adjust(top=0.9, bottom=0.23, left=0.13, right=0.9, 
    hspace=0.3)

    fig.legend(labels=labels, 
               handles=handles, loc='lower center', ncol = 4, fontsize=18)
    fig.text(0.08, 0.55, r'Frequency', va='center', rotation='vertical', fontsize=18)

    plt.savefig(os.path.join(save_dir,f"5_importance_new.pdf"))


def plot_reason_importance_one(df,llm,save_dir):
    # 数据
    motive_labels = df.columns[:3]
    section_labels = df.columns[3:8]
    motive_importance = np.dot(df.index,df.iloc[:,:3])
    section_importance = np.dot(df.index,df.iloc[:,3:8])

    # 创建绘图
    fig, ax = plt.subplots(figsize=(6, 5))

    # 定义柱的位置
    y_pos_motive = np.arange(len(motive_labels))
    y_pos_section = np.arange(len(section_labels)) + len(motive_labels) + 0.5  # 偏移
    # colors =["#FFB77F","#FBE8D5"]
    colors =["#E44A33","#4DBAD6"]
    # 绘制柱状图
    ax.barh(y_pos_motive, motive_importance, color=colors[0], label='Motive')
    ax.barh(y_pos_section, section_importance, color=colors[1],label='Section')

    # 添加标签
    ax.set_yticks(np.concatenate([y_pos_motive, y_pos_section]))
    ax.set_yticklabels([*motive_labels, *section_labels],fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xlabel('Importance',fontsize=18)
    # ax.set_title('Importance of Motives and Sections')

    # 显示图例
    ax.legend(fontsize=18)
    # 设置x轴和y轴的刻度标记及其字体大小
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    
    plt.tight_layout()
    # 显示图形
    plt.savefig(os.path.join(save_dir,f"5_{llm}_importance_new.pdf"))


if __name__ == "__main__":

    llms = ["gpt3.5", "gpt4","vllm","qwen2"]
    # # llms = ["vllm"]
    for llm in llms:
        plot_reason_visualize(save_dir="evaluate\Graph\graph10\plt_figures")
