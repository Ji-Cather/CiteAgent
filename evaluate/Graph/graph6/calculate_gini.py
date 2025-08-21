# paths = [
#     "LLMGraph/tasks/llm_agent_1/configs/search_shuffle_base_gpt3.5/evaluate/country_pub_nums.json",
#     "LLMGraph/tasks/llm_agent_1/configs/search_shuffle_no_author_country_gpt3.5/evaluate/country_pub_nums.json"
# ]
from sklearn.preprocessing import StandardScaler,MinMaxScaler

import pandas as pd
import numpy as np
def gini_coefficient(x):

    n = len(x)
    x = np.array(x)
    x_sum = np.sum(x)
    x = np.sort(x)
    index = np.arange(1, n + 1)
    return 2 * np.sum((2 * index - n - 1) * x) / (n * x_sum) - (n + 1) / n

paths = [
    "LLMGraph/tasks/llm_agent_1/configs/search_shuffle_base_gpt3.5/evaluate/in_degrees_country.json",
    "LLMGraph/tasks/llm_agent_2/configs/search_shuffle_base_gpt3.5/evaluate/in_degrees_country.json",
    "LLMGraph/tasks/llm_agent_1/configs/search_shuffle_no_country_gpt3.5/evaluate/in_degrees_country.json",
    "LLMGraph/tasks/llm_agent_2/configs/search_shuffle_no_country_gpt3.5/evaluate/in_degrees_country.json",
    "LLMGraph/tasks/llm_agent_1/configs/search_shuffle_anonymous_gpt3.5/evaluate/in_degrees_country.json",
    "LLMGraph/tasks/llm_agent_2/configs/search_shuffle_anonymous_gpt3.5/evaluate/in_degrees_country.json"
]

paths = [
    "LLMGraph/tasks/llm_agent_1/configs/search_shuffle_base_qwen2/evaluate/in_degrees_country.json",
    "LLMGraph/tasks/llm_agent_2/configs/search_shuffle_base_qwen2/evaluate/in_degrees_country.json",
    "LLMGraph/tasks/llm_agent_1/configs/search_shuffle_no_country_qwen2/evaluate/in_degrees_country.json",
    "LLMGraph/tasks/llm_agent_2/configs/search_shuffle_no_country_qwen2/evaluate/in_degrees_country.json",
    "LLMGraph/tasks/llm_agent_1/configs/search_shuffle_anonymous_qwen2/evaluate/in_degrees_country.json",
    "LLMGraph/tasks/llm_agent_2/configs/search_shuffle_anonymous_qwen2/evaluate/in_degrees_country.json"
]



from LLMGraph.utils.io import readinfo

def calculate_gini(data):
    # if len(data)<20:
    #     append_c = len(data)
    #     for i in range(20-append_c):
    #         data[f"{i}"] = 0
    cited_counts = sorted(data.values())
    # 计算累计份额。
    cum_cited = np.cumsum(cited_counts)
    cum_cited_share = cum_cited / cum_cited[-1]

    # 构建洛伦兹曲线的坐标。
    # lorenz_curve_x = np.arange(1, len(cited_counts) + 1) / len(cited_counts)
    lorenz_curve_x = np.insert(np.arange(1, len(cited_counts) + 1) / len(cited_counts), 0, 0)
    lorenz_curve_y = np.insert(cum_cited_share, 0, 0)  # 插入0在开始位置。

    # 计算对角线下的面积和洛伦兹曲线下的面积。
    area_under_diagonal = 0.5
    area_under_lorenz_curve = np.trapz(lorenz_curve_y, lorenz_curve_x)

    # 计算基尼系数。
    gini_index = (area_under_diagonal - area_under_lorenz_curve) / area_under_diagonal
    return gini_index

def gini_all_df():
    from evaluate.article.build_graph import build_citation_graph, build_country_citation_graph,update_citation_graph

    df = pd.DataFrame()
    llms = ["gpt3.5", 
            "gpt4-mini",
            "vllm",
            ]
    config_templates =[
                ("base","search_shuffle_base_{llm}"),
                ("anonymous","search_shuffle_anonymous_{llm}"),
        ]
    tasks = ["llm_agent_1", "llm_agent_2","llm_agent_3"]
    import powerlaw 
    threshold = 500

    dfs = []
    for task in tasks:
        for llm in llms:
            for config_type,config_template in config_templates:
                D =0
                gini =0
                rps =0
                alpha = 0
                try:
                    config = config_template.format(llm = llm)
                    path = f"LLMGraph/tasks/{task}/configs/{config}/evaluate/country_pub_nums.json"
                    rps = readinfo(f"LLMGraph/tasks/{task}/configs/{config}/evaluate/preferences_time.json")
                    cpr = rps["RPS"][-1][-1]
                    country_pub_nums = readinfo(path)
                    article_meta_info = readinfo(f"LLMGraph/tasks/{task}/configs/{config}/data/article_meta_info.pt")
                    if len(article_meta_info)<threshold:
                        continue
                    else:
                        article_meta_info = dict(list(article_meta_info.items())[:threshold])
                    author_info = readinfo(f"LLMGraph/tasks/{task}/configs/{config}/data/author.pt")
                    graph = build_citation_graph(article_meta_info)
                    graph = update_citation_graph(graph,article_meta_info,author_info)
                    
                    degree_list = [graph.degree(n) for n in graph.nodes()]

                    country_graph =  build_country_citation_graph(article_meta_info,
                                                             author_info,
                                                             graph)
                    country_degree_list = dict(country_graph.in_degree())
                    results = powerlaw.Fit(degree_list, discrete=True,
                                        xmin=3,
                            # fit_method="KS"
                            )
                    # gini,_,__= calculate_gini(list(country_pub_nums.values()))
                    gini = calculate_gini(country_degree_list)
                    gini_num = calculate_gini(country_pub_nums)
                    
                    D = results.power_law.D
                    alpha = results.power_law.alpha
                    dfs.append(pd.DataFrame([{"task": task, "llm":llm, "config":config, 
                            "gini_cite":gini,
                            "gini_num":gini_num,
                            "cpr":cpr,
                            "rps_core":rps["RPS"][-1][0],
                            "rps_ph":rps["RPS"][-1][1],
                            "rps_all":rps["RPS"][-1][2],
                            "D":D,
                            "alpha": alpha}]))
                except Exception as e:
                    pass
                
    df = pd.concat(dfs)
    df.set_index("llm",inplace=True)
    df.to_csv("evaluate\visualize\for_paper\Graph\graph6\gini.csv")


gini_all_df()