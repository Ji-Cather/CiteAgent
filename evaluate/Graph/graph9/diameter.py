import numpy as np
import scipy.stats as stats
import torch
import os
from tqdm import tqdm
import matplotlib.dates as mdates

def ks_critical_value(n, alpha):
    """
    计算单样本 K-S 检验的临界值

    参数：
    n (int)：样本的大小
    alpha (float)：显著性水平

    返回：
    critical_value (float)：临界值
    """
    # 计算临界值
    critical_value = np.sqrt(-0.5 * np.log(alpha) / n)
    return critical_value

def calculate_alpha(n, critical_value):
    alpha = np.exp(np.power(critical_value,2)*n/(-0.5))
    return alpha


def load_diameter(root_dir = "evaluate/visualize/for_paper"):
    from datetime import datetime
    llms = [
        "gpt3.5",
        "gpt4-mini",       
        "vllm",
    ]
    configs = [
        "fast_gpt3.5",
        "fast_gpt4-mini",       
        "fast_vllm",
        "gt"
    ]
    legend_size = 20
    tick_size = 18
    label_size = 18
    legend_size = 18
    tick_size = 16
    label_size = 18
    
    for llm in llms:
        import matplotlib.pyplot as plt
        path = f"LLMGraph/tasks/citeseer_1/configs/fast_{llm}/evaluate/article_citation_diameter.pt"
        diameter = torch.load(path)
        diameter = dict(filter(lambda x: isinstance(x[1],dict), diameter.items()))
        diameter = dict(filter(lambda x: datetime.strptime(x[0], "%Y-%m") >= datetime(2004, 1, 1), diameter.items()))
        dens_x, dens_y, dens_label = estimate_densification_plt(
            [v["nodes"] for v in diameter.values()], 
            [v["edges"] for v in diameter.values()], 
            )
        dens_x_04, dens_y_04, dens_label_04 = estimate_densification_plt(
            [v["nodes_04"] for v in diameter.values()], 
            [v["edges_04"] for v in diameter.values()], 
            )
        # dens_x_cc, dens_y_cc, dens_label_cc = estimate_densification_plt(
        #     [v["nodes_cc"] for v in diameter.values()], 
        #     [v["edges_cc"] for v in diameter.values()], 
        #     )
        plot_values = [{
            "args":{"x":"Time",
                    "y":"Effective Diameter"
                    },
            "data":{
            "Full Network":[v["diameter"] for v in diameter.values()],
            "Post '04 Network":[v["diameter_04"] for v in diameter.values()],
            # "diameter_cc":[v["diameter_cc"] for v in diameter.values()],
            # "diameter_lcc":[v["diameter_lcc"] for v in diameter.values()],
            # "diameter_lcc_cc":[v["diameter_lcc_cc"] for v in diameter.values()],
        }
        },
        {
            "args":{"x":"Time",
                    "y":"Vertice Fraction of GCC"
                    },
                "data":{
                # "clique_p":[v["clique_len"]/v["graph_len"] for v in diameter.values()],
                # "clique_p_cc":[v["clique_len_cc"]/v["graph_len_cc"] for v in diameter.values()],
                "Full Network":[v["gcc_len"]/v["graph_len"] for v in diameter.values()],
                "Post '04 Network":[v["gcc_len_04"]/v["graph_len_04"] for v in diameter.values()],
                # "gcc_p_cc":[v["gcc_len_cc"]/v["graph_len_cc"] for v in diameter.values()],
            }
        },
        {
            "args":{"x":"Number of Vertices",
                    "y":"Number of Edges"
                    },
                "data":{
                dens_label:{"x":dens_x, "y":dens_y},
                # dens_label_04:{"x":dens_x_04, "y":dens_y_04},
                },
            
        }
        ]
        # 创建一个1行2列的子图布局
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
        for plot_value_one,ax in zip(plot_values,axes):
            datas = plot_value_one["data"]
            # 转换日期字符串为 datetime 对象
            date_objects = [datetime.strptime(date, "%Y-%m") for date in list(diameter.keys())]
            styles = [
                ["-","o","#F08C55"],
                ["--","d","#6EC8C8"]
            ]
            for idx,(label, y_data) in enumerate(datas.items()):
                style = styles[idx]
                if isinstance(y_data, dict):
                    ax.plot(y_data["x"], 
                            y_data["y"],
                            label = label,
                            # marker='o',
                            linewidth=3, 
                            color = style[2]
                            )
                    ax.scatter(y_data["x"], 
                            y_data["y"],
                            label = "Full Network Edges",
                            s=24,
                            color = style[2]
                            )
                else:
                    # 创建折线图
                    ax.plot(date_objects, 
                        y_data,
                        label = label,
                        linestyle = style[0],
                        marker = style[1],
                        linewidth=3, 
                        markersize=8,
                        color = style[2]
                        )
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # 每隔3个月
                    fig.autofmt_xdate()
            ax.set_xlabel(plot_value_one["args"]["x"], fontsize=label_size)
            ax.set_ylabel(plot_value_one["args"]["y"], fontsize=label_size)
            # ax.tick_params(axis='x', rotation=20,labelsize=tick_size)
            ax.tick_params(axis='x', labelsize=tick_size)
            ax.tick_params(axis='y', labelsize=tick_size)
            ax.legend(fontsize=legend_size) 
        
        # 调整子图之间的间距
        plt.subplots_adjust(wspace=0, hspace=0)  # wspace: 水平方向间距, hspace: 垂直方向间距
        plt.tight_layout()
        plt.savefig(f"{root_dir}/diameter_{llm}.pdf")
        plt.clf()    
    return diameter

def load_ba_diameter():

    paths = {
        "citeseer_1":"LLMGraph/tasks/citeseer_1/evaluate/diameter.pt",
    }
    for dataset, path in paths.items():
        import matplotlib.pyplot as plt
        diameter = torch.load(path)
        # 创建折线图
        plt.plot(list(diameter["diameter_list"].keys()), 
                list(diameter["diameter_list"].values()),
                marker='o')
        
        # 添加标题和标签
        plt.title('shrinking diameter')
        plt.xlabel('graph size')
        plt.ylabel('effective diameter')
        # plt.legend()
        # 显示图形
        plt.savefig(f"evaluate/visualize/for_paper/diameter/diameter_{dataset}_ba.pdf")
        plt.clf()
    return diameter


def estimate_densification_plt(x, y):
    import numpy as np
    import scipy.optimize as optimize
    import matplotlib.pyplot as plt
    x = np.array(x)
    y = np.array(y)
   
    def power_law(x, k, a):
        return k * x**a

    # 使用非线性最小二乘法拟合数据
    params, params_covariance = optimize.curve_fit(power_law, x, y, p0=[1, 1])
    k_fit, a_fit = params

    # 计算拟合的预测值
    y_fit = power_law(x, k_fit, a_fit)

    # 计算R^2 值
    residuals = y - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return x, power_law(x, *params), f'= {k_fit:.1f}x$^{{{a_fit:.2f}}}$ $R^2$={r_squared:.2f}'


import powerlaw
def calculate_densification_power_law_exponent():
    llms = [
        "gpt3.5",
        "gpt4-mini",       
        "vllm",
    ]
    for llm in llms:
        import matplotlib.pyplot as plt
        path = f"LLMGraph/tasks/citeseer_1/configs/fast_{llm}/evaluate/article_citation_d_power_law.pt"
        diameter = torch.load(path)

        path = f"evaluate/Graphgraph9/diameter_{llm}.pdf"
        estimate_densification_plt(list(diameter["d_power_law"].keys()), 
                                   list(diameter["d_power_law"].values()))
    return diameter



if __name__ == "__main__":
    # calculate_densification_power_law_exponent()

    load_diameter("evaluate\Graph\graph9")