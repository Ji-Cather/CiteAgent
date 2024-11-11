import json
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
fontname = 'Times New Roman'  # Font for all text elements
bar_width = 0.3  # Width of the bars to control spacing between groups
alpha = 0.85  # Transparency for the bars
figsize = (10, 6)  # Size of the figure
# colors = ['#1f78b4', '#33a02c', '#e31a1c']  # Colors for Core, Periphery, All
colors = ['#EF767A', '#456990', '#48C0AA']  # Colors for Core, Periphery, All

# Font sizes
title_fontsize = 26
label_fontsize = 22
tick_fontsize = 18
legend_fontsize = 20

def load_data():
    # Load JSON data from the specified file path
    file_path = 'evaluate/Graph/graph7/rps_compare_country/all_ps.json'
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Load data
data = load_data()

# Define layout for a single row of plots
fig, ax = plt.subplots(figsize=figsize)  # Use the figsize hyperparameter

# Function to plot grouped bars for a set of groups
def plot_grouped_bars(ax, groups):
    x = np.arange(len(groups))  # Positions for each group

    # Plot bars for each RPS category: Core, Periphery, All
    for j, label in enumerate(['Core', 'Periphery', "Core + Periphery"]):
        rps_values = [data[group]['RPS'][j] for group in groups]
        err_values = [data[group]['RPS'][j+3] for group in groups ]
        ax.bar(x + bar_width*j - bar_width, 
               rps_values, 
               bar_width, 
               label=label, color=colors[j], alpha=alpha,
               yerr = err_values)

    ax.set_xticks(x)
    ax.set_xticklabels([llm_name_map.get(group,group) for group in groups ], 
                       fontname=fontname, fontsize=tick_fontsize)
    ax.axhline(y=1, color='#696969', linestyle='--',label='RPS = 1', alpha=0.7)
    ax.legend(prop={'family': fontname, 'size': legend_fontsize},ncols=2,
              frameon=False,
              loc = "upper left")

    # Adding gridlines for better readability
    ax.yaxis.grid(True, linestyle='--', which='both', color='gray', alpha=0.7)
    ax.set_axisbelow(True)


if __name__ == "__main__":
    ax.tick_params(axis='x', labelsize=tick_fontsize)  # x轴刻度字体大小
    ax.tick_params(axis='y', labelsize=tick_fontsize)  # x轴刻度字体大小
    # Arrange the groups with 'Cora' and 'CiteSeer' first

    llm_name_map ={
        "GPT-3.5":"GPT-3.5",
            "GPT-4o-mini":"GPT-4o.",
            "LLAMA-3-70B":"LLAMA-3.",
            # "QWEN2-70B":"QWEN2."
    }
    ordered_groups = ['PA', 'Cora', 'CiteSeer'] + [group for group in data.keys() if group not in ['Cora', 'CiteSeer','PA'] and group in llm_name_map.keys()]

    # Plot all groups in one row
    plot_grouped_bars(ax, ordered_groups)

    # Set title and y-axis label for clarity
    #ax.set_title('RPS Values by Group and Category', fontsize=title_fontsize, fontweight='bold', fontname=fontname)
    ax.set_ylabel('RPS', fontsize=label_fontsize, fontname=fontname)

    # # Adjust layout for better visualization
    # fig.tight_layout()

    # Save the plot
    plt.savefig("evaluate/Graph/graph7/8_rps_compare_one.pdf")