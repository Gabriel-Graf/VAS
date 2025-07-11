import os
from random import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
import seaborn as sns



colors = ['#68b582', '#42a087', '#288a86', '#22747d', '#2a5d6d', '#2f4858']  # Red to Green to Blue
custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", colors)


def plot_distributions(datacenters, dir_name):
    xs = np.linspace(0, 100, 1000)
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, dc in enumerate(datacenters):
        ax.plot(xs, norm.pdf(xs, dc.mean, dc.sigma), label=f"RZ {i + 1}")
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid()
    plt.savefig(f"figs/{dir_name}/distributions_{dir_name}.svg")
    plt.show()


def plot_revenue(revenue_per_round, total_revenue_per_round, datacenters, dir_name, suffix=""):
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(len(datacenters)):
        ax.plot(revenue_per_round[i], label=f"RZ {i + 1}")
    plt.plot(total_revenue_per_round, label="Total Revenue", linestyle="--")
    plt.title("Revenue per Round" + f" {suffix}")
    plt.xlabel('Round')
    plt.ylabel('Revenue')
    ax.set_yscale('log')
    ax.set_ylim([0.1, 10**3])
    # ax.set_xscale('log')

    plt.legend(loc="lower right")

    plt.grid()
    plt.tight_layout()
    plt.savefig(f"figs/{dir_name}/revenue_{dir_name}_{suffix}.svg")
    plt.show()


def plot_revenue_bar_for_one_rz(avg_revenue_per_round, total_revenue_per_round, datacenters, dir_name, rz_index=0, suffix=""):
    sns.set_theme()
    bar = sns.displot(avg_revenue_per_round, multiple="stack",)
    # for i in range(len(datacenters)):

        # mean = avg_revenue_per_round[:,i].mean()
        # sns.lineplot(x=[mean, mean], y=[0, 100], color='red', linestyle='--', label='Mean')



    #
    # fig, ax = plt.subplots(figsize=(10, 6))
    # plt.hist(revenue_per_round, 10, density=True, histtype='bar', stacked=True)
    # # plot(total_revenue_per_round, label="Total Revenue", linestyle="--")
    plt.title("Avg. Revenue per RZ" + f" {suffix}")
    plt.xlabel('Revenue')
    plt.ylabel('Count')
    # # ax.set_yscale('log')
    # # ax.set_ylim([0.1, 10**3])
    # # ax.set_xscale('log')
    #
    # # plt.legend(loc="upper right")
    #
    # plt.grid()
    # plt.tight_layout()
    plt.savefig(f"figs/{dir_name}/revenue_bar_{dir_name}_{suffix}.svg")
    plt.show()

def plot_alpha(alphas, offers_send_to, datacenters, dir_name, dc_index=3, section=[4000,5000], suffix=""):
    from matplotlib.lines import Line2D
    sns.set_theme()
    fig, axes = plt.subplots(nrows=len(datacenters)-dc_index, ncols=1, figsize=(16,9))  # 16:9 aspect ratio
    # Loop through each subplot and create a line plot

    palette = sns.color_palette()
    for i in range(dc_index, len(datacenters)):
        # Initialize a list to hold the change indices for each dimension

        # Find the indices where the values change
        change_indices = np.where(np.diff(offers_send_to[i]) != 0)[0] + 1  # +1 to adjust for the diff
        change_indices = np.append(change_indices, np.array([offers_send_to.shape[1]]))
        ax = i-dc_index
        color = palette[i]
        sns.lineplot(data=alphas[i,:], ax=axes[ax],  color=color)
        for j in range(0,len(change_indices)-1):
            color = palette[int(offers_send_to[i,change_indices[j]-1])]
            axes[ax].axvspan(change_indices[j], change_indices[j+1], color=color, alpha=0.3)  # Adjust alpha for transparency

        axes[ax].set_title(f'Alpha per Round for RZ {i + 1}', fontsize=12)  # Title font size
        axes[ax].set_xlabel('Round', fontsize=10)  # X-label font size
        axes[ax].set_ylabel('Alpha', fontsize=10)  # Y-label font size
        axes[ax].grid()
        axes[ax].set_xlim(section)
        axes[ax].set_ylim([-0.1, 1.1])

        # Collect the handle and label for the legend

    labels = [f'RZ {i + 1}' for i in range(6)]  # Create labels

    legend_handles = [Line2D([0], [0], color=palette[i], lw=4) for i in range(6)]
    # Create a single legend for all subplots
    fig.legend(legend_handles, labels, loc='upper right', fontsize='medium')

    # Adjust layout to reduce spacing
    plt.subplots_adjust(hspace=0.4)  # Adjust vertical spacing between plots

    # plt.legend()
    plt.tight_layout()
    # plt.legend(loc="upper right")
    plt.savefig(f"figs/{dir_name}/alpha_{dir_name}.svg")
    plt.show()

def plot_beta(beta_per_round, datacenters, dir_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(len(datacenters)):
        ax.plot(beta_per_round[i], label=f"RZ {i + 1}")
    plt.title("Beta per Round")
    plt.xlabel('Round')
    plt.ylabel('Beta')
    ax.set_yscale('log')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.legend(loc="lower right")
    plt.savefig(f"figs/{dir_name}/beta_{dir_name}.svg")
    plt.show()


def plot_bar_chart_data(values, legend_title):
    categories = [f'RZ {i + 1}' for i in range(0, values.shape[0])]
    # Define colors for the chunks
    colormap = cm.get_cmap('Dark2', values.shape[0])  # 'viridis' colormap with 6 distinct colors
    colors = [colormap(i) for i in range(colormap.N)]  # Extract colors from the colormap

    # colormap = cm.get_cmap(custom_cmap, values.shape[0])  # 'viridis' colormap with 6 distinct colors
    # colors = [colormap(i) for i in
    #           np.linspace(0, colormap.N, values.shape[0], dtype=int)]  # Extract colors from the colormap

    # Create a figure and axis

    fig, ax = plt.subplots(figsize=(19, 6))
    fig.set_figheight(9)
    fig.set_figwidth(15)
    # Set the y-ticks to the categories
    y_pos = np.arange(len(categories))
    # Create a set to track added colors for the legend
    added_colors = set()

    # Create a list to hold the legend handles
    legend_handles = []
    # Loop through each category and its values
    for i, (category, value) in enumerate(zip(categories, values)):
        for j, v in enumerate(value):
            if not np.isnan(v):  # Check if the value is not NaN
                color = colors[int(v) - 1]
                ax.barh(i, .75, left=j - (0.75/2), color=color)
                # Add a legend handle if it's the first occurrence of the color
    for c, l in zip(colors, categories):
        legend_handles.append(mpatches.Patch(color=c, label=l))
    ax.legend(handles=legend_handles, title=legend_title, loc='upper right')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)

    return fig, ax


def plot_send_to(values, dir_name, suffix=""):
    print("[INFO] Plotting 'Send To Plot'. MAy take a while!")

    fig, ax = plot_bar_chart_data(values, legend_title="Receiving RZ", )

    # Set labels and title
    ax.set_ylabel('Sending RZ')
    ax.set_xlabel('Rounds')
    ax.set_title('RZ i send to RZ j'+f" {suffix}")
    plt.tight_layout()
    # Show the plot
    plt.savefig(f"figs/{dir_name}/sendTo_{dir_name}_{suffix}.png")
    plt.show()


def plot_accepted_by(values, dir_name):
    print("[INFO] Plotting 'Send To Plot'. MAy take a while!")
    fig, ax = plot_bar_chart_data(values, "Accepting RZ")

    # Set labels and title
    ax.set_ylabel('Sending RZ')
    ax.set_xlabel('Rounds')
    ax.set_title('RZ i accepted by RZ j')
    plt.savefig(f"figs/{dir_name}/acceptedBy_{dir_name}.png")
    # Show the plot
    plt.show()


def get_matrix(values):
    rows = values.shape[0]
    count_matrix = np.zeros((rows, rows))

    # Count occurrences of values for each category
    for i in range(rows):
        for j in range(rows):
            if i != j:  # Only count occurrences between different categories
                count_matrix[i, j] = np.sum(values[i] == (j + 1))  # Count occurrences of (j + 1) in row i

    row_sums = np.sum(count_matrix, axis=1)
    count_matrix = np.column_stack((count_matrix, row_sums))

    # Calculate column sums and append as a new row
    col_sums = np.sum(count_matrix, axis=0)
    count_matrix = np.vstack((count_matrix, col_sums))

    return count_matrix


def plot_send_to_matrix(values):
    # Create a count matrix for the heatmap
    categories = [f'RZ {i + 1}' for i in range(0, values.shape[0])] + ["Sum"]
    count_matrix = get_matrix(values)

    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(count_matrix, annot=True, fmt=".0f", cmap='crest', xticklabels=categories, yticklabels=categories)
    plt.title('Offers sent by RZ i to RZ j')
    plt.ylabel('Sender')
    plt.xlabel('Receiver')
    plt.show()


def plot_accepted_by_matrix(values):
    # Create a count matrix for the heatmap
    categories = [f'RZ {i + 1}' for i in range(0, values.shape[0])] + ["Sum"]
    count_matrix = get_matrix(values)

    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(count_matrix, annot=True, fmt=".0f", cmap='crest', xticklabels=categories, yticklabels=categories)
    plt.title('Offers from RZ i accepted by RZ j')
    plt.ylabel('Sender')
    plt.xlabel('Accepted by')
    plt.show()


def plot_send_and_accepted(send_to, accepted_by, dir_name, suffix=""):
    categories = [f'RZ {i + 1}' for i in range(0, send_to.shape[0])] + ["Sum"]
    send_to_matrix = get_matrix(send_to)
    accepted_by_matrix = get_matrix(accepted_by)

    annot_matrix = np.empty(send_to_matrix.shape, dtype=object)
    for i in range(send_to_matrix.shape[0]):
        for j in range(send_to_matrix.shape[1]):
            annot_matrix[i, j] = f"{accepted_by_matrix[i, j]}/\n{send_to_matrix[i, j]:.2f}"

    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(send_to_matrix, annot=annot_matrix, fmt="", cmap=custom_cmap, xticklabels=categories,
                yticklabels=categories)
    plt.title('Offers sent by RZ i to RZ j and accepted by RZ j (accepted/send)'+f" {suffix}")
    plt.ylabel('Sender')
    plt.xlabel('Receiver')
    plt.tight_layout()
    plt.savefig(f"figs/{dir_name}/offers_{dir_name}_{suffix}.svg")
    plt.show()
