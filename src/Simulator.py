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

from BetaDatacenter import BetaDatacenter
from FixedCostDatacenter import FixedCostDatacenter


def initialize_datacenters(n, price, p_star, beta_tuning_freq, eta, alpha):
    # return [
    #     BetaDatacenter("Rechenzentrum 1", 1, 10, alpha, price, 8, p_star, beta_tuning_freq, eta),
    #     BetaDatacenter("Rechenzentrum 2", 2, 20, alpha, price, 30, p_star, beta_tuning_freq, eta),
    #     BetaDatacenter("Rechenzentrum 3", 3, 30, alpha, price, 4, p_star, beta_tuning_freq, eta),
    #     BetaDatacenter("Rechenzentrum 4", 4, 40, alpha, price, 20, p_star, beta_tuning_freq, eta),
    #     BetaDatacenter("Rechenzentrum 5", 5, 50, alpha, price, 15, p_star, beta_tuning_freq, eta),
    # ]
    return [
        BetaDatacenter("Rechenzentrum 1", 1, 10, alpha, price, 8, p_star, beta_tuning_freq, eta),
        BetaDatacenter("Rechenzentrum 2", 2, 20, alpha, price, 30, p_star, beta_tuning_freq, eta),
        BetaDatacenter("Rechenzentrum 3", 3, 30, alpha, price, 4, p_star, beta_tuning_freq, eta),
        BetaDatacenter("Rechenzentrum 4", 4, 40, alpha, price, 20, p_star, beta_tuning_freq, eta),
        BetaDatacenter("Rechenzentrum 5", 5, 50, alpha, price, 15, p_star, beta_tuning_freq, eta),
        # BetaDatacenter("Rechenzentrum 6", 6, 60, alpha, price, 15, p_star, beta_tuning_freq, eta),
    ]


def simulate(datacenters, max_rounds):
    revenue_per_round = np.empty((len(datacenters), 0), float)
    total_revenue_per_round = []
    betas_per_round = np.empty((len(datacenters), max_rounds), float)
    offers_send_to = np.empty((len(datacenters), max_rounds), float)
    offers_accepted_by = np.empty((len(datacenters), max_rounds), float)

    for r in range(max_rounds):
        for i, current in enumerate(datacenters):
            others = [d for d in datacenters if d != current]
            current.evaluate(others)
            betas_per_round[i, r] = current.beta
            offers_send_to[i, r] = current.send_to

        for i, current in enumerate(datacenters):
            current.check_offers()
            offers_accepted_by[i, r] = current.accepted_by

        overall = [dc.calculate_revenue() for dc in datacenters]
        # print(f"[TOTAL] Revenue for round {r + 1}: {np.sum(overall)}")
        # print(f"the revenue is {overall} with an overall of {sum(overall)}")
        revenue_per_round = np.column_stack((revenue_per_round, np.array(overall)))
        total_revenue_per_round.append(np.sum(overall))

        for dc in datacenters:
            dc.round_reset()

    return revenue_per_round, total_revenue_per_round, betas_per_round, offers_send_to, offers_accepted_by


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


def plot_revenue(revenue_per_round, total_revenue_per_round, datacenters, dir_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(len(datacenters)):
        ax.plot(revenue_per_round[i], label=f"DC {i + 1}")
    plt.plot(total_revenue_per_round, label="Total Revenue", linestyle="--")
    plt.title("Revenue per Round")
    plt.xlabel('Round')
    plt.ylabel('Revenue')
    ax.set_yscale('log')
    # ax.set_xscale('log')

    plt.legend(loc="lower right")

    plt.grid()
    plt.tight_layout()
    plt.savefig(f"figs/{dir_name}/revenue_{dir_name}.svg")
    plt.show()


def plot_beta(beta_per_round, datacenters, dir_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(len(datacenters)):
        ax.plot(beta_per_round[i], label=f"DC {i + 1}")
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
    # colors = ['#FF9999', '#66B3FF', '#99FF99']  # Colors for 1, 2, 3 /respectively
    colormap = cm.get_cmap('plasma', values.shape[0])  # 'viridis' colormap with 6 distinct colors
    colors = [colormap(i) for i in range(colormap.N)]  # Extract colors from the colormap
    labels_set = [False for l in categories]  # Labels for the legend
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
    plt.tight_layout()

    return fig, ax


def plot_send_to(values, dir_name):
    fig, ax = plot_bar_chart_data(values, legend_title="Receiving RZ", )

    # Set labels and title
    ax.set_ylabel('Sending RZ')
    ax.set_xlabel('Rounds')
    ax.set_title('RZ i send to RZ j')
    # Show the plot
    plt.savefig(f"figs/{dir_name}/sendTo_{dir_name}.png")
    plt.show()


def plot_accepted_by(values, dir_name):
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


def plot_send_and_accepted(send_to, accepted_by, dir_name):
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
    plt.title('Offers sent by RZ i to RZ j and accepted by RZ j (accepted/send)')
    plt.ylabel('Sender')
    plt.xlabel('Receiver')
    plt.tight_layout()
    plt.savefig(f"figs/{dir_name}/offers_{dir_name}.svg")
    plt.show()


def main():
    number_of_datacenters = 5  ## not used
    price = 75
    alpha = 0.8
    max_rounds = 10000
    p_star = 0.8
    beta_tuning_freq = 10
    eta = 0.1

    seed = 1308
    np.random.seed(seed)

    ### für mehrere runs
    # avgs = []
    # for i in range(10):
    #
    #     datacenters = initialize_datacenters(number_of_datacenters, price, p_star, beta_tuning_freq, eta, alpha)
    #     revenue_per_round, total_revenue_per_round, beta_per_round, offers_send_to, offers_accepted_by = simulate(
    #         datacenters, max_rounds)
    #
    #     avgs.append(np.mean(total_revenue_per_round[:-1000]))
    #     print(f"Avg. Total Revenue: {np.mean(total_revenue_per_round[:-1000])}")
    # print("AVG AVG: {:.2f}".format(np.mean(avgs)))

    datacenters = initialize_datacenters(number_of_datacenters, price, p_star, beta_tuning_freq, eta, alpha)
    revenue_per_round, total_revenue_per_round, beta_per_round, offers_send_to, offers_accepted_by = simulate(
        datacenters, max_rounds)
    save_dir_name = f"run_n{len(datacenters)}_p{price}_a{alpha}_b{beta_tuning_freq}_ps{p_star}_e{eta}_s{seed}"
    os.makedirs(f"figs/{save_dir_name}", exist_ok=True)


    plot_distributions(datacenters, save_dir_name)
    plot_revenue(revenue_per_round, total_revenue_per_round, datacenters, save_dir_name)
    plot_beta(beta_per_round, datacenters, save_dir_name)
    # plot_send_to(offers_send_to, save_dir_name)  # can take a while
    # plot_accepted_by(offers_accepted_by,save_dir_name) # can take a while
    # plot_send_to_matrix(offers_send_to)
    # plot_accepted_by_matrix(offers_accepted_by)
    plot_send_and_accepted(offers_send_to, offers_accepted_by, save_dir_name)




if __name__ == "__main__":
    main()
