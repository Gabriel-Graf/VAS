import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from BetaDatacenter import BetaDatacenter
from FixedCostDatacenter import FixedCostDatacenter

def initialize_datacenters(price, p_star, beta_tuning_freq, eta):
    return [
        BetaDatacenter("Rechenzentrum 1", 10, 0.4, price, 8, p_star, beta_tuning_freq, eta),
        BetaDatacenter("Rechenzentrum 2", 20, 0.5, price, 30, p_star, beta_tuning_freq, eta),
        BetaDatacenter("Rechenzentrum 3", 30, 0.6, price, 4, p_star, beta_tuning_freq, eta),
        BetaDatacenter("Rechenzentrum 4", 40, 0.7, price, 20, p_star, beta_tuning_freq, eta),
        BetaDatacenter("Rechenzentrum 5", 50, 0.8, price, 15, p_star, beta_tuning_freq, eta),
    ]

def simulate(datacenters, max_rounds):
    revenue_per_round = np.empty((len(datacenters), 0), float)
    total_revenue_per_round = []

    for r in range(max_rounds):
        for current in datacenters:
            others = [d for d in datacenters if d != current]
            current.evaluate(others)
        for dc in datacenters:
            dc.check_offers()
        overall = [dc.calculate_revenue() for dc in datacenters]
        print(f"[TOTAL] Revenue for round {r+1}: {np.sum(overall)}")
        print(f"the revenue is {overall} with an overall of {sum(overall)}")
        revenue_per_round = np.column_stack((revenue_per_round, np.array(overall)))
        total_revenue_per_round.append(np.sum(overall))
    return revenue_per_round, total_revenue_per_round

def plot_distributions(datacenters):
    xs = np.linspace(0, 100, 1000)
    plt.figure(figsize=(10, 6))
    for i, dc in enumerate(datacenters):
        plt.plot(xs, norm.pdf(xs, dc.mean, dc.variance), label=f"RZ {i+1}")
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid()
    plt.show()

def plot_revenue(revenue_per_round, total_revenue_per_round, datacenters):
    plt.figure(figsize=(10, 6))
    for i in range(len(datacenters)):
        plt.plot(revenue_per_round[i], label=f"DC {i+1}")
    plt.plot(total_revenue_per_round, label="Total Revenue", linestyle="--")
    plt.title("Revenue per Round")
    plt.xlabel('Round')
    plt.ylabel('Revenue')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    price = 50
    max_rounds = 200
    p_star = 0.8
    beta_tuning_freq = 10
    eta = 0.1

    datacenters = initialize_datacenters(price, p_star, beta_tuning_freq, eta)
    revenue_per_round, total_revenue_per_round = simulate(datacenters, max_rounds)
    plot_distributions(datacenters)
    plot_revenue(revenue_per_round, total_revenue_per_round, datacenters)

if __name__ == "__main__":
    main()