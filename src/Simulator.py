import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles
from scipy.stats import norm

from BetaDatacenter import BetaDatacenter
from FixedCostDatacenter import FixedCostDatacenter


def main():
    price = 50
    # datacenters = [
    #     FixedCostDatacenter("Rechenzentrum 1", 10, 0.5, price),
    #     FixedCostDatacenter("Rechenzentrum 2", 20, 0.5, price),
    #     FixedCostDatacenter("Rechenzentrum 3", 30, 0.5, price),
    #     FixedCostDatacenter("Rechenzentrum 4", 40, 0.5, price),
    #     FixedCostDatacenter("Rechenzentrum 5", 50, 0.5, price),
    # ]

    max_rounds = 200
    p_star = 0.8
    beta_tuning_freq = 10
    eta = 0.1



    datacenters = [
        BetaDatacenter("Rechenzentrum 1", 10, 0.4, price, 8, p_star,beta_tuning_freq,eta),
        BetaDatacenter("Rechenzentrum 2", 20, 0.5, price, 30, p_star,beta_tuning_freq,eta),
        BetaDatacenter("Rechenzentrum 3", 30, 0.6, price, 4, p_star,beta_tuning_freq,eta),
        BetaDatacenter("Rechenzentrum 4", 40, 0.7, price, 20, p_star,beta_tuning_freq,eta),
        BetaDatacenter("Rechenzentrum 5", 50, 0.8, price, 15, p_star,beta_tuning_freq,eta),
    ]

    revenue_per_round = np.empty((len(datacenters),0),float)
    total_revenue_per_round =[]

    for r in range(0, max_rounds):

        for current in datacenters:
            others = [d for d in datacenters if d != current]
            current.evaluate(others)

        for dc in datacenters:
            dc.check_offers()

        overall = [dc.calculate_revenue() for dc in datacenters]
        print(f"[TOTAL] Revenue for round {r+1}: {np.sum(overall)}")
        print(f"the revenue is {overall} with an overall of {sum(overall)}")

        revenue_per_round = np.column_stack((revenue_per_round, np.array(overall))) # anders stacken wegen reshape
        total_revenue_per_round.append(np.sum(overall))


    ## plot distributions
    Cs = []
    xs = np.linspace(0, 100, 1000)
    for dc in datacenters:

        Cs.append(norm.pdf(xs, dc.mean, dc.variance))

    plt.figure(figsize=(10, 6))
    for i,c in enumerate(Cs):
        plt.plot(xs, c, label=f"RZ {i+1}")

    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid()
    plt.show()

    ## plot revenue per round
    plt.figure(figsize=(10, 6))
    for i in range(0,len(datacenters)):
        plt.plot(revenue_per_round[i], label=f"DC {i+1}")
    plt.plot(total_revenue_per_round, label=f"Total Revenue", linestyle="--")
    plt.title("Revenue per Round")
    plt.xlabel('Round')
    plt.ylabel('Revenue')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
