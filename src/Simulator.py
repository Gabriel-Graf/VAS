import os

import numpy as np


from BetaDatacenter import BetaDatacenter
from FixedCostDatacenter import FixedCostDatacenter
from AlphaDatacenter import AlphaDatacenter
from plotting import *



def calc_optimal(dc_by_real_cost):
    dc_by_real_cost_sorted = sorted(dc_by_real_cost, key=lambda x: x.cost)

    revenue = []
    n = len(dc_by_real_cost_sorted)
    is_sending_to_dc_ids = [np.nan for _ in range(n)]
    is_sending_to = [np.nan for _ in range(n)]
    has_accepted = [np.nan for _ in range(n)]
    for i in range(int(n / 2)):
        ## i is accepting
        ## j is sending
        j = (n - (i)) -1
        is_sending_to[j] = i +1
        is_sending_to_dc_ids[j] = dc_by_real_cost_sorted[i].dc_id
        has_accepted[i] = j +1

    for i, dc in enumerate(dc_by_real_cost_sorted):
        revenue.append(calc_optimal_revenue(dc, is_sending_to[i], has_accepted[i], dc_by_real_cost_sorted))

    # revenue = sorted(revenue, key=lambda x: x.dc_id)
    ## sort back to original order
    revenue = [rev for _, rev in sorted(zip(dc_by_real_cost, revenue), key=lambda pair: pair[0].dc_id)]

    return revenue, np.array(is_sending_to_dc_ids), np.array(is_sending_to_dc_ids) # the second 'is_sending_to' is for who has accepted


def calc_optimal_revenue(dc_i, is_sending, has_accepted, dcs):
    if not np.isnan(is_sending):
        dc_j = dcs[is_sending-1]
        revenue_from_delegated_offer = (
                dc_i.price - dc_i.cost + np.ceil((dc_i.cost - dc_j.cost) * dc_i.alpha)
        )
    else:
        revenue_from_delegated_offer = 0.0

    own_revenue = revenue_from_delegated_offer if not np.isnan(is_sending) else dc_i.price - dc_i.cost

    secondary_revenue = 0.0
    if not np.isnan(has_accepted):
        dc_j = dcs[has_accepted-1]
        secondary_revenue = (dc_j.cost - dc_i.cost) * (1 - dc_i.alpha)
    return own_revenue + np.ceil(secondary_revenue)


def simulate(datacenters, max_rounds):
    '''
        Simulate
    '''
    revenue_per_round = np.empty((len(datacenters), 0), float)
    total_revenue_per_round = []
    betas_per_round = np.empty((len(datacenters), max_rounds), float)
    offers_send_to = np.empty((len(datacenters), max_rounds), float)
    offers_accepted_by = np.empty((len(datacenters), max_rounds), float)

    alphas = np.empty((len(datacenters), max_rounds), float)
    partners = np.empty((len(datacenters), max_rounds), float)

    optimal_revenue_per_round = np.empty((len(datacenters), 0), float)
    optimal_total_revenue_per_round = []
    optimal_offers_send_to = np.empty((len(datacenters), 0), int)
    optimal_offers_accepted_by = np.empty((len(datacenters), 0), int)



    if type(datacenters[0]) == AlphaDatacenter:
        for i, dc in enumerate(datacenters):
            dc.find_initial_partner(datacenters)



    for r in range(max_rounds):
        dc_by_real_cost = []
        for i, current in enumerate(datacenters):
            others = [d for d in datacenters if d != current]
            current.evaluate(others)
            # betas_per_round[i, r] = current.beta
            offers_send_to[i, r] = current.send_to
            alphas[i, r] = current.get_alpha()
            dc_by_real_cost.append(current)

        ### optimal ------------
        optimal_revenue, opt_send_to, opt_accepted_by = calc_optimal(dc_by_real_cost)
        optimal_revenue_per_round = np.column_stack((optimal_revenue_per_round, optimal_revenue))
        optimal_total_revenue_per_round.append(np.sum(optimal_revenue))
        optimal_offers_send_to = np.column_stack((optimal_offers_send_to, opt_send_to))
        optimal_offers_accepted_by = np.column_stack((optimal_offers_accepted_by, opt_accepted_by))
        ### optimal ------------

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

    return (revenue_per_round, np.array(total_revenue_per_round), betas_per_round, offers_send_to, offers_accepted_by, alphas,
            optimal_revenue_per_round, np.array(optimal_total_revenue_per_round), optimal_offers_send_to, optimal_offers_accepted_by)

def initialize_datacenters(n, price, p_star, beta_tuning_freq, eta, alpha):
    # return [
    #     BetaDatacenter("Rechenzentrum 1", 1, 5, alpha, price, 4, p_star, beta_tuning_freq, eta),
    #     BetaDatacenter("Rechenzentrum 2", 2, 20, alpha, price, 8, p_star, beta_tuning_freq, eta),
    #     BetaDatacenter("Rechenzentrum 3", 3, 30, alpha, price, 12, p_star, beta_tuning_freq, eta),
    #     BetaDatacenter("Rechenzentrum 4", 4, 40, alpha, price, 10, p_star, beta_tuning_freq, eta),
    #     BetaDatacenter("Rechenzentrum 5", 5, 65, alpha, price, 7, p_star, beta_tuning_freq, eta),
    #     BetaDatacenter("Rechenzentrum 6", 6, 80, alpha, price, 11, p_star, beta_tuning_freq, eta),
    # ]
    # return [
    #     BetaDatacenter("Rechenzentrum 1", 1, 10, alpha, price, 8, p_star, beta_tuning_freq, eta),
    #     BetaDatacenter("Rechenzentrum 2", 2, 20, alpha, price, 30, p_star, beta_tuning_freq, eta),
    #     BetaDatacenter("Rechenzentrum 3", 3, 30, alpha, price, 4, p_star, beta_tuning_freq, eta),
    #     BetaDatacenter("Rechenzentrum 4", 4, 40, alpha, price, 20, p_star, beta_tuning_freq, eta),
    #     BetaDatacenter("Rechenzentrum 5", 5, 50, alpha, price, 15, p_star, beta_tuning_freq, eta),
    #     BetaDatacenter("Rechenzentrum 6", 6, 60, alpha, price, 15, p_star, beta_tuning_freq, eta),
    # ]
    alpha_dcs = [
            AlphaDatacenter("Rechenzentrum 1", 1, 10, alpha, price),
            AlphaDatacenter("Rechenzentrum 2", 2, 20, alpha, price),
            AlphaDatacenter("Rechenzentrum 3", 3, 30, alpha, price),
            AlphaDatacenter("Rechenzentrum 4", 4, 40, alpha, price),
            AlphaDatacenter("Rechenzentrum 5", 5, 50, alpha, price),
            AlphaDatacenter("Rechenzentrum 6", 6, 60, alpha, price)
    ]
    # alpha_dcs = [
    #         AlphaDatacenter("Rechenzentrum 1", 1, 2, alpha, price),
    #         AlphaDatacenter("Rechenzentrum 2", 2, 10, alpha, price),
    #         AlphaDatacenter("Rechenzentrum 3", 3, 22, alpha, price),
    #         AlphaDatacenter("Rechenzentrum 4", 4, 36, alpha, price),
    #         AlphaDatacenter("Rechenzentrum 5", 5, 50, alpha, price),
    #         AlphaDatacenter("Rechenzentrum 6", 6, 57, alpha, price)
    # ]
    for dc in alpha_dcs:
        dc.add_datacenters(alpha_dcs)

    return alpha_dcs


def run_multiple():
    number_of_datacenters = 6  ## not used
    price = 75
    # price = 150
    # price = 80
    alpha = 0.8
    max_rounds = 10000
    p_star = 0.8
    beta_tuning_freq = 10
    eta = 0.1

    # seed = 1308
    # np.random.seed(seed)

    runs = 100
    ### für mehrere runs
    avgs = []
    opt_avgs = []

    avg_rev_per_run = np.empty((0,number_of_datacenters), float)
    avg_total_revs_per_run = np.empty((0,1), float)

    opt_avg_rev_per_run = np.empty((0, number_of_datacenters), float)
    opt_avg_total_revs_per_run = np.empty((0, 1), float)

    for i in range(runs):
        np.random.seed(i)
        datacenters = initialize_datacenters(number_of_datacenters, price, p_star, beta_tuning_freq, eta, alpha)
        (revenue_per_round, total_revenue_per_round, beta_per_round, offers_send_to,
         offers_accepted_by, alphas, optimal_revenue_per_round, optimal_total_revenue_per_round, optimal_offers_send_to,
         optimal_offers_accepted_by) = simulate(datacenters, max_rounds)

        avg_rev_per_run = np.vstack((avg_rev_per_run, revenue_per_round.mean(axis=1)))
        avg_total_revs_per_run = np.vstack((avg_total_revs_per_run, total_revenue_per_round.mean()))

        opt_avg_rev_per_run = np.vstack((opt_avg_rev_per_run, optimal_revenue_per_round.mean(axis=1)))
        opt_avg_total_revs_per_run = np.vstack((opt_avg_total_revs_per_run, optimal_total_revenue_per_round.mean()))

        avgs.append(np.mean(total_revenue_per_round))
        opt_avgs.append(np.mean(optimal_total_revenue_per_round))
        print(f"Avg. Total Revenue: {np.mean(total_revenue_per_round)}")
        print(f"Avg. Opt Total Revenue: {np.mean(optimal_total_revenue_per_round)}")
    print("AVG AVG: {:.2f}".format(np.mean(avgs)))
    print("OPT AVG AVG: {:.2f}".format(np.mean(opt_avgs)))

    save_dir_name = f"run_n{len(datacenters)}_p{price}_a{alpha}_b{beta_tuning_freq}_ps{p_star}_e{eta}_multi_{runs}"
    os.makedirs(f"figs/{save_dir_name}", exist_ok=True)
    plot_revenue_bar_for_one_rz(avg_rev_per_run, avg_total_revs_per_run, datacenters, save_dir_name)
    plot_revenue_bar_for_one_rz(opt_avg_rev_per_run, opt_avg_total_revs_per_run, datacenters, save_dir_name, suffix="optimal")

    for i, avg in enumerate(avg_rev_per_run.mean(axis=0)):
        print(f"RZ {i}: Total AVG: {avg}")

    for i, avg in enumerate(opt_avg_rev_per_run.mean(axis=0)):
        print(f"RZ {i}: Opt Total AVG: {avg}")

    #
    # # plot_distributions(datacenters, save_dir_name)
    # plot_revenue(revenue_per_round, total_revenue_per_round, datacenters, save_dir_name)
    # plot_revenue(optimal_revenue_per_round, optimal_total_revenue_per_round, datacenters, save_dir_name,
    #              suffix="optimal")
    # # plot_beta(beta_per_round, datacenters, save_dir_name)
    # # plot_send_to(offers_send_to, save_dir_name)  # can take a while
    # # plot_send_to(offers_send_to, save_dir_name)  # can take a while
    # # plot_send_to(optimal_offers_send_to, save_dir_name,suffix="optimal")  # can take a while
    # # plot_accepted_by(offers_accepted_by,save_dir_name) # can take a while
    # plot_send_and_accepted(offers_send_to, offers_accepted_by, save_dir_name)
    # plot_send_and_accepted(optimal_offers_send_to, optimal_offers_accepted_by, save_dir_name, suffix="optimal")


def main():
    number_of_datacenters = 6  ## not used
    price = 75
    # price = 80
    alpha = 0.8
    max_rounds = 10000
    p_star = 0.8
    beta_tuning_freq = 10
    eta = 0.1

    seed = 0
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
    (revenue_per_round, total_revenue_per_round, beta_per_round, offers_send_to, offers_accepted_by,
     alphas, optimal_revenue_per_round, optimal_total_revenue_per_round, optimal_offers_send_to, optimal_offers_accepted_by) = simulate(datacenters, max_rounds)

    print(f"Avg. Total Revenue: {np.mean(total_revenue_per_round[:-1000])}")
    print(f"Avg. Opt Total Revenue: {np.mean(optimal_total_revenue_per_round[:-1000])}")

    save_dir_name = f"run_n{len(datacenters)}_p{price}_a{alpha}_b{beta_tuning_freq}_ps{p_star}_e{eta}_s{seed}_config1"
    os.makedirs(f"figs/{save_dir_name}", exist_ok=True)

    # plot_distributions(datacenters, save_dir_name)
    plot_revenue(revenue_per_round, total_revenue_per_round, datacenters, save_dir_name)
    # plot_revenue(optimal_revenue_per_round, optimal_total_revenue_per_round, datacenters, save_dir_name,suffix="optimal")
    # plot_beta(beta_per_round, datacenters, save_dir_name)
    # plot_send_to(optimal_offers_send_to, save_dir_name,suffix="optimal")  # can take a while
    # plot_send_to(offers_send_to, save_dir_name)  # can take a while
    # plot_send_to(offers_send_to, save_dir_name)  # can take a while
    # plot_accepted_by(offers_accepted_by,save_dir_name) # can take a while
    plot_send_and_accepted(offers_send_to, offers_accepted_by, save_dir_name)
    # plot_send_and_accepted(optimal_offers_send_to, optimal_offers_accepted_by, save_dir_name, suffix="optimal")

    plot_alpha(alphas, offers_send_to, datacenters, save_dir_name)



if __name__ == "__main__":
    main()
    # run_multiple()
