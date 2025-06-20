from typing import List
from random import gauss
import math

import numpy as np

from Datacenter import Datacenter, Offer
from FixedCostDatacenter import FixedCostDatacenter

class BetaDatacenter(FixedCostDatacenter):
    def __init__(self, name: str, dc_id: int, mean: float, alpha: float, price: float, sigma: float, p_star: float, beta_tuning_freq: int, eta: float):
        super().__init__(name, dc_id, mean, alpha, price)

        self.price = price
        self.history = []  # type: List[bool]
        self.mean = mean
        self.sigma = sigma
        self.cost = 0
        self.name = name
        self.alpha = alpha
        self.beta = p_star
        self.p_star = p_star

        self.beta_tuning_freq = beta_tuning_freq
        self.round_count = 0
        self.eta = eta
        self.send_to = None
        self.accepted_by = None

    def round_reset(self):
        self.send_to = None
        self.accepted_by = None
    def true_false_ratio(self):
        ratio =  sum(self.history) / self.beta_tuning_freq
        self.history.clear()
        return ratio

    def get_gaussian_cost(self):
        # np.random.normal(self.mean, self.variance)
        return gauss(self.mean, self.sigma)

    def evaluate(self, others: List[Datacenter]):
        chosen_index = -1
        # others: List[BetaDatacenter] = [BetaDatacenter(dc) for dc in others]

        ## get cost
        self.cost = max(0,self.get_gaussian_cost())

        ## beta tuning
        if self.round_count > 0 and self.round_count % self.beta_tuning_freq == 0:
            p_hat = self.true_false_ratio()
            self.beta = self.beta + self.eta*(p_hat - self.p_star)
            # self.beta = min(self.beta, 1.0)  # ensure beta does not exceed 1.0
            # self.beta = max(self.beta, 0.01)  # ensure beta is at least 0.0

        risk_ordered = []
        for i,dc in enumerate(others):
            adjusted_cost = dc.mean + (self.beta * dc.sigma)
            risk_ordered.append((adjusted_cost, dc,i))
        try:
            # risk_ordered.sort()
            risk_ordered = sorted(risk_ordered, key=lambda x: x[0])
        except TypeError as e:
            print(e)
            print(risk_ordered)

        for (adjusted_cost,dc,i) in risk_ordered:
            ### determine predicted revenue
            predicted_additional_revenue = (self.cost - dc.mean) * self.alpha
            if predicted_additional_revenue > 0:
                chosen_index = i
                break ## select the first where the condition is true

        if chosen_index != -1:
            def accept(new_cost, dc_id):
                self.revenue_from_delegated_offer = (
                        self.price - self.cost + math.ceil((self.cost - new_cost) * self.alpha)
                )
                self.history.append(True)
                self.accepted_by = dc_id

            others[chosen_index].receive(Offer(self.name, self.cost, accept))
            # print(f"[SEND] {self.name} sends offer to {others[chosen_index].name} with own cost {self.cost}")
            self.send_to = others[chosen_index].dc_id

        self.round_count += 1


    def receive(self, offer: Offer):
        super().receive(offer)

    def check_offers(self):
        super().check_offers()

    def calculate_revenue(self) -> float:
        return super().calculate_revenue()
