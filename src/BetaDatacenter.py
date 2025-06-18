from typing import List
from random import gauss
import math

import numpy as np

from Datacenter import Datacenter, Offer
from FixedCostDatacenter import FixedCostDatacenter

class BetaDatacenter(FixedCostDatacenter):
    def __init__(self, name: str, mean: float, alpha: float, price: float, variance: float, p_star: float, beta_tuning_freq: int, eta: float ):
        super().__init__(name, mean, alpha, price)

        self.price = price
        self.history = []  # type: List[bool]
        self.mean = mean
        self.variance = variance
        self.cost = 0
        self.name = name
        self.alpha = alpha
        self.beta = 0
        self.p_star = p_star

        self.beta_tuning_freq = beta_tuning_freq
        self.round_count = 0
        self.eta = eta

    def true_false_ratio(self):
        total = len(self.history)
        if total == 0:
            return 0.0
        true_count = sum(self.history)
        return true_count / total

    def get_gaussian_cost(self):
        # np.random.normal(self.mean, self.variance)dc
        return gauss(self.mean, math.sqrt(self.variance))

    def evaluate(self, others: List[Datacenter]):
        chosen_index = -1
        # others: List[BetaDatacenter] = [BetaDatacenter(dc) for dc in others]

        ## get cost
        self.cost = self.get_gaussian_cost()

        ## beta tuning
        if self.round_count % self.beta_tuning_freq == 0:
            p_hat = self.true_false_ratio()
            self.beta = self.beta + self.eta*(p_hat - self.p_star)
            self.beta = min(self.beta, 1.0)  # ensure beta does not exceed 1.0
            self.beta = max(self.beta, 0.01)  # ensure beta is at least 0.0

        risk_ordered = []
        for i,dc in enumerate(others):
            adjusted_cost = dc.mean + (self.beta * dc.variance)
            risk_ordered.append((adjusted_cost, dc,i))

        risk_ordered.sort()
        for (adjusted_cost,dc,i) in risk_ordered:
            ### determine predicted revenue
            predicted_revenue = self.price - self.cost + math.ceil((self.cost - adjusted_cost) * self.alpha)
            if predicted_revenue > (self.price - self.cost):
                chosen_index = i
                break ## select the first where the condition is true

        if chosen_index != -1:
            def accept(new_cost):
                self.revenue_from_delegated_offer = (
                        self.price - self.cost + math.ceil((self.cost - new_cost) * self.alpha)
                )

            others[chosen_index].receive(Offer(self.name,self.cost, accept))
            print(f"[SEND] {self.name} sends offer to {others[chosen_index].name} with own cost {self.cost}")


    def receive(self, offer: Offer):
        super().receive(offer)

    def check_offers(self):
        super().check_offers()

    def calculate_revenue(self) -> float:
        return super().calculate_revenue()
