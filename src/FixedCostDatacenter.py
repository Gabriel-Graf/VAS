import random
import math
from typing import Optional, List

from Datacenter import Datacenter, Offer

class FixedCostDatacenter(Datacenter):
    def __init__(self, name: str, cost: float, alpha: float, price: float):
        self.name = name
        self.cost = cost
        self.alpha = alpha
        self.price = price
        self.revenue_from_delegated_offer: Optional[float] = None
        self.received_offers: List[Offer] = []
        self.selected_offer: Optional[Offer] = None

    def evaluate(self, others: List[Datacenter]):
        if not others:
            return
        chosen = random.randint(0, len(others) - 1)
        def accept(new_cost):
            self.revenue_from_delegated_offer = (
                self.price - self.cost + math.ceil((self.cost - new_cost) * self.alpha)
            )
        others[chosen].receive(Offer(self.name, self.cost, accept))

    def receive(self, offer: Offer):
        self.received_offers.append(offer)

    def check_offers(self):
        filtered = [o for o in self.received_offers if o.original_cost > self.cost]
        filtered.sort(key=lambda o: o.original_cost, reverse=True)
        self.selected_offer = filtered[0] if filtered else None
        if self.selected_offer:
            self.selected_offer.accept(self.cost)
            print(f"[SLCT] {self.name} accepts offer of {self.selected_offer.sender}")
        self.received_offers.clear()


    def calculate_revenue(self) -> float:
        own_revenue = self.revenue_from_delegated_offer if self.revenue_from_delegated_offer is not None else self.price - self.cost
        secondary_revenue = 0.0
        if self.selected_offer:
            secondary_revenue = (self.selected_offer.original_cost - self.cost) * (1 - self.alpha)
        print(f"[RVNE] {self.name} evalutes to ownRevenue: {own_revenue}, secondaryRevenue: {secondary_revenue}")
        return own_revenue + math.ceil(secondary_revenue)