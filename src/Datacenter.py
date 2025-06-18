from typing import Callable, List, Optional

class Offer:
    def __init__(self, sender, original_cost: float, accept: Callable[[float], None]):
        self.sender = sender
        self.original_cost = original_cost
        self.accept = accept



class Datacenter:
    def evaluate(self, others: List['Datacenter']):
        raise NotImplementedError

    def receive(self, offer: Offer):
        raise NotImplementedError

    def check_offers(self):
        raise NotImplementedError

    def calculate_revenue(self) -> int:
        raise NotImplementedError