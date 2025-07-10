import math
import random
from scipy.stats import norm
from FixedCostDatacenter import FixedCostDatacenter
from typing import List, Tuple, Callable

from src.Datacenter import Datacenter, Offer
import numpy as np

class BaseState:
    def __init__(self, name: str, sid: int, state, initial_j: int = None,  init_alpha: float = None, datacenters: List[FixedCostDatacenter] = None,
                 datacenter_self: FixedCostDatacenter = None):
        self.name = name
        self.sid = sid

        if state is None:
            self.j = initial_j
            self.prev_j = 0
            self.alpha_i = init_alpha
            self.r_c = 5
            self.history_alpha = {self.j: [self.alpha_i]}
            self.h_i = {self.j: -1 +1}  ## must be init with -1, the  "+1" is because we are "skipping" the entry for the first Entry
            ## will be set from outside
            self.datacenters = datacenters
            self.datacenter_self = datacenter_self
        else:
            self.j = state.j
            self.prev_j = state.prev_j
            self.alpha_i = state.alpha_i

            self.r_c = state.r_c
            self.history_alpha = state.history_alpha
            self.h_i = state.h_i
            self.datacenter_self = state.datacenter_self
            self.datacenters = state.datacenters

            ## only call entry, when its not initialized the first time
            self.entry()

    def accepted(self):
        self.r_c = max(0, self.r_c - 1)

    def declined(self):
        self.r_c = max(0, self.r_c - 1)

    def entry(self):
        pass

    def __str__(self):
        return f"{self.name}: j: {self.j}, alpha: {self.alpha_i}, h_alpha: {self.history_alpha[self.j]}, h_i: {self.h_i[self.j]}, r_c: {self.r_c}"


class SuccessOnce(BaseState):
    def __init__(self, state: BaseState = None, init_j: int = None, init_alpha: float = None,
                 datacenters: List[FixedCostDatacenter] = None,
                 datacenter_self: FixedCostDatacenter = None):
        super().__init__("SuccessOnce", 1, state, init_j,init_alpha, datacenters, datacenter_self)
               

    def entry(self):
        self.alpha_i = min(self.alpha_i + ((self.history_alpha[self.j][self.h_i[self.j] + 1] -
                                            self.history_alpha[self.j][self.h_i[self.j]]) / 2), 1)
        self.h_i[self.j] = self.h_i[self.j] + 1
        self.history_alpha[self.j][self.h_i[self.j]] = self.alpha_i

    def accepted(self):
        super().accepted()
        state = SuccessRep(self)
        return state

    def declined(self):
        super().declined()
        if len(self.history_alpha[self.j]) == 1:
            state = FailureNoHistory(self)
        else:
            state = FailureHistory(self)

        return state


class SuccessRep(BaseState):
    def __init__(self, state: BaseState = None):
        super().__init__("SuccessRep", 2, state)

    def entry(self):
        self.alpha_i = min(self.alpha_i * 2, 1)
        self.h_i[self.j] = self.h_i[self.j] + 1
        self.history_alpha[self.j].append(self.alpha_i)

    def accepted(self):
        super().accepted()

        if self.j != 1 and (self.alpha_i >= 0.9 and self.r_c == 0) :
            self.prev_j = self.j
            self.j = max(self.j - 1, 1)
            state = NewPartnerCheap(self)
        else:
            state = SuccessRep(self)
        
        return state

    def declined(self):
        super().declined()
        state = FailureHistory(self)
        
        return state


class FailureHistory(BaseState):
    def __init__(self, state: BaseState = None):
        super().__init__("FailureHistory", 3, state)

    def entry(self):
        self.h_i[self.j] = self.h_i[self.j] - 1
        self.alpha_i = self.history_alpha[self.j][self.h_i[self.j]]

        ## clean f penultimate (second last) alpha_i from history
        if self.h_i[self.j]+2 < len(self.history_alpha[self.j]):
            self.history_alpha[self.j].remove(self.history_alpha[self.j][self.h_i[self.j]+2])


    def accepted(self):
        super().accepted()
        self.h_i[self.j] = self.h_i[self.j] - 1 ### added in later
        state = SuccessOnce(self)
        return state


    def declined(self):
        super().declined()
        if self.h_i[self.j] <= 0:
            state = FailureNoHistory(self)
        else:
            state = FailureHistory(self)


        return state


class FailureNoHistory(BaseState):
    def __init__(self, state: BaseState = None):
        super().__init__("FailureNoHistory", 4, state)

    def entry(self):
        # self.history_alpha[self.j].clear()


        self.alpha_i *= 0.5
        self.history_alpha[self.j].insert(0,self.alpha_i) ## "append" in front

        self.history_alpha[self.j] = self.history_alpha[self.j][:2] ## only keep the two recent values

    def accepted(self):
        super().accepted()
        state = SuccessOnce(self)
        
        return state

    def declined(self):
        super().declined()
        if self.j < self.datacenter_self.dc_id -1 and (self.alpha_i <= 0.1 and self.r_c == 0):

            self.prev_j = self.j
            self.j = min(self.j + 1, self.datacenter_self.dc_id -1)  ### new Partner


            state = NewPartnerExp(self)
        else:
            state = FailureNoHistory(self)

        return state


class NewPartnerExp(BaseState):
    def __init__(self, state: BaseState = None):
        super().__init__("NewPartnerExp", 5, state)

    def entry(self):
        self.h_i[self.j] = -1  ## init history index for new partner

        self.alpha_i = 0.5
        self.history_alpha.setdefault(self.j,[]).append(self.alpha_i)
        self.h_i[self.j] = self.h_i[self.j] + 1
        self.r_c = 5

    def accepted(self):
        super().accepted()
        state = SuccessRep(self)
        
        return state

    def declined(self):
        super().declined()
        state = FailureNoHistory(self)
        
        return state


class NewPartnerCheap(BaseState):
    def __init__(self, state: BaseState = None):
        super().__init__("NewPartnerCheap", 6, state)

    def entry(self):
        self.h_i[self.j] = -1  ## init history index for new partner
        d_ij1 = self.datacenter_self.cost - self.datacenters[self.j + 1].cost

        d_ij = self.datacenter_self.cost - self.datacenters[self.j].cost

        self.alpha_i = d_ij1 * self.history_alpha[self.prev_j][self.h_i[self.prev_j]] / d_ij
        self.history_alpha.setdefault(self.j,[]).append(self.alpha_i)
        self.h_i[self.j] = self.h_i[self.j] + 1

    def accepted(self):
        super().accepted()
        state = SuccessRep(self)
        
        return state

    def declined(self):
        super().declined()
        self.j = self.j + 1
        self.r_c = 5
        self.h_i[self.j] = self.h_i[self.j] + 1  ## to balance out h_ij - 1 in "entry" of "FailureHistory"

        state = FailureHistory(self)
        
        return state


class AlphaOffer(Offer):
    def __init__(self, sender, original_cost: float, alpha: float, accept: Callable[[float, int], None],  decline: Callable[[], None]):
        super().__init__(sender,original_cost,accept)
        self.alpha = alpha
        self.decline = decline


class AlphaDatacenter(FixedCostDatacenter):

    def __init__(self, name: str, dc_id: int, cost: float, alpha: float, price: float):
        super().__init__(name, dc_id, cost, alpha, price)
        self.state = None
        self.prob_better_threshold = 0.99
        self.current_partner = None
        self.was_accepted = False
        self.send_to = None
        self.accepted_by = None

    def round_reset(self):
        self.send_to = None
        self.accepted_by = None

    def add_datacenters(self, datacenters: List["AlphaDatacenter"]):
        self.datacenters = datacenters



    def find_initial_partner(self, datacenters: List["AlphaDatacenter"]) -> "AlphaDatacenter":
        default_partner = None
        last_revenue = 0
        self.alpha = np.random.uniform(0.01,1)

        if self.dc_id > math.ceil(len(datacenters)/2): ## e.g. 4 is counted as lower half for 7 datacenters
            default_partner = (len(datacenters) - self.dc_id) +1
        elif self.dc_id !=1:
            default_partner = self.dc_id - 1

        initial_partner = default_partner
        competing_dc = datacenters.copy()
        # for target_datacenter in datacenters[default_partner if default_partner else len(datacenters): len(datacenters)]:
        for target_datacenter in datacenters[:default_partner-1 if default_partner else 0]:
            revenue, competing_dc = self.get_expected_revenue(target_datacenter, competing_dc)
            if revenue > last_revenue:
                initial_partner = target_datacenter.dc_id

        # self.current_partner = initial_partner
        if initial_partner is not None:
            self.state = SuccessOnce(init_j=initial_partner, init_alpha=self.alpha, datacenters=self.datacenters, datacenter_self=self)
            # print(f"RZ {self.dc_id}: initial partner is {initial_partner}")


    def get_expected_revenue(self, target_datacenter: "AlphaDatacenter", competing_dcs: List["AlphaDatacenter"]) -> Tuple[float, List["AlphaDatacenter"]]:
        prob_any_better = 1
        our_offer = self.alpha* abs(self.cost - target_datacenter.cost)
        strongest_competitor = [0,None]

        for competitor in competing_dcs:
            prob_better = self.get_prob_better(competitor, target_datacenter, our_offer)
            prob_any_better *= prob_better

            if strongest_competitor[0] < prob_better:
                strongest_competitor = [prob_better, competitor]

        # kick out dc which is most likely to take the current target dc for the next round
        if strongest_competitor[0] >= self.prob_better_threshold:
            competing_dcs.remove(strongest_competitor[1])

        expected_revenue = (1-prob_any_better) * our_offer
        return expected_revenue, competing_dcs

    @staticmethod
    def get_prob_better(competitor, target_datacenter, our_offer):
        revenue_comp = abs(competitor.cost - target_datacenter.cost)
        expected_value = revenue_comp * 0.5
        sigma = (revenue_comp*0.5)/3

        return 1 - norm.cdf(our_offer, expected_value, sigma)



    def evaluate(self, others: List[Datacenter]):
        if not others or self.state  == None:
            return


        # if not self.was_accepted:
        #     self.state = self.state.declined()
        # self.was_accepted = False
        
        ## choose by dc_id and not by index
        self.current_partner = [dc for dc in others if self.state.j == dc.dc_id ][0]


        def accept(new_cost, dc_id):
            self.revenue_from_delegated_offer = (
                    self.price - self.cost + math.ceil((self.cost - new_cost) * self.alpha)
            )
            self.state = self.state.accepted()
            self.was_accepted = True
            self.accepted_by = dc_id

        def decline():
            self.state = self.state.declined()
            self.was_accepted = False

        self.current_partner.receive(AlphaOffer(self.name, self.cost, 1.0 - self.alpha, accept, decline))
        self.send_to = self.current_partner.dc_id


    def check_offers(self):
        potential_accept = [o for o in self.received_offers if o.original_cost > self.cost]

        potential_accept.sort(key=lambda o: o.original_cost*o.alpha, reverse=True)


        self.selected_offer = potential_accept[0] if potential_accept else None
        if self.selected_offer:
            self.selected_offer.accept(self.cost, self.dc_id)
            # print(f"[SLCT] {self.name} accepts offer of {self.selected_offer.sender}")
            self.received_offers.remove(self.selected_offer)
        for o in self.received_offers:
            o.decline()

        self.received_offers.clear()