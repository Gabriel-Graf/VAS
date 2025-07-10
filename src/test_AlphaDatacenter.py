from unittest import TestCase
from AlphaDatacenter import AlphaDatacenter

class TestAlphaDatacenter(TestCase):

    def test_get_prob_better(self):
        competitor = AlphaDatacenter(
            name="competitor", # egal
            dc_id=0,           # egal
            cost=15,
            alpha=0,           # egal
            price=10           # egal
        )
        target = AlphaDatacenter(
            name="target",  # egal
            dc_id=1,        # egal
            cost=5,
            alpha=0,        # egal
            price=10        # egal
        )
        our_offer = 2.2

        prob = AlphaDatacenter.get_prob_better(competitor, target, our_offer)
        self.assertAlmostEqual( 0.9535213421362799, prob, places=4)
    
    def get_expected_revenue(self):

        target = AlphaDatacenter(
            name="target",  # egal
            dc_id=1,        # egal
            cost=5,
            alpha=0,        # egal
            price=10        # egal
        )
        competing_dcs = [
            AlphaDatacenter(
                name="target",  # egal
                dc_id=1,        # egal
                cost=5,
                alpha=0,        # egal
                price=10        # egal
            ),
            AlphaDatacenter(
                name="target",  # egal
                dc_id=1,        # egal
                cost=5,
                alpha=0,        # egal
                price=10        # egal
            ),
        ]