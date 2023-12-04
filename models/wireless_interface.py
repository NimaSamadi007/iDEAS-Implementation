
class WirelessInterface:
    def __init__(self, powers):
        self.powers = powers
        self.power = powers[0] # Current power level

    # No. of possible power values
    def __len__(self):
        return len(self.powers)

