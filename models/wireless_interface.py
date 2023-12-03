
class WirelessInterface:
    def __init__(self, powers):
        self.powers = powers

    # No. of possible power values
    def __len__(self):
        return len(self.powers)

