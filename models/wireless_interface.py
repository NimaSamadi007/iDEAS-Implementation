import numpy as np
from typing import List, Dict

from models.remote import EdgeServer
from models.task import Task

class WirelessInterface:
    def __init__(self, specs: Dict[str, List or int]):
        self.powers = specs["powers"]
        self.cg_sigma = specs["cg_sigma"]
        self.cn_mean = specs["cn_mean"]
        self.cn_std = specs["cn_std"]
        self.bandwidth = specs["bandwidth"]

        self.power = self.powers[0] # Current power level
        self.cg = 0. # channel gain ~ Rayleigh
        self.cn = 0. # channel noise ~ gaussian
        self.e_server = EdgeServer() # Edge server instance

    def offload(self, jobs: List[Task], in_power: float):
        print(f"Offloading task {jobs[0]}")
        print("-------------")
        for job in jobs:
            self.e_server.execute(job)

    #TODO: How frequent channel gain must be updated?
    def update_channel_gain(self):
        self.cg = np.random.rayleigh(scale=self.cg_sigma)
        return self.cg

    def update_channel_noise(self):
        self.cn = np.random.normal(loc=self.cn_mean,
                                   scale=self.cn_std)
        return self.cn

    def get_channel_rate(self) -> float:
        return self.bandwidth * np.log2(1+self.power*self.cg/self.cn)

    # No. of possible power values
    def __len__(self):
        return len(self.powers)
