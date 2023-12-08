import numpy as np
from typing import List, Dict

from models.remote import EdgeServer
from models.task import Task

class WirelessInterface:
    def __init__(self, specs: Dict[str, List or int]):
        # Power levels are represented in dbm and must be converted accordingly
        self.powers = specs["powers"]
        self.cg_sigma = specs["cg_sigma"]
        self.cn_mean = specs["cn_mean"]
        self.cn_std = specs["cn_std"]
        self.bandwidth = specs["bandwidth"]

        self.power = self.powers[0] # Current power level
        self.update_channel_gain() # channel gain ~ Rayleigh
        self.update_channel_noise() # channel noise ~ gaussian
        self.e_server = EdgeServer() # Edge server instance

    def offload(self, tasks: Dict[int, Task], acts: List[List]):
        for t_id, in_power in acts:
            for job in tasks[t_id]:
                if not in_power in self.powers:
                    raise ValueError(f"Unsupported wireless interface power: {in_power}")
                # consumed energy when offloading
                self.power = in_power
                job.cons_energy = (dbm_to_mw(self.power)*job.b)/self.get_channel_rate()
                #TODO: It's assuemd that remote can execute all tasks without missing deadlines
                job.deadline_missed = False
                self.e_server.execute(job)
                # update channel status
        #TODO: When channel status must be updated?
        self.update_channel_gain()
        self.update_channel_noise()

    #TODO: How frequent channel gain must be updated?
    def update_channel_gain(self):
        self.cg = np.random.rayleigh(scale=self.cg_sigma)
        return self.cg

    def update_channel_noise(self):
        self.cn = np.random.normal(loc=self.cn_mean,
                                   scale=self.cn_std)
        return self.cn

    def get_channel_rate(self):
        return self.bandwidth * np.log2(1+self.power*self.cg/self.cn)

    # No. of possible power values
    def __len__(self):
        return len(self.powers)


def dbm_to_mw(pow_dbm: float) -> float:
    return 10**(pow_dbm/10)
