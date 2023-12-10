import numpy as np
from typing import List, Dict

from models.remote import EdgeServer
from models.task import Task

class WirelessInterface:
    def __init__(self, specs: Dict[str, List or int]):
        # Power levels are represented in dbm and must be converted accordingly
        self.powers = specs["powers"]
        self.cg_sigma = specs["cg_sigma"]
        self.cn_power = specs["cn_power"]
        self.bandwidth = specs["bandwidth"]

        self.power = self.powers[0] # Current power level
        self.update_channel_state() # channel gain ~ Rayleigh
        self.e_server = EdgeServer() # Edge server instance

    def offload(self, tasks: Dict[int, Task], acts: List[List]):
        for t_id, in_power in acts:
            for job in tasks[t_id]:
                if not in_power in self.powers:
                    raise ValueError(f"Unsupported wireless interface power: {in_power}")
                # consumed energy when offloading
                self.power = in_power
                rate = self.get_channel_rate()*1e6
                if rate <= 0:
                    raise ValueError("Negative channel rate!")
                self.e_server.execute(job)
                job.aet += job.b/rate
                if job.aet > job.p:
                    job.deadline_missed = True
                else:
                    job.cons_energy = (dbm_to_w(self.power)*job.b)/(self.get_channel_rate()*1e6)

        #TODO: When channel status must be updated?

    #TODO: How frequent channel state must be updated?
    def update_channel_state(self):
        self.cg = np.random.rayleigh(scale=self.cg_sigma)
        return self.cg

    def get_channel_rate(self):
        return self.bandwidth * np.log2(1+dbm_to_w(self.power)*self.cg/self.cn_power)

    # No. of possible power values
    def __len__(self):
        return len(self.powers)


def dbm_to_w(pow_dbm: float) -> float:
    return (10**(pow_dbm/10))/1000
