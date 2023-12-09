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
        self.cn_var = specs["cn_var"]
        self.bandwidth = specs["bandwidth"]

        self.power = self.powers[0] # Current power level
        self._update_channel_gain() # channel gain ~ Rayleigh
        self._update_channel_noise() # channel noise ~ gaussian
        self.e_server = EdgeServer() # Edge server instance

    def offload(self, tasks: Dict[int, Task], acts: List[List]):
        for t_id, in_power in acts:
            for job in tasks[t_id]:
                if not in_power in self.powers:
                    raise ValueError(f"Unsupported wireless interface power: {in_power}")
                # consumed energy when offloading
                self.power = in_power
                job.cons_energy = (dbm_to_w(self.power)*job.b)/(self.get_channel_rate()*1e6)
                if job.cons_energy < 0:
                    # Transmission power is not set properly
                    job.deadline_missed = True
                else:
                    job.deadline_missed = False
                self.e_server.execute(job)
                #TODO: It's assuemd that remote can execute all tasks without missing deadlines
                # Time required to offload the task to edge server
                job.aet += job.b/(self.get_channel_rate()*1e6)

        #TODO: When channel status must be updated?
        self.update_channel_state()

    def update_channel_state(self):
        self._update_channel_gain()
        self._update_channel_noise()
        return (self.cg, self.cn)

    #TODO: How frequent channel gain must be updated?
    def _update_channel_gain(self):
        self.cg = np.random.rayleigh(scale=self.cg_sigma)

    def _update_channel_noise(self):
        self.cn = np.random.normal(loc=self.cn_mean,
                                   scale=np.sqrt(self.cn_var))

    def get_channel_rate(self):
        # FIXME: If bandwidth is negative, probably power is not enough
        return self.bandwidth * np.log2(1+dbm_to_w(self.power)*self.cg/self.cn)

    # No. of possible power values
    def __len__(self):
        return len(self.powers)


def dbm_to_w(pow_dbm: float) -> float:
    return (10**(pow_dbm/10))/1000
