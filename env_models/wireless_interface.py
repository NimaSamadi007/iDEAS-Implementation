import numpy as np
from typing import List, Dict

from env_models.remote import EdgeServer
from env_models.task import Task
from utils.utils import load_yaml


class WirelessInterface:
    def __init__(self, w_inter_conf_path):
        """
        Wireless interface model used in iDEAS

        Args:
            w_inter_conf_path (str): path to the wireless interface configuration file
        """
        specs = load_yaml(w_inter_conf_path)

        # Power levels are represented in dbm and must be converted accordingly
        self.powers = specs["powers"]  # must be sorted incrementally
        self.cg_sigma = specs["cg_sigma"]
        self.cn_power = specs["cn_power"]
        self.bandwidth = specs["bandwidth"]

        self.power = self.powers[0]  # Current power level
        self.update_channel_state()  # channel gain ~ Rayleigh
        self.e_server = EdgeServer()  # Edge server instance

    def set_cn(self, cn):
        self.cn_power = cn

    def offload(self, tasks: Dict[int, Task], acts: List[List]):
        for t_id, in_power in acts:
            for job in tasks[t_id]:
                if in_power not in self.powers:
                    raise ValueError(
                        f"Unsupported wireless interface power: {in_power}"
                    )
                # consumed energy when offloading
                self.power = in_power
                rate = self.get_channel_rate() * 1e6  # unit: bps
                if rate <= 0:
                    raise ValueError("Negative channel rate!")
                self.e_server.execute(job)
                job.aet += ((job.b * 1024 * 8) / rate) * 1e3  # aet unit: ms
                if job.aet > job.p:
                    job.deadline_missed = True
                job.cons_energy = (dbm_to_mw(self.power) * job.b * 1024 * 8) / rate

    def update_channel_state(self):
        self.cg = np.random.rayleigh(scale=self.cg_sigma)
        return self.cg

    def get_channel_rate(self):
        return self.bandwidth * np.log2(
            1 + dbm_to_w(self.power) * self.cg / self.cn_power
        )

    def get_min_energy(self, task: Task) -> float:
        min_power = self.powers[0]
        rate = self.get_channel_rate() * 1e6  # unit: bps
        min_energy = (dbm_to_mw(min_power) * task.b * 1024 * 8) / rate
        return min_energy

    def get_rate_bounds(self):
        max_cg = 2 * self.cg_sigma
        return [
            0,
            self.bandwidth * np.log2(1 + self.powers[-1] * max_cg / self.min_cn_power),
        ]

    def set_cn_power_bounds(self, min_cn_power, max_cn_power):
        self.min_cn_power = min_cn_power
        self.max_cn_power = max_cn_power

    # No. of possible power values
    def __len__(self):
        return len(self.powers)


class RRLOWirelessInterface(WirelessInterface):
    def __init__(self, w_inter_conf_path):
        """
        Wireless interface model used in RRLO algorithm

        Args:
            w_inter_conf_path (str): path to the wireless interface configuration file
        """
        super().__init__(w_inter_conf_path)

    def offload(self, tasks: Dict[int, Task], power_level: float):
        for t_id, job in tasks.items():
            for job in tasks[t_id]:
                if power_level not in self.powers:
                    raise ValueError(
                        f"Unsupported wireless interface power: {power_level}"
                    )
                # consumed energy when offloading
                self.power = power_level
                rate = self.get_channel_rate() * 1e6  # unit: bps
                if rate <= 0:
                    raise ValueError("Negative channel rate!")
                self.e_server.execute(job)
                job.aet += ((job.b * 1024 * 8) / rate) * 1e3  # aet unit: ms
                if job.aet > job.p:
                    job.deadline_missed = True
                job.cons_energy = (dbm_to_mw(self.power) * job.b * 1024 * 8) / rate


def dbm_to_w(pow_dbm: float) -> float:
    return (10 ** (pow_dbm / 10)) / 1000


def dbm_to_mw(pow_dbm: float) -> float:
    return 10 ** (pow_dbm / 10)
