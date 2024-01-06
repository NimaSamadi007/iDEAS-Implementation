import numpy as np
import json
from typing import List, Dict
import copy

from models.remote import EdgeServer, Cloud
from models.task import Task, TOMSTask

class WirelessInterface:
    def __init__(self, w_inter_conf_path):
        try:
            with open(w_inter_conf_path, "r") as f:
                specs = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Wireless interface configuration file not found at {w_inter_conf_path}")

        # Power levels are represented in dbm and must be converted accordingly
        self.powers = specs["powers"] # must be sorted incrementally
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
                rate = self.get_channel_rate()*1e6 # unit: bps
                if rate <= 0:
                    raise ValueError("Negative channel rate!")
                self.e_server.execute(job)
                job.aet += ((job.b*1024*8)/rate)*1e3 # aet unit: ms
                if job.aet > job.p:
                    job.deadline_missed = True
                else:
                    job.cons_energy = (dbm_to_w(self.power)*job.b*1024*8) / rate

        #TODO: When channel status must be updated?

    #TODO: How frequent channel state must be updated?
    def update_channel_state(self):
        self.cg = np.random.rayleigh(scale=self.cg_sigma)
        return self.cg

    def get_channel_rate(self):
        return self.bandwidth * np.log2(1+dbm_to_w(self.power)*self.cg/self.cn_power)

    def get_min_energy(self, task: Task) -> float:
        min_power = self.powers[0]
        rate = self.get_channel_rate()*1e6 # unit: bps
        min_energy = (dbm_to_w(min_power)*task.b*1024*8) / rate
        return min_energy

    # No. of possible power values
    def __len__(self):
        return len(self.powers)

class TOMSWirelessInterface:
    def __init__(self, conf_path):
        try:
            with open(conf_path, "r") as f:
                conf = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Wireless interface configuration file not found at {conf_path}")

        # Power levels are represented in dbm and must be converted accordingly
        self.up_rate_max, self.up_rate_min = conf["upload_rate"]
        self.down_rate_max, self.down_rate_min = conf["download_rate"]
        self.power = conf["power"]

        self.cloud = Cloud({"freq": conf["cloud_freq"],
                            "power_active": conf["cloud_power_active"],
                            "power_idle": conf["cloud_power_idle"]})
        self.update_channel_state()


    def offload(self, tasks: Dict[int, TOMSTask], acts: List[List]):
        for t_id,_ in acts:
            self._run_task(tasks[t_id])

    def _run_task(self, task: TOMSTask):
        # Must send the task input size to the cloud
        upload_time = task.in_size/self.uplink_rate
        # If cloud hasn't stored the task before, whole task must be sent, too
        if not self.is_offloaded_before(task):
            upload_time += (task.b/self.uplink_rate)
        # Execute the task
        task.aet = upload_time
        self.cloud.execute(task)
        # Finally, download the results from the cloud
        download_time = task.out_size/self.downlink_rate
        task.aet += download_time

        # Check if the task deadline have been missed
        if task.aet > task.p:
            task.deadline_missed = True
        else:
            task.cons_energy = self.power * (download_time+upload_time)

    def get_cons_energy(self, task: TOMSTask) -> float:
        # make a copy of the task
        task_copy = copy.deepcopy(task)
        self._run_task(task_copy)
        return task_copy.cons_energy

    def update_channel_state(self):
        self.uplink_rate = self.up_rate_min + np.random.randint(0, self.up_rate_max-self.up_rate_min+1)
        self.downlink_rate = self.down_rate_min + np.random.randint(0, self.down_rate_max-self.down_rate_min+1)
        return self.uplink_rate, self.downlink_rate

    def is_offloaded_before(self, task: TOMSTask):
        return self.cloud.is_executed_before(task)


def dbm_to_w(pow_dbm: float) -> float:
    return (10**(pow_dbm/10))/1000
