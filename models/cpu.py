import numpy as np

from typing import List, Dict
from models.task import Task

class CPU:
    def __init__(self, specs):
        self.freqs = np.asarray(specs['freqs']) # Mhz - sorted incrementally
        self.powers = np.asarray(specs['powers'])*1e-3 # mW -> W - sorted incrementally
        self.model = specs['model']
        self.ncore = specs['num_cores']
        self.cpu_type = specs['type']

        self.util = 0
        self.dynamic_slack = 0
        self.freq = self.freqs[-1] # Maximum frequency by default

    def __repr__(self):
        return f"'{self.model} {self.cpu_type}' CPU with {self.ncore} cores"

    # Assign tasks to execute
    def step(self, tasks: Dict[int, Task], acts: List[List]):
        # Check schedulability criteria
        total_util = 0
        for t_id, in_freq in acts:
            job = tasks[t_id][0]
            wcet_scaled = job.wcet * (self.freq/in_freq)
            total_util += wcet_scaled / job.p
        if total_util > 1: # Not schedulable
            #TODO: What's the best thing to do in case of unschedulable tasks
            for t_id, in_freq in acts:
                for job in tasks[t_id]:
                    job.deadline_missed = True
            return
        self.util = total_util

        # AET can only be set at the time of execution
        for t_id, in_freq in acts:
            for job in tasks[t_id]:
                job.gen_aet()
                # Check if deadline will be missed
                true_exec_time = (self.freq/in_freq)*job.aet
                if true_exec_time > job.p:
                    job.deadline_missed = True
                    continue
                # Calculate energy consumption (chip energy conusmption at given frequency)
                cons_power = self.powers[self.freqs == in_freq][0] # There should be only one element
                job.cons_energy = cons_power * (true_exec_time/1000)

    def get_min_energy(self, task: Task) -> float:
        max_freq = self.freqs[-1]
        for i, frq in enumerate(self.freqs):
            # If the task can be run, fetch the corresponding
            # power which show the minimum computation power
            wcet_scaled = task.wcet * (max_freq/frq)
            if wcet_scaled < task.p:
                min_energy = self.powers[i] * (wcet_scaled/1000)
                return min_energy
        raise ValueError(f"Unable to schedult task: {task}")
