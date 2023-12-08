from typing import List, Dict
from models.task import Task

class CPU:
    def __init__(self, specs):
        self.freqs = specs['freqs']
        self.powers = specs['powers']
        self.model = specs['model']
        self.ncore = specs['num_cores']
        self.cpu_type = specs['type']

        self.util = 0
        self.dynamic_slack = 0
        self.freq = self.freqs[-1] # Maximum frequency by default

    def __repr__(self):
        return f"'{self.model} {self.cpu_type}' CPU with {self.ncore} cores"

    # Assign tasks to execute
    def execute(self, tasks: Dict[int, Task], acts: List[List]):
        # print(f"Executing CPU {self.cpu_type} task {jobs[0]}")

        # Check schedulability criteria
        total_util = 0
        for t_id, in_freq in acts:
            job = tasks[t_id][0]
            wcet_scaled = job.wcet * (self.freq/in_freq)
            total_util += wcet_scaled / job.p
        print(f"{self.cpu_type} task util: {total_util}")
        if total_util > 1: # Not schedulable
            #TODO: What's the best thing to do in case of unschedulable tasks
            for t_id, in_freq in acts:
                for job in tasks[t_id]:
                    job.deadline_missed = True
            return

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
                job.cons_energy = self.powers[self.freqs.index(in_freq)] * true_exec_time
                job.cons_energy /= 1000. # Consumed energy must be in mJ
