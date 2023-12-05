from typing import List
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
    def execute(self, jobs: List[Task], in_freq: int):
        print(f"Executing CPU {self.cpu_type} task {jobs[0]}")
        print("-------------")
        # AET can only be set at the time of execution
        for job in jobs:
            job.gen_aet()
            print(f"Executing job {job}")
