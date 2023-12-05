from typing import List
from models.remote import EdgeServer
from models.task import Task

class WirelessInterface:
    def __init__(self, powers):
        self.powers = powers
        self.power = powers[0] # Current power level
        self.e_server = EdgeServer() # Edge server instance

    def offload(self, jobs: List[Task], in_power: float):
        print(f"Offloading task {jobs[0]}")
        print("-------------")
        for job in jobs:
            self.e_server.execute(job)

    # No. of possible power values
    def __len__(self):
        return len(self.powers)

