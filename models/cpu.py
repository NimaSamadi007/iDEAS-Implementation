
class CPU:
    def __init__(self, specs):
        self.freqs = specs['freqs']
        self.powers = specs['powers']
        self.model = specs['model']
        self.ncore = specs['num_cores']
        self.cpu_type = specs['type']
        self.util = 0
        self.dynamic_slack = 0

    def __repr__(self):
        return f"'{self.model} {self.cpu_type}' CPU with {self.ncore} cores"

    # Assign tasks to execute
    def execute(self, tasks):
        pass
