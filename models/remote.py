from models.task import Task

class EdgeServer:
    def __init__(self):
        pass

    def execute(self, job: Task):
        job.gen_aet()

# Used in TOMS
class Cloud:
    def __init__(self, conf):
        self.freq = conf["freq"]
        self.power_active = conf["power_active"]
        self.power_idle = conf["power_idle"]