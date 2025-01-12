from env_models.task import Task


class EdgeServer:
    def __init__(self):
        pass

    def execute(self, job: Task):
        # Edge server CPU has a 2800 MHz CPU frequency
        job.gen_aet(2800)
