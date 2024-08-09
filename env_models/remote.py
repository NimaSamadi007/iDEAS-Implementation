from env_models.task import Task


class EdgeServer:
    def __init__(self):
        pass

    def execute(self, job: Task):
        #TODO: Edge server CPU has a 3200 MHz CPU frequency
        job.gen_aet(3200)
