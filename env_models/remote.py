from env_models.task import Task


class EdgeServer:
    def __init__(self):
        pass

    def execute(self, job: Task):
        job.gen_aet()
        # TODO: Update execution model - Now I consider
        # Edge server has twice processing power of mobile
        # devices
        job.aet /= 2.0
