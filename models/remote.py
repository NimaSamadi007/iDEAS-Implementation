from models.task import Task

class EdgeServer:
    def __init__(self):
        pass

    def execute(self, job: Task):
        job.gen_aet()
        print(f"Executing job {job} at EServer")

