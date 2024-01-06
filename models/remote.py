import numpy as np
import collections
from typing import List
from models.task import Task, TOMSTask

class EdgeServer:
    def __init__(self):
        pass

    def execute(self, job: Task):
        job.gen_aet()

class TaskBuffer:
    def __init__(self, size):
        self.buf = collections.deque(maxlen=size)

    def add(self, task: TOMSTask):
        if task in self.buf:
            self.buf.remove(task)
        self.buf.append(task)

    def exists(self, task: TOMSTask):
        return task in self.buf

# Used in TOMS
class Cloud:
    def __init__(self, conf):
        self.freq = conf["freq"]
        self.power_active = conf["power_active"]
        self.power_idle = conf["power_idle"]
        self.task_buffer = TaskBuffer(32)

    def is_executed_before(self, task: TOMSTask):
        return self.task_buffer.exists(task)

    def execute(self, task: TOMSTask):
        self.task_buffer.add(task)
        pass
