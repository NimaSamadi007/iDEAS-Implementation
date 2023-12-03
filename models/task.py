import math
import copy
import numpy as np

class Task:
    def __init__(self, specs):
        self.p = specs['p'] # period frequency
        self.b = specs['b'] # task input data (bits)
        self.wcet = specs['w'] # worst-case execution time
        self.t_id = specs['task'] # task ID representing a unique task
        self.aet = 0
        self.exec_device = None

    def gen_aet(self):
        self.aet = np.random.uniform(self.wcet/2, self.wcet)

    def __repr__(self):
        return f"{self.t_id}: ({self.p}, {self.b}, {self.wcet}, {self.aet})"

class TaskGen:
    def __init__(self):
        pass

    def _set_hyper_period(self):
        task_periods = [task.p for task in self.task_set]
        self.hyper_period = math.lcm(*task_periods)

    def generate(self, task_set):
        self.task_set = task_set
        self._set_hyper_period()
        tasks = {}
        for ts in task_set:
            num_tasks = self.hyper_period // ts.p
            tasks_i = []
            for _ in range(num_tasks):
                tasks_i.append(copy.deepcopy(ts))
                tasks_i[-1].gen_aet()
            tasks[ts.t_id] = tasks_i

        return tasks
