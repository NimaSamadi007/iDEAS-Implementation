import math
import copy
import numpy as np

class Task:
    def __init__(self, specs):
        self.p = specs['p'] # period time, (ms)
        self.b = specs['b'] # task input data (KB)
        self.wcet = specs['w'] # worst-case execution time, (ms)
        self.t_id = specs['task'] # task ID representing a unique task
        self.aet = 0 #(s)
        self.cons_energy = 0. # consumed energy when executing the task in (J)
        self.deadline_missed = False

    def gen_aet(self):
        self.aet = np.random.uniform(self.wcet/2, self.wcet)

    def __repr__(self):
        return f"{self.t_id}: ({self.p}, {self.b}, {self.wcet}, {self.aet:.3f}, {self.cons_energy:.3f})"

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
            tasks[ts.t_id] = tasks_i

        return tasks
