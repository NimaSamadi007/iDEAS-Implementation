import math
import copy

class Task:
    def __init__(self, specs):
        self.p = specs['p'] # period frequency
        self.b = specs['b'] # task input data (bits)
        self.wcet = specs['w'] # worst-case execution time
        self.t_id = specs['task'] # task ID representing a unique task
        self.aet = 0
        self.exec_device = None

    def __repr__(self):
        return f"Task {self.t_id}: ({self.p}, {self.b}, {self.wcet})"

class TaskGen:
    def __init__(self):
        pass

    def _set_hyper_period(self):
        task_periods = [task.p for task in self.task_set]
        self.hyper_period = math.lcm(*task_periods)

    def generate(self, task_set):
        self.task_set = task_set
        self._set_hyper_period()
        tasks = {"little": [], "big": [], "remote": []}
        for ts in task_set:
            try:
                num_tasks = self.hyper_period // ts.p
                tasks[ts.exec_device].extend([copy.deepcopy(ts) for _ in range(num_tasks)])
            except KeyError:
                raise KeyError(f"Unable to find {ts.exec_device}, try again!")
        return tasks
