import json
import copy
import numpy as np
from typing import Dict, Any

class Task:
    def __init__(self, specs: Dict[str, Any]):
        self.p = specs['p'] # period time, (ms)
        self.b = specs['b'] # task input data (KB)
        self.wcet = specs['w'] # worst-case execution time, (ms)
        self.t_id = specs['task'] # task ID representing a unique task
        self.aet = -1 #(ms)
        self.cons_energy = -1 # consumed energy when executing the task in (J)
        self.deadline_missed = False

    def gen_aet(self):
        self.aet = np.random.uniform(self.wcet/2, self.wcet)

    def __repr__(self):
        return f"(P: {self.p}, W: {self.wcet}, A: {self.aet:.3f}, b: {self.b}, energy: {self.cons_energy:.3f})"

class TaskGen:
    def __init__(self, task_conf_path):
        self.time = 0
        self.task_set = []
        try:
            with open(task_conf_path, "r") as f:
                task_set_conf = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Task configuration file not found at {task_conf_path}")
        for i in range(len(task_set_conf)):
            self.task_set.append(Task(task_set_conf[i]))

    def step(self, time):
        t_s = self.time
        t_e = self.time + time
        gen_task = dict()
        for task in self.task_set:
            gen_task[task.t_id] = self._gen_task(task, t_s, t_e)
        self.time = t_e

        return gen_task

    def get_task_size_bound(self):
        task_sizes = [task.b for task in self.task_set]
        return min(task_sizes), max(task_sizes)

    def get_wcet_bound(self):
        task_wcets = [task.wcet for task in self.task_set]
        return min(task_wcets), max(task_wcets)

    def _gen_task(self, task, t_s, t_e):
        num_tasks = self._get_num_overlapping_tasks(task.p, t_s, t_e)
        return [copy.deepcopy(task) for _ in range(num_tasks)]

    def _get_num_overlapping_tasks(self, p, t_s, t_e):
        if t_s > t_e:
            raise ValueError("Start time must be less than end time")
        if p > t_e: # No task can overlap
            return 0

        q_s = t_s // p # Start quotient
        q_e = t_e // p # End quotient
        if t_s % p != 0:
            q_s += 1
        if t_e % p == 0:
            q_e -= 1
        return q_e-q_s+1 # No. tasks
