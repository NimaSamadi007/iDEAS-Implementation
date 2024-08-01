import json
import copy
import numpy as np
import math
from typing import Dict, Any


class Task:
    def __init__(self, specs: Dict[str, Any]):
        self.p = specs["p"]  # period time, (ms)
        self.b = specs["b"]  # task input data (KB)
        self.wcet = specs["w"]  # worst-case execution time, (ms)
        self.t_id = specs["task"]  # task ID representing a unique task
        self.aet = -1  # (ms)
        self.util = 0
        self.cons_energy = 0  # consumed energy when executing the task in (J)
        self.deadline_missed = False
        self.c_left = self.wcet

    def gen_aet(self):
        self.aet = np.random.uniform(self.wcet / 2, self.wcet)

    def __repr__(self):
        return f"(P: {self.p}, W: {self.wcet}, A: {self.aet:.3f}, b: {self.b}, energy: {self.cons_energy:.3f})"


class RRLOTask(Task):
    def __init__(self, specs: Dict[str, Any]):
        super().__init__(specs)
        self.base_freq = specs["cpu_freq"]

        self.executed_time = 0
        self.finished = False
        self.deadline = -1
        # Buffer holding the execution time and frequency of the task
        # so that the energy consumption can be calculated
        self.exec_time_history = []
        self.exec_freq_history = []

    def gen_aet(self, curr_freq=-1):
        if self.aet == -1:  # AET has not generated before
            self.aet = np.random.uniform(self.wcet / 2, self.wcet)
        # Scale AET and executed time based on the current CPU freq
        if curr_freq != -1:  # No scaling is required if curr_freq is not provided
            self.aet *= self.base_freq / curr_freq
            self.executed_time *= self.base_freq / curr_freq
            self.base_freq = curr_freq


class TaskGen:
    def __init__(self, task_conf_path):
        self.task_set = []
        try:
            with open(task_conf_path, "r") as f:
                task_set_conf = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Task configuration file not found at {task_conf_path}"
            )
        for i in range(len(task_set_conf)):
            self.task_set.append(Task(task_set_conf[i]))

    def step(self):
        time = self.get_hyper_period()
        gen_task = dict()
        for task in self.task_set:
            gen_task[task.t_id] = self._gen_task(task, time)

        return gen_task

    def get_task_size_bound(self):
        task_sizes = [task.b for task in self.task_set]
        return min(task_sizes), max(task_sizes)

    def get_wcet_bound(self):
        task_wcets = [task.wcet for task in self.task_set]
        return min(task_wcets), max(task_wcets)

    def _gen_task(self, task, time):
        num_tasks = time // task.p
        return [copy.deepcopy(task) for _ in range(num_tasks)]

    def get_hyper_period(self):
        return math.lcm(*[t.p for t in self.task_set])


class RRLOTaskGen(TaskGen):
    def __init__(self, cpu_freq, task_conf_path):
        self.task_set = []
        try:
            with open(task_conf_path, "r") as f:
                task_set_conf = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Task configuration file not found at {task_conf_path}"
            )
        for i in range(len(task_set_conf)):
            task_set_conf[i]["cpu_freq"] = cpu_freq
            self.task_set.append(RRLOTask(task_set_conf[i]))
