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
        self.base_freq = specs["base_freq"]  # base CPU frequency tasks are represented

        self.aet = -1  # (ms)
        self.util = 0
        self.cons_energy = 0  # consumed energy when executing the task in (J)
        self.deadline_missed = False
        self.c_left = self.wcet
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

    def __repr__(self):
        return f"(P: {self.p}, W: {self.wcet}, A: {self.aet:.3f}, b: {self.b}, energy: {self.cons_energy:.3f})"


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
        # Calculate hyper period of tasks
        time = math.lcm(*[t.p for t in self.task_set])
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


class RandomTaskGen:
    def __init__(self, task_conf_path):
        try:
            with open(task_conf_path, "r") as f:
                task_set_conf = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Task configuration file not found at {task_conf_path}"
            )
        self.num_tasks = task_set_conf["num_tasks"]
        self.p_min, self.p_max = task_set_conf["p"]
        self.w_min, self.w_max = task_set_conf["w"]
        self.b_min, self.b_max = task_set_conf["b"]
        self.base_freq = task_set_conf["base_freq"]

    def step(self, target_cpu_load, max_task_load):
        # Generate base tasks
        self._gen_base_tasks(target_cpu_load, max_task_load)
        time = math.lcm(*[t.p for t in self.task_set])
        gen_task = dict()
        for task in self.task_set:
            gen_task[task.t_id] = self._gen_task(task, time)

        return gen_task

    def _gen_task(self, task, time):
        num_tasks = time // task.p
        return [copy.deepcopy(task) for _ in range(num_tasks)]

    def _gen_base_tasks(self, target_cpu_load, max_task_load):
        single_task_load = target_cpu_load / self.num_tasks
        self.task_set = []
        for t_id in range(self.num_tasks):
            p = np.random.randint(self.p_min//20, self.p_max//20) * 50
            # Generate w based on p and task load while considering w ranges
            w = np.min([self.w_max, np.max([self.w_min, p * single_task_load])])
            b = np.min([self.b_max,np.max([self.b_min,np.random.randint(self.b_min//50, self.b_max//50)*50])])
            self.task_set.append(
                Task(
                    {"task": t_id, "p": p, "w": w, "b": b, "base_freq": self.base_freq}
                )
            )
        tasks_load = np.sum([t.wcet / t.p for t in self.task_set])
        if tasks_load >= max_task_load:
            raise ValueError(
                f"Generated tasks are non-schedulable! Task load: {tasks_load}"
            )

    def get_task_size_bound(self):
        return self.b_min, self.b_max

    def get_wcet_bound(self):
        return self.w_min, self.w_max
    



class NormalTaskGen:
    def __init__(self, task_conf_path):
        try:
            with open(task_conf_path, "r") as f:
                task_set_conf = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Task configuration file not found at {task_conf_path}"
            )
        self.num_tasks = task_set_conf["num_tasks"]
        self.p_min, self.p_max = task_set_conf["p"]
        self.w_min, self.w_max = task_set_conf["w"]
        self.b_min, self.b_max = task_set_conf["b"]
        self.base_freq = task_set_conf["base_freq"]

    def step(self, target_cpu_load,mean, max_task_load):
        # Generate base tasks
        self._gen_base_tasks(target_cpu_load,mean, max_task_load)
        time = math.lcm(*[t.p for t in self.task_set])
        gen_task = dict()
        for task in self.task_set:
            gen_task[task.t_id] = self._gen_task(task, time)

        return gen_task

    def _gen_task(self, task, time):
        num_tasks = time // task.p
        return [copy.deepcopy(task) for _ in range(num_tasks)]

    def _gen_base_tasks(self, target_cpu_load, mean, max_task_load):
        single_task_load = target_cpu_load / self.num_tasks
        std=20/3
        self.task_set = []
        for t_id in range(self.num_tasks):
            p = np.random.randint(self.p_min//20, self.p_max//20) * 50
            # Generate w based on p and task load while considering w ranges
            w = np.min([self.w_max, np.max([self.w_min, p * single_task_load])])
            b = np.min([self.b_max, np.max([self.b_min,np.random.normal(round(mean),std)])])
            if mean<100:
                b=100
            if b>500:
                b=500
            self.task_set.append(
                Task(
                    {"task": t_id, "p": p, "w": w, "b": b, "base_freq": self.base_freq}
                )
            )
        tasks_load = np.sum([t.wcet / t.p for t in self.task_set])
        if tasks_load >= max_task_load:
            raise ValueError(
                f"Generated tasks are non-schedulable! Task load: {tasks_load}"
            )

    def get_task_size_bound(self):
        return self.b_min, self.b_max

    def get_wcet_bound(self):
        return self.w_min, self.w_max
