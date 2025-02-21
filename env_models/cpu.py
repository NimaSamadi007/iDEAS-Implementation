"""
CPU model for iDEAS and RRLO algorithms
"""

import numpy as np
import math
import copy

from typing import List, Dict
from env_models.task import Task
from utils.utils import load_yaml


class CPU:
    def __init__(self, cpu_conf_path):
        """
        Generic CPU model used in iDEAS algorithm

        Args:
            cpu_conf_path (str): path to CPU configuration file
        """
        specs = load_yaml(cpu_conf_path)

        if sorted(specs["freqs"]) != specs["freqs"]:
            raise ValueError("CPU frequencies must be sorted incrementally")
        if sorted(specs["powers"]) != specs["powers"]:
            raise ValueError("CPU powers must be sorted incrementally")

        self.freqs = np.asarray(specs["freqs"])
        self.powers = np.asarray(specs["powers"])
        self.model = specs["model"]
        self.cpu_type = specs["type"]

        self.util = 0
        self.freq = self.freqs[-1]  # Maximum frequency by default

    def __repr__(self):
        return f"'{self.model} {self.cpu_type}' CPU"

    def step(self, tasks: Dict[int, Task], acts: List[List]):
        """
        Execute the assigned tasks based on the given actions

        Args:
            tasks (Dict[int, Task]): tasks to be executed
            acts (List[List]): actions to be taken
        """
        total_util = 0
        for t_id, in_freq in acts:
            aets = 0
            for job in tasks[t_id]:
                # AET can only be set at the time of execution
                job.gen_aet(in_freq)
                # Check if deadline will be missed
                if job.aet > job.p:
                    job.deadline_missed = True
                    job.aet = job.p  # If task is missed, it will be executed until the end of current cycle
                # Calculate energy consumption (chip energy conusmption at given frequency)
                cons_power = self.powers[self.freqs == in_freq][0]
                if job.aet < 0:
                    raise ValueError("AET is negative!")
                job.cons_energy = cons_power * (job.aet / 1000)

                aets += job.aet
            total_util += aets / (job.p * len(tasks[t_id]))
        self.util = total_util

    def get_min_energy(self, task: Task) -> float:
        """
        Minimum energy consumption for the given task
        if this task is the only task in the system
        """
        for i, frq in enumerate(self.freqs):
            # If the task can be executed, find the corresponding
            # power at which the minimum energy is consumed
            wcet_scaled = task.wcet * (task.base_freq / frq)
            if wcet_scaled < task.p:
                min_energy = self.powers[i] * (wcet_scaled / 1000)
                return min_energy


class CPU_CC(CPU):
    def __init__(self, cpu_conf_path):
        """
        CPU model that uses the cycle conserving (CC) algorithm for DVFS
        and EDF scheduling in task execution

        Args:
            cpu_conf_path (str): path to CPU configuration file
        """
        super().__init__(cpu_conf_path)

    def step(self, jobs: Dict[int, List[Task]]) -> List[Task]:
        """
        Execute the assigned jobs using CC and EDF scheduling

        Args:
            jobs (Dict[int, List[Task]]): jobs to be executed for each task
        """
        tasks = {t_id: copy.deepcopy(job[0]) for t_id, job in jobs.items()}
        self.hp = math.lcm(*[task.p for task in tasks.values()])
        self.tasks = tasks

        issue_times = self._cal_task_issue_times()
        curr_jobs = []
        finished_jobs = []
        for i, issued_info in enumerate(issue_times.items()):
            # Issue new tasks and order them based on their priorities (deadline)
            issue_time = issued_info[0]
            issued_tasks_id = issued_info[1]
            issued_jobs = [jobs[t_id].pop(0) for t_id in issued_tasks_id]
            for t in issued_jobs:
                t.deadline = issue_time + t.p
                # Call cycle conserving algorithm to assign CPU frequency
                self.freq = self._task_release(t.t_id)

            curr_jobs.extend(issued_jobs)
            curr_jobs.sort(key=lambda x: x.deadline)

            if i == len(issue_times) - 1:
                # This is the last cycle so we can execute tasks up until to the
                # next hyperperiod
                next_issue_time = self.hp
            else:
                next_issue_time = list(issue_times.keys())[i + 1]
            curr_time = issue_time
            remain_time = next_issue_time - issue_time
            for job in curr_jobs:
                # Execute tasks
                job.gen_aet(self.freq)
                if (job.aet - job.executed_time) <= remain_time:
                    # This job will be executed completely
                    job.exec_time_history.append(
                        [curr_time, curr_time + job.aet - job.executed_time]
                    )
                    job.exec_freq_history.append(job.base_freq)
                    job.finished = True
                    curr_time += job.aet - job.executed_time
                    remain_time -= job.aet - job.executed_time
                    job.executed_time = job.aet
                    self.freq = self._task_complete(job.t_id, job.aet)
                else:
                    # This job must continue to the next step
                    job.exec_time_history.append([curr_time, next_issue_time])
                    job.exec_freq_history.append(job.base_freq)
                    job.executed_time += remain_time
                    break
            # Remove finished tasks
            finished_jobs.extend([t for t in curr_jobs if t.finished])
            curr_jobs = [t for t in curr_jobs if not t.finished]

        # Check if any job is remained:
        for t in curr_jobs:
            t.deadline_missed = True
            for i in range(len(t.exec_time_history)):
                dt = t.exec_time_history[i][1] - t.exec_time_history[i][0]
                power = self.powers[self.freqs == t.exec_freq_history[i]][0]
                if dt < 0:
                    raise ValueError("Execution time is negative!")
                t.cons_energy += power * (dt / 1000)  # mJ

        # Calculate energy consumption
        for t in finished_jobs:
            # Task deadline is missed
            if t.exec_time_history[-1][1] > t.deadline:
                t.deadline_missed = True

            for i in range(len(t.exec_time_history)):
                dt = t.exec_time_history[i][1] - t.exec_time_history[i][0]
                power = self.powers[self.freqs == t.exec_freq_history[i]][0]
                if dt < 0:
                    raise ValueError("Execution time is negative!")
                t.cons_energy += power * (dt / 1000)  # mJ

        finished_jobs.extend(curr_jobs)
        return finished_jobs

    def _task_release(self, task_id: int) -> int:
        self.tasks[task_id].util = self.tasks[task_id].wcet / self.tasks[task_id].p
        return self._select_freq()

    def _task_complete(self, task_id: int, job_aet: float) -> int:
        self.tasks[task_id].util = job_aet / self.tasks[task_id].p
        return self._select_freq()

    def _select_freq(self):
        total_util = sum([j.util for j in self.tasks.values()])
        max_freq = self.freqs[-1]
        for freq in self.freqs:
            if total_util <= freq / max_freq:
                return freq
        return max_freq

    def _cal_task_issue_times(self):
        issue_times = dict()
        for task in self.tasks.values():
            for j in range(self.hp // task.p):
                if j * task.p in issue_times.keys():
                    issue_times[j * task.p].append(task.t_id)
                else:
                    issue_times[j * task.p] = [task.t_id]
        # Sort based on issue time
        sorted_issue_times = dict(sorted(issue_times.items()))
        return sorted_issue_times

    def _cal_total_util(self, tasks):
        total_util = 0
        for task in tasks.values():
            total_util += task.aet / task.p
        return total_util


class CPU_LA(CPU):
    def __init__(self, cpu_conf_path):
        """
        CPU model that uses the look ahead (LA) algorithm for DVFS
        and EDF scheduling in task execution

        Args:
            cpu_conf_path (str): path to CPU configuration file
        """
        super().__init__(cpu_conf_path)

    def step(self, jobs: Dict[int, List[Task]]) -> List[Task]:
        """
        Execute the assigned jobs using CC and EDF scheduling

        Args:
            jobs (Dict[int, List[Task]]): jobs to be executed for each task
        """
        tasks = {t_id: copy.deepcopy(job[0]) for t_id, job in jobs.items()}
        self.hp = math.lcm(*[task.p for task in tasks.values()])
        self.tasks = tasks

        issue_times = self._cal_task_issue_times()
        curr_jobs = []
        finished_jobs = []
        for i, issued_info in enumerate(issue_times.items()):
            issue_time = issued_info[0]
            issued_tasks_id = issued_info[1]
            issued_jobs = [jobs[t_id].pop(0) for t_id in issued_tasks_id]
            for t in issued_jobs:
                t.deadline = issue_time + t.p
            for t in issued_jobs:
                self.freq = self._task_release(
                    tasks=issued_jobs, task_id=t.t_id, curr_time=issue_time
                )

            curr_jobs.extend(issued_jobs)
            curr_jobs.sort(key=lambda x: x.deadline)

            if i == len(issue_times) - 1:
                next_issue_time = self.hp
            else:
                next_issue_time = list(issue_times.keys())[i + 1]
            curr_time = issue_time
            remain_time = next_issue_time - issue_time

            for job in curr_jobs:
                job.gen_aet(self.freq)
                if (job.aet - job.executed_time) < remain_time:
                    job.exec_time_history.append(
                        [curr_time, curr_time + job.aet - job.executed_time]
                    )
                    job.exec_freq_history.append(job.base_freq)
                    job.finished = True
                    curr_time += job.aet - job.executed_time
                    remain_time -= job.aet - job.executed_time
                    self.freq = self._task_complete(
                        tasks=issued_jobs, task_id=job.t_id, curr_time=curr_time
                    )
                else:
                    job.exec_time_history.append([curr_time, next_issue_time])
                    job.exec_freq_history.append(job.base_freq)
                    job.executed_time += remain_time
                    break

            finished_jobs.extend([t for t in curr_jobs if t.finished])
            curr_jobs = [t for t in curr_jobs if not t.finished]

        for t in curr_jobs:
            t.deadline_missed = True
            for i in range(len(t.exec_time_history)):
                dt = t.exec_time_history[i][1] - t.exec_time_history[i][0]
                power = self.powers[self.freqs == t.exec_freq_history[i]][0]
                if dt < 0:
                    raise ValueError("Execution time is negative!")
                t.cons_energy += power * (dt / 1000)  # mJ

        for t in finished_jobs:
            if t.exec_time_history[-1][1] > t.deadline:
                t.deadline_missed = True

            for i in range(len(t.exec_time_history)):
                dt = t.exec_time_history[i][1] - t.exec_time_history[i][0]
                power = self.powers[self.freqs == t.exec_freq_history[i]][0]
                if dt < 0:
                    raise ValueError("Execution time is negative!")
                t.cons_energy += power * (dt / 1000)

        finished_jobs.extend(curr_jobs)
        return finished_jobs

    def _task_release(self, tasks, task_id: int, curr_time) -> int:
        self.tasks[task_id].c_left = self.tasks[task_id].wcet
        return self._defer(tasks=tasks, current_time=curr_time)

    def _task_complete(self, tasks, curr_time, task_id: int) -> int:
        self.tasks[task_id].c_left = 0
        return self._defer(tasks=tasks, current_time=curr_time)

    def _select_freq(self, util):
        max_freq = self.freqs[-1]
        for freq in self.freqs[::-1]:
            if util <= freq / max_freq:
                return freq
        return max_freq

    def _defer(self, tasks, current_time: int) -> int:
        S = 0
        util_tot = self._cal_total_util(tasks)
        tsks = tasks
        tsks.sort(key=lambda x: x.deadline, reverse=True)
        Min_D = tsks[-1].deadline
        for task in tsks:
            util_tot -= task.wcet / task.p
            x = max(0, task.c_left - (1 - util_tot) * (task.deadline - Min_D))
            if task.deadline != Min_D:
                util_tot += (task.c_left - x) / (task.deadline - Min_D)
            S += x
        return self._select_freq(S / (Min_D - current_time))

    def _cal_task_issue_times(self):
        issue_times = dict()
        for task in self.tasks.values():
            for j in range(self.hp // task.p):
                if j * task.p in issue_times.keys():
                    issue_times[j * task.p].append(task.t_id)
                else:
                    issue_times[j * task.p] = [task.t_id]
        sorted_issue_times = dict(sorted(issue_times.items()))
        return sorted_issue_times

    def _cal_total_util(self, tasks):
        total_util = 0
        if isinstance(tasks, dict):
            task_list = tasks.values()
        else:
            task_list = tasks
        for task in task_list:
            total_util += task.aet / task.p
        return total_util
