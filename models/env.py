from typing import Dict, List, Any
import numpy as np

from models.cpu import CPU, TOMSCPU
from models.task import TaskGen, TOMSTaskGen
from models.wireless_interface import WirelessInterface, TOMSWirelessInterface
from configs import *

class Env:
    def __init__(self, confs: Dict[str, str]):
        self.task_gen = TaskGen(confs['task_set'])
        self.cpu_little = CPU(confs['cpu_little'])
        self.cpu_big = CPU(confs['cpu_big'])
        self.w_inter = WirelessInterface(confs['w_inter'])

        # (SU, U_big, U_little, WCET, B, h)
        wcet_bound = self.task_gen.get_wcet_bound()
        task_size_bound = self.task_gen.get_task_size_bound()
        self.min_state_vals = np.array([0, 0, 0, wcet_bound[0], task_size_bound[0], 0], dtype=float)
        self.max_state_vals = np.array([1, 1, 1, wcet_bound[1], task_size_bound[1], 2*self.w_inter.cg_sigma], dtype=float)

        # Initialize environment state
        self.curr_tasks = None
        self.curr_state = None

    def step(self, actions: Dict[str, List[int]]):
        self.cpu_little.step(self.curr_tasks, actions["little"])
        self.cpu_big.step(self.curr_tasks, actions["big"])
        self.w_inter.offload(self.curr_tasks, actions["offload"])

        return self._cal_reward()

    def observe(self, time: float):
        self.curr_tasks = self.task_gen.step(time)
        self.curr_state = self._get_system_state()
        is_final = len(self.curr_tasks)*[False]

        return self.curr_state, is_final

    def get_action_space(self):
        action_space = {"offload": self.w_inter.powers,
                        "big": self.cpu_big.freqs,
                        "little": self.cpu_little.freqs}
        return action_space

    def _get_system_state(self):
        states = np.zeros((len(self.curr_tasks), STATE_DIM), dtype=np.float32)
        su = 0.
        for i, task in enumerate(self.curr_tasks.values()):
            su += task[0].wcet/task[0].p
            states[i, 3] = task[0].wcet
            states[i, 4] = task[0].b
        states[:, 0] = su
        states[:, 1] = self.cpu_big.util
        states[:, 2] = self.cpu_little.util
        states[:, 5] = self.w_inter.update_channel_state()
        states = (states - self.min_state_vals)/(self.max_state_vals - self.min_state_vals)
        return states

    def _cal_reward(self):
        penalties = []
        min_penalties = []
        for task in self.curr_tasks.values():
            penalty = 0
            min_penalty = 0
            is_deadline_missed = False
            for job in task:
                # Calculate last execution penalty
                if job.deadline_missed:
                    is_deadline_missed = True
                    break
                else:
                    penalty += (job.cons_energy+LATENCY_ENERGY_COEFF*job.aet)

            min_penalty = np.min([self.cpu_big.get_min_energy(task[0]),
                                  self.cpu_little.get_min_energy(task[0]),
                                  self.w_inter.get_min_energy(task[0])])
            min_penalties.append(min_penalty/len(task))
            if not is_deadline_missed:
                penalties.append(penalty/len(task))
            else:
                penalties.append(DEADLINE_MISSED_PENALTY)

        # Calculate reward
        min_penalties = np.asarray(min_penalties, dtype=float)
        penalties = np.asarray(penalties, dtype=float)
        rewards = np.exp(-REWARD_COEFF*(penalties-min_penalties))
        return rewards, penalties, min_penalties

class TOMSEnv:
    def __init__(self, confs: Dict[str, str]):
        self.task_gen = TOMSTaskGen(confs['task_set'])
        self.cpu = TOMSCPU(confs['cpu_conf'])
        self.w_inter = TOMSWirelessInterface(confs['w_inter'])

        # (SU, U_cpu, WCET, Data)
        wcet_bound = self.task_gen.get_wcet_bound()
        task_size_bound = self.task_gen.get_task_size_bound()
        self.min_state_vals = np.array([0, 0, 0, wcet_bound[0], task_size_bound[0]], dtype=float)
        self.max_state_vals = np.array([1, 1, 1, wcet_bound[1], task_size_bound[1]], dtype=float)

        # Initialize environment state
        self.curr_tasks = None
        self.curr_state = None

    def observe(self, time: float):
        self.curr_tasks = self.task_gen.step(time)
        self.curr_state = self._get_system_state()
        is_final = len(self.curr_tasks)*[False]

        return self.curr_state, is_final
