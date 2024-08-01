from typing import Dict, List
import numpy as np

from env_models.cpu import CPU, CPU_CC, CPU_LA
from env_models.task import TaskGen, Task, RRLOTaskGen
from env_models.wireless_interface import WirelessInterface, RRLOWirelessInterface
from configs import (
    DQN_STATE_DIM,
    LATENCY_ENERGY_COEFF,
    DEADLINE_MISSED_PENALTY,
    RRLO_STATE_DIM,
    REWARD_COEFF,
)


class Env:
    def __init__(self, confs: Dict[str, str]):
        self.task_gen = TaskGen(confs["task_set"])
        self.cpu = CPU(confs["cpu_local"])
        self.w_inter = WirelessInterface(confs["w_inter"])

        # Initialize environment state
        self._init_state_bounds()
        self.curr_tasks = None
        self.curr_state = None

    def step(self, actions: Dict[str, List[int]]):
        self.cpu.step(self.curr_tasks, actions["local"])
        self.w_inter.offload(self.curr_tasks, actions["offload"])

        return self._cal_reward()

    def observe(self):
        self.curr_tasks = self.task_gen.step()
        self.curr_state = self._get_system_state()
        is_final = len(self.curr_tasks) * [True]

        return self.curr_state, is_final

    def get_action_space(self):
        action_space = {"offload": self.w_inter.powers, "local": self.cpu.freqs}
        return action_space

    def _init_state_bounds(self):
        # (SU, U_local, WCET, B, h)
        wcet_bound = self.task_gen.get_wcet_bound()
        task_size_bound = self.task_gen.get_task_size_bound()
        self.min_state_vals = np.array(
            [0, 0, wcet_bound[0], task_size_bound[0]], dtype=float
        )
        self.max_state_vals = np.array(
            [1, 1, wcet_bound[1], task_size_bound[1]], dtype=float
        )

    def _get_system_state(self):
        states = np.zeros((len(self.curr_tasks), DQN_STATE_DIM), dtype=np.float32)
        su = 0.0
        for i, task in enumerate(self.curr_tasks.values()):
            su += task[0].wcet / task[0].p
            states[i, 2] = task[0].wcet
            states[i, 3] = task[0].b
        states[:, 0] = su
        states[:, 1] = self.cpu.util
        # states[:, 4] = self.w_inter.update_channel_state()
        states = (states - self.min_state_vals) / (
            self.max_state_vals - self.min_state_vals
        )
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
                    penalty += job.cons_energy + LATENCY_ENERGY_COEFF * job.aet

            min_penalty = np.min(
                [self.cpu.get_min_energy(task[0]), self.w_inter.get_min_energy(task[0])]
            )
            min_penalties.append(min_penalty / len(task))
            if not is_deadline_missed:
                penalties.append(penalty / len(task))
            else:
                penalties.append(DEADLINE_MISSED_PENALTY)

        # Calculate reward
        min_penalties = np.asarray(min_penalties, dtype=float)
        penalties = np.asarray(penalties, dtype=float)
        rewards = np.exp(-REWARD_COEFF * (penalties - min_penalties))
        return rewards, penalties, min_penalties


class RRLOEnv:
    def __init__(self, confs: Dict[str, str]):
        self.cpu_cc = CPU_CC(confs["cpu_local"])
        self.cpu_la = CPU_LA(confs["cpu_local"])
        # Pass CPU frequency to RRLO task generator
        self.task_gen = RRLOTaskGen(self.cpu_cc.freq, confs["task_set"])
        self.w_inter = RRLOWirelessInterface(confs["w_inter"])

        # Initialize environment state
        self._init_state_bounds()
        self.curr_tasks = None
        self.curr_state = None

    def step(self, actions: Dict[str, int | List[int]]):
        # Execute local tasks
        if not (actions["dvfs_alg"] == 0 or actions["dvfs_alg"] == 1):
            raise ValueError("RRLO only supports CC and LA algorithms")
        local_tasks = {t_id: self.curr_tasks[t_id] for t_id in actions["local"]}
        if actions["dvfs_alg"] == 0:
            exec_local_tasks = self.cpu_cc.step(local_tasks)
        else:
            exec_local_tasks = self.cpu_la.step(local_tasks)
        # Execute offloaded tasks
        power_level = self.w_inter.powers[actions["power_level"]]
        offload_tasks = {t_id: self.curr_tasks[t_id] for t_id in actions["offload"]}
        self.w_inter.offload(offload_tasks, power_level)

        # Add locally execuated tasks to current tasks list as
        # current tasks list is modifier in DVFS algorithms
        for t in exec_local_tasks:
            self.curr_tasks[t.t_id].append(t)
        return self._cal_penalty(exec_local_tasks, offload_tasks)

    def observe(self):
        self.curr_tasks = self.task_gen.step()
        self._gen_aet(self.curr_tasks)
        self.curr_state = self._descretize_states(self._get_system_state())
        is_final = True

        return self.curr_state, is_final

    def get_state_bounds(self):
        return self.num_states

    def _cal_penalty(self, local_tasks, offload_tasks):
        penalty = 0
        for t in local_tasks:
            if t.deadline_missed:
                penalty += DEADLINE_MISSED_PENALTY
            else:
                penalty += t.cons_energy
        for tasks in offload_tasks.values():
            for job in tasks:
                if job.deadline_missed:
                    penalty += DEADLINE_MISSED_PENALTY
                else:
                    penalty += job.cons_energy
        return penalty

    def _init_state_bounds(self):
        self.num_states = np.array([16, 16, 8])
        self.min_state_vals = np.array([0, 0, 0])
        self.max_state_vals = np.array([1, 1, 2 * self.w_inter.cg_sigma])
        self.state_steps = (self.max_state_vals - self.min_state_vals) / (
            self.num_states - 1
        )

    def _gen_aet(self, tasks: Dict[int, List[Task]]):
        # FIXME: This is rather a weird choice as we're not
        # aware of task AET beforehand and it's only measured after
        # a task is executed
        for task in tasks.values():
            for job in task:
                job.gen_aet()

    def _get_system_state(self):
        states = np.zeros(RRLO_STATE_DIM, dtype=np.float32)
        h_p = self.task_gen.get_hyper_period()
        eth_p = 0.0
        ds_deonm = 0.0
        su = 0.0
        for task in self.curr_tasks.values():
            for job in task:
                eth_p += job.aet
            ds_deonm += h_p / task[0].p * task[0].wcet
            su += task[0].wcet / task[0].p

        ds = 1 - (eth_p / ds_deonm)
        states[0] = su
        states[1] = ds
        states[2] = self.w_inter.update_channel_state()
        return states

    def _descretize_states(self, states: np.ndarray):
        return np.floor((states - self.min_state_vals) / self.state_steps).astype(int)
