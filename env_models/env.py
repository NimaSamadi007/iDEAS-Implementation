"""
Environment models used in iDEAS and RRLO algorithms
"""

from typing import Dict, List
import numpy as np
import math

from env_models.cpu import CPU, CPU_CC, CPU_LA
from env_models.task import Task
from env_models.wireless_interface import WirelessInterface, RRLOWirelessInterface


class HetrogenEnv:
    def __init__(self, confs, cpu_load_bound, wcet_bound, task_size_bound, cn_bound):
        """
        Heterogeneous environment with two CPUs (big and little) and
        edge server for offloading tasks.

        Args:
            confs (Dict[str, str]): Environment configurations containing CPU and wireless interface configs
            cpu_load_bound (List[int]): CPU load bounds (min and max)
            wcet_bound (List[int]): WCET bounds (min and max)
            task_size_bound (List[int]): Task size bounds (min and max)
            cn_bound (List[int]): CN power bounds (min and max)

        Note:
            Bounds are used for state normalization
        """
        self.cpu_little = CPU(confs["cpus"]["little"])
        self.cpu_big = CPU(confs["cpus"]["big"])
        self.w_inter = WirelessInterface(confs["w_inter"])
        self.w_inter.set_cn_power_bounds(*cn_bound)

        self.latency_energy_coeff = confs["params"]["latency_energy_coeff"]
        self.deadline_missed_penalty = confs["params"]["deadline_missed_penalty"]
        self.reward_coeff = confs["params"]["reward_coeff"]

        # Initialize environment state
        self._init_state_bounds(
            cpu_load_bound, wcet_bound, task_size_bound, self.w_inter.get_rate_bounds()
        )
        self.state_dim = confs["params"]["dqn_state_dim"]
        self.curr_tasks = None
        self.curr_state = None

    def step(self, actions):
        if len(actions["little"]):
            self.cpu_little.step(self.curr_tasks, actions["little"])
        if len(actions["big"]):
            self.cpu_big.step(self.curr_tasks, actions["big"])
        if len(actions["offload"]):
            self.w_inter.offload(self.curr_tasks, actions["offload"])

        return self._cal_reward()

    def observe(self, tasks):
        self.curr_tasks = tasks
        self.curr_state = self._get_system_state()
        is_final = len(self.curr_tasks) * [False]

        return self.curr_state, is_final

    def get_action_space(self):
        action_space = {
            "offload": self.w_inter.powers,
            "little": self.cpu_little.freqs,
            "big": self.cpu_big.freqs,
        }
        return action_space

    def _init_state_bounds(
        self, cpu_load_bound, wcet_bound, task_size_bound, chan_rate_bound
    ):
        # (SU, U_little, U_big, WCET, B, chan_rate)
        self.min_state_vals = np.array(
            [
                cpu_load_bound[0],
                cpu_load_bound[0],
                cpu_load_bound[0],
                wcet_bound[0],
                task_size_bound[0],
                chan_rate_bound[0],
            ],
            dtype=float,
        )
        self.max_state_vals = np.array(
            [
                cpu_load_bound[1],
                cpu_load_bound[1],
                cpu_load_bound[1],
                wcet_bound[1],
                task_size_bound[1],
                chan_rate_bound[1],
            ],
            dtype=float,
        )

    def _get_system_state(self):
        # Update channel status when we are observing environment
        self.w_inter.update_channel_state()
        channel_rate = self.w_inter.get_channel_rate()

        states = np.zeros((len(self.curr_tasks), self.state_dim), dtype=np.float32)
        su = 0.0
        for i, task in enumerate(self.curr_tasks.values()):
            su += task[0].wcet / task[0].p
            states[i, 3] = task[0].wcet
            states[i, 4] = task[0].b
            states[i, 5] = channel_rate
        states[:, 0] = su
        states[:, 1] = self.cpu_little.util
        states[:, 2] = self.cpu_big.util

        # Clip the state values
        for i in range(states.shape[1]):
            states[:, i] = np.clip(
                states[:, i],
                np.full(states[:, i].shape, self.min_state_vals[i]),
                np.full(states[:, i].shape, self.max_state_vals[i]),
            )

        states = (states - self.min_state_vals) / (
            self.max_state_vals - self.min_state_vals
        )

        return states

    def _cal_reward(self):
        penalties = []
        for task in self.curr_tasks.values():
            penalty = 0
            for job in task:
                # Calculate last execution penalty
                if job.deadline_missed:
                    penalty += self.deadline_missed_penalty
                else:
                    penalty += job.cons_energy + self.latency_energy_coeff * job.aet

            if penalty / len(task) > self.deadline_missed_penalty:
                raise ValueError(f"Penalty {penalty / len(task)} excceds {self.deadline_missed_penalty}")
            penalties.append(penalty / len(task))

        # Calculate reward
        penalties = np.asarray(penalties, dtype=float)
        rewards = np.exp(-self.reward_coeff * (penalties))
        return rewards, penalties, None


class HomogenEnv:
    def __init__(
        self,
        confs: Dict[str, str],
        cpu_load_bound,
        wcet_bound,
        task_size_bound,
        cn_bound,
    ):
        """
        Homogeneous environment with a single CPU and edge server for offloading tasks.

        Args:
            confs (Dict[str, str]): Environment configurations containing CPU and wireless interface configs
            cpu_load_bound (List[int]): CPU load bounds (min and max)
            wcet_bound (List[int]): WCET bounds (min and max)
            task_size_bound (List[int]): Task size bounds (min and max)
            cn_bound (List[int]): CN power bounds (min and max)

        Note:
            Bounds are used for state normalization
        """
        self.cpu = CPU(confs["cpus"]["local"])
        self.w_inter = WirelessInterface(confs["w_inter"])
        self.w_inter.set_cn_power_bounds(*cn_bound)

        self.latency_energy_coeff = confs["params"]["latency_energy_coeff"]
        self.deadline_missed_penalty = confs["params"]["deadline_missed_penalty"]
        self.reward_coeff = confs["params"]["reward_coeff"]

        # Initialize environment state
        self.state_dim = confs["params"]["dqn_state_dim"]
        self._init_state_bounds(
            cpu_load_bound, wcet_bound, task_size_bound, self.w_inter.get_rate_bounds()
        )
        self.curr_tasks = None
        self.curr_state = None

    def step(self, actions: Dict[str, List[int]]):
        if len(actions["local"]):
            self.cpu.step(self.curr_tasks, actions["local"])
        if len(actions["offload"]):
            self.w_inter.offload(self.curr_tasks, actions["offload"])

        return self._cal_reward()

    def observe(self, tasks):
        self.curr_tasks = tasks
        self.curr_state = self._get_system_state()
        is_final = len(self.curr_tasks) * [False]

        return self.curr_state, is_final

    def get_action_space(self):
        action_space = {"offload": self.w_inter.powers, "local": self.cpu.freqs}
        return action_space

    def _init_state_bounds(
        self, cpu_load_bound, wcet_bound, task_size_bound, chan_rate_bound
    ):
        # (SU, U_local, WCET, B, r)
        self.min_state_vals = np.array(
            [
                cpu_load_bound[0],
                cpu_load_bound[0],
                wcet_bound[0],
                task_size_bound[0],
                chan_rate_bound[0],
            ],
            dtype=float,
        )
        self.max_state_vals = np.array(
            [
                cpu_load_bound[1],
                cpu_load_bound[1],
                wcet_bound[1],
                task_size_bound[1],
                chan_rate_bound[1],
            ],
            dtype=float,
        )

    def _get_system_state(self):
        # Update channel status when we are observing environment
        self.w_inter.update_channel_state()
        channel_rate = self.w_inter.get_channel_rate()

        # States: (SU, U_local, WCET, B, r)
        states = np.zeros((len(self.curr_tasks), self.state_dim), dtype=np.float32)
        su = 0.0
        for i, task in enumerate(self.curr_tasks.values()):
            su += task[0].wcet / task[0].p
            states[i, 2] = task[0].wcet
            states[i, 3] = task[0].b
            states[i, 4] = channel_rate
        states[:, 0] = su
        states[:, 1] = self.cpu.util

        # Clip the state values
        for i in range(states.shape[1]):
            states[:, i] = np.clip(
                states[:, i],
                np.full(states[:, i].shape, self.min_state_vals[i]),
                np.full(states[:, i].shape, self.max_state_vals[i]),
            )

        states = (states - self.min_state_vals) / (
            self.max_state_vals - self.min_state_vals
        )
        return states

    def _cal_reward(self):
        penalties = []
        for task in self.curr_tasks.values():
            penalty = 0
            for job in task:
                # Calculate last execution penalty
                if job.deadline_missed:
                    penalty += self.deadline_missed_penalty
                else:
                    penalty += job.cons_energy + self.latency_energy_coeff * job.aet

            if penalty / len(task) > self.deadline_missed_penalty:
                raise ValueError(f"Penalty {penalty / len(task)} excceds {self.deadline_missed_penalty}")
            penalties.append(penalty / len(task))

        # Calculate reward
        penalties = np.asarray(penalties, dtype=float)
        rewards = np.exp(-self.reward_coeff * (penalties))
        return rewards, penalties, None


class RRLOEnv:
    def __init__(self, confs: Dict[str, str]):
        """
        RRLO environment with a single CPU and edge server for offloading tasks. CPUs
        use CC and LA algorithms for DVFS.

        Args:
            confs (Dict[str, str]): Environment configurations containing CPU and wireless interface configs
        """
        self.cpu_cc = CPU_CC(confs["cpus"]["local"])
        self.cpu_la = CPU_LA(confs["cpus"]["local"])

        self.deadline_missed_penalty = confs["params"]["deadline_missed_penalty"]
        self.rrlo_state_dim = confs["params"]["rrlo_state_dim"]

        # Pass CPU frequency to RRLO task generator
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

    def observe(self, tasks):
        self.curr_tasks = tasks
        self._gen_aet(self.curr_tasks)
        self.curr_state = self._descretize_states(self._get_system_state())
        is_final = False

        return self.curr_state, is_final

    def get_state_bounds(self):
        return self.num_states

    def _cal_penalty(self, local_tasks, offload_tasks):
        penalty = 0
        for t in local_tasks:
            if t.deadline_missed:
                penalty += self.deadline_missed_penalty
            else:
                penalty += t.cons_energy
        for tasks in offload_tasks.values():
            for job in tasks:
                if job.deadline_missed:
                    penalty += self.deadline_missed_penalty
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
        for task in tasks.values():
            for job in task:
                job.gen_aet()

    def _get_system_state(self):
        states = np.zeros(self.rrlo_state_dim, dtype=np.float32)
        h_p = math.lcm(*[t[0].p for t in self.curr_tasks.values()])
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

        # Clip the state values
        for i in range(states.shape[0]):
            states[i] = np.clip(
                states[i], self.min_state_vals[i], self.max_state_vals[i]
            )

        return states

    def _descretize_states(self, states: np.ndarray):
        return np.floor((states - self.min_state_vals) / self.state_steps).astype(int)
