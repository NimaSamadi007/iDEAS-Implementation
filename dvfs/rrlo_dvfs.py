"""
RRLO DVFS algorithm implementation
"""

import numpy as np
import os


class RRLO_DVFS:
    def __init__(
        self,
        state_bounds,
        num_w_inter_powers,
        num_dvfs_algs,
        dvfs_algs,
        num_tasks,
        eps_decay=1.0 / 2000,
        max_eps=1.0,
        min_eps=0.1,
        eps_update_step=1e3,
    ):
        """
        Initialize the RRLO DVFS algorithm

        Args:
            state_bounds (np.ndarray): state space bounds (min and max values)
            num_w_inter_powers (int): number of possible wireless interface power levels
            num_dvfs_algs (int): number of possible (traditional) DVFS algorithms
            dvfs_algs (List[str]): list of DVFS algorithm names
            num_tasks (int): number of tasks
            eps_decay (float): epsilon decay rate
            max_eps (float): maximum epsilon value
            min_eps (float): minimum epsilon value
            eps_update_step (int): epsilon update step
        """
        self.state_bounds = state_bounds
        self.num_total_states = self.state_bounds.prod()
        self.num_dvfs_algs = num_dvfs_algs
        self.num_w_inter_powers = num_w_inter_powers
        self.dvfs_algs = dvfs_algs
        self.num_tasks = num_tasks
        self.eps = max_eps
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.eps_decay = eps_decay
        self.eps_update_step = eps_update_step
        self.act_bounds = (
            np.ones(self.num_tasks + 2, dtype=int) * 2
        )  # Initialize all with value 2
        self.act_bounds[-2] = self.num_w_inter_powers  # number of possible power levels
        self.act_bounds[-1] = self.num_dvfs_algs  # number of possible dvfs algs
        self.alpha = 0.8
        self.beta = 0.9
        self.Q_table_a = np.zeros(
            (
                self.num_total_states,
                2**self.num_tasks * self.num_dvfs_algs * self.num_w_inter_powers,
            )
        )
        self.Q_table_b = np.zeros_like(self.Q_table_a)
        self.eps_update_cnt = 0

    def execute(self, state: np.ndarray, eval_mode=False) -> np.ndarray:
        if not eval_mode and self.eps > np.random.random():
            # Select random action for exploration
            row_idx = np.random.randint(self.num_total_states)
        else:
            row_idx = self._conv_state_to_row(state)
        act_a = np.argmin(self.Q_table_a[row_idx, :])
        act_b = np.argmin(self.Q_table_b[row_idx, :])
        if self.Q_table_a[row_idx, act_a] < self.Q_table_b[row_idx, act_b]:
            return self._conv_col_to_act(act_a), act_a
        else:
            return self._conv_col_to_act(act_b), act_b

    def update(self, state, actions, penalty, next_state):
        # Update one of the Q-tables
        if np.random.random() < 0.5:
            Q_table = self.Q_table_a
        else:
            Q_table = self.Q_table_b
        state_row_idx = self._conv_state_to_row(state)
        next_state_row_idx = self._conv_state_to_row(next_state)
        next_action = np.argmin(Q_table[next_state_row_idx, :])
        Q_table[state_row_idx, actions] += self.alpha * (
            penalty
            + self.beta * Q_table[next_state_row_idx, next_action]
            - Q_table[state_row_idx, actions]
        )
        # Update epsilon
        self.eps_update_cnt += 1
        if self.eps_update_cnt % self.eps_update_step == 0:
            self.eps_update_cnt = 0
            self.eps = max(
                self.min_eps, self.eps - self.eps_decay * (self.max_eps - self.min_eps)
            )

    def save_model(self, path: str):
        """
        Save the trained models to path
        """
        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/rrlo_Q_table_a.npy", self.Q_table_a)
        np.save(f"{path}/rrlo_Q_table_b.npy", self.Q_table_b)

    def load_model(self, path: str):
        """
        Load pretrained model weights from the path
        """
        if os.path.exists(f"{path}/rrlo_Q_table_a.npy") and os.path.exists(
            f"{path}/rrlo_Q_table_b.npy"
        ):
            self.Q_table_a = np.load(f"{path}/rrlo_Q_table_a.npy")
            self.Q_table_b = np.load(f"{path}/rrlo_Q_table_b.npy")
        else:
            print("Model weights do not exist")

    def _conv_state_to_row(self, state: np.ndarray):
        row = 0
        for i in range(len(state) - 1):
            row += state[i] * self.state_bounds[i + 1 :].prod()
        row += state[-1]
        return row

    def _conv_col_to_act(self, act_idx: int) -> np.ndarray:
        """
        Converts the action index to the action dictionary
        based on number of power levels, DVFS algorithms, and target devices
        """
        local = []
        offload = []
        for i in range(len(self.act_bounds) - 1):
            multiple = self.act_bounds[i + 1 :].prod()
            q = act_idx // multiple
            act_idx = act_idx % multiple
            if i == len(self.act_bounds) - 2:  # power level
                power_level = q
            else:
                if q == 0:  # Execute locally
                    local.append(i)
                else:
                    offload.append(i)
        act = {
            "local": local,
            "offload": offload,
            "power_level": power_level,
            "dvfs_alg": act_idx,
        }
        return act
