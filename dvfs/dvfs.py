"""
DQN is implemented based on
https://github.com/Curt-Park/rainbow-is-all-you-need
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(Network, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
        self.net.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ReplayBuffer:
    def __init__(self, state_dim: int, size: int, batch_size: int):
        self.state_buf = np.zeros([size, state_dim], dtype=np.float32)
        self.next_state_buf = np.zeros_like(self.state_buf)
        self.actions_buf = np.zeros(size, dtype=np.float32)
        self.rewards_buf = np.zeros_like(self.actions_buf)
        self.final_action_buf = np.zeros(size, dtype=bool)
        self.batch_size = batch_size
        self.max_size = size
        self.end, self.size = 0, 0

    def __len__(self) -> int:
        return self.size

    def store(self,
              obsv: np.ndarray,
              action: np.ndarray,
              reward: float,
              n_obsv: np.ndarray,
              is_final: bool):

        self.state_buf[self.end] = obsv
        self.next_state_buf[self.end] = n_obsv
        self.actions_buf[self.end] = action
        self.rewards_buf[self.end] = reward
        self.final_action_buf[self.end] = is_final
        # Increase end element pointer and current size of the buffer
        self.end = (self.end+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample(self) -> Dict[str, np.ndarray]:
        # Uniform sampling
        indices = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(state=self.state_buf[indices],
                    n_state=self.next_state_buf[indices],
                    action=self.actions_buf[indices],
                    reward=self.rewards_buf[indices],
                    final=self.final_action_buf[indices])

class DQN_DVFS:
    def __init__(self,
                 state_dim: int,
                 act_space: Dict[str, List],
                 mem_size: int = 1000,
                 batch_size: int = 16,
                 update_target_net: int = 100,
                 eps_decay: float = 1.0/2000,
                 max_eps: float = 1.0,
                 min_eps: float = 0.1,
                 gamma: float = 0.9,
                 weight_decay: float = 1e-5):

        # Parameters
        self.state_dim = state_dim

        self.act_space = act_space
        self.act_type_boundry = []
        self.target_hard_name = list(self.act_space.keys())
        self.act_length = [len(arr) for _, arr in act_space.items()]
        for i in range(len(self.act_length)):
            self.act_type_boundry.append(sum(self.act_length[:i+1]))
        self.act_type_boundry.insert(0, 0)
        self.act_dim = self.act_type_boundry[-1]

        self.repl_buf = ReplayBuffer(state_dim, mem_size, batch_size)
        self.batch_size = batch_size
        self.eps = max_eps
        self.eps_decay = eps_decay
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.update_target_net = update_target_net
        self.gamma = gamma
        self.update_cnt = 1
        self.losses = []

        # Training device
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        print(f"Using {self.device} device for training")

        # Network initialization
        self.net = Network(state_dim, self.act_dim).to(self.device)
        self.target_net = Network(state_dim, self.act_dim).to(self.device)
        # Set target net from current net
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.net.parameters(), weight_decay=weight_decay)

    def train(self, states: np.ndarray,
                    actions: np.ndarray,
                    rewards: np.ndarray,
                    next_states: np.ndarray,
                    are_final: List[bool]) -> float:
        # Store state, action, and rewards to buffer
        for state, action, reward, next_state, is_final in zip(states, actions, rewards, next_states, are_final):
            self.repl_buf.store(state,
                                action,
                                reward,
                                next_state,
                                is_final)

        if len(self.repl_buf) >= self.batch_size:
            loss = self.update_model()
            self.losses.append(loss)
            self.update_cnt += 1

            # decrease epsilon
            self.eps = max(
                self.min_eps, self.eps-self.eps_decay*(self.max_eps-self.min_eps)
            )
            if self.update_cnt % self.update_target_net == 0:
                self.update_cnt = 1
                self._update_target_net()
            return loss
        return None

    def execute(self, states: np.ndarray) -> np.ndarray:
        actions = []
        for state in states:
            actions.append(self._sel_act(state))
        return np.asarray(actions)

    def update_model(self):
        samples = self.repl_buf.sample()
        loss = self._compute_net_loss(samples)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # Convert id-based actions to string type actions
    # so that they can be used in the environment
    def conv_acts(self, actions: np.ndarray):
        conv_actions = {key:[] for key in self.target_hard_name}
        for i, act in enumerate(actions):
            act = self._conv_act_id_to_type(act)
            conv_actions[act[0]].append([i, act[1]])
        return conv_actions

    def _compute_net_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        state = torch.FloatTensor(samples["state"]).to(self.device)
        next_state = torch.FloatTensor(samples["n_state"]).to(self.device)
        action = torch.FloatTensor(samples["action"].reshape(-1, 1)).to(self.device)
        reward = torch.FloatTensor(samples["reward"].reshape(-1, 1)).to(self.device)
        final = torch.FloatTensor(samples["final"].reshape(-1, 1)).to(self.device)

        curr_q_val = self.net(state).gather(1, action.type(torch.int64))
        next_q_val = self.target_net(next_state).max(dim=1, keepdim=True)[0].detach()
        mask = 1-final
        target_val = (reward + self.gamma*next_q_val*mask).to(self.device)

        loss = F.smooth_l1_loss(curr_q_val, target_val)
        return loss

    def _sel_act(self, state: np.ndarray):
        if self.eps > np.random.random():
            # First select between 3 possible targets
            target_id = np.random.randint(len(self.act_space))
            # Then select action from the selected target
            sel_act = np.random.randint(self.act_type_boundry[target_id],
                                        self.act_type_boundry[target_id+1])
        else:
            sel_act = self.net(
                torch.FloatTensor(state).to(self.device)
            ).argmax()
            sel_act = sel_act.detach().cpu().numpy()
        return sel_act

    def _update_target_net(self):
        # Update target network parameters
        self.target_net.load_state_dict(self.net.state_dict())

    # Convert action id resulting from network
    # to actual action type which is composed of
    # (execution target, frequency or power level)
    def _conv_act_id_to_type(self, act_id):
        target_id = 0
        rel_id = 0
        for i in range(len(self.act_type_boundry)-1):
            if act_id >= self.act_type_boundry[i] and \
               act_id < self.act_type_boundry[i+1]:
                target_id = i # target id [0..2]
                rel_id = act_id - self.act_type_boundry[i]
                break
        target_name = self.target_hard_name[target_id]
        return (target_name,
                self.act_space[target_name][rel_id])

class RRLO_DVFS:
    def __init__(self,
                 state_bounds,
                 num_w_inter_powers,
                 num_dvfs_algs,
                 dvfs_algs,
                 num_tasks):

        self.state_bounds = state_bounds
        self.num_total_states = self.state_bounds.prod()
        self.num_dvfs_algs = num_dvfs_algs
        self.num_w_inter_powers = num_w_inter_powers
        self.dvfs_algs = dvfs_algs
        self.num_tasks = num_tasks
        self.act_bounds = np.ones(self.num_tasks+2, dtype=int)*2 # Initialize all with value 2
        self.act_bounds[-2] = self.num_w_inter_powers # number of possible dvfs algs
        self.act_bounds[-1] = self.num_dvfs_algs # number of possible power levels
        self.alpha = 0.8
        self.beta = 0.9
        self.Q_table_a = np.zeros((self.num_total_states, 2**self.num_tasks * self.num_dvfs_algs * self.num_w_inter_powers))
        self.Q_table_b = np.zeros_like(self.Q_table_a)

    def execute(self, state: np.ndarray) -> np.ndarray:
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
        Q_table[state_row_idx, actions] += self.alpha * (penalty + self.beta*Q_table[next_state_row_idx, next_action] - Q_table[state_row_idx, actions])


    def _conv_state_to_row(self, state: np.ndarray):
        row = 0
        for i in range(len(state)-1):
            row += state[i] * self.state_bounds[i+1:].prod()
        row += state[-1]
        return row

    def _conv_col_to_act(self, act_idx: int) -> np.ndarray:
        local = []
        offload = []
        for i in range(len(self.act_bounds)-1):
            multiple = self.act_bounds[i+1:].prod()
            q = act_idx // multiple
            act_idx = act_idx % multiple
            if i == len(self.act_bounds)-2: # power level
                power_level = q
            else:
                if q == 0: # Execute locally
                    local.append(i)
                else:
                    offload.append(i)
        #TODO: Add more dvfs algorithm
        act = {'local': local, 'offload': offload, 'power_level': power_level, 'dvfs_alg': act_idx}
        return act
