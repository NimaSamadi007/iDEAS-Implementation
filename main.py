import json
import numpy as np
import torch
from typing import Dict
import matplotlib.pyplot as plt

from models.cpu import CPU
from models.wireless_interface import WirelessInterface
from models.task import Task, TaskGen
from models.remote import EdgeServer
from dvfs.dvfs import DVFS

###################################################
# Contants:
DEADLINE_MISSED_PENALTY = 1e3
NUM_ITR = int(1.5e5)
STATE_DIM = 4
REWARD_COEFF = 1
LATENCY_ENERGY_COEFF = 0.1
###################################################
# Utility functions
def load_wireless_interface_configs():
    with open("configs/wireless_interface_specs.json", "r") as f:
        specs = json.load(f)
    return WirelessInterface(specs)

def load_cpu_configs():
    with open("configs/cpu_specs.json", "r") as f:
        cpu_specs = json.load(f)
    assert len(cpu_specs) == 2, "CPU config must contain exactly 2 CPU types"
    if cpu_specs[0]['type'] == "big":
        cpu_big = CPU(cpu_specs[0])
        cpu_little = CPU(cpu_specs[1])
    else:
        cpu_big = CPU(cpu_specs[1])
        cpu_little = CPU(cpu_specs[0])
    return cpu_big, cpu_little

def load_task_set():
    task_set = []
    with open("configs/task_set.json", "r") as f:
        task_set_conf = json.load(f)
    for i in range(len(task_set_conf)):
        task_set.append(Task(task_set_conf[i]))
    return task_set

def observe_system_state(tasks, channel_gain):
    # State: (SU, WCET, B, h)
    states = np.zeros((len(tasks), STATE_DIM), dtype=np.float32)
    su = 0.
    for i, task in enumerate(tasks.values()):
        su += task[0].wcet/task[0].p
        states[i, 1] = task[0].wcet
        states[i, 2] = task[0].b
    states[:, 0] = su
    states[:, 3] = channel_gain
    return states

def cal_penalties(tasks: Dict[int, Task]) -> np.ndarray:
    penalties = []
    for task in tasks.values():
        penalty = 0
        is_deadline_missed = False
        for job in task:
            if job.deadline_missed:
                is_deadline_missed = True
                break
            else:
                penalty += (job.cons_energy+LATENCY_ENERGY_COEFF*job.aet)
        if not is_deadline_missed:
            penalties.append(penalty/len(task))
        else:
            penalties.append(DEADLINE_MISSED_PENALTY)

    return np.asarray(penalties, dtype=float)

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def moving_avg(arr, n):
    return np.convolve(arr, np.ones(n, dtype=arr.dtype), 'same') / n
###################################################
# Main function
if __name__ == '__main__':
    # Set random seed
    set_random_seed(42)

    ## Load tasks and CPU models
    cpu_big, cpu_little = load_cpu_configs()
    task_set = load_task_set()
    w_inter = load_wireless_interface_configs()

    ## Offloading and DVFS main cylce
    # Initialize RL network and other parameters
    tg = TaskGen()
    action_space = {"offload": [],
                    "big": [],
                    "little": []}
    action_space["offload"] = w_inter.powers
    action_space["big"] = cpu_big.freqs
    action_space["little"] = cpu_little.freqs
    dvfs_alg = DVFS(state_dim=STATE_DIM,
                    act_space=action_space,
                    batch_size=32,
                    gamma=0.95)

    cg = w_inter.update_channel_state()
    next_tasks = tg.generate(task_set)
    next_states = observe_system_state(next_tasks, cg)
    all_rewards = []
    for itr in range(NUM_ITR):
        # Assign current tasks from previous tasks
        tasks = next_tasks
        states = next_states

        # Run DVFS and offloader to assign tasks
        raw_actions = dvfs_alg.execute(states)
        actions = dvfs_alg.conv_raw_acts(raw_actions)

        # Execute tasks
        cpu_little.execute(tasks, actions['little'])
        cpu_big.execute(tasks, actions['big'])
        w_inter.offload(tasks, actions['offload'])

        # Calculate penalties
        penalties = cal_penalties(tasks)

        # Observe next state
        cg = w_inter.update_channel_state()
        next_tasks = tg.generate(task_set)
        next_states = observe_system_state(next_tasks, cg)

        # Update RL networks
        if itr == NUM_ITR-1:
            are_final = len(tasks)*[True]
        else:
            are_final = len(tasks)*[False]
        rewards = REWARD_COEFF*np.exp(-REWARD_COEFF*penalties)
        all_rewards.append(rewards.tolist())
        loss = dvfs_alg.train(states, raw_actions, rewards, next_states, are_final)
        if (itr+1) % 500 == 0:
            print(f"At {itr}, loss={loss:.3f}")
            print(f"Actions: {actions}")
            print(f"Rewards: {rewards}")
            print(f"Penalties: {penalties}")
            print(10*"-")

    print(f"Current eps val: {dvfs_alg.eps}")
    fig = plt.figure(figsize=(16, 12))
    plt.title("Loss function values")
    plt.plot(dvfs_alg.losses)
    plt.grid(True)
    fig.savefig("figs/loss_function.png")

    all_rewards = np.array(all_rewards)
    fig = plt.figure(figsize=(16, 12))
    plt.title("Reward value")
    plt.plot(moving_avg(all_rewards[:, 0], 100))
    plt.plot(moving_avg(all_rewards[:, 1], 100))
    plt.plot(moving_avg(all_rewards[:, 2], 100))
    plt.plot(moving_avg(all_rewards[:, 3], 100))
    plt.grid(True)
    fig.savefig("figs/all_reward.png")

    avg_reward = np.sum(all_rewards, axis=1)/all_rewards.shape[1]
    fig = plt.figure(figsize=(16, 12))
    plt.title("Average reward")
    plt.plot(moving_avg(avg_reward, 100))
    plt.grid(True)
    fig.savefig("figs/avg_reward.png")
