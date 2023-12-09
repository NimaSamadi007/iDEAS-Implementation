import json
import numpy as np
from typing import Dict

from models.cpu import CPU
from models.wireless_interface import WirelessInterface
from models.task import Task, TaskGen
from models.remote import EdgeServer
from dvfs.dvfs import DVFS

###################################################
# Contants:
DEADLINE_MISSED_PENALTY = 1e3
NUM_ITR = 10
STATE_DIM = 4

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
        states[i, 1] = task[0].p
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
                penalty += job.cons_energy
        if not is_deadline_missed:
            penalties.append(penalty)
        else:
            penalties.append(DEADLINE_MISSED_PENALTY)

    return np.asarray(penalties, dtype=float)

###################################################
# Main function
if __name__ == '__main__':
    # Set numpy random seed
    RND_SEED = 81
    np.random.seed(RND_SEED)

    ## Load tasks and CPU models
    cpu_big, cpu_little = load_cpu_configs()
    print(cpu_big)
    print(cpu_little)

    task_set = load_task_set()
    for t in task_set:
        print(t)

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
                    seed=RND_SEED)
    print("-------------")
    cg, _ = w_inter.update_channel_state()
    next_tasks = tg.generate(task_set)
    next_states = observe_system_state(next_tasks, cg)
    for itr in range(NUM_ITR):
        # Assign current tasks from previous tasks
        tasks = next_tasks
        states = next_states

        # Run DVFS and offloader to assign tasks
        raw_actions = dvfs_alg.execute(states)
        actions = dvfs_alg.conv_raw_acts(raw_actions)
        print(f"Actions:\n{actions}")

        # Execute tasks
        print("Executing LITTLE core tasks...")
        cpu_little.execute(tasks, actions['little'])
        print("Executing big core tasks...")
        cpu_big.execute(tasks, actions['big'])
        print("Offloading tasks to edge server...")
        w_inter.offload(tasks, actions['offload'])

        # Calculate penalties
        penalties = cal_penalties(tasks)

        # Observe next state
        cg, _ = w_inter.update_channel_state()
        next_tasks = tg.generate(task_set)
        next_states = observe_system_state(next_tasks, cg)

        # Update RL networks
        if itr == NUM_ITR-1:
            are_final = len(tasks)*[True]
        else:
            are_final = len(tasks)*[False]
        dvfs_alg.train(states, raw_actions, -penalties, next_states, are_final)

        print(10*"*")
