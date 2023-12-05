import json
import numpy as np

from models.cpu import CPU
from models.wireless_interface import WirelessInterface
from models.task import Task, TaskGen
from models.remote import EdgeServer
from dvfs.dvfs import DVFS

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

def observe_system_state(tasks):
    # State: (SU, WCET, B)
    # TODO: Add channel randomness to state
    states = np.zeros((len(tasks), 3), dtype=np.float32)
    su = 0.
    for i, task in enumerate(tasks.values()):
        su += task[0].wcet/task[0].p
        states[i, 1] = task[0].p
        states[i, 2] = task[0].b
    states[:, 0] = su
    return states

######################## Main function #######################
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
    dvfs_alg = DVFS(state_dim=3,
                    act_space=action_space,
                    seed=RND_SEED)
    print("-------------")
    while True:
        # Generate tasks for one hyper period:
        tasks = tg.generate(task_set)

        # Current state value:
        states = observe_system_state(tasks)
        print(f"States shape: {states.shape}")

        # Run DVFS and offloader to assign tasks
        actions = dvfs_alg.execute(states)
        print(f"Actions:\n{actions}")

        # Execute tasks
        for t_id, act in actions['little']:
            cpu_little.execute(tasks[t_id], act)
        print(10*"=")
        for t_id, act in actions['big']:
            cpu_big.execute(tasks[t_id], act)
        print(10*"=")
        for t_id, act in actions['offload']:
            w_inter.offload(tasks[t_id], act)
        print(10*"=")

        # Calculate rewards

        # Update RL networks

        break
