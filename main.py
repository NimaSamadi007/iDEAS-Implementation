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
    return WirelessInterface(specs["powers"])

def load_cpu_configs():
    with open("configs/cpu_specs.json", "r") as f:
        cpu_specs = json.load(f)
    assert len(cpu_specs) == 2, "CPU config must contain exactly 2 CPU types"
    if cpu_specs[0]['model'] == "big":
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

def observe_system_state(tasks, hp):
    # System utilization (SU)
    su, ds_denom, aet_total = 0, 0, 0
    for tid in tasks:
        for t_i in tasks[tid]:
            aet_total += t_i.aet
        su += tasks[tid][0].wcet/tasks[tid][0].p
        ds_denom += hp*tasks[tid][0].wcet/tasks[tid][0].p
    ds = 1-aet_total/ds_denom
    #TODO: Use b_i in states
    state = np.array([su, ds, 0])
    return np.repeat(state.reshape(1, -1), len(tasks), axis=0)

######################## Main function #######################
if __name__ == '__main__':
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
    dvfs_alg = DVFS(state_dim=3, #(SU, DS, R)
                    act_dim=len(w_inter)+
                            len(cpu_big.freqs)+
                            len(cpu_little.freqs))
    edge_server = EdgeServer()
    while True:
        # Generate tasks for one hyper period:
        tasks = tg.generate(task_set)

        # Current state value:
        states = observe_system_state(tasks, tg.hyper_period)
        print(f"States shape: {states.shape}")

        # Run DVFS and offloader to assign tasks
        actions = dvfs_alg.execute(states)
        print(f"Actions:\n{actions}")

        # Execute tasks

        # Calculate rewards

        # Update RL networks

        break
