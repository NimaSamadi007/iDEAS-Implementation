import json

from models.cpu import CPU
from models.task import Task, TaskGen
from models.remote import EdgeServer
from offloading.offloader import Offloader
from dvfs.dvfs import DVFS

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

if __name__ == '__main__':
    ## Load tasks and CPU models
    cpu_big, cpu_little = load_cpu_configs()
    print(cpu_big)
    print(cpu_little)

    task_set = load_task_set()
    for t in task_set:
        print(t)

    ## Offloading and DVFS main cylce
    # 0. Initialize RL network and other parameters
    tg = TaskGen()
    offloader = Offloader()
    dvfs_alg = DVFS()
    edge_server = EdgeServer()
    while True:
        # 1. Assign tasks to LITTLE, big, or Remote
        state = ()
        offloader.assign(state, task_set)

        # 2. Generate tasks for one hyper period:
        tasks = tg.generate(task_set)

        # 3. Execute tasks and run DVFS
        dvfs_alg.execute(tasks["little"], cpu_little)
        dvfs_alg.execute(tasks["big"], cpu_big)
        edge_server.execute(tasks["remote"])

        # 4. Calculate rewards

        # 5. Update RL networks

        break
