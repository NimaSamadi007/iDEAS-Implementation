import json

from models.cpu import CPU
from models.task import Task, TaskGen
from offloading.offloader import Offloader

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
    while True:
        # 1. Assign tasks to LITTLE, big, or Remote
        state = ()
        offloader.assign(state, task_set)

        # 2. Generate tasks for one hyper period:
        tasks = tg.generate(task_set)
        # print(len(tasks["little"]))
        # print(tasks["little"])
        # print(len(tasks["big"]))
        # print(tasks["big"])
        # print(len(tasks["remote"]))
        # print(tasks["remote"])

        # 3. Run DVFS

        # 4. Calculate rewards

        # 5. Update RL networks

        break
