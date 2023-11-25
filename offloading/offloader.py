
class Offloader:
    def __init__(self):
        pass

    # Assign a set of task using RL
    def assign(self, state, task_set):
        task_set[0].exec_device = "big"
        task_set[1].exec_device = "little"
        task_set[2].exec_device = "big"
        task_set[3].exec_device = "remote"

    def update(self, state, reward):
        pass
