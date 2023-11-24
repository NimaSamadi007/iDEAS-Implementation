
class Task:
    def __init__(self, specs):
        self.p = specs['p'] # period frequency
        self.b = specs['b'] # task input data (bits)
        self.w = specs['w'] # worst-case execution time
        self.t_id = specs['task']

    def __repr__(self):
        return f"Task t_{self.t_id}: ({self.p}, {self.b}, {self.w})"

    def exectue(self, dt):
        pass

class GenTask:
    def __init__(self):
        pass
