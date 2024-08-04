from train import train
from evaluate import evaluate
from utils.utils import plot_res
import numpy as np

def results():

    train_configs = {
        "task_set": "configs/task_set_train.json",
        "cpu_local": "configs/cpu_local.json",
        "w_inter": "configs/wireless_interface.json",
    }

    test_configs = {
        "task_set": "configs/task_set_eval.json",
        "cpu_local": "configs/cpu_local.json",
        "w_inter": "configs/wireless_interface.json",
    }
    #train(train_configs)
    energy_consumption_train = evaluate(train_configs)
    energy_consumption_test = evaluate(test_configs)
    alg_set= ['Local','Remote','RRLO', 'DQN']
    plot_res(alg_set,energy_consumption_train,energy_consumption_test, '', 'Energy Consumption', '', 'firstTry')

if __name__ == "__main__":
    results()
