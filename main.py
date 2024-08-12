from train import train_rrlo_scenario, train_dqn_scenario
from evaluate import evaluate_rrlo_scenario, evaluate_dqn_scenario
from utils.utils import plot_res
import numpy as np
import matplotlib.pyplot as plt


def compare_dqn_rrlo(eval_itr=10000, iter=100):
    DQN_STATE_DIM = 4
    train_configs = {
        "task_set": "configs/task_set_train.json",
        "cpu_local": "configs/cpu_local.json",
        "w_inter": "configs/wireless_interface.json",
        "dqn_state_dim": DQN_STATE_DIM,
    }

    eval_configs = {
        "task_set": "configs/task_set_eval.json",
        "cpu_local": "configs/cpu_local.json",
        "w_inter": "configs/wireless_interface.json",
        "dqn_state_dim": DQN_STATE_DIM,
    }
    test_configs = {
        "task_set": "configs/task_set_eval2.json",
        "cpu_local": "configs/cpu_local.json",
        "w_inter": "configs/wireless_interface.json",
        "dqn_state_dim": DQN_STATE_DIM,
    }

    #train_rrlo_scenario(train_configs)



    all_energy_eval1 = np.empty((0, 5))
    all_energy_eval2 = np.empty((0, 5))
    all_deadline_eval1 = np.empty((0, 5))
    all_deadline_eval2 = np.empty((0, 5))

    for j in range(iter):
        energy_consumption_eval1, missed_deadline_eval1, energy_improvement1,deadline_improvement1 = evaluate_rrlo_scenario(
        eval_configs, eval_itr
        )
        all_energy_eval1=np.vstack((all_energy_eval1,energy_consumption_eval1))
        all_deadline_eval1=np.vstack((all_deadline_eval1,missed_deadline_eval1))
        energy_consumption_eval2, missed_deadline_eval2, energy_improvement2,deadline_improvement2 = evaluate_rrlo_scenario(
        test_configs, eval_itr
        )
        all_energy_eval2=np.vstack((all_energy_eval2,energy_consumption_eval2))
        all_deadline_eval2=np.vstack((all_deadline_eval2,missed_deadline_eval2))
    # Calculate mean and standard deviation column-wise
    mean_energy_consumption_eval1 = np.mean(all_energy_eval1, axis=0)
    std_energy_consumption_eval1 = np.std(10*np.log10(all_energy_eval1/1e-3), axis=0)

    mean_missed_deadline_eval1 = np.mean(all_deadline_eval1, axis=0)
    std_missed_deadline_eval1 = np.std(all_deadline_eval1, axis=0)

    # Repeat the same process for eval2
    mean_energy_consumption_eval2 = np.mean(all_energy_eval2, axis=0)
    std_energy_consumption_eval2 = np.std(10*np.log10(all_energy_eval2/1e-3), axis=0)

    mean_missed_deadline_eval2 = np.mean(all_deadline_eval2, axis=0)
    std_missed_deadline_eval2 = np.std(all_deadline_eval2, axis=0)






    print("*"*20)
    print(mean_energy_consumption_eval1)
    print(mean_energy_consumption_eval2)
    print(mean_missed_deadline_eval1)
    print(mean_missed_deadline_eval2)
    print("*"*20)
    print(std_energy_consumption_eval1)
    print(std_energy_consumption_eval2)
    print(std_missed_deadline_eval1)
    print(std_missed_deadline_eval2)
    alg_set = ["Random", "Local", "Remote", "RRLO", "DQN"]
    plot_res(
        alg_set,
        10 * np.log10(mean_energy_consumption_eval1 / 1e-3),
        10 * np.log10(mean_energy_consumption_eval2 / 1e-3),
        std_energy_consumption_eval1,
        std_energy_consumption_eval2,
        "Algorithm",
        "Energy Consumption (dBm)",
        "Energy Consumption of Different Algorithms in RRLO SScenario",
        "Energy_Consumption_RRLO",
    )
    plot_res(
        alg_set,
        mean_missed_deadline_eval1,
        mean_missed_deadline_eval2,
        std_missed_deadline_eval1,
        std_missed_deadline_eval2,
        "Algorithm",
        "Dropped Tasks (%) ",
        "Dropped Tasks of Different Algorithms in RRLO SScenario",
        "Missed_Deadline_RRLO",
    )


def compare_dqn_base(eval_itr=10000,iter=100):
    DQN_STATE_DIM = 5
    train_configs = {
        "task_set": "configs/task_set_train.json",
        "cpu_little": "configs/cpu_little.json",
        "cpu_big": "configs/cpu_big.json",
        "w_inter": "configs/wireless_interface.json",
        "dqn_state_dim": DQN_STATE_DIM,
    }

    eval_configs = {
        "task_set": "configs/task_set_eval.json",
        "cpu_little": "configs/cpu_little.json",
        "cpu_big": "configs/cpu_big.json",
        "w_inter": "configs/wireless_interface.json",
        "dqn_state_dim": DQN_STATE_DIM,
    }

    test_configs = {
        "task_set": "configs/task_set_eval2.json",
        "cpu_little": "configs/cpu_little.json",
        "cpu_big": "configs/cpu_big.json",
        "w_inter": "configs/wireless_interface.json",
        "dqn_state_dim": DQN_STATE_DIM,
    }

    #train_dqn_scenario(train_configs)
    all_energy_eval1 = np.empty((0, 4))
    all_energy_eval2 = np.empty((0, 4))
    all_deadline_eval1 = np.empty((0, 4))
    all_deadline_eval2 = np.empty((0, 4))

    for j in range(iter):
        energy_consumption_eval1, missed_deadline_eval1 = evaluate_dqn_scenario(
        eval_configs, eval_itr
        )
        all_energy_eval1=np.vstack((all_energy_eval1,energy_consumption_eval1))
        all_deadline_eval1=np.vstack((all_deadline_eval1,missed_deadline_eval1))
        energy_consumption_eval2, missed_deadline_eval2 = evaluate_dqn_scenario(
        test_configs, eval_itr
        )
        all_energy_eval2=np.vstack((all_energy_eval2,energy_consumption_eval2))
        all_deadline_eval2=np.vstack((all_deadline_eval2,missed_deadline_eval2))
    # Calculate mean and standard deviation column-wise
    mean_energy_consumption_eval1 = np.mean(all_energy_eval1, axis=0)
    std_energy_consumption_eval1 = np.std(10*np.log10(all_energy_eval1/1e-3), axis=0)

    mean_missed_deadline_eval1 = np.mean(all_deadline_eval1, axis=0)
    std_missed_deadline_eval1 = np.std(all_deadline_eval1, axis=0)

    # Repeat the same process for eval2
    mean_energy_consumption_eval2 = np.mean(all_energy_eval2, axis=0)
    std_energy_consumption_eval2 = np.std(10*np.log10(all_energy_eval2/1e-3), axis=0)

    mean_missed_deadline_eval2 = np.mean(all_deadline_eval2, axis=0)
    std_missed_deadline_eval2 = np.std(all_deadline_eval2, axis=0)
    print("*"*20)
    print(mean_energy_consumption_eval1)
    print(mean_energy_consumption_eval2)
    print(mean_missed_deadline_eval1)
    print(mean_missed_deadline_eval2)
    print("*"*20)
    print(std_energy_consumption_eval1)
    print(std_energy_consumption_eval2)
    print(std_missed_deadline_eval1)
    print(std_missed_deadline_eval2)
    alg_set = ["Random", "Local", "Remote","DQN"]
    plot_res(
        alg_set,
        10 * np.log10(mean_energy_consumption_eval1 / 1e-3),
        10 * np.log10(mean_energy_consumption_eval2 / 1e-3),
        std_energy_consumption_eval1,
        std_energy_consumption_eval2,
        "Algorithm",
        "Energy Consumption (dBm)",
        "Energy Consumption of Different Algorithms in DQN SScenario",
        "Energy_Consumption_DQN",
    )
    plot_res(
        alg_set,
        mean_missed_deadline_eval1,
        mean_missed_deadline_eval2,
        std_missed_deadline_eval1,
        std_missed_deadline_eval2,
        "Algorithm",
        "Dropped Tasks (%) ",
        "Dropped Tasks of Different Algorithms in DQN SScenario",
        "Missed_Deadline_DQN",
    )


if __name__ == "__main__":
    compare_dqn_rrlo()
    compare_dqn_base()
