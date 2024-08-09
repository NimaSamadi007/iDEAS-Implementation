from train import train
from evaluate import evaluate
from utils.utils import plot_res
import numpy as np
import matplotlib.pyplot as plt


def results(eval_itr=10000):
    train_configs = {
        "task_set": "configs/task_set_train.json",
        "cpu_local": "configs/cpu_local.json",
        "w_inter": "configs/wireless_interface.json",
    }

    eval_configs = {
        "task_set": "configs/task_set_eval.json",
        "cpu_local": "configs/cpu_local.json",
        "w_inter": "configs/wireless_interface.json",
    }
    test_configs = {
        "task_set": "configs/task_set_eval2.json",
        "cpu_local": "configs/cpu_local.json",
        "w_inter": "configs/wireless_interface.json",
    }

    train(train_configs)
    energy_consumption_eval1, missed_deadline_eval1, percent1 = evaluate(
        eval_configs, eval_itr
    )
    energy_consumption_eval2, missed_deadline_eval2, percent2 = evaluate(
        test_configs, eval_itr
    )

    # per1=[]
    # per2=[]
    # final1_mean=np.zeros(10)
    # final2_mean=np.zeros(10)
    # final1_std=np.zeros(10)
    # final2_std=np.zeros(10)
    # for idx,iter in enumerate(range(5000,55000,5000)):
    #   for j in range(10):
    #      energy_consumption_eval1, missed_deadline_eval1,percent1 = evaluate(eval_configs,iter)
    #     energy_consumption_eval2, missed_deadline_eval2,percent2 = evaluate(test_configs,iter)
    #    per1.append(percent1)
    #   per2.append(percent2)
    # final1_mean[idx]= np.mean(per1)
    # final2_mean[idx]=np.mean(per2)
    # final1_std[idx]=np.std(per1,ddof=1)
    # final2_std[idx]=np.std(per2,ddof=1)

    # print(f"task set 1 mean: {final1_mean} %")
    # print(f"task set 1 std: {final1_std} ")
    # print(f"task set 2 mean: {final2_mean} %")
    # print(f"task set 2 std: {final2_std} ")

    # fig = plt.figure(figsize=(16, 12))
    # plt.subplot(1,2,1)
    # plt.plot(range(len(final1_mean)), final1_mean, label='Taskset1', color='r')
    # plt.plot(range(len(final2_mean)), final2_mean, label='Taskset2', color='b')
    # plt.legend()
    # plt.subplot(1,2,2)
    # plt.plot(range(len(final1_std)), final1_std, label='Taskset1', color='r')
    # plt.plot(range(len(final2_std)), final2_std, label='Taskset2', color='b')
    # plt.legend()

    # plt.show()

    alg_set = ["Random", "Local", "Remote", "RRLO", "DQN"]
    plot_res(
        alg_set,
        10 * np.log10(energy_consumption_eval1 / 1e-3),
        10 * np.log10(energy_consumption_eval2 / 1e-3),
        "Algorithm",
        "Energy Consumption (dBm)",
        "Energy Consumption of Different Algorithms",
        "Energy_Consumption",
    )
    plot_res(
        alg_set,
        missed_deadline_eval1,
        missed_deadline_eval2,
        "Algorithm",
        "Dropped Tasks (%) ",
        "Dropped Taks of Different Algorithms",
        "Missed_Deadline",
    )


if __name__ == "__main__":
    results()
