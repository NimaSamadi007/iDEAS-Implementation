import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from evaluate import iDEAS_evaluate, RRLO_evaluate, big_LITTLE_evaluate
from utils.utils import (
    plot_res,
    print_improvement,
    plot_loss_function,
    line_plot_res,
    stack_bar_res,
    plot_all_rewards,
    load_yaml
)

from train.trainer import iDEAS_BaseTrainer, iDEAS_RRLOTrainer

def iDEAS_Base(configs):
    params = configs["params"]

    if params["do_train"]:
        trainer = iDEAS_BaseTrainer(configs)
        dqn_loss, dqn_rewards = trainer.run()
        dqn_loss = np.array(dqn_loss)
        dqn_rewards = np.array(dqn_rewards)

        plot_loss_function(dqn_loss, "iDEAS", "iterations", "loss", "iDEAS_Loss")
        plot_all_rewards(dqn_rewards, "iDEAS", "iterations", "rewards", "iDEAS_Rewards")

    cpuloads = np.linspace(0.05, 3, 10)
    tasksizes = np.round(np.linspace(110, 490, 11))
    cns = np.logspace(np.log10(2e-11), np.log10(2e-6), num=11, base=10)
    all_energy_eval = np.empty((0, params["num_tasks"], 2))
    all_deadline_eval = np.empty((0, params["num_tasks"], 2))
    all_cpu_energy = np.empty((0, params["num_tasks"], len(cpuloads)))
    all_cpu_deadline = np.empty((0, params["num_tasks"], len(cpuloads)))
    all_task_energy = np.empty((0, params["num_tasks"], len(tasksizes) - 1))
    all_task_deadline = np.empty((0, params["num_tasks"], len(tasksizes) - 1))
    all_cns_energy = np.empty((0, params["num_tasks"], len(cns)))
    all_cns_deadline = np.empty((0, params["num_tasks"], len(cns)))

    for _ in tqdm(range(params["eval_cycle"])):
        result = iDEAS_evaluate(configs, cpuloads, tasksizes, cns)

        all_energy_eval = np.vstack((all_energy_eval, [result["taskset_energy"]]))
        all_deadline_eval = np.vstack((all_deadline_eval, [result["taskset_drop"]]))
        all_cpu_energy = np.vstack((all_cpu_energy, [result["cpu_energy"]]))
        all_cpu_deadline = np.vstack((all_cpu_deadline, [result["cpu_drop"]]))
        all_task_energy = np.vstack((all_task_energy, [result["task_energy"]]))
        all_task_deadline = np.vstack((all_task_deadline, [result["task_drop"]]))
        all_cns_energy = np.vstack((all_cns_energy, [result["cn_energy"]]))
        all_cns_deadline = np.vstack((all_cns_deadline, [result["cn_drop"]]))

    mean_energy_eval = np.mean(all_energy_eval, axis=0)
    mean_drop_eval = np.mean(all_deadline_eval, axis=0)
    mean_cpu_energy = np.mean(all_cpu_energy, axis=0)
    mean_cpu_deadline = np.mean(all_cpu_deadline, axis=0)
    mean_task_energy = np.mean(all_task_energy, axis=0)
    mean_task_deadline = np.mean(all_task_deadline, axis=0)
    mean_cns_energy = np.mean(all_cns_energy, axis=0)
    mean_cns_deadline = np.mean(all_cns_deadline, axis=0)

    alg_set = ["big", "LITTLE", "offload", "Total"]
    stack_bar_res(
        labels=alg_set,
        data1=mean_energy_eval,
        x_val=["Task set 1", "Task set 2"],
        xlabel="Task sets",
        ylabel="Consumed Energy (J) ",
        title="Energy Consumption Levels on Different Task sets",
        fig_name="iDEAS_energy_levels_tasksets",
    )
    stack_bar_res(
        alg_set,
        mean_drop_eval,
        ["Task set 1", "Task set 2"],
        "Task sets",
        "Dropped Tasks (\%) ",
        "Dropped Tasks levels on Different Task sets",
        "iDEAS_dropped_tasks_tasksets",
    )

    stack_bar_res(
        alg_set,
        mean_cpu_energy,
        cpuloads,
        "Task Load",
        "Consumed Energy (J) ",
        "Energy Consumption Levels for Different Task Loads",
        "iDEAS_cpu_energy",
        numbered=True,
    )
    stack_bar_res(
        alg_set,
        mean_cpu_deadline,
        cpuloads,
        "Task Load",
        "Dropped Tasks (\%) ",
        "Dropped Tasks levels for Different Task Loads",
        "iDEAS_cpu_drop",
        numbered=True,
    )

    stack_bar_res(
        alg_set,
        mean_task_energy,
        tasksizes[:-1],
        "Task Size (KB)",
        "Consumed Energy (J) ",
        "Energy Consumption Levels for Different Task Sizes",
        "iDEAS_task_energy",
        numbered=True,
    )
    stack_bar_res(
        alg_set,
        mean_task_deadline,
        tasksizes[:-1],
        "Task Size(KB)",
        "Dropped Tasks (\%) ",
        "Dropped Tasks levels for Different Task Sizes",
        "iDEAS_task_drop",
        numbered=True,
    )
    stack_bar_res(
        alg_set,
        mean_cns_energy,
        cns,
        "Channel Noise",
        "Consumed Energy (J) ",
        "Energy Consumption Levels for Different Channel Noises",
        "iDEAS_cns_energy",
        numbered=True,
        xlog=True,
    )
    stack_bar_res(
        alg_set,
        mean_cns_deadline,
        cns,
        "Channel Noise",
        "Dropped Tasks (\%) ",
        "Dropped Tasks levels for Different Channel Noises",
        "iDEAS_cns_drop",
        numbered=True,
        xlog=True,
    )


def iDEAS_RRLO(configs):
    params = configs["params"]

    if params["do_train"]:
        trainer = iDEAS_RRLOTrainer(configs)
        dqn_loss, all_rewards = trainer.run()
        dqn_loss = np.array(dqn_loss)
        dqn_rewards = np.array([all_reward['ideas'] for all_reward in all_rewards])

        plot_loss_function(dqn_loss, "iDEAS", "iterations", "loss", "RRLO_loss")
        plot_all_rewards(dqn_rewards, "iDEAS", "iterations", "rewards", "RRLO_rewards")

    cpuloads = np.linspace(0.05, 2.05, 10)
    tasksizes = np.round(np.linspace(110, 490, 11))
    cns = np.logspace(np.log10(2e-11), np.log10(2e-6), num=10, base=10)
    all_energy_eval = np.empty((0, 5, 2))
    all_deadline_eval = np.empty((0, 5, 2))
    all_improvement_eval = np.empty((0, 4, 2))
    all_cpu_energy = np.empty((0, 5, len(cpuloads)))
    all_cpu_deadline = np.empty((0, 5, len(cpuloads)))
    all_task_energy = np.empty((0, 5, len(tasksizes) - 1))
    all_task_deadline = np.empty((0, 5, len(tasksizes) - 1))
    all_cns_energy = np.empty((0, 5, len(cns)))
    all_cns_deadline = np.empty((0, 5, len(cns)))

    for _ in tqdm(range(params["eval_cycle"])):
        result = RRLO_evaluate(configs, cpuloads, tasksizes, cns)
        all_energy_eval = np.vstack((all_energy_eval, [result["taskset_energy"]]))
        all_deadline_eval = np.vstack((all_deadline_eval, [result["taskset_drop"]]))
        all_improvement_eval = np.vstack(
            (all_improvement_eval, [result["taskset_improvement"]])
        )
        all_cpu_energy = np.vstack((all_cpu_energy, [result["cpu_energy"]]))
        all_cpu_deadline = np.vstack((all_cpu_deadline, [result["cpu_drop"]]))
        all_task_energy = np.vstack((all_task_energy, [result["task_energy"]]))
        all_task_deadline = np.vstack((all_task_deadline, [result["task_drop"]]))
        all_cns_energy = np.vstack((all_cns_energy, [result["cn_energy"]]))
        all_cns_deadline = np.vstack((all_cns_deadline, [result["cn_drop"]]))
    mean_energy_eval = np.mean(all_energy_eval, axis=0)
    mean_drop_eval = np.mean(all_deadline_eval, axis=0)
    mean_improvement_eval = np.mean(all_improvement_eval, axis=0)
    mean_cpu_energy = np.mean(all_cpu_energy, axis=0)
    mean_cpu_deadline = np.mean(all_cpu_deadline, axis=0)
    mean_task_energy = np.mean(all_task_energy, axis=0)
    mean_task_deadline = np.mean(all_task_deadline, axis=0)
    mean_cns_energy = np.mean(all_cns_energy, axis=0)
    mean_cns_deadline = np.mean(all_cns_deadline, axis=0)

    alg_set = ["Random", "RRLO", "iDEAS", "Local", "Edge Only"]
    alg_set2 = ["Random", "RRLO", "Local", "Edge Only"]
    plot_res(
        alg_set,
        mean_energy_eval[:, 0],
        mean_energy_eval[:, 1],
        "Scheme",
        "Energy Consumption (J)",
        "Energy Consumption of Different Baseline Single Core Schemes",
        "RRLO_energy_tasksets",
        ylog=True,
    )
    plot_res(
        alg_set,
        mean_drop_eval[:, 0],
        mean_drop_eval[:, 1],
        "Scheme",
        "Dropped Tasks (\%) ",
        "Dropped Tasks of Different Baseline Single Core Schemes  ",
        "RRLO_dropped_tasksets",
    )

    print(" Energy Consumption Improvements.......")
    print("")
    print("")
    print(
        print_improvement(
            alg_set2, mean_improvement_eval[:, 0], mean_improvement_eval[:, 1], 4, 4
        )
    )

    line_plot_res(
        alg_set,
        mean_cpu_energy,
        cpuloads,
        "Task Load",
        "Energy Consumption (J)",
        "Energy Consumption of Different Single Core Schemes With Respect to Various Task Loads",
        "RRLO_cpu_energy",
        ylog=True,
    )
    line_plot_res(
        alg_set,
        mean_cpu_deadline,
        cpuloads,
        "Task Load",
        "Dropped Tasks (\%) ",
        "Dropped Tasks of Different Baseline Single Core Scheme With Respect to Various Task Loads",
        "RRLO_cpu_drop",
    )

    line_plot_res(
        alg_set,
        mean_task_energy,
        tasksizes[:-1],
        "Task Size(KB)",
        "Energy Consumption (J)",
        "Energy Consumption of Different Single Core Schemes With Respect to Various Task Sizes",
        "RRLO_task_energy",
        ylog=True,
    )
    line_plot_res(
        alg_set,
        mean_task_deadline,
        tasksizes[:-1],
        "Task Size (KB)",
        "Dropped Tasks (\%) ",
        "Dropped Tasks of Different Baseline Single Core Scheme With Respect to Various Task Sizes",
        "RRLO_task_drop",
    )

    line_plot_res(
        alg_set,
        mean_cns_energy,
        cns,
        "Channel Noise",
        "Energy Consumption (J)",
        "Energy Consumption of Different Single Core Schemes With Respect to Various Channel Noises",
        "RRLO_cns_energy",
        ylog=True,
        xlog=True,
    )
    line_plot_res(
        alg_set,
        mean_cns_deadline,
        cns,
        "Channel Noise",
        "Dropped Tasks (\%) ",
        "Dropped Tasks of Different Baseline Single Core Scheme With Respect to Various Channel Noises",
        "RRLO_cns_drop",
        xlog=True,
    )


def big_little_main(eval_itr=10000, iter=1, Train=False):
    DQN_STATE_DIM = 5
    configs = {
        "task_set1": "configs/task_set_eval.json",
        "task_set2": "configs/task_set_eval2.json",
        "task_set3": "configs/task_set_train.json",
        "cpu_little": "configs/cpu_little.json",
        "cpu_big": "configs/cpu_big.json",
        "cpu_local": "configs/cpu_local.json",
        "w_inter": "configs/wireless_interface.json",
        "dqn_state_dim": DQN_STATE_DIM,
    }

    if Train:
        dqn_loss, dqn_rewards = iDEAS_train(configs)

        plot_loss_function(dqn_loss, "iDEAS", "iterations", "loss", "iDEAS_Loss")
        plot_all_rewards(dqn_rewards, "iDEAS", "iterations", "rewards", "iDEAS_Rewards")

    cpuloads = np.linspace(0.05, 3.05, 10)
    tasksizes = np.round(np.linspace(110, 490, 11))
    cns = np.logspace(np.log10(2e-11), np.log10(2e-6), num=10, base=10)
    all_energy_eval = np.empty((0, 4, 2))
    all_deadline_eval = np.empty((0, 4, 2))
    all_improvement_eval = np.empty((0, 3, 2))
    all_cpu_energy = np.empty((0, 4, len(cpuloads)))
    all_cpu_deadline = np.empty((0, 4, len(cpuloads)))
    all_task_energy = np.empty((0, 4, len(tasksizes) - 1))
    all_task_deadline = np.empty((0, 4, len(tasksizes) - 1))
    all_cns_energy = np.empty((0, 4, len(cns)))
    all_cns_deadline = np.empty((0, 4, len(cns)))

    for j in tqdm(range(iter)):
        result = big_LITTLE_evaluate(configs, cpuloads, tasksizes, cns, eval_itr)
        all_energy_eval = np.vstack((all_energy_eval, [result["taskset_energy"]]))
        all_deadline_eval = np.vstack((all_deadline_eval, [result["taskset_drop"]]))
        all_improvement_eval = np.vstack(
            (all_improvement_eval, [result["taskset_improvement"]])
        )
        all_cpu_energy = np.vstack((all_cpu_energy, [result["cpu_energy"]]))
        all_cpu_deadline = np.vstack((all_cpu_deadline, [result["cpu_drop"]]))
        all_task_energy = np.vstack((all_task_energy, [result["task_energy"]]))
        all_task_deadline = np.vstack((all_task_deadline, [result["task_drop"]]))
        all_cns_energy = np.vstack((all_cns_energy, [result["cn_energy"]]))
        all_cns_deadline = np.vstack((all_cns_deadline, [result["cn_drop"]]))
    mean_energy_eval = np.mean(all_energy_eval, axis=0)
    mean_drop_eval = np.mean(all_deadline_eval, axis=0)
    mean_improvement_eval = np.mean(all_improvement_eval, axis=0)
    mean_cpu_energy = np.mean(all_cpu_energy, axis=0)
    mean_cpu_deadline = np.mean(all_cpu_deadline, axis=0)
    mean_task_energy = np.mean(all_task_energy, axis=0)
    mean_task_deadline = np.mean(all_task_deadline, axis=0)
    mean_cns_energy = np.mean(all_cns_energy, axis=0)
    mean_cns_deadline = np.mean(all_cns_deadline, axis=0)

    alg_set = ["Random", "iDEAS", "Local", "Edge Only"]
    alg_set2 = ["Random", "RRLO", "Local", "Edge Only"]
    plot_res(
        alg_set,
        mean_energy_eval[:, 0],
        mean_energy_eval[:, 1],
        "Scheme",
        "Energy Consumption (J)",
        "Energy Consumption of Different Baseline big.LITTLE Schemes",
        "BL_energy_tasksets",
        ylog=True,
    )
    plot_res(
        alg_set,
        mean_drop_eval[:, 0],
        mean_drop_eval[:, 1],
        "Scheme",
        "Dropped Tasks (\%) ",
        "Dropped Tasks of Different Baseline big.LITTLE Schemes  ",
        "BL_dropped_tasksets",
    )

    print(" Energy Consumption Improvements.......")
    print("")
    print("")
    print(
        print_improvement(
            alg_set2, mean_improvement_eval[:, 0], mean_improvement_eval[:, 1], 3, 3
        )
    )

    line_plot_res(
        alg_set,
        mean_cpu_energy,
        cpuloads,
        "Task Load",
        "Energy Consumption (J)",
        "Energy Consumption of Different big.LITTLE Schemes With Respect to Various Task Loads",
        "BL_cpu_energy",
        ylog=True,
    )
    line_plot_res(
        alg_set,
        mean_cpu_deadline,
        cpuloads,
        "Task Load",
        "Dropped Tasks (\%) ",
        "Dropped Tasks of Different Baseline big.LITTLE Schemes With Respect to Various Task Loads",
        "BL_cpu_drop",
    )

    line_plot_res(
        alg_set,
        mean_task_energy,
        tasksizes[:-1],
        "Task Size(KB)",
        "Energy Consumption (J)",
        "Energy Consumption of Different Single big.LITTLE Schemes With Respect to Various Task Sizes",
        "BL_task_energy",
        ylog=True,
    )
    line_plot_res(
        alg_set,
        mean_task_deadline,
        tasksizes[:-1],
        "Task Size (KB)",
        "Dropped Tasks (\%) ",
        "Dropped Tasks of Different Baseline big.LITTLE Schemes With Respect to Various Task Sizes",
        "BL_task_drop",
    )

    line_plot_res(
        alg_set,
        mean_cns_energy,
        cns,
        "Channel Noise",
        "Energy Consumption (J)",
        "Energy Consumption of Different big.LITTLE Schemes With Respect to Various Channel Noises",
        "BL_cns_energy",
        ylog=True,
        xlog=True,
    )
    line_plot_res(
        alg_set,
        mean_cns_deadline,
        cns,
        "Channel Noise",
        "Dropped Tasks (\%) ",
        "Dropped Tasks of Different Baseline big.LITTLE Schemes With Respect to Various Channel Noises",
        "BL_cns_drop",
        xlog=True,
    )


if __name__ == "__main__":
    # configs_ideas_base = load_yaml("./configs/iDEAS_Base.yaml")
    # iDEAS_Base(configs_ideas_base)

    configs_ideas_rrlo = load_yaml("./configs/iDEAS_RRLO.yaml")
    iDEAS_RRLO(configs_ideas_rrlo)

    # big_little_main(Train=False)
