import numpy as np
from tqdm import tqdm

from eval.evaluate import big_LITTLE_evaluate
from utils.utils import (
    plot_res,
    print_improvement,
    plot_loss_function,
    line_plot_res,
    stack_bar_res,
    plot_all_rewards,
    load_yaml,
)

from train.trainer import iDEAS_BaseTrainer, iDEAS_RRLOTrainer
from eval.evaluator import iDEAS_BaseEvaluator, iDEAS_RRLOEvaluator

def iDEAS_Base(configs):
    params = configs["params"]
    num_eval_cycles = params["eval_cycle"]

    if params["do_train"]:
        trainer = iDEAS_BaseTrainer(configs)
        dqn_loss, dqn_rewards = trainer.run()
        dqn_loss = np.array(dqn_loss)
        dqn_rewards = np.array(dqn_rewards)
        print("Training completed")
        print(100 * "-")
        plot_loss_function(dqn_loss, "iDEAS", "iterations", "loss", "iDEAS_Base_loss")
        plot_all_rewards(dqn_rewards, "iDEAS", "iterations", "rewards", "iDEAS_Base_rewards")


    cpuloads = np.linspace(0.05, 3, 10)
    tasksizes = np.round(np.linspace(110, 490, 11))
    cns = np.logspace(np.log10(2e-11), np.log10(2e-6), num=11, base=10)

    all_results = {}
    for i in tqdm(range(num_eval_cycles)):
        evaluator = iDEAS_BaseEvaluator(configs, cpuloads, tasksizes, cns)
        result = evaluator.run()

        for scenario in result:
            if scenario not in all_results:
                # Create result holder array
                all_results[scenario] = np.zeros(
                    (num_eval_cycles, *result[scenario].shape)
                )
            all_results[scenario][i, :] = result[scenario]

    plot_infos = {
        "fixed_taskset_energy": [
            ["Task set 1", "Task set 2"],
            "Task sets",
            "Consumed Energy (J) ",
            "Energy Consumption Levels on Different Task sets",
            "iDEAS_Base_fixed_taskset_energy",
        ],
        "fixed_taskset_drop": [
            ["Task set 1", "Task set 2"],
            "Task sets",
            "Dropped Tasks (\%) ",
            "Dropped Tasks levels on Different Task sets",
            "iDEAS_Base_fixed_taskset_drop",
        ],
        "varied_cpuload_energy": [
            cpuloads,
            "Task Load",
            "Consumed Energy (J) ",
            "Energy Consumption Levels for Different Task Loads",
            "iDEAS_Base_varied_cpuload_energy",
            True,
        ],
        "varied_cpuload_drop": [
            cpuloads,
            "Task Load",
            "Dropped Tasks (\%) ",
            "Dropped Tasks levels for Different Task Loads",
            "iDEAS_Base_varied_cpuload_drop",
            True,
        ],
        "varied_tasksize_energy": [
            tasksizes[:-1],
            "Task Size (KB)",
            "Consumed Energy (J) ",
            "Energy Consumption Levels for Different Task Sizes",
            "iDEAS_Base_varied_tasksize_energy",
            True,
        ],
        "varied_tasksize_drop": [
            tasksizes[:-1],
            "Task Size(KB)",
            "Dropped Tasks (\%) ",
            "Dropped Tasks levels for Different Task Sizes",
            "iDEAS_Base_varied_tasksize_drop",
            True,
        ],
        "varied_channel_energy": [
            cns,
            "Channel Noise",
            "Consumed Energy (J) ",
            "Energy Consumption Levels for Different Channel Noises",
            "iDEAS_Base_varied_channel_energy",
            True,
            True,
        ],
        "varied_channel_drop": [
            cns,
            "Channel Noise",
            "Dropped Tasks (\%) ",
            "Dropped Tasks levels for Different Channel Noises",
            "iDEAS_Base_varied_channel_drop",
            True,
            True,
        ],
    }

    # Plot all results
    alg_set = ["big", "LITTLE", "offload", "Total"]
    for scenario in all_results:
        mean_result = np.mean(all_results[scenario], axis=0)
        stack_bar_res(alg_set, mean_result, *plot_infos[scenario])


def iDEAS_RRLO(configs):
    params = configs["params"]
    num_eval_cycles = params["eval_cycle"]

    if params["do_train"]:
        trainer = iDEAS_RRLOTrainer(configs)
        dqn_loss, all_rewards = trainer.run()
        dqn_loss = np.array(dqn_loss)
        dqn_rewards = np.array([all_reward["ideas"] for all_reward in all_rewards])

        print("Training completed")
        print(100 * "-")

        plot_loss_function(dqn_loss, "iDEAS", "iterations", "loss", "iDEAS_RRLO_loss")
        plot_all_rewards(dqn_rewards, "iDEAS", "iterations", "rewards", "iDEAS_RRLO_rewards")

    cpuloads = np.linspace(0.05, 2.05, 10)
    tasksizes = np.round(np.linspace(110, 490, 11))
    cns = np.logspace(np.log10(2e-11), np.log10(2e-6), num=10, base=10)

    all_results = {}
    for i in tqdm(range(num_eval_cycles)):
        evaluator = iDEAS_RRLOEvaluator(configs, cpuloads, tasksizes, cns)
        result = evaluator.run()

        for scenario in result:
            if scenario not in all_results:
                # Create result holder array
                all_results[scenario] = np.zeros(
                    (num_eval_cycles, *result[scenario].shape)
                )
            all_results[scenario][i, :] = result[scenario]

    plot_infos = {
        "fixed_taskset_energy": [
            "Energy Consumption (J)",
            "Energy Consumption of Different Baseline Single Core Schemes",
            "iDEAS_RRLO_fixed_taskset_energy",
            True,
        ],
        "fixed_taskset_drop": [
            "Dropped Tasks (\%) ",
            "Dropped Tasks of Different Baseline Single Core Schemes  ",
            "iDEAS_RRLO_fixed_taskset_drop",
        ],
        "varied_cpuload_energy": [
            cpuloads,
            "Task Load",
            "Energy Consumption (J)",
            "Energy Consumption of Different Single Core Schemes With Respect to Various Task Loads",
            "iDEAS_RRLO_varied_cpuload_energy",
            True,
        ],
        "varied_cpuload_drop": [
            cpuloads,
            "Task Load",
            "Dropped Tasks (\%) ",
            "Dropped Tasks of Different Baseline Single Core Scheme With Respect to Various Task Loads",
            "iDEAS_RRLO_varied_cpuload_drop",
        ],
        "varied_tasksize_energy": [
            tasksizes[:-1],
            "Task Size(KB)",
            "Energy Consumption (J)",
            "Energy Consumption of Different Single Core Schemes With Respect to Various Task Sizes",
            "iDEAS_RRLO_varied_tasksize_energy",
            True,
            False,
        ],
        "varied_tasksize_drop": [
            tasksizes[:-1],
            "Task Size (KB)",
            "Dropped Tasks (\%) ",
            "Dropped Tasks of Different Baseline Single Core Scheme With Respect to Various Task Sizes",
            "iDEAS_RRLO_varied_tasksize_drop",
        ],
        "varied_channel_energy": [
            cns,
            "Channel Noise",
            "Energy Consumption (J)",
            "Energy Consumption of Different Single Core Schemes With Respect to Various Channel Noises",
            "iDEAS_RRLO_varied_channel_energy",
            True,
            True,
        ],
        "varied_channel_drop": [
            cns,
            "Channel Noise",
            "Dropped Tasks (\%) ",
            "Dropped Tasks of Different Baseline Single Core Scheme With Respect to Various Channel Noises",
            "iDEAS_RRLO_varied_channel_drop",
            False,
            True,
        ],
    }

    alg_set = ["Random", "RRLO", "iDEAS", "Local", "Edge Only"]
    alg_set2 = ["Random", "RRLO", "Local", "Edge Only"]
    for scenario in plot_infos:
        mean_values = np.mean(all_results[scenario], axis=0)
        if scenario in ["fixed_taskset_energy", "fixed_taskset_drop"]:
            plot_res(
                alg_set,
                mean_values[:, 0],
                mean_values[:, 1],
                "Scheme",
                *plot_infos[scenario],
            )
        else:
            line_plot_res(alg_set, mean_values, *plot_infos[scenario])

    # Calculate improvement:
    energy_vals = all_results["fixed_taskset_energy"]
    random_energy = energy_vals[:, 0, :]
    rrlo_energy = energy_vals[:, 1, :]
    ideas_energy = energy_vals[:, 2, :]
    local_energy = energy_vals[:, 3, :]
    remote_energy = energy_vals[:, 4, :]

    random_improvement = (ideas_energy - random_energy) / random_energy * 100
    rrlo_improvement = (ideas_energy - rrlo_energy) / rrlo_energy * 100
    local_improvement = (ideas_energy - local_energy) / local_energy * 100
    remote_improvement = (ideas_energy - remote_energy) / remote_energy * 100

    taskset_improvement = np.stack(
        [random_improvement, rrlo_improvement, local_improvement, remote_improvement],
        axis=1,
    )
    improvement_eval = np.mean(taskset_improvement, axis=0)
    print("iDEAS energy consumption improvements:")
    print(
        print_improvement(
            alg_set2, improvement_eval[:, 0], improvement_eval[:, 1], 4, 4
        )
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
    configs_ideas_base = load_yaml("./configs/iDEAS_Base.yaml")
    iDEAS_Base(configs_ideas_base)

    configs_ideas_rrlo = load_yaml("./configs/iDEAS_RRLO.yaml")
    iDEAS_RRLO(configs_ideas_rrlo)

    # big_little_main(Train=False)
