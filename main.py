"""
Main entry point function to train iDEAS scenarios
and evaluate each algorithm's performance.
"""

import numpy as np
import os

from utils.utils import (
    plot_res,
    print_improvement,
    plot_loss_function,
    line_plot_res,
    stack_bar_res,
    plot_all_rewards,
    plot_loss_and_reward,
    load_yaml,
)

from train.trainer import iDEAS_MainTrainer, iDEAS_RRLO_DRLDOTrainer
from eval.evaluator import (
    iDEAS_MainEvaluator,
    iDEAS_RRLO_DRLDOEvaluator,
    iDEAS_BaselineEvaluator,
)


def iDEAS_Main(configs):
    """
    iDEAS main scenario which evaluates algorithm convergence

    Args:
        configs (dict): Configuration parameters read from yaml file

    Returns:
        None
    """
    params = configs["params"]
    num_eval_cycles = params["eval_cycle"]
    results_dir = "results/ideas_main"

    if params["do_train"]:
        trainer = iDEAS_MainTrainer(configs)
        loss, rewards = trainer.run()
        loss = np.array(loss)
        rewards = np.array([reward["ideas"] for reward in rewards])
        print("Training completed")
        print(100 * "-")
        plot_loss_function(loss, "iDEAS", "iterations", "loss", "iDEAS_Main_loss")
        plot_all_rewards(
            rewards, "iDEAS", "iterations", "rewards", "iDEAS_Main_rewards"
        )
        plot_loss_and_reward(
            loss,
            rewards,
            "iDEAS",
            "iterations",
            "loss",
            "rewards",
            "iDEAS_Main_loss_reward",
        )

        os.makedirs(results_dir, exist_ok=True)
        np.save(f"{results_dir}/loss.npy", loss)
        np.save(f"{results_dir}/reward.npy", rewards)

    loss = (
        np.load(f"{results_dir}/loss.npy")
        if os.path.exists(f"{results_dir}/loss.npy")
        else None
    )
    rewards = (
        np.load(f"{results_dir}/reward.npy")
        if os.path.exists(f"{results_dir}/reward.npy")
        else None
    )
    if loss is not None and rewards is not None:
        plot_loss_and_reward(
            loss,
            rewards,
            "iDEAS",
            "iterations",
            "loss",
            "rewards",
            "iDEAS_Main_loss_reward",
        )

    linspace_values = np.linspace(
        1, params["max_task_load_eval"], 7
    )  # Create the desired array with the first value as min_task_load_eval
    cpuloads = np.insert(linspace_values, 0, params["min_task_load_eval"])
    cpuloads= np.ceil(cpuloads * 2) / 2
    tasksizes = np.round(
        np.linspace(params["min_task_size"], params["max_task_size"], 10)
    )
    cns = np.logspace(
        np.log10(params["min_cn_power"]),
        np.log10(params["max_cn_power"]),
        num=10,
        base=10,
    )

    os.makedirs(results_dir, exist_ok=True)
    np.save(f"{results_dir}/cpuloads.npy", cpuloads)
    np.save(f"{results_dir}/tasksizes.npy", tasksizes)
    np.save(f"{results_dir}/cns.npy", cns)

    all_results = {}
    for i in range(num_eval_cycles):
        evaluator = iDEAS_MainEvaluator(configs, cpuloads, tasksizes, cns)
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
            ["Task Set I", "Task Set II"],
            "Task Sets",
            "Energy Consumption (mJ) ",
            "Energy Consumption Levels on Different Task sets",
            "iDEAS_Main_fixed_taskset_energy",
        ],
        "fixed_taskset_drop": [
            ["Task Set 1", "Task Set 2"],
            "Task Sets",
            "Dropped Tasks (\%) ",
            "Dropped Tasks levels on Different Task sets",
            "iDEAS_Main_fixed_taskset_drop",
        ],
        "varied_cpuload_energy": [
            cpuloads/4.0,
            "Utilization",
            "Energy Consumption (mJ) ",
            "Energy Consumption Levels for Different Utilization",
            "iDEAS_Main_varied_cpuload_energy",
            True,
        ],
        "varied_cpuload_drop": [
            cpuloads/4.0,
            "Utilization",
            "Dropped Tasks (\%) ",
            "Dropped Tasks levels for Different Task Loads",
            "iDEAS_Main_varied_cpuload_drop",
            True,
        ],
        "varied_tasksize_energy": [
            tasksizes[:-1],
            "Task Size (KB)",
            "Energy Consumption (mJ) ",
            "Energy Consumption Levels for Different Task Sizes",
            "iDEAS_Main_varied_tasksize_energy",
            True,
        ],
        "varied_tasksize_drop": [
            tasksizes[:-1],
            "Task Size(KB)",
            "Dropped Tasks (\%) ",
            "Dropped Tasks levels for Different Task Sizes",
            "iDEAS_Main_varied_tasksize_drop",
            True,
        ],
        "varied_channel_energy": [
            cns,
            "Channel Noise",
            "Energy Consumption (mJ) ",
            "Energy Consumption Levels for Different Channel Noises",
            "iDEAS_Main_varied_channel_energy",
            True,
            True,
        ],
        "varied_channel_drop": [
            cns,
            "Channel Noise",
            "Dropped Tasks (\%) ",
            "Dropped Tasks levels for Different Channel Noises",
            "iDEAS_Main_varied_channel_drop",
            True,
            True,
        ],
    }

    # Plot all results
    alg_set = ["big", "LITTLE", "offload", "Total"]
    for scenario in all_results:
        mean_result = np.mean(all_results[scenario], axis=0)
        stack_bar_res(alg_set, mean_result, *plot_infos[scenario])


def iDEAS_RRLO_DRLDO(configs):
    """
    iDEAS comparison with RRLO scenario which evaluates iDEAS performance
    compared to RLLO

    Args:
        configs (dict): Configuration parameters read from yaml file

    Returns:
        None
    """
    params = configs["params"]
    num_eval_cycles = params["eval_cycle"]
    results_dir = "results/ideas_rrlo_drldo"

    if params["do_train"]:
        trainer = iDEAS_RRLO_DRLDOTrainer(configs)
        dqn_loss, rewards = trainer.run()
        dqn_loss = np.array(dqn_loss)
        rewards = np.array([reward["ideas"] for reward in rewards])

        print("Training completed")
        print(100 * "-")

        plot_loss_function(dqn_loss, "iDEAS", "iterations", "loss", "iDEAS_RRLO_loss")
        plot_all_rewards(
            rewards, "iDEAS", "iterations", "rewards", "iDEAS_RRLO_rewards"
        )
        plot_loss_and_reward(
            dqn_loss,
            rewards,
            "iDEAS",
            "iterations",
            "loss",
            "rewards",
            "iDEAS_RRLO_loss_reward",
        )
        os.makedirs(results_dir, exist_ok=True)

        np.save(f"{results_dir}/loss.npy", dqn_loss)
        np.save(f"{results_dir}/reward.npy", rewards)

    # Check if results exist
    loss = (
        np.load(f"{results_dir}/loss.npy")
        if os.path.exists(f"{results_dir}/loss.npy")
        else None
    )
    rewards = (
        np.load(f"{results_dir}/reward.npy")
        if os.path.exists(f"{results_dir}/reward.npy")
        else None
    )
    if loss is not None and rewards is not None:
        plot_loss_and_reward(
            loss,
            rewards,
            "iDEAS",
            "iterations",
            "loss",
            "rewards",
            "iDEAS_RRLO_loss_reward",
        )

    linspace_values = np.linspace(
        1, params["max_task_load_eval"], 7
    )  # Create the desired array with the first value as min_task_load_eval
    cpuloads = np.insert(linspace_values, 0, params["min_task_load_eval"])
    cpuloads= np.ceil(cpuloads * 2) / 2
    tasksizes = np.round(
        np.linspace(params["min_task_size"], params["max_task_size"], 10)
    )
    cns = np.logspace(
        np.log10(params["min_cn_power"]),
        np.log10(params["max_cn_power"]),
        num=10,
        base=10,
    )
    os.makedirs(results_dir, exist_ok=True)
    np.save(f"{results_dir}/cpuloads.npy", cpuloads)
    np.save(f"{results_dir}/tasksizes.npy", tasksizes)
    np.save(f"{results_dir}/cns.npy", cns)
    all_results = {}
    for i in range(num_eval_cycles):
        evaluator = iDEAS_RRLO_DRLDOEvaluator(configs, cpuloads, tasksizes, cns)
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
            "Energy Consumption (mJ)",
            "Energy Consumption of Different Baseline Single Core Schemes",
            "iDEAS_RRLO_DRLDO_fixed_taskset_energy",
            True,
        ],
        "fixed_taskset_drop": [
            "Dropped Tasks (\%) ",
            "Dropped Tasks of Different Baseline Single Core Schemes  ",
            "iDEAS_RRLO_DRLDO_fixed_taskset_drop",
        ],
        "varied_cpuload_energy": [
            cpuloads/4.0,
            "Utilization",
            "Energy Consumption (mJ)",
            "Energy Consumption of Different Single Core Schemes With Respect to Various Utilization",
            "iDEAS_RRLO_DRLDO_varied_cpuload_energy",
            [2, 0, 1],
            True,
            False,
        ],
        "varied_cpuload_drop": [
            cpuloads/4.0,
            "Utilization",
            "Dropped Tasks (\%) ",
            "Dropped Tasks of Different Baseline Single Core Scheme With Respect to Various Utilization",
            "iDEAS_RRLO_DRLDO_varied_cpuload_drop",
            [2, 0, 1],
            False,
            False,
        ],
        "varied_tasksize_energy": [
            tasksizes[:-1],
            "Task Size(KB)",
            "Energy Consumption (mJ)",
            "Energy Consumption of Different Single Core Schemes With Respect to Various Task Sizes",
            "iDEAS_RRLO_DRLDO_varied_tasksize_energy",
            None,
            True,
            False,
        ],
        "varied_tasksize_drop": [
            tasksizes[:-1],
            "Task Size (KB)",
            "Dropped Tasks (\%) ",
            "Dropped Tasks of Different Baseline Single Core Scheme With Respect to Various Task Sizes",
            "iDEAS_RRLO_DRLDO_varied_tasksize_drop",
            None,
            False,
            False,
        ],
        "varied_channel_energy": [
            cns,
            "Channel Noise",
            "Energy Consumption (mJ)",
            "Energy Consumption of Different Single Core Schemes With Respect to Various Channel Noises",
            "iDEAS_RRLO_DRLDO_varied_channel_energy",
            None,
            True,
            True,
        ],
        "varied_channel_drop": [
            cns,
            "Channel Noise",
            "Dropped Tasks (\%) ",
            "Dropped Tasks of Different Baseline Single Core Scheme With Respect to Various Channel Noises",
            "iDEAS_RRLO_DRLDO_varied_channel_drop",
            None,
            False,
            True,
        ],
    }

    alg_set = ["RRLO", "iDEAS", "DRLDO"]
    alg_set2 = ["RRLO", "DRLDO"]
    for scenario in plot_infos:
        if scenario in all_results:
            mean_values = np.mean(all_results[scenario], axis=0)
        else:
            print(f"Scenario {scenario} not found in results")
            continue
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
    if "fixed_taskset_energy" not in all_results:
        print("Scenario fixed_taskset_energy not found in results")
        return
    energy_vals = all_results["fixed_taskset_energy"]
    rrlo_energy = energy_vals[:, 0, :]
    ideas_energy = energy_vals[:, 1, :]
    drldo_energy = energy_vals[:, 2, :]

    rrlo_improvement = (ideas_energy - rrlo_energy) / rrlo_energy * 100
    drldo_improvement = (ideas_energy - drldo_energy) / drldo_energy * 100

    taskset_improvement = np.stack(
        [rrlo_improvement, drldo_improvement],
        axis=1,
    )
    improvement_eval = np.mean(taskset_improvement, axis=0)

    with open("results/RRLO_DRLDO.txt", "w") as file:
        file.write(
            print_improvement(
                alg_set2, improvement_eval[:, 0], improvement_eval[:, 1]
            )
        )
    print("iDEAS energy consumption improvements:")
    print(
        print_improvement(
            alg_set2, improvement_eval[:, 0], improvement_eval[:, 1]
        )
    )


def iDEAS_Baseline(configs):
    """
    iDEAS comparison with baseline algorithms scenario which evaluates iDEAS performance
    compared to these baselines

    Args:
        configs (dict): Configuration parameters read from yaml file

    Returns:
        None
    """
    params = configs["params"]
    num_eval_cycles = params["eval_cycle"]
    results_dir = "results/ideas_baseline"

    if params["do_train"]:
        trainer = iDEAS_MainTrainer(configs)
        loss, rewards = trainer.run()
        loss = np.array(loss)
        rewards = np.array([reward["ideas"] for reward in rewards])

        print("Training completed")
        print(100 * "-")

        plot_loss_function(loss, "iDEAS", "iterations", "loss", "iDEAS_Baseline_loss")
        plot_all_rewards(
            rewards, "iDEAS", "iterations", "rewards", "iDEAS_Baseline_rewards"
        )
        plot_loss_and_reward(
            loss,
            rewards,
            "iDEAS",
            "iterations",
            "loss",
            "rewards",
            "iDEAS_baseline_loss_reward",
        )
        os.makedirs(results_dir, exist_ok=True)
        np.save(f"{results_dir}/loss.npy", loss)
        np.save(f"{results_dir}/reward.npy", rewards)

    linspace_values = np.linspace(
        1, params["max_task_load_eval"], 7
    )  # Create the desired array with the first value as min_task_load_eval
    cpuloads = np.insert(linspace_values, 0, params["min_task_load_eval"])
    cpuloads= np.ceil(cpuloads * 2) / 2
    tasksizes = np.round(
        np.linspace(params["min_task_size"], params["max_task_size"], 10)
    )
    cns = np.logspace(
        np.log10(params["min_cn_power"]),
        np.log10(params["max_cn_power"]),
        num=11,
        base=10,
    )

    os.makedirs(results_dir, exist_ok=True)
    np.save(f"{results_dir}/cpuloads.npy", cpuloads)
    np.save(f"{results_dir}/tasksizes.npy", tasksizes)
    np.save(f"{results_dir}/cns.npy", cns)
    all_results = {}
    for i in range(num_eval_cycles):
        print(f"Cycle {i + 1}:")
        evaluator = iDEAS_BaselineEvaluator(configs, cpuloads, tasksizes, cns)
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
            "Energy Consumption (mJ)",
            "Energy Consumption of Different Baseline big.LITTLE Schemes",
            "iDEAS_Baseline_fixed_taskset_energy",
            True,
        ],
        "fixed_taskset_drop": [
            "Dropped Tasks (\%) ",
            "Dropped Tasks of Different Baseline big.LITTLE Schemes  ",
            "iDEAS_Baseline_fixed_taskset_drop",
        ],
        "varied_cpuload_energy": [
            cpuloads/4.0,
            "Utilization",
            "Energy Consumption (mJ)",
            "Energy Consumption of Different big.LITTLE Schemes With Respect to Various Utilization",
            "iDEAS_Baseline_varied_cpuload_energy",
            [2, 1, 3, 0],
            True,
        ],
        "varied_cpuload_drop": [
            cpuloads/4.0,
            "Utilization",
            "Dropped Tasks (\%) ",
            "Dropped Tasks of Different Baseline big.LITTLE Scheme With Respect to Various Utilization",
            "iDEAS_Baseline_varied_cpuload_drop",
            [1, 3, 0, 2],
        ],
        "varied_tasksize_energy": [
            tasksizes[:-1],
            "Task Size(KB)",
            "Energy Consumption (mJ)",
            "Energy Consumption of Different big.LITTLE Schemes With Respect to Various Task Sizes",
            "iDEAS_Baseline_varied_tasksize_energy",
            [3, 2, 1, 0],
            True,
            False,
        ],
        "varied_tasksize_drop": [
            tasksizes[:-1],
            "Task Size (KB)",
            "Dropped Tasks (\%) ",
            "Dropped Tasks of Different Baseline big.LITTLE Scheme With Respect to Various Task Sizes",
            "iDEAS_Baseline_varied_tasksize_drop",
            [1, 0, 2, 3],
        ],
    }

    alg_set = ["iDEAS", "Random", "Local", "Edge Only"]
    alg_set2 = ["Random", "Local", "Edge Only"]
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
    random_energy = energy_vals[:, 1, :]
    ideas_energy = energy_vals[:, 0, :]
    local_energy = energy_vals[:, 2, :]
    remote_energy = energy_vals[:, 3, :]

    random_improvement = (ideas_energy - random_energy) / random_energy * 100
    local_improvement = (ideas_energy - local_energy) / local_energy * 100
    remote_improvement = (ideas_energy - remote_energy) / remote_energy * 100

    taskset_improvement = np.stack(
        [random_improvement, local_improvement, remote_improvement],
        axis=1,
    )
    improvement_eval = np.mean(taskset_improvement, axis=0)
    with open("results/baseline.txt", "w") as file:
        file.write(
            print_improvement(
                alg_set2, improvement_eval[:, 0], improvement_eval[:, 1], 3, 3
            )
        )
    print("iDEAS energy consumption improvements:")
    print(
        print_improvement(
            alg_set2, improvement_eval[:, 0], improvement_eval[:, 1], 3, 3
        )
    )


if __name__ == "__main__":
    # configs_ideas_main = load_yaml("./configs/iDEAS_Main.yaml")
    # iDEAS_Main(configs_ideas_main)

    configs_ideas_rrlo = load_yaml("./configs/iDEAS_RRLO_DRLDO.yaml")
    iDEAS_RRLO_DRLDO(configs_ideas_rrlo)

    # configs_ideas_baseline = load_yaml("./configs/iDEAS_Baseline.yaml")
    # iDEAS_Baseline(configs_ideas_baseline)
