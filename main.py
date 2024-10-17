import numpy as np

from utils.utils import (
    plot_res,
    print_improvement,
    plot_loss_function,
    line_plot_res,
    stack_bar_res,
    plot_all_rewards,
    load_yaml,
)

from train.trainer import iDEAS_MainTrainer, iDEAS_RRLOTrainer
from eval.evaluator import (
    iDEAS_MainEvaluator,
    iDEAS_RRLOEvaluator,
    iDEAS_BaselineEvaluator,
)


def iDEAS_Main(configs):
    params = configs["params"]
    num_eval_cycles = params["eval_cycle"]

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

    cpuloads = np.linspace(
        params["min_task_load_eval"], params["max_task_load_eval"], 10
    )
    tasksizes = np.round(
        np.linspace(params["min_task_size"], params["max_task_size"], 10)
    )
    cns = np.logspace(
        np.log10(params["min_cn_power"]),
        np.log10(params["max_cn_power"]),
        num=10,
        base=10,
    )

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
            ["Task set 1", "Task set 2"],
            "Task sets",
            "Consumed Energy (mJ) ",
            "Energy Consumption Levels on Different Task sets",
            "iDEAS_Main_fixed_taskset_energy",
        ],
        "fixed_taskset_drop": [
            ["Task set 1", "Task set 2"],
            "Task sets",
            "Dropped Tasks (\%) ",
            "Dropped Tasks levels on Different Task sets",
            "iDEAS_Main_fixed_taskset_drop",
        ],
        "varied_cpuload_energy": [
            cpuloads,
            "Task Load",
            "Consumed Energy (mJ) ",
            "Energy Consumption Levels for Different Task Loads",
            "iDEAS_Main_varied_cpuload_energy",
            True,
        ],
        "varied_cpuload_drop": [
            cpuloads,
            "Task Load",
            "Dropped Tasks (\%) ",
            "Dropped Tasks levels for Different Task Loads",
            "iDEAS_Main_varied_cpuload_drop",
            True,
        ],
        "varied_tasksize_energy": [
            tasksizes[:-1],
            "Task Size (KB)",
            "Consumed Energy (mJ) ",
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
            "Consumed Energy (mJ) ",
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


def iDEAS_RRLO(configs):
    params = configs["params"]
    num_eval_cycles = params["eval_cycle"]

    if params["do_train"]:
        trainer = iDEAS_RRLOTrainer(configs)
        dqn_loss, rewards = trainer.run()
        dqn_loss = np.array(dqn_loss)
        rewards = np.array([reward["ideas"] for reward in rewards])

        print("Training completed")
        print(100 * "-")

        plot_loss_function(dqn_loss, "iDEAS", "iterations", "loss", "iDEAS_RRLO_loss")
        plot_all_rewards(
            rewards, "iDEAS", "iterations", "rewards", "iDEAS_RRLO_rewards"
        )

    cpuloads = np.linspace(
        params["min_task_load_eval"], params["max_task_load_eval"], 10
    )
    tasksizes = np.round(
        np.linspace(params["min_task_size"], params["max_task_size"], 10)
    )
    cns = np.logspace(
        np.log10(params["min_cn_power"]),
        np.log10(params["max_cn_power"]),
        num=10,
        base=10,
    )

    all_results = {}
    for i in range(num_eval_cycles):
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
            "Energy Consumption (mJ)",
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
            "Energy Consumption (mJ)",
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
            "Energy Consumption (mJ)",
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
            "Energy Consumption (mJ)",
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


def iDEAS_Baseline(configs):
    params = configs["params"]
    num_eval_cycles = params["eval_cycle"]

    if params["do_train"]:
        trainer = iDEAS_MainTrainer(configs)
        loss, rewards = trainer.run()
        loss = np.array(loss)
        rewards = np.array([reward["ideas"] for reward in rewards])

        print("Training completed")
        print(100 * "-")

        plot_loss_function(
            loss, "iDEAS", "iterations", "loss", "iDEAS_Baseline_loss"
        )
        plot_all_rewards(
            rewards, "iDEAS", "iterations", "rewards", "iDEAS_Baseline_rewards"
        )

    cpuloads = np.linspace(
        params["min_task_load_eval"], params["max_task_load_eval"], 10
    )
    tasksizes = np.round(
        np.linspace(params["min_task_size"], params["max_task_size"], 10)
    )
    cns = np.logspace(
        np.log10(params["min_cn_power"]),
        np.log10(params["max_cn_power"]),
        num=11,
        base=10,
    )

    all_results = {}
    for i in range(num_eval_cycles):
        print(f"Cycle {i+1}:")
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
            cpuloads,
            "Task Load",
            "Energy Consumption (mJ)",
            "Energy Consumption of Different big.LITTLE Schemes With Respect to Various Task Loads",
            "iDEAS_Baseline_varied_cpuload_energy",
            True,
        ],
        "varied_cpuload_drop": [
            cpuloads,
            "Task Load",
            "Dropped Tasks (\%) ",
            "Dropped Tasks of Different Baseline big.LITTLE Scheme With Respect to Various Task Loads",
            "iDEAS_Baseline_varied_cpuload_drop",
        ],
        "varied_tasksize_energy": [
            tasksizes[:-1],
            "Task Size(KB)",
            "Energy Consumption (mJ)",
            "Energy Consumption of Different big.LITTLE Schemes With Respect to Various Task Sizes",
            "iDEAS_Baseline_varied_tasksize_energy",
            True,
            False,
        ],
        "varied_tasksize_drop": [
            tasksizes[:-1],
            "Task Size (KB)",
            "Dropped Tasks (\%) ",
            "Dropped Tasks of Different Baseline big.LITTLE Scheme With Respect to Various Task Sizes",
            "iDEAS_Baseline_varied_tasksize_drop",
        ],
        "varied_channel_energy": [
            cns,
            "Channel Noise",
            "Energy Consumption (mJ)",
            "Energy Consumption of Different big.LITTLE Schemes With Respect to Various Channel Noises",
            "iDEAS_Baseline_varied_channel_energy",
            True,
            True,
        ],
        "varied_channel_drop": [
            cns,
            "Channel Noise",
            "Dropped Tasks (\%) ",
            "Dropped Tasks of Different Baseline big.LITTLE Scheme With Respect to Various Channel Noises",
            "iDEAS_Baseline_varied_channel_drop",
            False,
            True,
        ],
    }

    alg_set = ["Random", "iDEAS", "Local", "Edge Only"]
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
    random_energy = energy_vals[:, 0, :]
    ideas_energy = energy_vals[:, 1, :]
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
    print("iDEAS energy consumption improvements:")
    print(
        print_improvement(
            alg_set2, improvement_eval[:, 0], improvement_eval[:, 1], 3, 3
        )
    )


if __name__ == "__main__":
    configs_ideas_main = load_yaml("./configs/iDEAS_Main.yaml")
    iDEAS_Main(configs_ideas_main)

    configs_ideas_rrlo = load_yaml("./configs/iDEAS_RRLO.yaml")
    iDEAS_RRLO(configs_ideas_rrlo)

    configs_ideas_baseline = load_yaml("./configs/iDEAS_Baseline.yaml")
    iDEAS_Baseline(configs_ideas_baseline)
