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

from train.trainer import iDEAS_MainTrainer, iDEAS_RRLOTrainer
from eval.evaluator import (
    iDEAS_MainEvaluator,
    iDEAS_RRLOEvaluator,
    iDEAS_BaselineEvaluator,
)
import json

def iDEAS_Main(configs):

        
    loss = np.load("iDEAS conv/loss.npy")
    rewards=np.load("iDEAS conv/reward.npy")
    plot_loss_and_reward(loss, rewards, "iDEAS", "iterations", "loss", "rewards", "iDEAS_Main_loss_reward")


    cpuloads=np.load("iDEAS_res/cpuloads.npy")
    tasksizes=np.load("iDEAS_res/tasksizes.npy")
    cns=np.load("iDEAS_res/cns.npy")
    #all_results=np.load("iDEAS_res/all_results.npy",allow_pickle=True)
    #with open('iDEAS_res/all_results.json', 'w') as json_file:
     #   json.dump(all_results, json_file, indent=4)
    with open('iDEAS_res/all_results.json', 'r') as json_file:
        all_results = json.load(json_file)


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
            cpuloads,
            "Utilization",
            "Energy Consumption (mJ) ",
            "Energy Consumption Levels for Different Utilization",
            "iDEAS_Main_varied_cpuload_energy",
            True,
        ],
        "varied_cpuload_drop": [
            cpuloads,
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


def iDEAS_RRLO(configs):


    loss = np.load("RRLO conv/loss.npy")
    rewards=np.load("RRLO conv/reward.npy")
    plot_loss_and_reward(loss, rewards, "iDEAS", "iterations", "loss", "rewards", "iDEAS_Main_loss_reward")


    cpuloads=np.load("RRLO_res/cpuloads.npy")
    tasksizes=np.load("RRLO_res/tasksizes.npy")
    cns=np.load("RRLO_res/cns.npy")
    #all_results=np.load("RRLO_res/all_results.npy")
    #with open('RRLO_res/all_results.json', 'w') as json_file:
     #   json.dump(all_results, json_file, indent=4)
    with open('RRLO_res/all_results.json', 'r') as json_file:
        all_results = json.load(json_file)

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
            "Utilization",
            "Energy Consumption (mJ)",
            "Energy Consumption of Different Single Core Schemes With Respect to Various Utilization",
            "iDEAS_RRLO_varied_cpuload_energy",
            None,
            True,
            False,
        ],
        "varied_cpuload_drop": [
            cpuloads,
            "Utilization",
            "Dropped Tasks (\%) ",
            "Dropped Tasks of Different Baseline Single Core Scheme With Respect to Various Utilization",
            "iDEAS_RRLO_varied_cpuload_drop",
            None,
            False,
            False
        ],
        "varied_tasksize_energy": [
            tasksizes[:-1],
            "Task Size(KB)",
            "Energy Consumption (mJ)",
            "Energy Consumption of Different Single Core Schemes With Respect to Various Task Sizes",
            "iDEAS_RRLO_varied_tasksize_energy",
            None,
            True,
            False,
        ],
        "varied_tasksize_drop": [
            tasksizes[:-1],
            "Task Size (KB)",
            "Dropped Tasks (\%) ",
            "Dropped Tasks of Different Baseline Single Core Scheme With Respect to Various Task Sizes",
            "iDEAS_RRLO_varied_tasksize_drop",
            None,
            False,
            False
        ],
        "varied_channel_energy": [
            cns,
            "Channel Noise",
            "Energy Consumption (mJ)",
            "Energy Consumption of Different Single Core Schemes With Respect to Various Channel Noises",
            "iDEAS_RRLO_varied_channel_energy",
            None,
            True,
            True,
        ],
        "varied_channel_drop": [
            cns,
            "Channel Noise",
            "Dropped Tasks (\%) ",
            "Dropped Tasks of Different Baseline Single Core Scheme With Respect to Various Channel Noises",
            "iDEAS_RRLO_varied_channel_drop",
            None,
            False,
            True,
        ],
    }

    alg_set = ["RRLO", "iDEAS"]
    alg_set2 = [ "RRLO"]
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
    #random_energy = energy_vals[:, 0, :]
    rrlo_energy = energy_vals[:, 0, :]
    ideas_energy = energy_vals[:, 1, :]
    #local_energy = energy_vals[:, 3, :]
    #remote_energy = energy_vals[:, 4, :]

    #random_improvement = (ideas_energy - random_energy) / random_energy * 100
    rrlo_improvement = (ideas_energy - rrlo_energy) / rrlo_energy * 100
    #local_improvement = (ideas_energy - local_energy) / local_energy * 100
    #remote_improvement = (ideas_energy - remote_energy) / remote_energy * 100

    taskset_improvement = np.stack(
       # [random_improvement, rrlo_improvement, local_improvement, remote_improvement],
       [rrlo_improvement],
        axis=1,
    )
    improvement_eval = np.mean(taskset_improvement, axis=0)

    with open("results/RRLO.text", 'w') as file:
        file.write(
            print_improvement(
                alg_set2, improvement_eval[:, 0], improvement_eval[:, 1], 1, 1
            )
        )
    print("iDEAS energy consumption improvements:")
    print(
        print_improvement(
            alg_set2, improvement_eval[:, 0], improvement_eval[:, 1], 1, 1
        )
    )


def iDEAS_Baseline(configs):


    loss = np.load("baseline conv/loss.npy")
    rewards=np.load("baseline conv/reward.npy")
    plot_loss_and_reward(loss, rewards, "iDEAS", "iterations", "loss", "rewards", "iDEAS_Main_loss_reward")


    cpuloads=np.load("baseline_res/cpuloads.npy")
    tasksizes=np.load("baseline_res/tasksizes.npy")
    cns=np.load("baseline_res/cns.npy")
    #all_results=np.load("baseline_res/all_results.npy")
    #with open('baseline_res/all_results.json', 'w') as json_file:
     #   json.dump(all_results, json_file, indent=4)
    with open('baseline_res/all_results.json', 'r') as json_file:
        all_results = json.load(json_file)
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
            "Utilization",
            "Energy Consumption (mJ)",
            "Energy Consumption of Different big.LITTLE Schemes With Respect to Various Utilization",
            "iDEAS_Baseline_varied_cpuload_energy",
            [2,1,3,0],
            True,
        ],
        "varied_cpuload_drop": [
            cpuloads,
            "Utilization",
            "Dropped Tasks (\%) ",
            "Dropped Tasks of Different Baseline big.LITTLE Scheme With Respect to Various Utilization",
            "iDEAS_Baseline_varied_cpuload_drop",
            [1,3,0,2],
        ],
        "varied_tasksize_energy": [
            tasksizes[:-1],
            "Task Size(KB)",
            "Energy Consumption (mJ)",
            "Energy Consumption of Different big.LITTLE Schemes With Respect to Various Task Sizes",
            "iDEAS_Baseline_varied_tasksize_energy",
            [3,2,1,0],
            True,
            False,
        ],
        "varied_tasksize_drop": [
            tasksizes[:-1],
            "Task Size (KB)",
            "Dropped Tasks (\%) ",
            "Dropped Tasks of Different Baseline big.LITTLE Scheme With Respect to Various Task Sizes",
            "iDEAS_Baseline_varied_tasksize_drop",
            [1,0,2,3],
        ],
        #"varied_channel_energy": [
         #   cns,
          #  "Channel Noise",
           # "Energy Consumption (mJ)",
            #"Energy Consumption of Different big.LITTLE Schemes With Respect to Various Channel Noises",
            #"iDEAS_Baseline_varied_channel_energy",
            #True,
            #True,
        #],
        #"varied_channel_drop": [
         #   cns,
          #  "Channel Noise",
           # "Dropped Tasks (\%) ",
           # "Dropped Tasks of Different Baseline big.LITTLE Scheme With Respect to Various Channel Noises",
           # "iDEAS_Baseline_varied_channel_drop",
            #False,
            #True,
        #],
    }

    alg_set = ["iDEAS","Random", "Local", "Edge Only"]
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
    with open("results/baseline.txt", 'w') as file: 
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
    configs_ideas_main = load_yaml("./configs/iDEAS_Main.yaml")
    iDEAS_Main(configs_ideas_main)

    #configs_ideas_rrlo = load_yaml("./configs/iDEAS_RRLO.yaml")
    #iDEAS_RRLO(configs_ideas_rrlo)

    configs_ideas_baseline = load_yaml("./configs/iDEAS_Baseline.yaml")
    iDEAS_Baseline(configs_ideas_baseline)
