import numpy as np
import torch
import matplotlib.pyplot as plt
import os


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def moving_avg(arr, n):
    return np.convolve(arr, np.ones(n, dtype=arr.dtype), "same") / n


def plot_loss_function(losses):
    fig = plt.figure(figsize=(16, 12))
    plt.title("Loss function values")
    plt.plot(losses)
    plt.grid(True)
    fig.savefig("figs/loss_function.png")


def plot_all_rewards(all_rewards):
    fig = plt.figure(figsize=(16, 12))
    plt.title("Reward value")
    plt.plot(moving_avg(all_rewards[:, 0], 500))
    plt.plot(moving_avg(all_rewards[:, 1], 500))
    plt.plot(moving_avg(all_rewards[:, 2], 500))
    plt.plot(moving_avg(all_rewards[:, 3], 500))
    plt.grid(True)
    fig.savefig("figs/all_reward.png")


def plot_penalty(penalty, min_penalty, t_id):
    fig = plt.figure(figsize=(16, 12))
    plt.title(f"Penalties task{t_id}")
    plt.plot(moving_avg(penalty, 100))
    plt.plot(moving_avg(min_penalty, 100))
    plt.grid(True)
    plt.legend(["actual", "min"])
    fig.savefig(f"figs/pen{t_id}.png")


def plot_res(alg_set, taskset1, taskset2, std1,std2, xlabel, ylabel,title, fig_name ):
    # Data for plotting
    #algorithms = ['Local Scheduling', 'RRLO [8]', 'Our Algorithm']

    # Positioning of bars on x-axis
    ind = range(len(alg_set))
    # Plotting both tasksets
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, fig_name + ".png")
    fig = plt.figure(figsize=(16, 12))
    plt.bar(ind, taskset1, width=0.4, label='Taskset1', color='r',yerr=std1,edgecolor='black', capsize=20)
    plt.bar([i + 0.4 for i in ind], taskset2, width=0.4, label='Taskset2', color='b',yerr=std2,edgecolor='black',capsize=20)

    for i, value in enumerate(taskset1):
        plt.text(i, value, f"{value:.3f}", ha='center', va='bottom', fontweight='bold')
    for i, value in enumerate(taskset2):
        plt.text(i+0.4, value, f"{value:.3f}", ha='center', va='bottom', fontweight='bold')
    # Labels and Title
    plt.xlabel(xlabel,fontweight='bold')
    plt.ylabel(ylabel,fontweight='bold')
    plt.title(title,fontweight='bold')
    # X-axis tick labels positioning
    plt.xticks([i + 0.2 for i in ind], alg_set,fontweight='bold')
    # Adding legend to specify which color represents which task set
    plt.legend()
    # Displaying the plot
    plt.grid(True)
    fig.savefig(file_path)
