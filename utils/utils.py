import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from tabulate import tabulate
from matplotlib.colors import ListedColormap


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def moving_avg(arr, n):
    return np.convolve(arr, np.ones(n)/n, "same")


def plot_loss_function(losses, alg, xlabel, ylabel,fig_name):
    plt.rcParams['font.size'] = 20
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, fig_name + ".png")
    fig = plt.figure(figsize=(16, 12))
    plt.title(f"{alg} Loss function values")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(losses)
    plt.tight_layout()
    plt.grid(True)
    fig.savefig(file_path)


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


def plot_res(alg_set, taskset1, taskset2, std1,std2,  xlabel, ylabel,title, fig_name, ylog=False ):
    # Data for plotting
    #algorithms = ['Local Scheduling', 'RRLO [8]', 'Our Algorithm']

    # Positioning of bars on x-axis
    plt.rcParams['font.size'] = 20
    ind = range(len(alg_set))
    # Plotting both tasksets
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, fig_name + ".png")
    fig = plt.figure(figsize=(16, 12))
    plt.bar(ind, taskset1, width=0.4, label='Taskset1', color='#17becf',yerr=std1,edgecolor='black', capsize=20)
    plt.bar([i + 0.4 for i in ind], taskset2, width=0.4, label='Taskset2', color='#ffbb78',yerr=std2,edgecolor='black',capsize=20)
    if ylog:
        plt.yscale('log')

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
    legend=plt.legend()
    for text in legend.get_texts():
        text.set_fontweight('bold')  # Set legend text to bold
    plt.tight_layout()
    # Displaying the plot
    plt.grid(True)
    fig.savefig(file_path)

def line_plot_res(alg_set,data1,y_val , xlabel, ylabel,title, fig_name, ylog=False ):
    # Data for plotting
    #algorithms = ['Local Scheduling', 'RRLO [8]', 'Our Algorithm']

    # Positioning of bars on x-axis
    plt.rcParams['font.size'] = 16
    ind = range(len(alg_set))
    colors = ListedColormap(
        [
            '#ffbb78',  # Light orange
            '#17becf',  # Cyan
            '#2ca02c',  # Green
            '#d62728',  # Red
            '#1f77b4',  # Blue
            '#9467bd',  # Purple
            '#ff7f0e',  # Orange
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Gray
            '#bcbd22',  # Yellow-green
            '#aec7e8'   # Light blue
        ]
    ).colors

    markers = ['o', 's', 'D', '^', 'v', '<', '>', '*', '+', 'x', 'p', 'H']
    
    # Plotting both tasksets
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, fig_name + ".png")
    fig = plt.figure(figsize=(16, 12))
    if ylog:
        plt.yscale('log')
    for i in range(data1.shape[0]):
        plt.plot(y_val, data1[i], label=alg_set[i], color=colors[i],linewidth=5, marker=markers[i],markersize=10)


    # Labels and Title
    plt.xlabel(xlabel,fontweight='bold')
    plt.ylabel(ylabel,fontweight='bold')
    plt.title(title,fontweight='bold')
    # X-axis tick labels positioning
    #plt.xticks([i + 0.2 for i in ind], alg_set,fontweight='bold')
    # Adding legend to specify which color represents which task set
    legend=plt.legend()
    for text in legend.get_texts():
        text.set_fontweight('bold')  # Set legend text to bold
    plt.tight_layout()
    # Displaying the plot
    plt.grid(True)
    fig.savefig(file_path)

def print_improvement(alg_set,improvements_task1, improvements_task2,num1,num2):
    #algorithms = ['Random', 'Local', 'Remote', 'RRLO']
    
    if len(improvements_task1) != num1 or len(improvements_task2) != num2:
        raise ValueError(f" input 1 array should contain exactly {num1} elements and input 2 array should contain exactly {num2} elements.")
    
    # Prepare data for the table
    table_data = [
        [alg, f"{imp1:.2f}%", f"{imp2:.2f}%"] 
        for alg, imp1, imp2 in zip(alg_set, improvements_task1, improvements_task2)
    ]
    
    
    # Create the table
    table = tabulate(table_data, 
                     headers=['Algorithm', 'DQN Improvement (Task Set 1)', 'DQN Improvement (Task Set 2)'],
                     tablefmt='grid')
    
    return table
