import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import yaml
from tabulate import tabulate
from matplotlib.colors import ListedColormap


def load_yaml(path):
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"YAML file not found at {path}")
    return data


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def moving_avg(arr, n):
    return np.convolve(arr, np.ones(n) / n, "same")


def plot_loss_function(losses, alg, xlabel, ylabel, fig_name):
    plt.rcParams["font.size"] = 30
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["text.usetex"] = True
    os.makedirs("results", exist_ok=True)
    file_path = f"results/{fig_name}.png"

    colors = ListedColormap(
        [
            "#ffbb78",  # Light orange
            "#17becf",  # Cyan
            "#2ca02c",  # Green
            "#d62728",  # Red
            "#1f77b4",  # Blue
            "#9467bd",  # Purple
            "#ff7f0e",  # Orange
            "#8c564b",  # Brown
            "#e377c2",  # Pink
            "#7f7f7f",  # Gray
            "#bcbd22",  # Yellow-green
            "#aec7e8",  # Light blue
        ]
    ).colors

    fig = plt.figure(figsize=(20, 12))
    plt.title(rf"{alg} Loss function values")
    plt.xlabel(rf"{xlabel}")
    plt.ylabel(rf"{ylabel}")
    plt.plot(moving_avg(losses, 10000), label=rf"Loss", color=colors[4], linewidth=5)
    # x0,x1 = plt.xlim()
    # visible= [t for t in plt.xticks() if t>=x0 and t<= x1]
    # plt.xticks(visible,list(map(str,visible)))
    # y0,y1 = plt.ylim()
    # visible= [t for t in plt.yticks() if t>=y0 and t<= y1]
    # plt.yticks(visible,list(map(str,visible)))
    plt.tight_layout()
    plt.grid(True)
    fig.savefig(file_path)


def plot_all_rewards(all_rewards, alg, xlabel, ylabel, fig_name):
    plt.rcParams["font.size"] = 30
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["text.usetex"] = True
    os.makedirs("results", exist_ok=True)
    file_path = f"results/{fig_name}.png"

    colors = ListedColormap(
        [
            "#ffbb78",  # Light orange
            "#17becf",  # Cyan
            "#2ca02c",  # Green
            "#d62728",  # Red
            "#1f77b4",  # Blue
            "#9467bd",  # Purple
            "#ff7f0e",  # Orange
            "#8c564b",  # Brown
            "#e377c2",  # Pink
            "#7f7f7f",  # Gray
            "#bcbd22",  # Yellow-green
            "#aec7e8",  # Light blue
        ]
    ).colors

    mean_all_rewards = np.mean(all_rewards[:, :4], axis=1)

    fig = plt.figure(figsize=(20, 12))
    plt.title(rf"{alg} Reward value")
    plt.plot(
        moving_avg(mean_all_rewards, 10000),
        label=rf"Rewards",
        color=colors[3],
        linewidth=5,
    )
    # plt.plot(moving_avg(all_rewards[:, 0], 1000))
    # plt.plot(moving_avg(all_rewards[:, 1], 1000))
    # plt.plot(moving_avg(all_rewards[:, 2], 1000))
    # plt.plot(moving_avg(all_rewards[:, 3], 1000))
    plt.xlabel(rf"{xlabel}")
    plt.ylabel(rf"{ylabel}")
    # x0,x1 = plt.xlim()
    # visible= [t for t in plt.xticks() if t>=x0 and t<= x1]
    # plt.xticks(visible,list(map(str,visible)))
    # y0,y1 = plt.ylim()
    # visible= [t for t in plt.yticks() if t>=y0 and t<= y1]
    # plt.yticks(visible,list(map(str,visible)))
    plt.tight_layout()
    plt.grid(True)
    fig.savefig(file_path)



def plot_loss_and_reward(losses, rewards, alg, xlabel, ylabel1, ylabel2, fig_name):
    plt.rcParams["font.size"] = 30
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["text.usetex"] = True
    os.makedirs("results", exist_ok=True)
    file_path = f"results/{fig_name}.png"

    mean_all_rewards = np.mean(rewards[:, :4], axis=1)

    colors = ListedColormap(
        [
            "#ffbb78",  # Light orange
            "#17becf",  # Cyan
            "#2ca02c",  # Green
            "#d62728",  # Red
            "#1f77b4",  # Blue
            "#9467bd",  # Purple
            "#ff7f0e",  # Orange
            "#8c564b",  # Brown
            "#e377c2",  # Pink
            "#7f7f7f",  # Gray
            "#bcbd22",  # Yellow-green
            "#aec7e8",  # Light blue
        ]
    ).colors

    fig, ax1 = plt.subplots(figsize=(20, 12))
    plt.title(rf"{alg} Loss and Reward values")

    ax1.set_xlabel(rf"{xlabel}")
    ax1.set_ylabel(rf"{ylabel1}", color=colors[4])
    ax1.plot(moving_avg(losses, 10000)[:-10000], label=rf"Loss", color=colors[4], linewidth=5)
    ax1.tick_params(axis='y', labelcolor=colors[4])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel(rf"{ylabel2}", color=colors[3])  # we already handled the x-label with ax1
    ax2.plot(moving_avg(mean_all_rewards, 10000)[:-10000], label=rf"Rewards", color=colors[3], linewidth=5)
    ax2.tick_params(axis='y', labelcolor=colors[3])

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.grid(True)
    fig.savefig(file_path)
    #plt.show()



def plot_penalty(penalty, min_penalty, t_id):
    fig = plt.figure(figsize=(20, 12))
    plt.title(f"Penalties task{t_id}")
    plt.plot(moving_avg(penalty, 100))
    plt.plot(moving_avg(min_penalty, 100))
    plt.grid(True)
    plt.legend(["actual", "min"])
    fig.savefig(f"figs/pen{t_id}.png")


def plot_res(alg_set, taskset1, taskset2, xlabel, ylabel, title, fig_name, ylog=False):
    # Data for plotting
    # algorithms = ['Local Scheduling', 'RRLO [8]', 'Our Algorithm']

    # Positioning of bars on x-axis
    plt.rcParams["font.size"] = 30
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["text.usetex"] = True
    ind = range(len(alg_set))
    # Plotting both tasksets
    os.makedirs("results", exist_ok=True)
    file_path = f"results/{fig_name}.png"
    fig = plt.figure(figsize=(20, 12))
    plt.bar(ind, taskset1, width=0.4, label=r"Taskset1", color="#17becf")
    plt.bar(
        [i + 0.4 for i in ind], taskset2, width=0.4, label=r"Taskset2", color="#ffbb78"
    )
    if ylog:
        plt.yscale("log")

    for i, value in enumerate(taskset1):
        plt.text(i, value, rf"{value:.3f}", ha="center", va="bottom")
    for i, value in enumerate(taskset2):
        plt.text(i + 0.4, value, rf"{value:.3f}", ha="center", va="bottom")
    # Labels and Title
    plt.xlabel(rf"{xlabel}")
    plt.ylabel(rf"{ylabel}")
    plt.title(rf"{title}")
    # X-axis tick labels positioning
    plt.xticks([i + 0.2 for i in ind], alg_set)

    # y0,y1 = plt.ylim()
    # visible= [t for t in plt.yticks() if t>=y0 and t<= y1]
    # plt.yticks(visible,list(map(str,visible)))
    # Adding legend to specify which color represents which task set
    legend = plt.legend()
    # for text in legend.get_texts():
    #   text.set_fontweight('bold')  # Set legend text to bold
    plt.tight_layout()
    # Displaying the plot
    plt.grid(True)
    fig.savefig(file_path)


def line_plot_res(
    alg_set, data1, y_val, xlabel, ylabel, title, fig_name, ylog=False, xlog=False
):
    # Data for plotting
    # algorithms = ['Local Scheduling', 'RRLO [8]', 'Our Algorithm']

    # Positioning of bars on x-axis
    plt.rcParams["font.size"] = 30
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["text.usetex"] = True
    ind = range(len(alg_set))
    colors = ListedColormap(
        [
            "#ffbb78",  # Light orange
            "#17becf",  # Cyan
            "#2ca02c",  # Green
            "#d62728",  # Red
            "#1f77b4",  # Blue
            "#9467bd",  # Purple
            "#ff7f0e",  # Orange
            "#8c564b",  # Brown
            "#e377c2",  # Pink
            "#7f7f7f",  # Gray
            "#bcbd22",  # Yellow-green
            "#aec7e8",  # Light blue
        ]
    ).colors

    markers = ["o", "s", "D", "^", "v", "<", ">", "*", "+", "x", "p", "H"]

    # Plotting both tasksets
    os.makedirs("results", exist_ok=True)
    file_path = f"results/{fig_name}.png"
    fig = plt.figure(figsize=(20, 12))
    if ylog:
        plt.yscale("log")
    if xlog:
        plt.xscale("log")
    for i in range(data1.shape[0]):
        plt.plot(
            y_val,
            data1[i],
            label=rf"{alg_set[i]}",
            color=colors[i],
            linewidth=5,
            marker=markers[i],
            markersize=20,
        )

    # Labels and Title
    plt.xlabel(rf"{xlabel}")
    plt.ylabel(rf"{ylabel}")
    plt.title(rf"{title}")

    # x0,x1 = plt.xlim()
    # visible= [t for t in plt.xticks() if t>=x0 and t<= x1]
    # plt.xticks(visible,list(map(str,visible)))
    # y0,y1 = plt.ylim()
    # visible= [t for t in plt.yticks() if t>=y0 and t<= y1]
    # plt.yticks(visible,list(map(str,visible)))
    # X-axis tick labels positioning
    # plt.xticks([i + 0.2 for i in ind], alg_set,fontweight='bold')
    # Adding legend to specify which color represents which task set
    legend = plt.legend()
    # for text in legend.get_texts():
    #   text.set_fontweight('bold')  # Set legend text to bold
    plt.tight_layout()
    # Displaying the plot
    plt.grid(True)
    fig.savefig(file_path)


def stack_bar_res(
    labels,
    data1,
    x_val,
    xlabel,
    ylabel,
    title,
    fig_name,
    numbered=False,
    xlog=False,
    ylog=False,
):
    # Data for plotting

    # Positioning of bars on x-axis
    plt.rcParams["font.size"] = 30
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["text.usetex"] = True

    colors = ListedColormap(
        [
            "#ffbb78",  # Light orange
            "#17becf",  # Cyan
            "#2ca02c",  # Green
            "#d62728",  # Red
            "#1f77b4",  # Blue
            "#9467bd",  # Purple
            "#ff7f0e",  # Orange
            "#8c564b",  # Brown
            "#e377c2",  # Pink
            "#7f7f7f",  # Gray
            "#bcbd22",  # Yellow-green
            "#aec7e8",  # Light blue
        ]
    ).colors

    ind = range(len(x_val))
    if numbered:
        step_sizes = np.diff(x_val)
        if xlog:
            bar_widths = step_sizes * 0.4
            bar_widths=np.append(bar_widths, bar_widths[-1])
        else:
            bar_widths = step_sizes * 0.7
            bar_widths=np.append(bar_widths, bar_widths[-1])
    else:
        bar_widths = 0.5
    # Plotting both tasksets
    os.makedirs("results", exist_ok=True)
    file_path = f"results/{fig_name}.png"
    fig, ax = plt.subplots(figsize=(20, 12))

    max_hight = np.max(data1[-1])

    if ylog:
        ax.set_yscale("log")
    if xlog:
        ax.set_xscale("log")

    bottom = np.zeros(data1.shape[1])  # Initialize the bottom array to zeros

    for values, color, label in zip(
        [data1[i] for i in range(data1.shape[0]-1)],
        colors[: data1.shape[0] - 1],
        labels[:-1],
    ):
        if numbered:
            ax.bar(
                x_val,
                values,
                width=bar_widths,
                bottom=bottom,
                label=rf"{label}",
                color=color,
            )
            bottom += values  # Update the bottom for the next stack

        else:
            ax.bar(
                x_val,
                values,
                width=bar_widths,
                bottom=bottom,
                label=rf"{label}",
                color=color,
            )
            bottom += values  # Update the bottom for the next stack

    # Labels and Title

    # ax.margins(y=1)
    if numbered:
        for x, value in zip(x_val, data1[-1]):
            ax.text(x, value, rf"{value:.3f}", ha="center", va="bottom")
    else:
        for i, value in enumerate(data1[-1]):
            ax.text(i, value, rf"{value:.3f}", ha="center", va="bottom")
    ax.set_xlabel(rf"{xlabel}")
    ax.set_ylabel(rf"{ylabel}")
    ax.set_title(rf"{title}")

    if numbered:
        ax.set_xticks(x_val)
        #ax.set_xticklabels([rf"{x:.2f}" for x in x_val])
        x0, x1 = ax.get_xlim()
        visible = [t for t in ax.get_xticks() if t >= x0 and t <= x1]
        ax.set_xticks(np.round(visible,2), list(map(str, np.round(visible,2))))
        y0, y1 = ax.get_ylim()
        visible = [t for t in ax.get_yticks() if t >= y0 and t <= y1]
        ax.set_yticks(np.round(visible,2), list(map(str, np.round(visible,2))))
    else:
        ax.set_xticks(ind, x_val)
        y0, y1 = ax.get_ylim()
        visible = [t for t in ax.get_yticks() if t >= y0 and t <= y1]
        ax.set_yticks(visible, list(map(str, visible)))

    legend = ax.legend()
    # for text in legend.get_texts():
    #   text.set_fontweight('bold')  # Set legend text to bold

    # ax.tick_params(axis='x', labelsize=30, labelcolor='black')
    # ax.tick_params(axis='y', labelsize=30, labelcolor='black')
    ax.set_ylim(0, max_hight * 1.1)
    # fig.tight_layout()
    # Displaying the plot
    ax.grid(True)
    fig.savefig(file_path)


def print_improvement(alg_set, improvements_task1, improvements_task2, num1, num2):
    # algorithms = ['Random', 'Local', 'Remote', 'RRLO']

    if len(improvements_task1) != num1 or len(improvements_task2) != num2:
        raise ValueError(
            f" input 1 array should contain exactly {num1} elements and input 2 array should contain exactly {num2} elements."
        )

    # Prepare data for the table
    table_data = [
        [alg, f"{imp1:.2f}%", f"{imp2:.2f}%"]
        for alg, imp1, imp2 in zip(alg_set, improvements_task1, improvements_task2)
    ]

    # Create the table
    table = tabulate(
        table_data,
        headers=[
            "Algorithm",
            "DQN Improvement (Task Set 1)",
            "DQN Improvement (Task Set 2)",
        ],
        tablefmt="grid",
    )

    return table
