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
    plt.rcParams["font.size"] = 36
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["text.usetex"] = True
    os.makedirs("results", exist_ok=True)
    file_path = f"results/{fig_name}.png"
    file_path1 = f"results/{fig_name}.pdf"
    file_path2 = f"results/{fig_name}.eps"

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
    plt.xlabel(rf"{xlabel}")
    plt.ylabel(rf"{ylabel}")
    plt.plot(moving_avg(losses, 10000), label=r"Loss", color=colors[4], linewidth=5)
    plt.tight_layout()
    plt.grid(True)
    fig.savefig(file_path)
    fig.savefig(file_path1, format="pdf")
    fig.savefig(file_path2, format="eps")


def plot_all_rewards(all_rewards, alg, xlabel, ylabel, fig_name):
    plt.rcParams["font.size"] = 36
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["text.usetex"] = True
    os.makedirs("results", exist_ok=True)
    file_path = f"results/{fig_name}.png"
    file_path1 = f"results/{fig_name}.pdf"
    file_path2 = f"results/{fig_name}.eps"

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
    # plt.title(rf"{alg} Reward value")
    plt.plot(
        moving_avg(mean_all_rewards, 10000),
        label=r"Rewards",
        color=colors[3],
        linewidth=5,
    )
    plt.xlabel(rf"{xlabel}")
    plt.ylabel(rf"{ylabel}")
    plt.tight_layout()
    plt.grid(True)
    fig.savefig(file_path)
    fig.savefig(file_path1, format="pdf")
    fig.savefig(file_path2, format="eps")

def plot_loss_and_reward(losses, rewards, alg, xlabel, ylabel1, ylabel2, fig_name):
    plt.rcParams["font.size"] = 36
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["text.usetex"] = True
    os.makedirs("results", exist_ok=True)
    file_path = f"results/{fig_name}.png"
    file_path1 = f"results/{fig_name}.pdf"
    file_path2 = f"results/{fig_name}.eps"

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

    ax1.set_xlabel(rf"{xlabel}")
    ax1.set_ylabel(rf"{ylabel1}", color=colors[4])
    (line1,) = ax1.plot(
        moving_avg(losses, 10000)[:-10000], label=r"Loss", color=colors[4], linewidth=5
    )
    ax1.tick_params(axis="y", labelcolor=colors[4])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel(
        rf"{ylabel2}", color=colors[3]
    )  # we already handled the x-label with ax1
    (line2,) = ax2.plot(
        moving_avg(mean_all_rewards, 10000)[:-10000],
        label=r"Rewards",
        color=colors[3],
        linewidth=5,
    )
    ax2.tick_params(axis="y", labelcolor=colors[3])
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.grid(True)
    fig.savefig(file_path)
    fig.savefig(file_path1, format="pdf")
    fig.savefig(file_path2, format="eps")


def plot_res(alg_set, taskset1, taskset2, xlabel, ylabel, title, fig_name, ylog=False):
    # Data for plotting

    # Positioning of bars on x-axis
    plt.rcParams["font.size"] = 36
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["text.usetex"] = True
    ind = range(len(alg_set))
    # Plotting both tasksets
    os.makedirs("results", exist_ok=True)
    file_path = f"results/{fig_name}.png"
    file_path1 = f"results/{fig_name}.pdf"
    file_path2 = f"results/{fig_name}.eps"
    fig = plt.figure(figsize=(20, 12))
    plt.bar(ind, taskset1, width=0.5 * 0.4, label=r"Task set I", color="#17becf")
    plt.bar(
        [i + 0.4 for i in ind],
        taskset2,
        width=0.5 * 0.4,
        label=r"Task set II",
        color="#ffbb78",
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
    # X-axis tick labels positioning
    plt.xticks([i + 0.2 for i in ind], alg_set)

    _ = plt.legend()
    plt.tight_layout()
    plt.grid(True)
    fig.savefig(file_path)
    fig.savefig(file_path1, format="pdf")
    fig.savefig(file_path2, format="eps")


def line_plot_res(
    alg_set,
    data1,
    y_val,
    xlabel,
    ylabel,
    title,
    fig_name,
    legend_order=None,
    ylog=False,
    xlog=False,
):
    # Data for plotting

    # Positioning of bars on x-axis
    plt.rcParams["font.size"] = 36
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["text.usetex"] = True
    _ = range(len(alg_set))
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
    file_path1 = f"results/{fig_name}.pdf"
    file_path2 = f"results/{fig_name}.eps"
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

    if legend_order is None:
        _ = plt.legend()
    else:
        handles, labels = (
            plt.gca().get_legend_handles_labels()
        )  # Specify the order you want for the legend
        plt.legend(
            [handles[idx] for idx in legend_order],
            [labels[idx] for idx in legend_order],
        )

    plt.tight_layout()
    # Displaying the plot
    plt.grid(True)
    fig.savefig(file_path)
    fig.savefig(file_path1, format="pdf")
    fig.savefig(file_path2, format="eps")


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
    plt.rcParams["font.size"] = 36
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

    _ = range(len(x_val))
    if numbered:
        step_sizes = np.diff(x_val)
        if xlog:
            bar_widths = step_sizes * 0.4 * 0.5
            bar_widths = np.append(bar_widths, bar_widths[-1])
        else:
            bar_widths = step_sizes * 0.7 * 0.5
            bar_widths = np.append(bar_widths, bar_widths[-1])
    else:
        bar_widths = 0.4
    # Plotting both tasksets
    os.makedirs("results", exist_ok=True)
    file_path = f"results/{fig_name}.png"
    file_path1 = f"results/{fig_name}.pdf"
    file_path2 = f"results/{fig_name}.eps"
    fig, ax = plt.subplots(figsize=(20, 12))

    max_hight = np.max(data1[-1])

    if ylog:
        ax.set_yscale("log")
    if xlog:
        ax.set_xscale("log")

    bottom = np.zeros(data1.shape[1])  # Initialize the bottom array to zeros

    for values, color, label in zip(
        [data1[i] for i in range(data1.shape[0] - 1)],
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

#    print(len(x_val))
#    index = np.arange(len(x_val))
#    print(index)

    ax.set_xticks(x_val)  # Ensure ticks are at the bar positions
    ax.set_xticklabels(x_val,ha='center')
    _ = ax.legend()

    ax.set_ylim(0, max_hight * 1.1)
    # Displaying the plot
    ax.grid(True)
    fig.savefig(file_path)
    fig.savefig(file_path1, format="pdf")
    fig.savefig(file_path2, format="eps")


def print_improvement(alg_set, improvements_task1, improvements_task2, num1, num2):
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
