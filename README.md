# Implementation of DVFS article
This repository contains the implementation of "iDEAS: Intelligent DVFS for Energy-Efficient Task Scheduling in Mobile Devices with big.LITTLE Computing Architecture" paper. This paper proposes a joint DVFS and offloading algorithm designed for heterogenous big.LITTLE computing architecture. The proposed algorithm outperforms several baselines and RRLO algorithm in the literature in multiple test scenarios.

## How to Run?
First you need to create a virtual environment.

**Requirements:**
- Python 3.12.8
- TexLive (for plotting purposes. disable `usetex` in the code if you don't want to use LaTex)

If you want to run the code on CPU and don't have a GPU, use `requirements-cpu.txt` to create your virtual environment. We have used `venv` for environment creation.

After creating your environment, you can train and evaluated your pretrained models. Most configurations sed in the code are in `configs` directory. In total, we have 3 scenarios as below:
- `iDEAS_Baseline`: This scenario compares the performance of `iDEAS` algorithm with baseline methods like random, local only, and edge only execution.
- `iDEAS_Main`: This scenario evaluated `iDEAS` convergence properties.
- `iDEAS_RRLO`: As its name suggests, `iDEAS` is compared to `RRLO` in this sceanrio.

In each of this scenario, you may train the models from scratch or use the pretrained weights stored under `models/<scenario_name>` directory. You can configure this behaviour by modifying `do_train` parameter in scneario configuration file under `configs` directory.

In each scenario, we have 4 evaluation tests that can be enabled or disabled as needed. These tests are:
- Taskset: Two fixed tasksets are used for evaluating each algorithm's performance ($T_1$ and $T_2$ in the paper).
- CPU load: CPU load is varied and algorithms' performance is evaluated.
- Task size: Task sizes are changing in a specific range and model performance is analyzed.
- Channel: Channel impact is evaluated.

Each scenario has its own parameters that you may want to change to analyze each algorithm's results. Most parameters are in scenario configuraion file and are explained what each one does. Apart from scenario configs, you can find environment, task, and channel configurations under `configs` directory.

To run the code, from repo root directory, run `python main.py` and all three scenarios will be started.

## Results
Results along with an analysis of them are presented in the paper. The model weights corresponding to paper resutls are in `paper-model-weight` directory. To use them, transfer the folders and weights inside this directory to `models` directory. Then, only evaluate the models and do not train them as it will replace the weights.
