import numpy as np
from tqdm import tqdm
import copy

from env_models.env import Env, RRLOEnv
from dvfs.dqn_dvfs import DQN_DVFS
from env_models.task import TaskGen
from dvfs.rrlo_dvfs import RRLO_DVFS
from configs import DQN_STATE_DIM


def evaluate(configs, eval_itr=10000):
    task_gen = TaskGen(configs["task_set"])
    dqn_env = Env(configs, task_gen.get_wcet_bound(), task_gen.get_task_size_bound())
    local_env = Env(configs, task_gen.get_wcet_bound(), task_gen.get_task_size_bound())
    remote_env = Env(configs, task_gen.get_wcet_bound(), task_gen.get_task_size_bound())
    random_env = Env(configs, task_gen.get_wcet_bound(), task_gen.get_task_size_bound())
    rand_freq_list = random_env.cpu.freqs
    rand_powers_list = random_env.w_inter.powers
    rrlo_env = RRLOEnv(configs)

    dqn_dvfs = DQN_DVFS(state_dim=DQN_STATE_DIM, act_space=dqn_env.get_action_space())
    rrlo_dvfs = RRLO_DVFS(
        state_bounds=rrlo_env.get_state_bounds(),
        num_w_inter_powers=len(rrlo_env.w_inter.powers),
        num_dvfs_algs=2,
        dvfs_algs=["cc", "la"],
        num_tasks=4,
    )

    dqn_dvfs.load_model("models")
    rrlo_dvfs.load_model("models")

    # Evaluate the trained model
    print("Evaluating the trained model...")
    dqn_task_energy_cons = np.zeros(4)
    dqn_num_tasks = np.zeros(4)
    rrlo_task_energy_cons = np.zeros(4)
    rrlo_num_tasks = np.zeros(4)

    local_task_energy_cons = np.zeros(4)
    local_num_tasks = np.zeros(4)
    remote_task_energy_cons = np.zeros(4)
    remote_num_tasks = np.zeros(4)
    random_task_energy_cons = np.zeros(4)
    random_num_tasks = np.zeros(4)
    local_task_deadline_missed = np.zeros(4)
    random_task_deadline_missed = np.zeros(4)
    remote_task_deadline_missed = np.zeros(4)
    dqn_task_deadline_missed = np.zeros(4)
    rrlo_task_deadline_missed = np.zeros(4)

    tasks = task_gen.step()
    dqn_state, _ = dqn_env.observe(copy.deepcopy(tasks))
    rrlo_state, _ = rrlo_env.observe(copy.deepcopy(tasks))
    local_env.observe(copy.deepcopy(tasks))
    remote_env.observe(copy.deepcopy(tasks))
    random_env.observe(copy.deepcopy(tasks))

    for _ in tqdm(range(eval_itr)):
        actions_dqn = dqn_dvfs.execute(dqn_state, eval_mode=True)
        actions_dqn_str = dqn_dvfs.conv_acts(actions_dqn)
        actions_local_str = {
            "offload": [],
            "local": [[0, 1820], [1, 1820], [2, 1820], [3, 1820]],
        }
        actions_remote_str = {
            "offload": [[0, 28], [1, 28], [2, 28], [3, 28]],
            "local": [],
        }
        actions_random_str = RandomPolicyGen(rand_freq_list, rand_powers_list)
        actions_rrlo, _ = rrlo_dvfs.execute(rrlo_state)

        dqn_env.step(actions_dqn_str)
        rrlo_env.step(actions_rrlo)
        local_env.step(actions_local_str)
        remote_env.step(actions_remote_str)
        random_env.step(actions_random_str)

        # Gather energy consumption values
        for jobs in local_env.curr_tasks.values():
            for j in jobs:
                # print(j)
                local_task_energy_cons[j.t_id] += j.cons_energy
                if j.deadline_missed:
                    local_task_deadline_missed[j.t_id] += 1
            local_num_tasks[j.t_id] += len(jobs)

        for jobs in remote_env.curr_tasks.values():
            for j in jobs:
                # print(j)
                remote_task_energy_cons[j.t_id] += j.cons_energy
                if j.deadline_missed:
                    remote_task_deadline_missed[j.t_id] += 1
            remote_num_tasks[j.t_id] += len(jobs)

        for jobs in random_env.curr_tasks.values():
            for j in jobs:
                # print(j)
                random_task_energy_cons[j.t_id] += j.cons_energy
                if j.deadline_missed:
                    random_task_deadline_missed[j.t_id] += 1
            random_num_tasks[j.t_id] += len(jobs)

        for jobs in dqn_env.curr_tasks.values():
            for j in jobs:
                # print(j)
                dqn_task_energy_cons[j.t_id] += j.cons_energy
                if j.deadline_missed:
                    dqn_task_deadline_missed[j.t_id] += 1
            dqn_num_tasks[j.t_id] += len(jobs)

        for jobs in rrlo_env.curr_tasks.values():
            for j in jobs:
                # print(j)
                rrlo_task_energy_cons[j.t_id] += j.cons_energy
                if j.deadline_missed:
                    rrlo_task_deadline_missed[j.t_id] += 1
            rrlo_num_tasks[j.t_id] += len(jobs)

        tasks = task_gen.step()
        next_state_dqn, _ = dqn_env.observe(copy.deepcopy(tasks))
        next_state_rrlo, _ = rrlo_env.observe(copy.deepcopy(tasks))
        local_env.observe(copy.deepcopy(tasks))
        remote_env.observe(copy.deepcopy(tasks))
        random_env.observe(copy.deepcopy(tasks))

        # Update current state
        dqn_state = next_state_dqn
        rrlo_state = next_state_rrlo

    # Average energy consumption:
    print(dqn_num_tasks)
    print(rrlo_num_tasks)
    print(local_num_tasks)
    print(remote_num_tasks)
    print(random_num_tasks)
    np.set_printoptions(suppress=True)
    dqn_avg_task_energy_cons = dqn_task_energy_cons / dqn_num_tasks
    local_avg_task_energy_cons = local_task_energy_cons / local_num_tasks
    remote_avg_task_energy_cons = remote_task_energy_cons / remote_num_tasks
    random_avg_task_energy_cons = random_task_energy_cons / random_num_tasks
    rrlo_avg_task_energy_cons = rrlo_task_energy_cons / rrlo_num_tasks

    print(f"DQN energy consumption: {dqn_avg_task_energy_cons}")
    print(f"local energy consumption: {local_avg_task_energy_cons}")
    print(f"remote energy consumption: {remote_avg_task_energy_cons}")
    print(f"random energy consumption: {random_avg_task_energy_cons}")
    print(f"RRLO energy consumption: {rrlo_avg_task_energy_cons}")

    dqn_percent_task_missed = dqn_task_deadline_missed / dqn_num_tasks * 100
    local_percent_task_missed = local_task_deadline_missed / local_num_tasks * 100
    remote_percent_task_missed = remote_task_deadline_missed / remote_num_tasks * 100
    random_percent_task_missed = random_task_deadline_missed / random_num_tasks * 100
    rrlo_percent_task_missed = rrlo_task_deadline_missed / rrlo_num_tasks * 100

    print(f"DQN deadline missed: {dqn_task_deadline_missed}")
    print(f"local deadline missed: {local_task_deadline_missed}")
    print(f"remote deadline missed: {remote_task_deadline_missed}")
    print(f"random deadline missed: {random_task_deadline_missed}")
    print(f"RRLO deadline missed: {rrlo_task_deadline_missed}")

    print(f"DQN deadline missed %: {dqn_percent_task_missed}")
    print(f"local deadline missed %: {local_percent_task_missed}")
    print(f"remote deadline missed %: {remote_percent_task_missed}")
    print(f"random deadline missed %: {random_percent_task_missed}")
    print(f"RRLO deadline missed %: {rrlo_percent_task_missed}")

    avg_dqn_energy_cons = np.sum(dqn_task_energy_cons) / np.sum(dqn_num_tasks)
    avg_local_energy_cons = np.sum(local_task_energy_cons) / np.sum(local_num_tasks)
    avg_remote_energy_cons = np.sum(remote_task_energy_cons) / np.sum(remote_num_tasks)
    avg_random_energy_cons = np.sum(random_task_energy_cons) / np.sum(random_num_tasks)
    avg_rrlo_energy_cons = np.sum(rrlo_task_energy_cons) / np.sum(rrlo_num_tasks)
    print(f"DQN task set avg energy consumption: {avg_dqn_energy_cons:.3e}")
    print(f"local task set avg energy consumption: {avg_local_energy_cons:.3e}")
    print(f"remote task set avg energy consumption: {avg_remote_energy_cons:.3e}")
    print(f"random task set avg energy consumption: {avg_random_energy_cons:.3e}")
    print(f"RRLO task set avg energy consumption: {avg_rrlo_energy_cons:.3e}")

    total_dqn_missed_task = (
        np.sum(dqn_task_deadline_missed) / np.sum(dqn_num_tasks) * 100
    )
    total_local_missed_task = (
        np.sum(local_task_deadline_missed) / np.sum(local_num_tasks) * 100
    )
    total_remote_missed_task = (
        np.sum(remote_task_deadline_missed) / np.sum(remote_num_tasks) * 100
    )
    total_random_missed_task = (
        np.sum(random_task_deadline_missed) / np.sum(random_num_tasks) * 100
    )
    total_rrlo_missed_task = (
        np.sum(rrlo_task_deadline_missed) / np.sum(rrlo_num_tasks) * 100
    )
    print(f"DQN tasks deadline missed: {total_dqn_missed_task:.3e}%")
    print(f"local tasks deadline missed: {total_local_missed_task:.3e}%")
    print(f"remote tasks deadline missed: {total_remote_missed_task:.3e}%")
    print(f"random tasks deadline missed: {total_random_missed_task:.3e}%")
    print(f"RRLO tasks deadline missed: {total_rrlo_missed_task:.3e}%")

    dqn_deadline_improvement = (
        np.sum(dqn_task_deadline_missed-rrlo_task_deadline_missed) / np.sum(rrlo_task_deadline_missed) * 100
    )


    dqn_improvement = (
        (avg_dqn_energy_cons - avg_rrlo_energy_cons) / (avg_rrlo_energy_cons) * 100
    )
    dqn_task_improvement = (
        (dqn_avg_task_energy_cons - rrlo_avg_task_energy_cons)
        / rrlo_avg_task_energy_cons
        * 100
    )
    print(f"DQN per task energy usage: {dqn_task_improvement} %")
    print(f"DQN avg energy usage: {dqn_improvement:.3f} %")
    print(f"DQN missed deadline improvement: {dqn_deadline_improvement:.3f} %")
    return (
        np.array(
            [
                avg_random_energy_cons,
                avg_local_energy_cons,
                avg_remote_energy_cons,
                avg_rrlo_energy_cons,
                avg_dqn_energy_cons,
            ]
        ),
        np.array(
            [
                total_random_missed_task,
                total_local_missed_task,
                total_remote_missed_task,
                total_rrlo_missed_task,
                total_dqn_missed_task,
            ]
        ),
        dqn_improvement,
        dqn_deadline_improvement
    )


def RandomPolicyGen(freqs, powers):
    offload = np.random.randint(0, 5)
    actions = {"offload": [], "local": []}
    random_freq_idx = np.random.choice(len(freqs), size=4-offload, replace=True)
    random_power_idx = np.random.choice(len(powers), size=offload, replace=True)
    actions["offload"] = [
            [i, powers[idx]] for i, idx in enumerate(random_power_idx)
    ]
    actions["local"] = [[i+offload, freqs[idx]] for i, idx in enumerate(random_freq_idx)]

    return actions


if __name__ == "__main__":
    configs = {
        "task_set": "configs/task_set_eval.json",
        "cpu_local": "configs/cpu_local.json",
        "w_inter": "configs/wireless_interface.json",
    }
    evaluate(configs, 5000)
    #freqs=[384, 480, 632, 767,
     #           863, 959, 1247, 1341,
      #          1440, 1534, 1632, 1690,
       #         1820]
    #powers=[
     #   0, 4, 8, 12, 16, 20, 24, 28
    #]
    #print(RandomPolicyGen(freqs,powers))
