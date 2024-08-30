import numpy as np
from tqdm import tqdm
import copy
from itertools import cycle

from env_models.env import BaseDQNEnv, DQNEnv, RRLOEnv
from dvfs.dqn_dvfs import DQN_DVFS
from env_models.task import TaskGen,RandomTaskGen
from dvfs.rrlo_dvfs import RRLO_DVFS
from dvfs.conference_dvfs import conference_DVFS
from utils.utils import set_random_seed


def evaluate_rrlo_scenario(configs, eval_itr=10000):
    task_gen = TaskGen(configs["task_set"])
    dqn_env = DQNEnv(configs, task_gen.get_wcet_bound(), task_gen.get_task_size_bound())
    local_env = DQNEnv(configs, task_gen.get_wcet_bound(), task_gen.get_task_size_bound())
    remote_env = DQNEnv(configs, task_gen.get_wcet_bound(), task_gen.get_task_size_bound())
    random_env = DQNEnv(configs, task_gen.get_wcet_bound(), task_gen.get_task_size_bound())
    rand_freq_list = random_env.cpu.freqs
    rand_powers_list = random_env.w_inter.powers
    rrlo_env = RRLOEnv(configs)
    conference_env = RRLOEnv(configs)

    dqn_dvfs = DQN_DVFS(state_dim=configs["dqn_state_dim"], act_space=dqn_env.get_action_space())
    rrlo_dvfs = RRLO_DVFS(
        state_bounds=rrlo_env.get_state_bounds(),
        num_w_inter_powers=len(rrlo_env.w_inter.powers),
        num_dvfs_algs=2,
        dvfs_algs=["cc", "la"],
        num_tasks=4,
    )
    conference_dvfs = conference_DVFS(
        state_bounds=conference_env.get_state_bounds(),
        num_w_inter_powers=len(conference_env.w_inter.powers),
        num_dvfs_algs=1,
        dvfs_algs=["cc"],
        num_tasks=4,
    )

    dqn_dvfs.load_model("models/rrlo_scenario")
    rrlo_dvfs.load_model("models/rrlo_scenario")
    conference_dvfs.load_model("models/rrlo_scenario")

    # Evaluate the trained model
    #print("Evaluating the trained model...")
    dqn_task_energy_cons = np.zeros(4)
    dqn_num_tasks = np.zeros(4)
    rrlo_task_energy_cons = np.zeros(4)
    rrlo_num_tasks = np.zeros(4)
    conference_task_energy_cons = np.zeros(4)
    conference_num_tasks = np.zeros(4)

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
    conference_task_deadline_missed = np.zeros(4)

    tasks = task_gen.step()
    dqn_state, _ = dqn_env.observe(copy.deepcopy(tasks))
    rrlo_state, _ = rrlo_env.observe(copy.deepcopy(tasks))
    conference_state, _ = conference_env.observe(copy.deepcopy(tasks))
    local_env.observe(copy.deepcopy(tasks))
    remote_env.observe(copy.deepcopy(tasks))
    random_env.observe(copy.deepcopy(tasks))

    for _ in range(eval_itr):
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
        actions_conference, _ = conference_dvfs.execute(conference_state)

        dqn_env.step(actions_dqn_str)
        rrlo_env.step(actions_rrlo)
        conference_env.step(actions_conference)
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
        for jobs in conference_env.curr_tasks.values():
            for j in jobs:
                # print(j)
                conference_task_energy_cons[j.t_id] += j.cons_energy
                if j.deadline_missed:
                    conference_task_deadline_missed[j.t_id] += 1
            conference_num_tasks[j.t_id] += len(jobs)

        tasks = task_gen.step()
        next_state_dqn, _ = dqn_env.observe(copy.deepcopy(tasks))
        next_state_rrlo, _ = rrlo_env.observe(copy.deepcopy(tasks))
        next_state_conference, _ = conference_env.observe(copy.deepcopy(tasks))
        local_env.observe(copy.deepcopy(tasks))
        remote_env.observe(copy.deepcopy(tasks))
        random_env.observe(copy.deepcopy(tasks))

        # Update current state
        dqn_state = next_state_dqn
        rrlo_state = next_state_rrlo
        conference_state = next_state_conference

    # Average energy consumption:
    #print(dqn_num_tasks)
    #print(rrlo_num_tasks)
    #print(conference_num_tasks)
    #print(local_num_tasks)
    #print(remote_num_tasks)
    #print(random_num_tasks)
    np.set_printoptions(suppress=True)
    dqn_avg_task_energy_cons = dqn_task_energy_cons / dqn_num_tasks
    local_avg_task_energy_cons = local_task_energy_cons / local_num_tasks
    remote_avg_task_energy_cons = remote_task_energy_cons / remote_num_tasks
    random_avg_task_energy_cons = random_task_energy_cons / random_num_tasks
    rrlo_avg_task_energy_cons = rrlo_task_energy_cons / rrlo_num_tasks
    conference_avg_task_energy_cons = conference_task_energy_cons / conference_num_tasks

    #print(f"DQN energy consumption: {dqn_avg_task_energy_cons}")
    #print(f"local energy consumption: {local_avg_task_energy_cons}")
    #print(f"remote energy consumption: {remote_avg_task_energy_cons}")
    #print(f"random energy consumption: {random_avg_task_energy_cons}")
    #print(f"RRLO energy consumption: {rrlo_avg_task_energy_cons}")
    #print(f"Conference energy consumption: {conference_avg_task_energy_cons}")

    dqn_percent_task_missed = dqn_task_deadline_missed / dqn_num_tasks * 100
    local_percent_task_missed = local_task_deadline_missed / local_num_tasks * 100
    remote_percent_task_missed = remote_task_deadline_missed / remote_num_tasks * 100
    random_percent_task_missed = random_task_deadline_missed / random_num_tasks * 100
    rrlo_percent_task_missed = rrlo_task_deadline_missed / rrlo_num_tasks * 100
    conference_percent_task_missed = conference_task_deadline_missed / conference_num_tasks * 100

    #print(f"DQN deadline missed: {dqn_task_deadline_missed}")
    #print(f"local deadline missed: {local_task_deadline_missed}")
    #print(f"remote deadline missed: {remote_task_deadline_missed}")
    #print(f"random deadline missed: {random_task_deadline_missed}")
    #print(f"RRLO deadline missed: {rrlo_task_deadline_missed}")
    #print(f"Conference deadline missed: {conference_task_deadline_missed}")

    #print(f"DQN deadline missed %: {dqn_percent_task_missed}")
    #print(f"local deadline missed %: {local_percent_task_missed}")
    #print(f"remote deadline missed %: {remote_percent_task_missed}")
    #print(f"random deadline missed %: {random_percent_task_missed}")
    #print(f"RRLO deadline missed %: {rrlo_percent_task_missed}")
    #print(f"Conference deadline missed %: {conference_percent_task_missed}")

    avg_dqn_energy_cons = np.sum(dqn_task_energy_cons) / np.sum(dqn_num_tasks)
    avg_local_energy_cons = np.sum(local_task_energy_cons) / np.sum(local_num_tasks)
    avg_remote_energy_cons = np.sum(remote_task_energy_cons) / np.sum(remote_num_tasks)
    avg_random_energy_cons = np.sum(random_task_energy_cons) / np.sum(random_num_tasks)
    avg_rrlo_energy_cons = np.sum(rrlo_task_energy_cons) / np.sum(rrlo_num_tasks)
    avg_conference_energy_cons = np.sum(conference_task_energy_cons) / np.sum(conference_num_tasks)
    #print(f"DQN task set avg energy consumption: {avg_dqn_energy_cons:.3e}")
    #print(f"local task set avg energy consumption: {avg_local_energy_cons:.3e}")
    #print(f"remote task set avg energy consumption: {avg_remote_energy_cons:.3e}")
    #print(f"random task set avg energy consumption: {avg_random_energy_cons:.3e}")
    #print(f"RRLO task set avg energy consumption: {avg_rrlo_energy_cons:.3e}")
    #print(f"Conference task set avg energy consumption: {avg_conference_energy_cons:.3e}")

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
    total_conference_missed_task = (
        np.sum(conference_task_deadline_missed) / np.sum(conference_num_tasks) * 100
    )
    #print(f"DQN tasks deadline missed: {total_dqn_missed_task:.3e}%")
    #print(f"local tasks deadline missed: {total_local_missed_task:.3e}%")
    #print(f"remote tasks deadline missed: {total_remote_missed_task:.3e}%")
    #print(f"random tasks deadline missed: {total_random_missed_task:.3e}%")
    #print(f"RRLO tasks deadline missed: {total_rrlo_missed_task:.3e}%")
    #print(f"Conference tasks deadline missed: {total_conference_missed_task:.3e}%")
    #dqn_random_deadline_improvement = (
     #   np.sum(dqn_task_deadline_missed-random_task_deadline_missed) / np.sum(random_task_deadline_missed) * 100
    #)

    #dqn_local_deadline_improvement = (
     #   np.sum(dqn_task_deadline_missed-local_task_deadline_missed) / np.sum(local_task_deadline_missed) * 100
    #)

    #dqn_remote_deadline_improvement = (
     #   np.sum(dqn_task_deadline_missed-remote_task_deadline_missed) / np.sum(remote_task_deadline_missed) * 100
    #)


    dqn_random_improvement = (
        (avg_dqn_energy_cons - avg_random_energy_cons) / (avg_random_energy_cons) * 100
    )

    dqn_local_improvement = (
        (avg_dqn_energy_cons - avg_local_energy_cons) / (avg_local_energy_cons) * 100
    )

    dqn_remote_improvement = (
        (avg_dqn_energy_cons - avg_remote_energy_cons) / (avg_remote_energy_cons) * 100
    )
    
    dqn_conference_improvement = (
        (avg_dqn_energy_cons - avg_conference_energy_cons) / (avg_conference_energy_cons) * 100
    )
    
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
    #print(f"DQN per task energy usage: {dqn_task_improvement} %")
    #print(f"DQN avg energy usage: {dqn_improvement:.3f} %")
    #print(f"DQN missed deadline improvement: {dqn_deadline_improvement:.3f} %")
    return (
        np.array(
            [
                avg_random_energy_cons,
                avg_local_energy_cons,
                avg_remote_energy_cons,
                avg_rrlo_energy_cons,
                avg_conference_energy_cons,
                avg_dqn_energy_cons,
            ]
        ),
        np.array(
            [
                total_random_missed_task,
                total_local_missed_task,
                total_remote_missed_task,
                total_rrlo_missed_task,
                total_conference_missed_task,
                total_dqn_missed_task,
            ]
        ),
        np.array(
            [
                dqn_random_improvement,
                dqn_local_improvement,
                dqn_remote_improvement,
                dqn_improvement,
                dqn_conference_improvement,
            ]
        )
        #np.array(
         #   [
          #      dqn_random_deadline_improvement,
           #     dqn_local_deadline_improvement,
            #    dqn_remote_deadline_improvement,
             #   dqn_deadline_improvement
            #]
        #)
    )


def evaluate_dqn_scenario(configs, eval_itr=10000):
    task_gen = TaskGen(configs["task_set"])
    dqn_env = BaseDQNEnv(configs, task_gen.get_wcet_bound(), task_gen.get_task_size_bound())
    local_env = BaseDQNEnv(configs, task_gen.get_wcet_bound(), task_gen.get_task_size_bound())
    remote_env = BaseDQNEnv(configs, task_gen.get_wcet_bound(), task_gen.get_task_size_bound())
    random_env = BaseDQNEnv(configs, task_gen.get_wcet_bound(), task_gen.get_task_size_bound())
    conference_env = RRLOEnv(configs)
    rand_littlefreq_list = random_env.cpu_little.freqs
    rand_bigfreq_list = random_env.cpu_big.freqs
    rand_powers_list = random_env.w_inter.powers

    dqn_dvfs = DQN_DVFS(state_dim=configs["dqn_state_dim"], act_space=dqn_env.get_action_space())
    conference_dvfs = conference_DVFS(
        state_bounds=conference_env.get_state_bounds(),
        num_w_inter_powers=len(conference_env.w_inter.powers),
        num_dvfs_algs=1,
        dvfs_algs=["cc"],
        num_tasks=4,
    )

    conference_dvfs = conference_DVFS(
        state_bounds=conference_env.get_state_bounds(),
        num_w_inter_powers=len(conference_env.w_inter.powers),
        num_dvfs_algs=1,
        dvfs_algs=["cc"],
        num_tasks=4,
    )
    dqn_dvfs.load_model("models/dqn_scenario")
    conference_dvfs.load_model("models/dqn_scenario")

    # Evaluate the trained model
    #print("Evaluating the trained model...")
    dqn_task_energy_cons = np.zeros(4)
    dqn_num_tasks = np.zeros(4)

    local_task_energy_cons = np.zeros(4)
    local_num_tasks = np.zeros(4)

    remote_task_energy_cons = np.zeros(4)
    remote_num_tasks = np.zeros(4)

    random_task_energy_cons = np.zeros(4)
    random_num_tasks = np.zeros(4)

    conference_task_energy_cons = np.zeros(4)
    conference_num_tasks = np.zeros(4)

    dqn_task_deadline_missed = np.zeros(4)
    local_task_deadline_missed = np.zeros(4)
    remote_task_deadline_missed = np.zeros(4)
    random_task_deadline_missed = np.zeros(4)
    conference_task_deadline_missed = np.zeros(4)

    tasks = task_gen.step()
    dqn_state, _ = dqn_env.observe(copy.deepcopy(tasks))
    local_env.observe(copy.deepcopy(tasks))
    remote_env.observe(copy.deepcopy(tasks))
    random_env.observe(copy.deepcopy(tasks))
    conference_state, _ = conference_env.observe(copy.deepcopy(tasks))

    for _ in range(eval_itr):
        actions_dqn = dqn_dvfs.execute(dqn_state, eval_mode=True)
        actions_dqn_str = dqn_dvfs.conv_acts(actions_dqn)
        #FIXME: Check out local actions
        actions_local_str = {
            "offload": [],
            "little": [],
            "big": [[0, 1820], [1, 1820], [2, 1820], [3, 1820]],
        }
        actions_remote_str = {
            "offload": [[0, 28], [1, 28], [2, 28], [3, 28]],
            "little": [],
            "big": [],
        }
        actions_random_str = RandomDQNPolicyGen(rand_littlefreq_list,rand_bigfreq_list, rand_powers_list)
        actions_conference, _ = conference_dvfs.execute(conference_state)

        dqn_env.step(actions_dqn_str)
        local_env.step(actions_local_str)
        remote_env.step(actions_remote_str)
        random_env.step(actions_random_str)
        conference_env.step(actions_conference)

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

        for jobs in conference_env.curr_tasks.values():
            for j in jobs:
                # print(j)
                conference_task_energy_cons[j.t_id] += j.cons_energy
                if j.deadline_missed:
                    conference_task_deadline_missed[j.t_id] += 1
            conference_num_tasks[j.t_id] += len(jobs)

        tasks = task_gen.step()
        next_state_dqn, _ = dqn_env.observe(copy.deepcopy(tasks))
        local_env.observe(copy.deepcopy(tasks))
        remote_env.observe(copy.deepcopy(tasks))
        random_env.observe(copy.deepcopy(tasks))
        next_state_conference, _ = conference_env.observe(copy.deepcopy(tasks))

        # Update current state
        dqn_state = next_state_dqn
        conference_state = next_state_conference

    # Average energy consumption:
    #print(dqn_num_tasks)
    #print(local_num_tasks)
    #print(remote_num_tasks)
    #print(random_num_tasks)
    np.set_printoptions(suppress=True)
    dqn_avg_task_energy_cons = dqn_task_energy_cons / dqn_num_tasks
    local_avg_task_energy_cons = local_task_energy_cons / local_num_tasks
    remote_avg_task_energy_cons = remote_task_energy_cons / remote_num_tasks
    random_avg_task_energy_cons = random_task_energy_cons / random_num_tasks
    conference_avg_task_energy_cons = conference_task_energy_cons / conference_num_tasks


    #print(f"DQN energy consumption: {dqn_avg_task_energy_cons}")
    #print(f"local energy consumption: {local_avg_task_energy_cons}")
    #print(f"remote energy consumption: {remote_avg_task_energy_cons}")
    #print(f"random energy consumption: {random_avg_task_energy_cons}")

    dqn_percent_task_missed = dqn_task_deadline_missed / dqn_num_tasks * 100
    local_percent_task_missed = local_task_deadline_missed / local_num_tasks * 100
    remote_percent_task_missed = remote_task_deadline_missed / remote_num_tasks * 100
    random_percent_task_missed = random_task_deadline_missed / random_num_tasks * 100
    conference_percent_task_missed = conference_task_deadline_missed / conference_num_tasks * 100

    #print(f"DQN deadline missed: {dqn_task_deadline_missed}")
    #print(f"local deadline missed: {local_task_deadline_missed}")
    #print(f"remote deadline missed: {remote_task_deadline_missed}")
    #print(f"random deadline missed: {random_task_deadline_missed}")

    #print(f"DQN deadline missed %: {dqn_percent_task_missed}")
    #print(f"local deadline missed %: {local_percent_task_missed}")
    #print(f"remote deadline missed %: {remote_percent_task_missed}")
    #print(f"random deadline missed %: {random_percent_task_missed}")

    avg_dqn_energy_cons = np.sum(dqn_task_energy_cons) / np.sum(dqn_num_tasks)
    avg_local_energy_cons = np.sum(local_task_energy_cons) / np.sum(local_num_tasks)
    avg_remote_energy_cons = np.sum(remote_task_energy_cons) / np.sum(remote_num_tasks)
    avg_random_energy_cons = np.sum(random_task_energy_cons) / np.sum(random_num_tasks)
    avg_conference_energy_cons = np.sum(conference_task_energy_cons) / np.sum(conference_num_tasks)
    #print(f"DQN task set avg energy consumption: {avg_dqn_energy_cons:.3e}")
    #print(f"local task set avg energy consumption: {avg_local_energy_cons:.3e}")
    #print(f"remote task set avg energy consumption: {avg_remote_energy_cons:.3e}")
    #print(f"random task set avg energy consumption: {avg_random_energy_cons:.3e}")

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
    total_conference_missed_task = (
         np.sum(conference_task_deadline_missed) / np.sum(conference_num_tasks) * 100
    )

    #print(f"DQN tasks deadline missed: {total_dqn_missed_task:.3e}%")
    #print(f"local tasks deadline missed: {total_local_missed_task:.3e}%")
    #print(f"remote tasks deadline missed: {total_remote_missed_task:.3e}%")
    #print(f"random tasks deadline missed: {total_random_missed_task:.3e}%")


    #FIXME: Fixe DQN improvement
    #dqn_improvement=0
    #dqn_random_deadline_improvement = (
     #   np.sum(dqn_task_deadline_missed-random_task_deadline_missed) / np.sum(random_task_deadline_missed) * 100
    #)

    #dqn_local_deadline_improvement = (
     #   np.sum(dqn_task_deadline_missed-local_task_deadline_missed) / np.sum(local_task_deadline_missed) * 100
    #)

    #dqn_remote_deadline_improvement = (
     #   np.sum(dqn_task_deadline_missed-remote_task_deadline_missed) / np.sum(remote_task_deadline_missed) * 100
    #)


    dqn_random_improvement = (
        (avg_dqn_energy_cons - avg_random_energy_cons) / (avg_random_energy_cons) * 100
    )

    dqn_local_improvement = (
        (avg_dqn_energy_cons - avg_local_energy_cons) / (avg_local_energy_cons) * 100
    )

    dqn_remote_improvement = (
        (avg_dqn_energy_cons - avg_remote_energy_cons) / (avg_remote_energy_cons) * 100
    )

    dqn_conference_improvement = (
        (avg_dqn_energy_cons - avg_conference_energy_cons) / (avg_conference_energy_cons) * 100
    )
    #dqn_task_improvement = (
     #   (dqn_avg_task_energy_cons - rrlo_avg_task_energy_cons)
      #  / rrlo_avg_task_energy_cons
       # * 100
    #)
    #print(f"DQN per task energy usage: {dqn_task_improvement} %")
    #print(f"DQN avg energy usage: {dqn_improvement:.3f} %")
    #print(f"DQN missed deadline improvement: {dqn_deadline_improvement:.3f} %")
    return (
        np.array(
            [
                avg_random_energy_cons,
                avg_local_energy_cons,
                avg_remote_energy_cons,
                avg_conference_energy_cons,
                avg_dqn_energy_cons
            ]
        ),
        np.array(
            [
                total_random_missed_task,
                total_local_missed_task,
                total_remote_missed_task,
                total_conference_missed_task,
                total_dqn_missed_task
            ]
        ),
        np.array(
            [
                dqn_random_improvement,
                dqn_local_improvement,
                dqn_remote_improvement,
                dqn_conference_improvement
            ]
        )
        #np.array(
         #   [
          #      dqn_random_deadline_improvement,
           #     dqn_local_deadline_improvement,
            #    dqn_remote_deadline_improvement
            #]
        #)
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

def RandomDQNPolicyGen(littlefreqs, bigfreqs, powers):
    offload = np.random.randint(0, 5)
    little = np.random.randint(0,5-offload)
    big=4-offload-little
    actions = {"offload": [], "little": [], "big": []}
    random_littlefreq_idx = np.random.choice(len(littlefreqs), size=little, replace=True)
    random_bigfreq_idx = np.random.choice(len(bigfreqs), size=big, replace=True)
    random_power_idx = np.random.choice(len(powers), size=offload, replace=True)
    actions["offload"] = [
            [i, powers[idx]] for i, idx in enumerate(random_power_idx)
    ]
    actions["little"] = [[i+offload, littlefreqs[idx]] for i, idx in enumerate(random_littlefreq_idx)]
    actions["big"] = [[i+offload+little, bigfreqs[idx]] for i, idx in enumerate(random_bigfreq_idx)]

    return actions



    #FIXME: Implement random DQN policy
    # offload = np.random.randint(0, 2)
    # actions = {"offload": [], "little": [], "big": []}
    # random_freq_idx = np.random.choice(len(freqs), size=4, replace=True)
    # random_power_idx = np.random.choice(len(powers), size=4, replace=True)
    # if offload:
    #     actions["offload"] = [
    #         [i, powers[idx]] for i, idx in enumerate(random_power_idx)
    #     ]
    # else:
    #     actions["local"] = [[i, freqs[idx]] for i, idx in enumerate(random_freq_idx)]

    # return actions



def evaluate_cpu_load_scenario(configs, cpu_load_val, eval_itr=10000):
    
    set_random_seed(42)
    #cpu_load_generator = cycle(cpu_load_val)
    #target_cpu_load = next(cpu_load_generator)

    task_gen = RandomTaskGen(configs["task_set"])
    #task_gen = TaskGen(configs["task_set"])
    dqn_env = DQNEnv(configs, task_gen.get_wcet_bound(), task_gen.get_task_size_bound())
    local_env = DQNEnv(configs, task_gen.get_wcet_bound(), task_gen.get_task_size_bound())
    remote_env = DQNEnv(configs, task_gen.get_wcet_bound(), task_gen.get_task_size_bound())
    random_env = DQNEnv(configs, task_gen.get_wcet_bound(), task_gen.get_task_size_bound())
    rand_freq_list = random_env.cpu.freqs
    rand_powers_list = random_env.w_inter.powers
    rrlo_env = RRLOEnv(configs)
    conference_env = RRLOEnv(configs)

    dqn_dvfs = DQN_DVFS(state_dim=configs["dqn_state_dim"], act_space=dqn_env.get_action_space())
    rrlo_dvfs = RRLO_DVFS(
        state_bounds=rrlo_env.get_state_bounds(),
        num_w_inter_powers=len(rrlo_env.w_inter.powers),
        num_dvfs_algs=2,
        dvfs_algs=["cc", "la"],
        num_tasks=4,
    )
    conference_dvfs = conference_DVFS(
        state_bounds=conference_env.get_state_bounds(),
        num_w_inter_powers=len(conference_env.w_inter.powers),
        num_dvfs_algs=1,
        dvfs_algs=["cc"],
        num_tasks=4,
    )

    dqn_dvfs.load_model("models/rrlo_scenario")
    rrlo_dvfs.load_model("models/rrlo_scenario")
    conference_dvfs.load_model("models/rrlo_scenario")

    # Evaluate the trained model
    #print("Evaluating the trained model...")
    dqn_task_energy_cons = np.zeros((4,len(cpu_load_val)))
    dqn_num_tasks = np.zeros((4,len(cpu_load_val)))
    rrlo_task_energy_cons = np.zeros((4,len(cpu_load_val)))
    rrlo_num_tasks = np.zeros((4,len(cpu_load_val)))
    conference_task_energy_cons = np.zeros((4,len(cpu_load_val)))
    conference_num_tasks = np.zeros((4,len(cpu_load_val)))

    local_task_energy_cons = np.zeros((4,len(cpu_load_val)))
    local_num_tasks = np.zeros((4,len(cpu_load_val)))
    remote_task_energy_cons = np.zeros((4,len(cpu_load_val)))
    remote_num_tasks = np.zeros((4,len(cpu_load_val)))
    random_task_energy_cons = np.zeros((4,len(cpu_load_val)))
    random_num_tasks = np.zeros((4,len(cpu_load_val)))

    local_task_deadline_missed = np.zeros((4,len(cpu_load_val)))
    random_task_deadline_missed = np.zeros((4,len(cpu_load_val)))
    remote_task_deadline_missed = np.zeros((4,len(cpu_load_val)))
    dqn_task_deadline_missed = np.zeros((4,len(cpu_load_val)))
    rrlo_task_deadline_missed = np.zeros((4,len(cpu_load_val)))
    conference_task_deadline_missed = np.zeros((4,len(cpu_load_val)))
    
    
    
    
    for i in range(len(cpu_load_val)):

        target_cpu_load=cpu_load_val[i]
        tasks = task_gen.step(target_cpu_load)
        dqn_state, _ = dqn_env.observe(copy.deepcopy(tasks))
        rrlo_state, _ = rrlo_env.observe(copy.deepcopy(tasks))
        conference_state, _ = conference_env.observe(copy.deepcopy(tasks))
        local_env.observe(copy.deepcopy(tasks))
        remote_env.observe(copy.deepcopy(tasks))
        random_env.observe(copy.deepcopy(tasks))
        for _ in range(eval_itr):
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
            actions_conference, _ = conference_dvfs.execute(conference_state)
            
            dqn_env.step(actions_dqn_str)
            rrlo_env.step(actions_rrlo)
            conference_env.step(actions_conference)
            local_env.step(actions_local_str)
            remote_env.step(actions_remote_str)
            random_env.step(actions_random_str)
            # Gather energy consumption values
            for jobs in local_env.curr_tasks.values():
                for j in jobs:
                    # print(j)
                    local_task_energy_cons[j.t_id,i] += j.cons_energy
                    if j.deadline_missed:
                        local_task_deadline_missed[j.t_id,i] += 1
                local_num_tasks[j.t_id,i] += len(jobs)

            for jobs in remote_env.curr_tasks.values():
                for j in jobs:
                    # print(j)
                    remote_task_energy_cons[j.t_id,i] += j.cons_energy
                    if j.deadline_missed:
                        remote_task_deadline_missed[j.t_id,i] += 1
                remote_num_tasks[j.t_id,i] += len(jobs)

            for jobs in random_env.curr_tasks.values():
                for j in jobs:
                    # print(j)
                    random_task_energy_cons[j.t_id,i] += j.cons_energy
                    if j.deadline_missed:
                        random_task_deadline_missed[j.t_id,i] += 1
                random_num_tasks[j.t_id,i] += len(jobs)

            for jobs in dqn_env.curr_tasks.values():
                for j in jobs:
                    # print(j)
                    dqn_task_energy_cons[j.t_id,i] += j.cons_energy
                    if j.deadline_missed:
                        dqn_task_deadline_missed[j.t_id,i] += 1
                dqn_num_tasks[j.t_id,i] += len(jobs)

            for jobs in rrlo_env.curr_tasks.values():
                for j in jobs:
                    # print(j)
                    rrlo_task_energy_cons[j.t_id,i] += j.cons_energy
                    if j.deadline_missed:
                        rrlo_task_deadline_missed[j.t_id,i] += 1
                rrlo_num_tasks[j.t_id,i] += len(jobs)
            for jobs in conference_env.curr_tasks.values():
                for j in jobs:
                    # print(j)
                    conference_task_energy_cons[j.t_id,i] += j.cons_energy
                    if j.deadline_missed:
                        conference_task_deadline_missed[j.t_id,i] += 1
                conference_num_tasks[j.t_id,i] += len(jobs)

            tasks = task_gen.step(target_cpu_load)
            next_state_dqn, _ = dqn_env.observe(copy.deepcopy(tasks))
            next_state_rrlo, _ = rrlo_env.observe(copy.deepcopy(tasks))
            next_state_conference, _ = conference_env.observe(copy.deepcopy(tasks))
            local_env.observe(copy.deepcopy(tasks))
            remote_env.observe(copy.deepcopy(tasks))
            random_env.observe(copy.deepcopy(tasks))

            # Update current state
            dqn_state = next_state_dqn
            rrlo_state = next_state_rrlo
            conference_state = next_state_conference


    np.set_printoptions(suppress=True)
    dqn_avg_task_energy_cons = dqn_task_energy_cons / dqn_num_tasks
    local_avg_task_energy_cons = local_task_energy_cons / local_num_tasks
    remote_avg_task_energy_cons = remote_task_energy_cons / remote_num_tasks
    random_avg_task_energy_cons = random_task_energy_cons / random_num_tasks
    rrlo_avg_task_energy_cons = rrlo_task_energy_cons / rrlo_num_tasks
    conference_avg_task_energy_cons = conference_task_energy_cons / conference_num_tasks



    dqn_percent_task_missed = dqn_task_deadline_missed / dqn_num_tasks * 100
    local_percent_task_missed = local_task_deadline_missed / local_num_tasks * 100
    remote_percent_task_missed = remote_task_deadline_missed / remote_num_tasks * 100
    random_percent_task_missed = random_task_deadline_missed / random_num_tasks * 100
    rrlo_percent_task_missed = rrlo_task_deadline_missed / rrlo_num_tasks * 100
    conference_percent_task_missed = conference_task_deadline_missed / conference_num_tasks * 100



    avg_dqn_energy_cons = np.sum(dqn_task_energy_cons,axis=0) / np.sum(dqn_num_tasks,axis=0)
    avg_local_energy_cons = np.sum(local_task_energy_cons,axis=0) / np.sum(local_num_tasks,axis=0)
    avg_remote_energy_cons = np.sum(remote_task_energy_cons,axis=0) / np.sum(remote_num_tasks,axis=0)
    avg_random_energy_cons = np.sum(random_task_energy_cons,axis=0) / np.sum(random_num_tasks,axis=0)
    avg_rrlo_energy_cons = np.sum(rrlo_task_energy_cons,axis=0) / np.sum(rrlo_num_tasks,axis=0)
    avg_conference_energy_cons = np.sum(conference_task_energy_cons,axis=0) / np.sum(conference_num_tasks,axis=0)


    total_dqn_missed_task = (
        np.sum(dqn_task_deadline_missed,axis=0) / np.sum(dqn_num_tasks,axis=0) * 100
    )
    total_local_missed_task = (
        np.sum(local_task_deadline_missed,axis=0) / np.sum(local_num_tasks,axis=0) * 100
    )
    total_remote_missed_task = (
        np.sum(remote_task_deadline_missed,axis=0) / np.sum(remote_num_tasks,axis=0) * 100
    )
    total_random_missed_task = (
        np.sum(random_task_deadline_missed,axis=0) / np.sum(random_num_tasks,axis=0) * 100
    )
    total_rrlo_missed_task = (
        np.sum(rrlo_task_deadline_missed,axis=0) / np.sum(rrlo_num_tasks,axis=0) * 100
    )
    total_conference_missed_task = (
        np.sum(conference_task_deadline_missed,axis=0) / np.sum(conference_num_tasks,axis=0) * 100
    )


    dqn_random_improvement = (
        (avg_dqn_energy_cons - avg_random_energy_cons) / (avg_random_energy_cons) * 100
    )

    dqn_local_improvement = (
        (avg_dqn_energy_cons - avg_local_energy_cons) / (avg_local_energy_cons) * 100
    )

    dqn_remote_improvement = (
        (avg_dqn_energy_cons - avg_remote_energy_cons) / (avg_remote_energy_cons) * 100
    )
    
    dqn_conference_improvement = (
        (avg_dqn_energy_cons - avg_conference_energy_cons) / (avg_conference_energy_cons) * 100
    )
    
    #dqn_deadline_improvement = (
        #   np.sum(dqn_task_deadline_missed-rrlo_task_deadline_missed) / np.sum(rrlo_task_deadline_missed) * 100
    #)


    dqn_improvement = (
        (avg_dqn_energy_cons - avg_rrlo_energy_cons) / (avg_rrlo_energy_cons) * 100
    )
    dqn_task_improvement = (
        (dqn_avg_task_energy_cons - rrlo_avg_task_energy_cons)
        / rrlo_avg_task_energy_cons
        * 100
    )

    return (
        np.array(
            [
                avg_random_energy_cons,
                avg_local_energy_cons,
                avg_remote_energy_cons,
                avg_rrlo_energy_cons,
                avg_conference_energy_cons,
                avg_dqn_energy_cons,
            ]
        ),
        np.array(
            [
                total_random_missed_task,
                total_local_missed_task,
                total_remote_missed_task,
                total_rrlo_missed_task,
                total_conference_missed_task,
                total_dqn_missed_task,
            ]
        ),
        np.array(
            [
                dqn_random_improvement,
                dqn_local_improvement,
                dqn_remote_improvement,
                dqn_improvement,
                dqn_conference_improvement,
            ]
        )

    )

if __name__ == "__main__":
    rrlo_scenario_configs = {
        "task_set": "configs/task_set_eval.json",
        "cpu_local": "configs/cpu_local.json",
        "w_inter": "configs/wireless_interface.json",
        "dqn_state_dim": 4,
    }
    evaluate_rrlo_scenario(rrlo_scenario_configs, 5000)

    dqn_scenario_configs = {
        "task_set": "configs/task_set_eval.json",
        "cpu_little": "configs/cpu_little.json",
        "cpu_big": "configs/cpu_big.json",
        "w_inter": "configs/wireless_interface.json",
        "dqn_state_dim": 5,
    }
    evaluate_dqn_scenario(dqn_scenario_configs, 5000)
    #littlefreq=[384, 461, 600, 670,
      #          786, 863, 959, 1239,
     #           1440]
    #bigfreq=[384, 480, 632, 767,
       #         863, 959, 1247, 1341,
      #          1440, 1534, 1632, 1690,
     #           1820]
    #powers=[
     #   0, 4, 8, 12, 16, 20, 24, 28
    #]
    #print(RandomDQNPolicyGen(littlefreq,bigfreq,powers))
