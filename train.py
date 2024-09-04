import numpy as np
from tqdm import tqdm
import copy
import os
from itertools import cycle

from env_models.env import BaseDQNEnv, DQNEnv, RRLOEnv
from env_models.task import RandomTaskGen, NormalTaskGen
from dvfs.dqn_dvfs import DQN_DVFS
from dvfs.rrlo_dvfs import RRLO_DVFS
from dvfs.conference_dvfs import conference_DVFS
from utils.utils import set_random_seed



# Main function
def train_rrlo_scenario(configs):
    # Set random seed
    set_random_seed(42)
    
    cpu_load_values=np.arange(0.01, 1, 0.01)
    cpu_load_generator = cycle(cpu_load_values)
    task_mean_values=np.arange(100, 505, 4)
    task_mean_generator = cycle(task_mean_values)

    cpu_generate=True

    task_gen_cpu = RandomTaskGen(configs["task_set"])
    task_gen_task= NormalTaskGen(configs["task_set"])
    dqn_env = DQNEnv(configs, task_gen_cpu.get_wcet_bound(), task_gen_cpu.get_task_size_bound())
    rrlo_env = RRLOEnv(configs)
    #conference_env=RRLOEnv(configs)

    dqn_loss=[]

    # Initialize DVFS algorithms
    dqn_dvfs = DQN_DVFS(
        state_dim=configs["dqn_state_dim"],
        act_space=dqn_env.get_action_space(),
        batch_size=64,
        gamma=0.95,
        mem_size=1000,
        update_target_net=1000,
        eps_decay=1 / 200,
        min_eps=0,
    )

    rrlo_dvfs = RRLO_DVFS(
        state_bounds=rrlo_env.get_state_bounds(),
        num_w_inter_powers=len(rrlo_env.w_inter.powers),
        num_dvfs_algs=2,
        dvfs_algs=["cc", "la"],
        num_tasks=4,
    )
    #conference_dvfs = conference_DVFS(
     #   state_bounds=conference_env.get_state_bounds(),
      #  num_w_inter_powers=len(conference_env.w_inter.powers),
       # num_dvfs_algs=1,
        #dvfs_algs=["cc"],
        #num_tasks=4,
    #)
    # Initial state observation
    if cpu_generate:
        target_cpu_load = next(cpu_load_generator)
        tasks = task_gen_cpu.step(target_cpu_load)
    else:
        target_cpu_load = next(cpu_load_generator)
        target_task_mean=next(task_mean_generator)
        tasks= task_gen_task.step(target_cpu_load,target_task_mean)
    dqn_state, _ = dqn_env.observe(copy.deepcopy(tasks))
    rrlo_state, _ = rrlo_env.observe(copy.deepcopy(tasks))
    #conference_state, _ = conference_env.observe(copy.deepcopy(tasks))

    for itr in tqdm(range(int(6e5))):

        if (itr + 1) % 100 == 0:
            cpu_generate= not cpu_generate
        # Run DVFS to assign tasks
        actions_dqn = dqn_dvfs.execute(dqn_state)
        actions_dqn_str = dqn_dvfs.conv_acts(actions_dqn)
        actions_rrlo, actions_rrlo_col = rrlo_dvfs.execute(rrlo_state)
        #actions_conference, actions_conference_col = conference_dvfs.execute(conference_state)

        # Execute tasks and get reward
        rewards_dqn, penalties_dqn, _ = dqn_env.step(actions_dqn_str)
        penalty_rrlo = rrlo_env.step(actions_rrlo)
        #penalty_conference = conference_env.step(actions_conference)

        # Observe next state
        if cpu_generate:
            target_cpu_load = next(cpu_load_generator)
            tasks = task_gen_cpu.step(target_cpu_load)
        else:
            target_cpu_load = next(cpu_load_generator)
            target_task_mean=next(task_mean_generator)
            tasks= task_gen_task.step(target_cpu_load,target_task_mean)
        next_state_dqn, is_final_dqn = dqn_env.observe(copy.deepcopy(tasks))
        next_state_rrlo, _ = rrlo_env.observe(copy.deepcopy(tasks))
        #next_state_conference, _ = conference_env.observe(copy.deepcopy(tasks))

        # Update RL network
        loss = dqn_dvfs.train(
            dqn_state, actions_dqn, rewards_dqn, next_state_dqn, is_final_dqn
        )
        dqn_loss.append(loss)
        rrlo_dvfs.update(rrlo_state, actions_rrlo_col, penalty_rrlo, next_state_rrlo)
        #conference_dvfs.update(conference_state, actions_conference_col, penalty_conference, next_state_conference)

        # Update current state
        dqn_state = next_state_dqn
        rrlo_state = next_state_rrlo
        #conference_state = next_state_conference

        # Print results
        #if (itr + 1) % 1000 == 0:
         #   tqdm.write(f"At {itr+1}, DQN loss={loss:.5f}")
          #  tqdm.write(
           #     f"Penalties DQN sum: {np.sum(penalties_dqn):.3e}, all: {penalties_dqn}"
            #)
            #tqdm.write(f"Penalties RRLO: {penalty_rrlo:.3e}")
            #tqdm.write(f"Penalties conference: {penalty_conference:.3e}")
            #tqdm.write(10 * "-")

    print("Saving trained model...")
    os.makedirs("models/rrlo_scenario", exist_ok=True)
    dqn_dvfs.save_model("models/rrlo_scenario")
    rrlo_dvfs.save_model("models/rrlo_scenario")
    #conference_dvfs.save_model("models/rrlo_scenario")
    return np.array(dqn_loss)

def train_dqn_scenario(configs):
    # Set random seed
    cpu_load_values=np.arange(0.01, 1.01, 0.01)
    cpu_load_generator = cycle(cpu_load_values)
    task_mean_values=np.arange(100, 505, 4)
    task_mean_generator = cycle(task_mean_values)

    cpu_generate=True

    task_gen_cpu = RandomTaskGen(configs["task_set"])
    task_gen_task= NormalTaskGen(configs["task_set"])
    dqn_env = BaseDQNEnv(configs, task_gen_cpu.get_wcet_bound(), task_gen_cpu.get_task_size_bound())
    #conference_env=RRLOEnv(configs)

    dqn_loss=[]

    # Initialize DVFS algorithms
    dqn_dvfs = DQN_DVFS(
        state_dim=configs["dqn_state_dim"],
        act_space=dqn_env.get_action_space(),
        batch_size=64,
        gamma=0.95,
        mem_size=1000,
        update_target_net=1000,
        eps_decay=1 / 200,
        min_eps=0,
    )

    #conference_dvfs = conference_DVFS(
     #   state_bounds=conference_env.get_state_bounds(),
      #  num_w_inter_powers=len(conference_env.w_inter.powers),
      #  num_dvfs_algs=1,
      #  dvfs_algs=["cc"],
       # num_tasks=4,
    #)

    # Initial state observation
    
    if cpu_generate:
        target_cpu_load = next(cpu_load_generator)
        tasks = task_gen_cpu.step(target_cpu_load)
    else:
        target_cpu_load = next(cpu_load_generator)
        target_task_mean=next(task_mean_generator)
        tasks= task_gen_task.step(target_cpu_load,target_task_mean)

    dqn_state, _ = dqn_env.observe(copy.deepcopy(tasks))
    #conference_state, _ = conference_env.observe(copy.deepcopy(tasks))

    for itr in tqdm(range(int(6e5))):

        if (itr + 1) % 100 == 0:
            cpu_generate= not cpu_generate

        # Run DVFS to assign tasks
        actions_dqn = dqn_dvfs.execute(dqn_state)
        actions_dqn_str = dqn_dvfs.conv_acts(actions_dqn)
        #actions_conference, actions_conference_col = conference_dvfs.execute(conference_state)

        # Execute tasks and get reward
        rewards_dqn, penalties_dqn, _ = dqn_env.step(actions_dqn_str)
        #penalty_conference = conference_env.step(actions_conference)


        # Observe next state
        if cpu_generate:
            target_cpu_load = next(cpu_load_generator)
            tasks = task_gen_cpu.step(target_cpu_load)
        else:
            target_cpu_load = next(cpu_load_generator)
            target_task_mean=next(task_mean_generator)
            tasks= task_gen_task.step(target_cpu_load,target_task_mean)
        next_state_dqn, is_final_dqn = dqn_env.observe(copy.deepcopy(tasks))
        #next_state_conference, _ = conference_env.observe(copy.deepcopy(tasks))

        # Update RL network
        loss = dqn_dvfs.train(
            dqn_state, actions_dqn, rewards_dqn, next_state_dqn, is_final_dqn
        )
        dqn_loss.append(loss)
        #conference_dvfs.update(conference_state, actions_conference_col, penalty_conference, next_state_conference)

        # Update current state
        dqn_state = next_state_dqn
        #conference_state = next_state_conference

        # Print results
       # if (itr + 1) % 1000 == 0:
        #    tqdm.write(f"At {itr+1}, DQN loss={loss:.5f}")
         #   tqdm.write(
          #      f"Penalties DQN sum: {np.sum(penalties_dqn):.3e}, all: {penalties_dqn}"
           # )
            #tqdm.write(f"Penalties conference: {penalty_conference:.3e}")
            #tqdm.write(10 * "-")

    print("Saving trained model...")
    os.makedirs("models/dqn_scenario", exist_ok=True)
    dqn_dvfs.save_model("models/dqn_scenario")
    #conference_dvfs.save_model("models/dqn_scenario")
    return np.array(dqn_loss)

if __name__ == "__main__":
    configs_rrlo_scenario = {
        "task_set": "configs/task_set_train.json",
        "cpu_local": "configs/cpu_local.json",
        "w_inter": "configs/wireless_interface.json",
        "dqn_state_dim": 4,
    }
    train_rrlo_scenario(configs_rrlo_scenario)

    configs_dqn_scenario = {
        "task_set": "configs/task_set_train.json",
        "cpu_little": "configs/cpu_little.json",
        "cpu_big": "configs/cpu_big.json",
        "w_inter": "configs/wireless_interface.json",
        "dqn_state_dim": 5,
    }

    train_dqn_scenario(configs_dqn_scenario)
