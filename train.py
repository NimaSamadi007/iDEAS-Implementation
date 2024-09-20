import numpy as np
from tqdm import tqdm
import copy
import os
from itertools import cycle,product

from env_models.env import BaseDQNEnv, DQNEnv, RRLOEnv
from env_models.task import RandomTaskGen, NormalTaskGen
from dvfs.dqn_dvfs import DQN_DVFS
from dvfs.rrlo_dvfs import RRLO_DVFS
from dvfs.conference_dvfs import conference_DVFS
from utils.utils import set_random_seed



def iDEAS_train(configs):
    # Set random seed
    max_task_load=5
    default_cn=1e-9
    cpu_load_values=np.arange(0.01, max_task_load, 0.2)
    cpu_load_generator=cycle(cpu_load_values)
    task_mean_values=np.arange(100, 505, 20)
    cn_values=np.logspace(np.log10(2e-13), np.log10(2e-4), num=50,base=10)
    generator=cycle(product(cpu_load_values,task_mean_values,cn_values))


    cpu_generate=False


    task_gen_cpu = RandomTaskGen(configs["task_set3"])
    task_gen_task= NormalTaskGen(configs["task_set3"])
    dqn_env = BaseDQNEnv(configs, task_gen_cpu.get_wcet_bound(), task_gen_cpu.get_task_size_bound())

    dqn_loss=[]
    all_rewards=[]

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

    

    if cpu_generate:
        target_cpu_load = next(cpu_load_generator)
        dqn_env.w_inter.cn_setter(default_cn)
        tasks = task_gen_cpu.step(target_cpu_load,max_task_load)
    
    else:
        target_cpu_load,target_task_mean,cn=next(generator)
        dqn_env.w_inter.cn_setter(cn)
        tasks=task_gen_task.step(target_cpu_load,target_task_mean,max_task_load)

    
    dqn_state, _ = dqn_env.observe(copy.deepcopy(tasks))

    for itr in tqdm(range(int(1e6))):

        if (itr % 50000) < 25000:
            cpu_generate = False
        else:
            cpu_generate = True

        # Run DVFS to assign tasks
        actions_dqn = dqn_dvfs.execute(dqn_state)
        actions_dqn_str = dqn_dvfs.conv_acts(actions_dqn)

        # Execute tasks and get reward
        rewards_dqn, penalties_dqn, _ = dqn_env.step(actions_dqn_str)


        # Observe next state
        if cpu_generate:
            target_cpu_load = next(cpu_load_generator)
            dqn_env.w_inter.cn_setter(default_cn)
            tasks = task_gen_cpu.step(target_cpu_load,max_task_load)
    
        else:
            target_cpu_load,target_task_mean,cn=next(generator)
            dqn_env.w_inter.cn_setter(cn)
            tasks=task_gen_task.step(target_cpu_load,target_task_mean,max_task_load)
        next_state_dqn, is_final_dqn = dqn_env.observe(copy.deepcopy(tasks))

        # Update RL network
        loss = dqn_dvfs.train(
            dqn_state, actions_dqn, rewards_dqn, next_state_dqn, is_final_dqn
        )
        dqn_loss.append(loss)
        all_rewards.append(rewards_dqn.tolist())

        # Update current state
        dqn_state = next_state_dqn
        
        # Print results
       # if (itr + 1) % 1000 == 0:
        #    tqdm.write(f"At {itr+1}, DQN loss={loss:.5f}")
         #   tqdm.write(
          #      f"Penalties DQN sum: {np.sum(penalties_dqn):.3e}, all: {penalties_dqn}"
           # )
            #tqdm.write(f"Penalties conference: {penalty_conference:.3e}")
            #tqdm.write(10 * "-")
    lossloss=dqn_dvfs.losses
    print("Saving trained model...")
    os.makedirs("models/iDEAS_train", exist_ok=True)
    dqn_dvfs.save_model("models/iDEAS_train")
    return np.array(lossloss), np.array(all_rewards)



def rrlo_train(configs):

    # Set random seed
    set_random_seed(42)
    max_task_load=4
    default_cn=1e-9
    cpu_load_values=np.arange(0.01, max_task_load, 0.2)
    cpu_load_generator=cycle(cpu_load_values)
    task_mean_values=np.arange(100, 505, 20)
    cn_values=np.logspace(np.log10(2e-13), np.log10(2e-4), num=50,base=10)
    generator=cycle(product(cpu_load_values,task_mean_values,cn_values))

    cpu_generate=False

    task_gen_cpu = RandomTaskGen(configs["task_set3"])
    task_gen_task= NormalTaskGen(configs["task_set3"])
    dqn_env = DQNEnv(configs, task_gen_cpu.get_wcet_bound(), task_gen_cpu.get_task_size_bound())
    rrlo_env = RRLOEnv(configs)

    dqn_loss=[]
    all_rewards=[]

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
    
    # Initial state observation
    if cpu_generate:
        target_cpu_load = next(cpu_load_generator)
        dqn_env.w_inter.cn_setter(default_cn)
        rrlo_env.w_inter.cn_setter(default_cn)
        tasks = task_gen_cpu.step(target_cpu_load,max_task_load)
    
    else:
        target_cpu_load,target_task_mean,cn=next(generator)
        dqn_env.w_inter.cn_setter(cn)
        rrlo_env.w_inter.cn_setter(cn)
        tasks=task_gen_task.step(target_cpu_load,target_task_mean,max_task_load)
    dqn_state, _ = dqn_env.observe(copy.deepcopy(tasks))
    rrlo_state, _ = rrlo_env.observe(copy.deepcopy(tasks))

    for itr in tqdm(range(int(1e6))):
        
        if (itr % 50000) < 25000:
            cpu_generate = False
        else:
            cpu_generate = True
        # Run DVFS to assign tasks
        actions_dqn = dqn_dvfs.execute(dqn_state)
        actions_dqn_str = dqn_dvfs.conv_acts(actions_dqn)
        actions_rrlo, actions_rrlo_col = rrlo_dvfs.execute(rrlo_state)

        # Execute tasks and get reward
        rewards_dqn, penalties_dqn, _ = dqn_env.step(actions_dqn_str)
        penalty_rrlo = rrlo_env.step(actions_rrlo)

        # Observe next state
        if cpu_generate:
            target_cpu_load = next(cpu_load_generator)
            dqn_env.w_inter.cn_setter(default_cn)
            rrlo_env.w_inter.cn_setter(default_cn)
            tasks = task_gen_cpu.step(target_cpu_load,max_task_load)
    
        else:
            target_cpu_load,target_task_mean,cn=next(generator)
            dqn_env.w_inter.cn_setter(cn)
            rrlo_env.w_inter.cn_setter(cn)
            tasks=task_gen_task.step(target_cpu_load,target_task_mean,max_task_load)
        next_state_dqn, is_final_dqn = dqn_env.observe(copy.deepcopy(tasks))
        next_state_rrlo, _ = rrlo_env.observe(copy.deepcopy(tasks))

        # Update RL network
        loss = dqn_dvfs.train(
            dqn_state, actions_dqn, rewards_dqn, next_state_dqn, is_final_dqn
        )
        dqn_loss.append(loss)
        all_rewards.append(rewards_dqn.tolist())
        rrlo_dvfs.update(rrlo_state, actions_rrlo_col, penalty_rrlo, next_state_rrlo)

        # Update current state
        dqn_state = next_state_dqn
        rrlo_state = next_state_rrlo

        # Print results
        #if (itr + 1) % 1000 == 0:
         #   tqdm.write(f"At {itr+1}, DQN loss={loss:.5f}")
          #  tqdm.write(
           #     f"Penalties DQN sum: {np.sum(penalties_dqn):.3e}, all: {penalties_dqn}"
            #)
            #tqdm.write(f"Penalties RRLO: {penalty_rrlo:.3e}")
            #tqdm.write(f"Penalties conference: {penalty_conference:.3e}")
            #tqdm.write(10 * "-")
    lossloss=dqn_dvfs.losses
    print("Saving trained model...")
    os.makedirs("models/RRLO_train", exist_ok=True)
    dqn_dvfs.save_model("models/RRLO_train")
    rrlo_dvfs.save_model("models/RRLO_train")
    return np.array(lossloss), np.array(all_rewards)



if __name__ == "__main__":
    configs_rrlo = {
        "task_set": "configs/task_set_train.json",
        "cpu_local": "configs/cpu_local.json",
        "w_inter": "configs/wireless_interface.json",
        "dqn_state_dim": 4,
    }
    rrlo_train(configs_rrlo)

    configs_iDEAS = {
        "task_set": "configs/task_set_train.json",
        "cpu_little": "configs/cpu_little.json",
        "cpu_big": "configs/cpu_big.json",
        "w_inter": "configs/wireless_interface.json",
        "dqn_state_dim": 5,
    }

    iDEAS_train(configs_iDEAS)
