import numpy as np

from configs import *
from models.env import Env, RRLOEnv
from dvfs.dvfs import DQN_DVFS, RRLO_DVFS
from utils.utils import *

# Main function
if __name__ == '__main__':
    # Set random seed
    set_random_seed(42)

    ## Load tasks and CPU models
    configs = {"task_set": "configs/task_set.json",
               "cpu_local": "configs/cpu_local.json",
               "w_inter": "configs/wireless_interface.json"
               }

    dqn_env = Env(configs)
    rrlo_env = RRLOEnv(configs)

    # Initialize DVFS algorithms
    dqn_dvfs = DQN_DVFS(state_dim=DQN_STATE_DIM,
                        act_space=dqn_env.get_action_space(),
                        batch_size=32,
                        gamma=0.90,
                        update_target_net= 1000,
                        eps_decay = 1/2000,
                        min_eps=0.1)

    rrlo_dvfs = RRLO_DVFS(state_bounds=rrlo_env.get_state_bounds(),
                          num_w_inter_powers=len(rrlo_env.w_inter.powers),
                          num_dvfs_algs=1,
                          dvfs_algs=["cc"],
                          num_tasks=4)

    # Initial state observation
    dqn_state,_ = dqn_env.observe()
    rrlo_state,_ = rrlo_env.observe()

    for itr in range(int(1.5e5)):
        # Run DVFS to assign tasks
        actions_dqn = dqn_dvfs.execute(dqn_state)
        actions_dqn_str = dqn_dvfs.conv_acts(actions_dqn)
        actions_rrlo, actions_rrlo_col = rrlo_dvfs.execute(rrlo_state)

        # print(f"DQN actions: {actions_dqn_str}")
        # print(f"RRLO actions: {actions_rrlo}")

        # # Execute tasks and get reward
        rewards_dqn, penalties_dqn, min_penalties_dqn = dqn_env.step(actions_dqn_str)
        penalty_rrlo = rrlo_env.step(actions_rrlo)
        # for t in finished_tasks:
        #     print(f"T_{t.t_id} exec time:")
        #     for i, e_time in enumerate(t.exec_time_history):
        #         print(f"\t{i}: {e_time[0]:.3f} -- {e_time[1]:.3f} @ {t.exec_freq_history[i]}")
        #     print(f"\tconsumed {t.cons_energy:.3f} mJ")
        #     print(20*'-')

        # Observe next state
        next_state_dqn, is_final_dqn = dqn_env.observe()
        next_state_rrlo,_ = rrlo_env.observe()

        # Update RL network
        loss = dqn_dvfs.train(dqn_state,
                              actions_dqn,
                              rewards_dqn,
                              next_state_dqn,
                              is_final_dqn)
        rrlo_dvfs.update(rrlo_state, actions_rrlo_col, penalty_rrlo, next_state_rrlo)

        # Update current state
        dqn_state = next_state_dqn
        rrlo_state = next_state_rrlo

        # Print results
        # all_rewards.append(rewards.tolist())
        # all_penalties.append(penalties.tolist())
        # all_min_penalties.append(min_penalties.tolist())
        if (itr+1) % 500 == 0:
            print(f"At {itr+1}, loss={loss:.3f}")
            # print(f"Actions: {actions_str}")
            # print(f"Rewards: {rewards}")
            # print(f"Penalties: {penalties}")
            # print(f"Min penalties: {min_penalties}")
            print(10*"-")

    # all_rewards = np.array(all_rewards)
    # all_penalties = np.array(all_penalties)
    # all_min_penalties = np.array(all_min_penalties)

    # # Plot results
    # print(f"Current eps val: {dvfs_alg.eps}")
    # plot_all_rewards(all_rewards)
    # plot_loss_function(dvfs_alg.losses)
    # for i in range(4):
    #     plot_penalty(all_penalties[:, i], all_min_penalties[:, i], i)
    # np.save("q_a.npy", rrlo_dvfs.Q_table_a)
    # np.save("q_b.npy", rrlo_dvfs.Q_table_b)

    # Evaluate the trained model
    print("Evaluating the trained model...")
    dqn_task_energy_cons = np.zeros(4)
    dqn_num_tasks = np.zeros(4)
    rrlo_task_energy_cons = np.zeros(4)
    rrlo_num_tasks = np.zeros(4)

    dqn_state,_ = dqn_env.observe()
    rrlo_state,_ = rrlo_env.observe()
    for itr in range(5000):
        actions_dqn = dqn_dvfs.execute(dqn_state, eval_mode=True)
        actions_dqn_str = dqn_dvfs.conv_acts(actions_dqn)
        actions_rrlo, actions_rrlo_col = rrlo_dvfs.execute(rrlo_state)

        rewards_dqn, penalties_dqn, min_penalties_dqn = dqn_env.step(actions_dqn_str)
        penalty_rrlo = rrlo_env.step(actions_rrlo)

        # Save energy consumption
        for jobs in dqn_env.curr_tasks.values():
            for j in jobs:
                dqn_task_energy_cons[j.t_id] = j.cons_energy
            dqn_num_tasks[j.t_id] += len(jobs)

        for jobs in rrlo_env.curr_tasks.values():
            for j in jobs:
                rrlo_task_energy_cons[j.t_id] = j.cons_energy
            rrlo_num_tasks[j.t_id] += len(jobs)

        next_state_dqn, is_final_dqn = dqn_env.observe()
        next_state_rrlo,_ = rrlo_env.observe()

        # Update current state
        dqn_state = next_state_dqn
        rrlo_state = next_state_rrlo

    # Average energy consumption:
    print(dqn_num_tasks)
    print(rrlo_num_tasks)
    np.set_printoptions(suppress=True)
    print(f"DQN energy consumption: {dqn_task_energy_cons/dqn_num_tasks}")
    print(f"RRLO energy consumption: {rrlo_task_energy_cons/rrlo_num_tasks}")

    print(f"DQN task set avg energy consumption: {np.sum(dqn_task_energy_cons)/np.sum(dqn_num_tasks)}")
    print(f"RRLO task set avg energy consumption: {np.sum(rrlo_task_energy_cons)/np.sum(rrlo_num_tasks)}")
