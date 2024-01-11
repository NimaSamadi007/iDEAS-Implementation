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
                        min_eps=0)

    rrlo_dvfs = RRLO_DVFS(state_bounds=rrlo_env.get_state_bounds(),
                          num_dvfs_algs=4,
                          num_tasks=4)

    dqn_state,_ = dqn_env.observe()
    rrlo_state,_ = rrlo_env.observe()
    print(f"Number of states: {rrlo_env.state_steps}")
    print(f"State values: {rrlo_state}")
    print(f"Row id: {rrlo_dvfs._conv_state_to_row(rrlo_state)}")

    print(f"Q table shape: {rrlo_dvfs.Q_table_a.shape}")
    print(f"Q table b shape: {rrlo_dvfs.Q_table_b.shape}")
    """
    rrlo_dvfs = RRLO_DVFS()

    # Initial state observation
    state, _ = our_env.observe()

    for itr in range(NUM_ITR):
        # Run DVFS to assign tasks
        actions = dvfs_alg.execute(state)
        actions_str = dvfs_alg.conv_acts(actions)
        # Execute tasks and get reward
        rewards, penalties, min_penalties = env.step(actions_str)
        # Observe next state
        next_state, is_final = env.observe(100)
        # Update RL network
        loss = dvfs_alg.train(state,
                              actions,
                              rewards,
                              next_state,
                              is_final)
        # Update current state
        state = next_state

        # Print results
        # all_rewards.append(rewards.tolist())
        # all_penalties.append(penalties.tolist())
        # all_min_penalties.append(min_penalties.tolist())
        # if (itr+1) % 500 == 0:
        #     print(f"At {itr+1}, loss={loss:.3f}")
        #     print(f"Actions: {actions_str}")
        #     print(f"Rewards: {rewards}")
        #     print(f"Penalties: {penalties}")
        #     print(f"Min penalties: {min_penalties}")
        #     print(10*"-")

    # all_rewards = np.array(all_rewards)
    # all_penalties = np.array(all_penalties)
    # all_min_penalties = np.array(all_min_penalties)

    # # Plot results
    # print(f"Current eps val: {dvfs_alg.eps}")
    # plot_all_rewards(all_rewards)
    # plot_loss_function(dvfs_alg.losses)
    # for i in range(4):
    #     plot_penalty(all_penalties[:, i], all_min_penalties[:, i], i)
    """