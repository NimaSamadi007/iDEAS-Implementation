import numpy as np

from configs import *
from models.env import TOMSEnv
from dvfs.dvfs import DVFS
from utils.utils import *

# Main function
if __name__ == '__main__':
    # Set random seed
    set_random_seed(42)

    ## Load tasks and CPU models
    configs = {"task_set": "configs/TOMS/task_set.json",
               "cpu_conf": "configs/TOMS/cpu_conf.json",
               "w_inter": "configs/TOMS/wireless_interface.json"}
    env = TOMSEnv(configs)

    # Initialize RL network
    dvfs_alg = DVFS(state_dim=STATE_DIM,
                    act_space=env.get_action_space(),
                    batch_size=32,
                    gamma=0.90,
                    update_target_net=1000,
                    eps_update_step=2000,
                    eps_decay = 1e-2,
                    min_eps=0)

    all_rewards = []
    all_penalties = []
    all_min_penalties = []

    # Initial state observation
    state, _ = env.observe()
    for itr in range(int(2.5e5)):
        # Run DVFS to assign tasks
        actions = dvfs_alg.execute(state)
        actions_str = dvfs_alg.conv_acts(actions)
        # Execute tasks and get reward
        rewards, penalties, min_penalties = env.step(actions_str)
        # Observe next state
        next_state, is_final = env.observe()
        # Update RL network
        loss = dvfs_alg.train(state,
                            actions,
                            rewards,
                            next_state,
                            is_final)
        # Update current state
        state = next_state

        # Print results
        all_rewards.append(rewards.tolist())
        all_penalties.append(penalties.tolist())
        all_min_penalties.append(min_penalties.tolist())
        if (itr+1) % 500 == 0:
            print(f"At {itr+1}, loss={loss:.3f}")
            print(f"eps: {dvfs_alg.eps:.2f}")
            print(f"Rewards: {rewards}")
            print(f"Penalties: {penalties}")
            print(f"Min penalties: {min_penalties}")
            print(10*"-")

    # Save model:
    dvfs_alg.save_model("models/TOMS/dvfs.pt")

    # Plot results
    all_rewards = np.array(all_rewards)
    all_penalties = np.array(all_penalties)
    all_min_penalties = np.array(all_min_penalties)

    plot_all_rewards(all_rewards)
    plot_loss_function(dvfs_alg.losses)
    for i in range(all_penalties.shape[1]):
        plot_penalty(all_penalties[:, i], all_min_penalties[:, i], i)
