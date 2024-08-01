import numpy as np

from configs import DQN_STATE_DIM
from env_models.env import Env, RRLOEnv
from dvfs.dvfs import DQN_DVFS, RRLO_DVFS
from utils.utils import set_random_seed


# Main function
def train():
    # Set random seed
    set_random_seed(42)

    ## Load tasks and CPU models
    configs = {
        "task_set": "configs/task_set_train.json",
        "cpu_local": "configs/cpu_local.json",
        "w_inter": "configs/wireless_interface.json",
    }
    dqn_env = Env(configs)
    rrlo_env = RRLOEnv(configs)

    # Initialize DVFS algorithms
    dqn_dvfs = DQN_DVFS(
        state_dim=DQN_STATE_DIM,
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
    dqn_state, _ = dqn_env.observe()
    rrlo_state, _ = rrlo_env.observe()

    for itr in range(int(4e3)):
        # Run DVFS to assign tasks
        actions_dqn = dqn_dvfs.execute(dqn_state)
        actions_dqn_str = dqn_dvfs.conv_acts(actions_dqn)
        actions_rrlo, actions_rrlo_col = rrlo_dvfs.execute(rrlo_state)

        # Execute tasks and get reward
        rewards_dqn, penalties_dqn, min_penalties_dqn = dqn_env.step(actions_dqn_str)
        penalty_rrlo = rrlo_env.step(actions_rrlo)

        # Observe next state
        next_state_dqn, is_final_dqn = dqn_env.observe()
        next_state_rrlo, _ = rrlo_env.observe()

        # Update RL network
        loss = dqn_dvfs.train(
            dqn_state, actions_dqn, rewards_dqn, next_state_dqn, is_final_dqn
        )
        rrlo_dvfs.update(rrlo_state, actions_rrlo_col, penalty_rrlo, next_state_rrlo)

        # Update current state
        dqn_state = next_state_dqn
        rrlo_state = next_state_rrlo

        # Print results
        if (itr + 1) % 1000 == 0:
            print(f"At {itr+1}, DQN loss={loss:.5f}")
            print(
                f"Penalties DQN sum: {np.sum(penalties_dqn):.3e}, all: {penalties_dqn}"
            )
            print(f"Penalties RRLO: {penalty_rrlo:.3e}")
            print(10 * "-")

    print("Saving trained model...")
    dqn_dvfs.save_model("models")
    rrlo_dvfs.save_model("models")


if __name__ == "__main__":
    train()
