import abc
import numpy as np
from tqdm import tqdm
import copy
import os
from itertools import cycle, product

from env_models.env import BaseDQNEnv, DQNEnv, RRLOEnv
from env_models.task import RandomTaskGen, NormalTaskGen
from dvfs.dqn_dvfs import DQN_DVFS
from dvfs.rrlo_dvfs import RRLO_DVFS
from utils.utils import set_random_seed


class Trainer(abc.ABC):
    def __init__(self, configs):
        # Set random seed
        set_random_seed(42)

        # Load and set parameters
        self.configs = configs
        self.task_configs = configs["tasks"]
        self.params = configs["params"]
        self.max_task_load, self.cpu_load_values, self.cn_values = self._load_params()

        # Initialize task generator
        self.task_gen = RandomTaskGen(self.task_configs["train"])

        # Init environments
        self._init_envs()

        # Init DVFS algorithm
        self._init_algs()

    def run(self):
        # Reset envs:
        states, _ = self.reset()

        all_rewards = []
        all_losses = []
        NUM_TRAIN_ITR = int(self.params["train_itr"])
        for itr in tqdm(range(NUM_TRAIN_ITR)):
            # Run DVFS algorithm
            actions = self._run_algs(states)

            if itr % 100 == 0:
                print(states)
                print(actions)

            # Step environment
            rewards = self._step_envs(actions)

            # Observe next state
            next_states, is_final = self._observe()

            # Train DVFS algorithm
            loss = self._train_algs(states, actions, rewards, next_states, is_final)

            if loss:
                all_rewards.append(rewards)
                all_losses.append(loss)

            # Make transition to the next state
            states = next_states

        # Save trained models
        self._save_algs()

        return np.array(all_losses), np.array(all_rewards)

    def _load_params(self):
        max_task_load = self.params["max_task_load_train"]
        cpu_load_values = np.arange(0.01, max_task_load, 0.1)
        cn_values = np.logspace(np.log10(2e-11), np.log10(2e-6), num=10, base=10)

        return max_task_load, cpu_load_values, cn_values

    @abc.abstractmethod
    def _init_envs(self):
        pass

    @abc.abstractmethod
    def _init_algs(self):
        pass

    @abc.abstractmethod
    def _observe(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def _run_algs(self, states):
        pass

    @abc.abstractmethod
    def _step_envs(self, actions):
        pass

    @abc.abstractmethod
    def _train_algs(self, states, actions, rewards, next_states, is_final):
        pass

    @abc.abstractmethod
    def _save_algs(self):
        pass


class iDEAS_BaseTrainer(Trainer):
    def _init_envs(self):
        self.env = BaseDQNEnv(
            self.configs,
            self.task_gen.get_wcet_bound(),
            self.task_gen.get_task_size_bound(),
        )

    def _init_algs(self):
        self.alg = DQN_DVFS(
            state_dim=self.params["dqn_state_dim"],
            act_space=self.env.get_action_space(),
            batch_size=self.params["batch_size"],
            gamma=self.params["gamma"],
            mem_size=self.params["mem_size"],
            update_target_net=self.params["update_target_net"],
            eps_decay=self.params["eps_decay"],
            min_eps=self.params["min_eps"],
        )

    def _observe(self):
        target_cpu_load = np.random.choice(self.cpu_load_values)
        cn = np.random.choice(self.cn_values)
        self.env.w_inter.set_cn(cn)
        self.tasks = self.task_gen.step(target_cpu_load, self.max_task_load)
        state, is_final = self.env.observe(self.tasks)

        return state, is_final

    def reset(self):
        return self._observe()

    def _run_algs(self, states):
        actions_raw = self.alg.execute(states)
        actions_str = self.alg.conv_acts(actions_raw)
        actions = {'raw': actions_raw, 'str': actions_str}
        return actions

    def _step_envs(self, actions):
        rewards, penalites, min_penalties = self.env.step(actions['str'])
        return rewards

    def _train_algs(self, states, actions, rewards, next_states, is_final):
        return self.alg.train(states, actions['raw'], rewards, next_states, is_final)

    def _save_algs(self):
        os.makedirs("models/iDEAS_Base", exist_ok=True)
        self.alg.save_model("models/iDEAS_Base")


class iDEAS_RRLOTrainer(Trainer):
    def _init_envs(self):
        pass

    def _init_algs(self):
        pass


def rrlo_train(configs):
    # Set random seed
    set_random_seed(42)
    max_task_load = 4
    default_cn = 1e-9
    cpu_load_values = np.arange(0.01, max_task_load, 0.2)
    cpu_load_generator = cycle(cpu_load_values)
    task_mean_values = np.arange(100, 505, 20)
    cn_values = np.logspace(np.log10(2e-13), np.log10(2e-4), num=50, base=10)
    generator = cycle(product(cpu_load_values, task_mean_values, cn_values))

    cpu_generate = False

    task_gen_cpu = RandomTaskGen(configs["train"])
    task_gen_task = NormalTaskGen(configs["train"])
    dqn_env = DQNEnv(
        configs, task_gen_cpu.get_wcet_bound(), task_gen_cpu.get_task_size_bound()
    )
    rrlo_env = RRLOEnv(configs)

    dqn_loss = []
    all_rewards = []

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
        dqn_env.w_inter.set_cn(default_cn)
        rrlo_env.w_inter.set_cn(default_cn)
        tasks = task_gen_cpu.step(target_cpu_load, max_task_load)

    else:
        target_cpu_load, target_task_mean, cn = next(generator)
        dqn_env.w_inter.set_cn(cn)
        rrlo_env.w_inter.set_cn(cn)
        tasks = task_gen_task.step(target_cpu_load, target_task_mean, max_task_load)
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
            dqn_env.w_inter.set_cn(default_cn)
            rrlo_env.w_inter.set_cn(default_cn)
            tasks = task_gen_cpu.step(target_cpu_load, max_task_load)

        else:
            target_cpu_load, target_task_mean, cn = next(generator)
            dqn_env.w_inter.set_cn(cn)
            rrlo_env.w_inter.set_cn(cn)
            tasks = task_gen_task.step(target_cpu_load, target_task_mean, max_task_load)
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
        # if (itr + 1) % 1000 == 0:
        #   tqdm.write(f"At {itr+1}, DQN loss={loss:.5f}")
        #  tqdm.write(
        #     f"Penalties DQN sum: {np.sum(penalties_dqn):.3e}, all: {penalties_dqn}"
        # )
        # tqdm.write(f"Penalties RRLO: {penalty_rrlo:.3e}")
        # tqdm.write(f"Penalties conference: {penalty_conference:.3e}")
        # tqdm.write(10 * "-")
    lossloss = dqn_dvfs.losses
    print("Saving trained model...")
    os.makedirs("models/RRLO_train", exist_ok=True)
    dqn_dvfs.save_model("models/RRLO_train")
    rrlo_dvfs.save_model("models/RRLO_train")
    return np.array(lossloss), np.array(all_rewards)
