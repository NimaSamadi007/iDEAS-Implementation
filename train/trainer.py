import abc
import numpy as np
from tqdm import tqdm
import copy
import os

from env_models.env import HetrogenEnv, HomogenEnv, RRLOEnv
from env_models.task import RandomTaskGen
from dvfs.dqn_dvfs import DQN_DVFS
from dvfs.rrlo_dvfs import RRLO_DVFS
from utils.utils import set_random_seed


class Trainer(abc.ABC):
    def __init__(self, configs):
        # Set random seed
        set_random_seed(42)

        # Load and set parameters
        self.configs = configs
        self.params = configs["params"]
        self.task_configs = configs["tasks"]
        self.num_tasks = self.params["num_tasks"]
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

            if itr % 500 == 0:
                tqdm.write(f"States:\n{str(states)}")
                tqdm.write(f"Actions:\n{str(actions)}")
                tqdm.write(20 * "=")

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

        return all_losses, all_rewards

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


class iDEAS_MainTrainer(Trainer):
    def _init_envs(self):
        self.env = HetrogenEnv(
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
        actions = {"raw": actions_raw, "str": actions_str}
        return actions

    def _step_envs(self, actions):
        rewards, penalites, min_penalties = self.env.step(actions["str"])
        return rewards

    def _train_algs(self, states, actions, rewards, next_states, is_final):
        return self.alg.train(states, actions["raw"], rewards, next_states, is_final)

    def _save_algs(self):
        os.makedirs("models/iDEAS_Main", exist_ok=True)
        self.alg.save_model("models/iDEAS_Main")

class iDEAS_BaselineTrainer(iDEAS_MainTrainer):
    def _save_algs(self):
        os.makedirs("models/iDEAS_Baseline", exist_ok=True)
        self.alg.save_model("models/iDEAS_Baseline")

class iDEAS_RRLOTrainer(Trainer):
    def _init_envs(self):
        self.ideas_env = HomogenEnv(
            self.configs,
            self.task_gen.get_wcet_bound(),
            self.task_gen.get_task_size_bound(),
        )
        self.rrlo_env = RRLOEnv(self.configs)

    def _init_algs(self):
        self.ideas_dvfs = DQN_DVFS(
            state_dim=self.params["dqn_state_dim"],
            act_space=self.ideas_env.get_action_space(),
            batch_size=self.params["batch_size"],
            gamma=self.params["gamma"],
            mem_size=self.params["mem_size"],
            update_target_net=self.params["update_target_net"],
            eps_decay=self.params["eps_decay"],
            min_eps=self.params["min_eps"],
        )
        self.rrlo_dvfs = RRLO_DVFS(
            state_bounds=self.rrlo_env.get_state_bounds(),
            num_w_inter_powers=len(self.rrlo_env.w_inter.powers),
            num_dvfs_algs=2,
            dvfs_algs=["cc", "la"],
            num_tasks=self.num_tasks,
        )

    def _observe(self):
        states = {}
        is_final = {}

        target_cpu_load = np.random.choice(self.cpu_load_values)
        cn = np.random.choice(self.cn_values)
        self.ideas_env.w_inter.set_cn(cn)
        self.rrlo_env.w_inter.set_cn(cn)
        self.tasks = self.task_gen.step(target_cpu_load, self.max_task_load)

        states_tmp, is_final_tmp = self.ideas_env.observe(copy.deepcopy(self.tasks))
        states["ideas"] = states_tmp
        is_final["ideas"] = is_final_tmp

        states_tmp, _ = self.rrlo_env.observe(copy.deepcopy(self.tasks))
        states["rrlo"] = states_tmp

        return states, is_final

    def reset(self):
        return self._observe()

    def _run_algs(self, states):
        # FIXME: Return two action types as a tuple
        actions_ideas_raw = self.ideas_dvfs.execute(states["ideas"])
        actions_ideas_str = self.ideas_dvfs.conv_acts(actions_ideas_raw)

        actions_rrlo_str, actions_rrlo_raw = self.rrlo_dvfs.execute(states["rrlo"])

        actions = {
            "ideas": {"raw": actions_ideas_raw, "str": actions_ideas_str},
            "rrlo": {"raw": actions_rrlo_raw, "str": actions_rrlo_str},
        }
        return actions

    def _step_envs(self, actions):
        rewards_ideas, penalites, min_penalties = self.ideas_env.step(
            actions["ideas"]["str"]
        )
        penalty_rrlo = self.rrlo_env.step(actions["rrlo"]["str"])

        # For RRLO, penalty is used for updating the Q-table and using
        # it in 'rewrads' variable may be misleading
        rewards = {"ideas": rewards_ideas, "rrlo": penalty_rrlo}
        return rewards

    def _train_algs(self, states, actions, rewards, next_states, is_final):
        loss_ideas = self.ideas_dvfs.train(
            states["ideas"],
            actions["ideas"]["raw"],
            rewards["ideas"],
            next_states["ideas"],
            is_final["ideas"],
        )
        # This is misleading but we are actually passing penalty, not reward
        # for RRLO
        penalty_rrlo = rewards["rrlo"]
        self.rrlo_dvfs.update(
            states["rrlo"],
            actions["rrlo"]["raw"],
            penalty_rrlo,
            next_states["rrlo"],
        )
        return loss_ideas

    def _save_algs(self):
        os.makedirs("models/iDEAS_RRLO", exist_ok=True)
        self.ideas_dvfs.save_model("models/iDEAS_RRLO")

        os.makedirs("models/iDEAS_RRLO", exist_ok=True)
        self.rrlo_dvfs.save_model("models/iDEAS_RRLO")
