import abc
import copy
import numpy as np
from tqdm import tqdm

from env_models.env import HetrogenEnv, HomogenEnv, RRLOEnv
from env_models.task import TaskGen, NormalTaskGen, RandomTaskGen
from dvfs.dqn_dvfs import DQN_DVFS
from dvfs.rrlo_dvfs import RRLO_DVFS


class Evaluator(abc.ABC):
    def __init__(self, configs, cpu_loads, task_sizes, cns):
        self.configs = configs
        self.params = self.configs["params"]
        self.tasks_conf = self.configs["tasks"]

        self.do_taskset_eval = self.params["do_taskset_eval"]
        self.do_cpu_load_eval = self.params["do_cpu_load_eval"]
        self.do_task_size_eval = self.params["do_task_size_eval"]
        self.do_channel_eval = self.params["do_channel_eval"]
        self.eval_itr = self.params["eval_itr"]
        self.num_tasks = self.params["num_tasks"]

        self.cpu_loads = cpu_loads
        self.task_sizes = task_sizes
        self.cns = cns

        np.set_printoptions(suppress=True)

    def run(self):
        results = {}
        if self.do_taskset_eval:
            print("Evaluating fixed taskset scenario:")
            result = self._eval_fixed_taskset()
            results.update(result)
            print(100 * "-")
        if self.do_cpu_load_eval:
            print("Evaluating cpu load variation:")
            result = self._eval_varied_cpuload()
            results.update(result)
            print(100 * "-")
        if self.do_task_size_eval:
            print("Evaluating task size variation:")
            result = self._eval_varied_tasksize()
            results.update(result)
            print(100 * "-")
        if self.do_channel_eval:
            print("Evaluating varied channel scenario:")
            result = self._eval_varied_channel()
            results.update(result)
            print(100 * "-")

        return results

    def _eval_fixed_taskset(self):
        # Results container
        scenario_name = "fixed_taskset"
        self._init_results_container(scenario_name)

        for task_id in range(2):
            print(f"Taskset {task_id+1}")
            self.task_gen = TaskGen(self.tasks_conf[f"eval_{task_id+1}"])
            # Create environments to evaluate algorithms on
            self.envs = self._init_envs()
            # Init DVFS algorithms
            self.algs = self._init_algs()
            # Generate tasks
            self.tasks = self.task_gen.step()

            # Observe initial state
            states = self._observe(self.tasks)
            for _ in tqdm(range(self.eval_itr)):
                # Run DVFS algorithms and baselines
                self.actions = self._run_algs(states)

                # Apply actions on the environments
                self._step_envs(self.actions)

                # Gather results
                self._process_results(task_id)

                # Observe next states
                self.tasks = self.task_gen.step()
                next_states = self._observe(self.tasks)
                # Make transition
                states = next_states

        return self._get_results(scenario_name)

    def _eval_varied_cpuload(self):
        # Results container
        scenario_name = "varied_cpuload"
        max_task_load = self.params["max_task_load_eval"]
        self._init_results_container(scenario_name)

        self.task_gen = RandomTaskGen(self.tasks_conf["train"])
        # Create environments to evaluate algorithms on
        self.envs = self._init_envs()
        # Init DVFS algorithms
        self.algs = self._init_algs()

        for i in range(len(self.cpu_loads)):
            print(f"CPU Load {self.cpu_loads[i]:.3f}")
            target_cpu_load = self.cpu_loads[i]
            # Generate tasks
            self.tasks = self.task_gen.step(target_cpu_load, max_task_load)
            # Observe initial state
            states = self._observe(self.tasks)
            for _ in tqdm(range(self.eval_itr)):
                # Run DVFS algorithms and baselines
                self.actions = self._run_algs(states)

                # Apply actions on the environments
                self._step_envs(self.actions)

                # Gather results
                self._process_results(i)

                # Observe next states
                self.tasks = self.task_gen.step(target_cpu_load, max_task_load)
                next_states = self._observe(self.tasks)
                # Make transition
                states = next_states

        return self._get_results(scenario_name)

    def _eval_varied_tasksize(self):
        # Results container
        scenario_name = "varied_tasksize"
        target_cpu_load = self.params["target_cpu_load"]
        max_task_load = self.params["max_task_load_eval"]
        self._init_results_container(scenario_name)

        self.task_gen = NormalTaskGen(self.tasks_conf["train"])
        # Create environments to evaluate algorithms on
        self.envs = self._init_envs()
        # Init DVFS algorithms
        self.algs = self._init_algs()

        # Generate tasks
        self.tasks = self.task_gen.step(
            target_cpu_load, self.task_sizes[0], max_task_load
        )
        # Observe initial state
        states = self._observe(self.tasks)
        for i in range(len(self.task_sizes) - 1):
            print(f"Task Size {self.task_sizes[i]:.3f}")
            for _ in tqdm(range(self.eval_itr)):
                # Run DVFS algorithms and baselines
                self.actions = self._run_algs(states)

                # Apply actions on the environments
                self._step_envs(self.actions)

                # Gather results
                self._process_results(i)

                # Observe next states
                if i == len(self.task_sizes) - 2:
                    self.tasks = self.task_gen.step(
                        target_cpu_load, self.task_sizes[i], max_task_load
                    )
                else:
                    self.tasks = self.task_gen.step(
                        target_cpu_load, self.task_sizes[i + 1], max_task_load
                    )
                next_states = self._observe(self.tasks)
                # Make transition
                states = next_states

        return self._get_results(scenario_name)

    def _eval_varied_channel(self):
        # Results container
        scenario_name = "varied_channel"
        target_cpu_load = self.params["target_cpu_load"]
        max_task_load = self.params["max_task_load_eval"]
        self._init_results_container(scenario_name)

        self.task_gen = RandomTaskGen(self.tasks_conf["train"])
        # Create environments to evaluate algorithms on
        self.envs = self._init_envs()
        # Init DVFS algorithms
        self.algs = self._init_algs()

        for i in range(len(self.cns)):
            print(f"Channel noise: {self.cns[i]}")
            # Generate tasks
            self.tasks = self.task_gen.step(target_cpu_load, max_task_load)
            # Change channel state
            for env_tmp in self.envs.values():
                env_tmp.w_inter.set_cn(self.cns[i])

            # Observe initial state
            states = self._observe(self.tasks)
            for _ in tqdm(range(self.eval_itr)):
                # Run DVFS algorithms and baselines
                self.actions = self._run_algs(states)

                # Apply actions on the environments
                self._step_envs(self.actions)

                # Gather results
                self._process_results(i)

                # Observe next states
                self.tasks = self.task_gen.step(target_cpu_load, max_task_load)
                next_states = self._observe(self.tasks)
                # Make transition
                states = next_states

        return self._get_results(scenario_name)

    @abc.abstractmethod
    def _get_results(self, scenario_name):
        pass

    def _step_envs(self, actions):
        for method in actions:
            self.envs[method].step(actions[method])

    @abc.abstractmethod
    def _init_results_container(self, scenario):
        pass

    @abc.abstractmethod
    def _process_results(self, idx):
        pass

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
    def _run_algs(self, states):
        pass


class iDEAS_MainEvaluator(Evaluator):
    def _init_results_container(self, scenario):
        if scenario == "fixed_taskset":
            self.num_results_item = 2
        elif scenario == "varied_cpuload":
            self.num_results_item = len(self.cpu_loads)
        elif scenario == "varied_tasksize":
            self.num_results_item = len(self.task_sizes) - 1
        elif scenario == "varied_channel":
            self.num_results_item = len(self.cns)
        else:
            raise ValueError(f"Unknown Scenario! {scenario}")

        # 0: energy, 1: num_tasks, 2: missed deadline
        total_stat = np.zeros((3, self.num_tasks, self.num_results_item))
        big_stat = np.zeros((3, self.num_tasks, self.num_results_item))
        little_stat = np.zeros((3, self.num_tasks, self.num_results_item))
        offload_stat = np.zeros((3, self.num_tasks, self.num_results_item))

        self.raw_results = {
            "big": big_stat,
            "little": little_stat,
            "offload": offload_stat,
            "total": total_stat,
        }

    def _process_results(self, idx):
        for target in ["little", "big", "offload"]:
            for t_id, _ in self.actions["ideas"][target]:
                stat = self.raw_results[target]
                total_stat = self.raw_results["total"]
                for job in self.envs["ideas"].curr_tasks[t_id]:
                    stat[0, job.t_id, idx] += job.cons_energy
                    total_stat[0, job.t_id, idx] += job.cons_energy
                    if job.deadline_missed:
                        stat[2, job.t_id, idx] += 1
                        total_stat[2, job.t_id, idx] += 1
                stat[1, job.t_id, idx] += len(self.envs["ideas"].curr_tasks[t_id])
                total_stat[1, job.t_id, idx] += len(self.envs["ideas"].curr_tasks[t_id])

    def _get_results(self, scenario_name):
        results = {
            f"{scenario_name}_{scen}": np.zeros(
                (len(self.raw_results), self.num_results_item)
            )
            for scen in ["energy", "drop"]
        }

        num_tasks = self.raw_results["total"][1]
        for i, raw_result in enumerate(self.raw_results.values()):
            energy = raw_result[0]
            missed_deadline = raw_result[2]
            results[f"{scenario_name}_energy"][i] = (
                np.sum(energy, axis=0) / np.sum(num_tasks, axis=0) * 100
            )
            results[f"{scenario_name}_drop"][i] = (
                np.sum(missed_deadline, axis=0) / np.sum(num_tasks, axis=0) * 100
            )

        return results

    def _init_envs(self):
        env_tmp = HetrogenEnv(
            self.configs,
            self.task_gen.get_wcet_bound(),
            self.task_gen.get_task_size_bound(),
        )
        return {"ideas": env_tmp}

    def _init_algs(self):
        dqn_dvfs = DQN_DVFS(
            state_dim=self.params["dqn_state_dim"],
            act_space=self.envs["ideas"].get_action_space(),
        )
        dqn_dvfs.load_model("models/iDEAS_Main")
        return {"ideas": dqn_dvfs}

    def _observe(self, tasks):
        # FIXME: Make sure if copying tasks is needed here
        states, _ = self.envs["ideas"].observe(copy.deepcopy(tasks))
        return {"ideas": states}

    def _run_algs(self, states):
        actions_raw = self.algs["ideas"].execute(states["ideas"])
        actions_str = self.algs["ideas"].conv_acts(actions_raw)
        return {"ideas": actions_str}


class iDEAS_RRLOEvaluator(Evaluator):
    def _init_envs(self):
        envs = {}
        env_temp = HomogenEnv(
            self.configs,
            self.task_gen.get_wcet_bound(),
            self.task_gen.get_task_size_bound(),
        )
        envs["random"] = copy.deepcopy(env_temp)
        envs["rrlo"] = RRLOEnv(self.configs)
        envs["ideas"] = copy.deepcopy(env_temp)
        envs["local"] = copy.deepcopy(env_temp)
        envs["remote"] = copy.deepcopy(env_temp)

        return envs

    def _init_algs(self):
        dqn_dvfs = DQN_DVFS(
            state_dim=self.params["dqn_state_dim"],
            act_space=self.envs["ideas"].get_action_space(),
        )
        dqn_dvfs.load_model("models/iDEAS_RRLO")

        rrlo_dvfs = RRLO_DVFS(
            state_bounds=self.envs["rrlo"].get_state_bounds(),
            num_w_inter_powers=len(self.envs["rrlo"].w_inter.powers),
            num_dvfs_algs=2,
            dvfs_algs=["cc", "la"],
            num_tasks=self.num_tasks,
        )
        rrlo_dvfs.load_model("models/iDEAS_RRLO")

        random_policy = RandomPolicy(
            self.envs["random"].cpu.freqs, self.envs["random"].w_inter.powers
        )
        # FIXME: Probably need to change local policy
        local_policy = {
            "offload": [],
            "local": [[0, 1820], [1, 1820], [2, 1820], [3, 1820]],
        }
        remote_policy = {
            "offload": [[0, 28], [1, 28], [2, 28], [3, 28]],
            "local": [],
        }

        return {
            "random": random_policy,
            "rrlo": rrlo_dvfs,
            "ideas": dqn_dvfs,
            "local": local_policy,
            "remote": remote_policy,
        }

    def _init_results_container(self, scenario):
        if scenario == "fixed_taskset":
            self.num_results_item = 2
        elif scenario == "varied_cpuload":
            self.num_results_item = len(self.cpu_loads)
        elif scenario == "varied_tasksize":
            self.num_results_item = len(self.task_sizes) - 1
        elif scenario == "varied_channel":
            self.num_results_item = len(self.cns)
        else:
            raise ValueError(f"Unknown Scenario! {scenario}")

        # 0: energy, 1: num_tasks, 2: missed deadline
        ideas_stat = np.zeros((3, self.num_tasks, self.num_results_item))
        rrlo_stat = np.zeros((3, self.num_tasks, self.num_results_item))
        local_stat = np.zeros((3, self.num_tasks, self.num_results_item))
        remote_stat = np.zeros((3, self.num_tasks, self.num_results_item))
        random_stat = np.zeros((3, self.num_tasks, self.num_results_item))

        self.raw_results = {
            "random": random_stat,
            "rrlo": rrlo_stat,
            "ideas": ideas_stat,
            "local": local_stat,
            "remote": remote_stat,
        }

    def _process_results(self, idx):
        for scen, env in self.envs.items():
            stat = self.raw_results[scen]
            for jobs in env.curr_tasks.values():
                for j in jobs:
                    stat[0, j.t_id, idx] += j.cons_energy
                    if j.deadline_missed:
                        stat[2, j.t_id, idx] += 1
                stat[1, j.t_id, idx] += len(jobs)

    def _get_results(self, scenario_name):
        results = {
            f"{scenario_name}_{scen}": np.zeros(
                (len(self.raw_results), self.num_results_item)
            )
            for scen in ["energy", "drop"]
        }

        for i, raw_result in enumerate(self.raw_results.values()):
            energy = raw_result[0]
            num_tasks = raw_result[1]
            missed_deadline = raw_result[2]
            results[f"{scenario_name}_energy"][i] = (
                np.sum(energy, axis=0) / np.sum(num_tasks, axis=0) * 100
            )
            results[f"{scenario_name}_drop"][i] = (
                np.sum(missed_deadline, axis=0) / np.sum(num_tasks, axis=0) * 100
            )

        return results

    def _observe(self, tasks):
        ideas_state, _ = self.envs["ideas"].observe(copy.deepcopy(tasks))
        rrlo_state, _ = self.envs["rrlo"].observe(copy.deepcopy(tasks))
        self.envs["local"].observe(copy.deepcopy(tasks))
        self.envs["remote"].observe(copy.deepcopy(tasks))
        self.envs["random"].observe(copy.deepcopy(tasks))

        return {"ideas": ideas_state, "rrlo": rrlo_state}

    def _run_algs(self, states):
        actions_raw = self.algs["ideas"].execute(states["ideas"])
        actions_str = self.algs["ideas"].conv_acts(actions_raw)
        actions_rrlo, _ = self.algs["rrlo"].execute(states["rrlo"])

        actions = {
            "random": self.algs["random"].generate(),
            "rrlo": actions_rrlo,
            "ideas": actions_str,
            "local": self.algs["local"],
            "remote": self.algs["remote"],
        }

        return actions


class iDEAS_BaselineEvaluator(iDEAS_RRLOEvaluator):
    def _init_envs(self):
        envs = {}
        env_temp = HetrogenEnv(
            self.configs,
            self.task_gen.get_wcet_bound(),
            self.task_gen.get_task_size_bound(),
        )
        envs["random"] = copy.deepcopy(env_temp)
        envs["ideas"] = copy.deepcopy(env_temp)
        envs["local"] = copy.deepcopy(env_temp)
        envs["remote"] = copy.deepcopy(env_temp)

        return envs

    def _init_algs(self):
        dqn_dvfs = DQN_DVFS(
            state_dim=self.params["dqn_state_dim"],
            act_space=self.envs["ideas"].get_action_space(),
        )
        dqn_dvfs.load_model("models/iDEAS_Baseline")

        cpu_freqs = {
            "little": self.envs["random"].cpu_little.freqs,
            "big": self.envs["random"].cpu_big.freqs,
        }
        random_policy = RandomIdeasPolicy(cpu_freqs, self.envs["random"].w_inter.powers)

        # FIXME: Probably need to change local policy
        local_policy = {
            "offload": [],
            "little": [],
            "big": [[0, 1820], [1, 1820], [2, 1820], [3, 1820]],
        }
        remote_policy = {
            "offload": [[0, 28], [1, 28], [2, 28], [3, 28]],
            "little": [],
            "big": [],
        }

        return {
            "random": random_policy,
            "ideas": dqn_dvfs,
            "local": local_policy,
            "remote": remote_policy,
        }

    def _init_results_container(self, scenario):
        if scenario == "fixed_taskset":
            self.num_results_item = 2
        elif scenario == "varied_cpuload":
            self.num_results_item = len(self.cpu_loads)
        elif scenario == "varied_tasksize":
            self.num_results_item = len(self.task_sizes) - 1
        elif scenario == "varied_channel":
            self.num_results_item = len(self.cns)
        else:
            raise ValueError(f"Unknown Scenario! {scenario}")

        # 0: energy, 1: num_tasks, 2: missed deadline
        ideas_stat = np.zeros((3, self.num_tasks, self.num_results_item))
        local_stat = np.zeros((3, self.num_tasks, self.num_results_item))
        remote_stat = np.zeros((3, self.num_tasks, self.num_results_item))
        random_stat = np.zeros((3, self.num_tasks, self.num_results_item))

        self.raw_results = {
            "random": random_stat,
            "ideas": ideas_stat,
            "local": local_stat,
            "remote": remote_stat,
        }

    def _observe(self, tasks):
        ideas_state, _ = self.envs["ideas"].observe(copy.deepcopy(tasks))
        self.envs["local"].observe(copy.deepcopy(tasks))
        self.envs["remote"].observe(copy.deepcopy(tasks))
        self.envs["random"].observe(copy.deepcopy(tasks))

        return {"ideas": ideas_state}

    def _run_algs(self, states):
        actions_raw = self.algs["ideas"].execute(states["ideas"])
        actions_str = self.algs["ideas"].conv_acts(actions_raw)

        actions = {
            "random": self.algs["random"].generate(),
            "ideas": actions_str,
            "local": self.algs["local"],
            "remote": self.algs["remote"],
        }

        return actions


# FIXME: This class is probably wrong
class RandomPolicy:
    def __init__(self, freqs, powers):
        self.freqs = freqs
        self.powers = powers

    def generate(self):
        offload = np.random.randint(0, 5)
        actions = {"offload": [], "local": []}
        random_freq_idx = np.random.choice(
            len(self.freqs), size=4 - offload, replace=True
        )
        random_power_idx = np.random.choice(
            len(self.powers), size=offload, replace=True
        )
        actions["offload"] = [
            [i, self.powers[idx]] for i, idx in enumerate(random_power_idx)
        ]
        actions["local"] = [
            [i + offload, self.freqs[idx]] for i, idx in enumerate(random_freq_idx)
        ]

        return actions


# FIXME: Check this class
class RandomIdeasPolicy(RandomPolicy):
    def generate(self):
        offload = np.random.randint(0, 5)
        little = np.random.randint(0, 5 - offload)
        big = 4 - offload - little
        actions = {"offload": [], "little": [], "big": []}
        random_littlefreq_idx = np.random.choice(
            len(self.freqs["little"]), size=little, replace=True
        )
        random_bigfreq_idx = np.random.choice(
            len(self.freqs["big"]), size=big, replace=True
        )
        random_power_idx = np.random.choice(
            len(self.powers), size=offload, replace=True
        )
        actions["offload"] = [
            [i, self.powers[idx]] for i, idx in enumerate(random_power_idx)
        ]
        actions["little"] = [
            [i + offload, self.freqs["little"][idx]]
            for i, idx in enumerate(random_littlefreq_idx)
        ]
        actions["big"] = [
            [i + offload + little, self.freqs["big"][idx]]
            for i, idx in enumerate(random_bigfreq_idx)
        ]

        return actions
