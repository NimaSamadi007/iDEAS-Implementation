import numpy as np
from tqdm import tqdm
import copy
from itertools import cycle

from env_models.env import BaseDQNEnv, DQNEnv, RRLOEnv
from dvfs.dqn_dvfs import DQN_DVFS
from env_models.task import TaskGen,RandomTaskGen,NormalTaskGen
from dvfs.rrlo_dvfs import RRLO_DVFS
from dvfs.conference_dvfs import conference_DVFS
from utils.utils import set_random_seed



def iDEAS_evaluate(
        configs,
        cpu_loads,
        task_sizes, 
        CNs, 
        eval_itr=10000,
        taskset_eval=True,
        CPU_load_eval=True, 
        task_size_eval=True, 
        CN_eval= True
    ):


    results={ 
        "taskset_energy":[], "taskset_drop":[],
        "cpu_energy":[], "cpu_drop":[],
        "task_energy":[], "task_drop":[],
        "cn_energy":[], "cn_drop":[]
    }
    dqn_energy = np.zeros((4,2))
    dqn_num_tasks = np.zeros((4,2))
    big_energy = np.zeros((4,2))
    big_num_tasks = np.zeros((4,2))
    little_energy = np.zeros((4,2))
    little_num_tasks = np.zeros((4,2))
    offload_energy = np.zeros((4,2))
    offload_num_tasks = np.zeros((4,2))

    dqn_deadline_missed = np.zeros((4,2))
    big_deadline_missed = np.zeros((4,2))
    little_deadline_missed = np.zeros((4,2))
    offload_deadline_missed = np.zeros((4,2))
    if taskset_eval:
        for i in range(2):
            if i==0:
                task_gen = TaskGen(configs["task_set1"])
            if i==1:
                task_gen = TaskGen(configs["task_set2"])

            dqn_env = BaseDQNEnv(configs, task_gen.get_wcet_bound(), task_gen.get_task_size_bound())
            dqn_dvfs = DQN_DVFS(state_dim=configs["dqn_state_dim"], act_space=dqn_env.get_action_space())
            dqn_dvfs.load_model("models/iDEAS_train")

            tasks = task_gen.step()
            dqn_state, _ = dqn_env.observe(copy.deepcopy(tasks))


            for _ in range(eval_itr):
                actions_dqn = dqn_dvfs.execute(dqn_state, eval_mode=True)
                actions_dqn_str = dqn_dvfs.conv_acts(actions_dqn)

                dqn_env.step(actions_dqn_str)


                for t_id, _ in actions_dqn_str["little"]:
                    for job in dqn_env.curr_tasks[t_id]:
                        little_energy[job.t_id,i]+=job.cons_energy
                        if job.deadline_missed:
                            little_deadline_missed[job.t_id,i] += 1
                    little_num_tasks[job.t_id,i] += len(dqn_env.curr_tasks[t_id])

                for t_id, _ in actions_dqn_str["big"]:
                    for job in dqn_env.curr_tasks[t_id]:
                        big_energy[job.t_id,i]+=job.cons_energy
                        if job.deadline_missed:
                            big_deadline_missed[job.t_id,i] += 1
                    big_num_tasks[job.t_id,i] += len(dqn_env.curr_tasks[t_id])
        
                for t_id, _ in actions_dqn_str["offload"]:
                    for job in dqn_env.curr_tasks[t_id]:
                        offload_energy[job.t_id,i]+=job.cons_energy
                        if job.deadline_missed:
                            offload_deadline_missed[job.t_id,i] += 1
                    offload_num_tasks[job.t_id,i] += len(dqn_env.curr_tasks[t_id])



                for jobs in dqn_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        dqn_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            dqn_deadline_missed[j.t_id,i] += 1
                    dqn_num_tasks[j.t_id,i] += len(jobs)


                tasks = task_gen.step()
                next_state_dqn, _ = dqn_env.observe(copy.deepcopy(tasks))
        

                # Update current state
                dqn_state = next_state_dqn

        np.set_printoptions(suppress=True)
        dqn_avg_energy = dqn_energy / dqn_num_tasks
        little_avg_energy = little_energy / dqn_num_tasks
        big_avg_energy = big_energy / dqn_num_tasks
        offload_avg_energy = offload_energy / dqn_num_tasks






        dqn_percent_missed = dqn_deadline_missed / dqn_num_tasks * 100
        big_percent_missed = big_deadline_missed / dqn_num_tasks * 100
        little_percent_missed = little_deadline_missed / dqn_num_tasks * 100
        offload_percent_missed = offload_deadline_missed / dqn_num_tasks * 100

        avg_dqn_energy = np.sum(dqn_energy,axis=0) / np.sum(dqn_num_tasks,axis=0)
        avg_dqn_little_energy = np.sum(little_energy,axis=0) / np.sum(dqn_num_tasks,axis=0)
        avg_dqn_big_energy = np.sum(big_energy,axis=0) / np.sum(dqn_num_tasks,axis=0)
        avg_dqn_offload_energy = np.sum(offload_energy,axis=0) / np.sum(dqn_num_tasks,axis=0)


        total_dqn_missed = (
            np.sum(dqn_deadline_missed,axis=0) / np.sum(dqn_num_tasks,axis=0) * 100
        )

        total_big_missed = (
            np.sum(big_deadline_missed,axis=0) / np.sum(dqn_num_tasks,axis=0) * 100
        )

        total_little_missed = (
            np.sum(little_deadline_missed,axis=0) / np.sum(dqn_num_tasks,axis=0) * 100
        )

        total_offload_missed = (
            np.sum(dqn_deadline_missed,axis=0) / np.sum(dqn_num_tasks,axis=0) * 100
        )



        results["taskset_energy"] = np.array(
            [
                
                avg_dqn_big_energy,
                avg_dqn_little_energy,
                avg_dqn_offload_energy,
                avg_dqn_energy
            ]
        )

        results["taskset_drop"] = np.array(
            [
                total_big_missed,
                total_little_missed,
                total_offload_missed,
                total_dqn_missed,
            ]
        )





    if CPU_load_eval:
        
        set_random_seed(42)
        max_task_load=3
        
        random_task_gen = RandomTaskGen(configs["task_set3"])
        dqn_env = BaseDQNEnv(configs, random_task_gen.get_wcet_bound(), random_task_gen.get_task_size_bound())
    
        dqn_dvfs = DQN_DVFS(state_dim=configs["dqn_state_dim"], act_space=dqn_env.get_action_space())
        dqn_dvfs.load_model("models/iDEAS_train")

        dqn_cpu_energy = np.zeros((4,len(cpu_loads)))
        dqn_cpu_num_tasks = np.zeros((4,len(cpu_loads)))
        big_cpu_energy = np.zeros((4,len(cpu_loads)))
        big_cpu_num_tasks = np.zeros((4,len(cpu_loads)))
        little_cpu_energy = np.zeros((4,len(cpu_loads)))
        little_cpu_num_tasks = np.zeros((4,len(cpu_loads)))
        offload_cpu_energy = np.zeros((4,len(cpu_loads)))
        offload_cpu_num_tasks = np.zeros((4,len(cpu_loads)))

        dqn_cpu_deadline_missed =np.zeros((4,len(cpu_loads)))
        big_cpu_deadline_missed =np.zeros((4,len(cpu_loads)))
        little_cpu_deadline_missed =np.zeros((4,len(cpu_loads)))
        offload_cpu_deadline_missed =np.zeros((4,len(cpu_loads)))
    
    
        for i in range(len(cpu_loads)):

            target_cpu_load=cpu_loads[i]
            tasks = random_task_gen.step(target_cpu_load,max_task_load)
            dqn_state, _ = dqn_env.observe(copy.deepcopy(tasks))
            for _ in range(eval_itr):
                actions_dqn = dqn_dvfs.execute(dqn_state, eval_mode=True)
                actions_dqn_str = dqn_dvfs.conv_acts(actions_dqn)
            
                dqn_env.step(actions_dqn_str)

            
                for t_id, _ in actions_dqn_str["little"]:
                    for job in dqn_env.curr_tasks[t_id]:
                        little_cpu_energy[job.t_id,i]+=job.cons_energy
                        if job.deadline_missed:
                            little_cpu_deadline_missed[job.t_id,i] += 1
                    little_cpu_num_tasks[job.t_id,i] += len(dqn_env.curr_tasks[t_id])

                for t_id, _ in actions_dqn_str["big"]:
                    for job in dqn_env.curr_tasks[t_id]:
                        big_cpu_energy[job.t_id,i]+=job.cons_energy
                        if job.deadline_missed:
                            big_cpu_deadline_missed[job.t_id,i] += 1
                    big_cpu_num_tasks[job.t_id,i] += len(dqn_env.curr_tasks[t_id])
        
                for t_id, _ in actions_dqn_str["offload"]:
                    for job in dqn_env.curr_tasks[t_id]:
                        offload_cpu_energy[job.t_id,i]+=job.cons_energy
                        if job.deadline_missed:
                            offload_cpu_deadline_missed[job.t_id,i] += 1
                    offload_cpu_num_tasks[job.t_id,i] += len(dqn_env.curr_tasks[t_id])

                for jobs in dqn_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        dqn_cpu_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            dqn_cpu_deadline_missed[j.t_id,i] += 1
                    dqn_cpu_num_tasks[j.t_id,i] += len(jobs)


                tasks = random_task_gen.step(target_cpu_load,max_task_load)
                next_state_dqn, _ = dqn_env.observe(copy.deepcopy(tasks))


                # Update current state
                dqn_state = next_state_dqn
                #conference_state = next_state_conference


        np.set_printoptions(suppress=True)
        dqn_avg_cpu_energy = dqn_cpu_energy / dqn_cpu_num_tasks
        big_avg_cpu_energy = big_cpu_energy / dqn_cpu_num_tasks
        little_avg_cpu_energy = little_cpu_energy / dqn_cpu_num_tasks
        offload_avg_cpu_energy = offload_cpu_energy / dqn_cpu_num_tasks



        dqn_percent_cpu_missed = dqn_cpu_deadline_missed / dqn_cpu_num_tasks * 100
        big_percent_cpu_missed = big_cpu_deadline_missed / dqn_cpu_num_tasks * 100
        little_percent_cpu_missed = little_cpu_deadline_missed / dqn_cpu_num_tasks * 100
        offload_percent_cpu_missed = offload_cpu_deadline_missed / dqn_cpu_num_tasks * 100



        avg_dqn_cpu_energy = np.sum(dqn_cpu_energy,axis=0) / np.sum(dqn_cpu_num_tasks,axis=0)
        avg_big_cpu_energy = np.sum(big_cpu_energy,axis=0) / np.sum(dqn_cpu_num_tasks,axis=0)
        avg_little_cpu_energy = np.sum(little_cpu_energy,axis=0) / np.sum(dqn_cpu_num_tasks,axis=0)
        avg_offload_cpu_energy = np.sum(offload_cpu_energy,axis=0) / np.sum(dqn_cpu_num_tasks,axis=0)



        total_dqn_missed_cpu = (
            np.sum(dqn_cpu_deadline_missed,axis=0) / np.sum(dqn_cpu_num_tasks,axis=0) * 100
        )
        total_big_missed_cpu = (
            np.sum(big_cpu_deadline_missed,axis=0) / np.sum(dqn_cpu_num_tasks,axis=0) * 100
        )
        total_little_missed_cpu = (
            np.sum(little_cpu_deadline_missed,axis=0) / np.sum(dqn_cpu_num_tasks,axis=0) * 100
        )
        total_offload_missed_cpu = (
            np.sum(offload_cpu_deadline_missed,axis=0) / np.sum(dqn_cpu_num_tasks,axis=0) * 100
        )

        results["cpu_energy"] = np.array(
            [
                
                avg_big_cpu_energy,
                avg_little_cpu_energy,
                avg_offload_cpu_energy,
                avg_dqn_cpu_energy
            ]
        )

        results["cpu_drop"] = np.array(
            [
                
                total_big_missed_cpu,
                total_little_missed_cpu,
                total_offload_missed_cpu,
                total_dqn_missed_cpu
            ]
        )





    if task_size_eval:
        set_random_seed(42)
        target_cpu_load = 0.35
        max_task_load=3
        normal_task_gen = NormalTaskGen(configs["task_set3"])

        dqn_env = BaseDQNEnv(configs, normal_task_gen.get_wcet_bound(), normal_task_gen.get_task_size_bound())
    
        dqn_dvfs = DQN_DVFS(state_dim=configs["dqn_state_dim"], act_space=dqn_env.get_action_space())
        dqn_dvfs.load_model("models/iDEAS_train")

        dqn_task_energy = np.zeros((4,len(task_sizes)-1))
        dqn_task_num_tasks = np.zeros((4,len(task_sizes)-1))
        big_task_energy = np.zeros((4,len(task_sizes)-1))
        big_task_num_tasks = np.zeros((4,len(task_sizes)-1))
        little_task_energy = np.zeros((4,len(task_sizes)-1))
        little_task_num_tasks = np.zeros((4,len(task_sizes)-1))
        offload_task_energy = np.zeros((4,len(task_sizes)-1))
        offload_task_num_tasks = np.zeros((4,len(task_sizes)-1))

        dqn_task_deadline_missed = np.zeros((4,len(task_sizes)-1))
        big_task_deadline_missed = np.zeros((4,len(task_sizes)-1))
        little_task_deadline_missed = np.zeros((4,len(task_sizes)-1))
        offload_task_deadline_missed = np.zeros((4,len(task_sizes)-1))

    
    
    
        tasks = normal_task_gen.step(target_cpu_load,task_sizes[0],max_task_load)
        for i in range(len(task_sizes)-1):
            dqn_state, _ = dqn_env.observe(copy.deepcopy(tasks))
            for _ in range(eval_itr):
                actions_dqn = dqn_dvfs.execute(dqn_state, eval_mode=True)
                actions_dqn_str = dqn_dvfs.conv_acts(actions_dqn)
                dqn_env.step(actions_dqn_str)
                # Gather energy consumption values
                for t_id, _ in actions_dqn_str["little"]:
                    for job in dqn_env.curr_tasks[t_id]:
                        little_task_energy[job.t_id,i]+=job.cons_energy
                        if job.deadline_missed:
                            little_task_deadline_missed[job.t_id,i] += 1
                    little_task_num_tasks[job.t_id,i] += len(dqn_env.curr_tasks[t_id])

                for t_id, _ in actions_dqn_str["big"]:
                    for job in dqn_env.curr_tasks[t_id]:
                        big_task_energy[job.t_id,i]+=job.cons_energy
                        if job.deadline_missed:
                            big_task_deadline_missed[job.t_id,i] += 1
                    big_task_num_tasks[job.t_id,i] += len(dqn_env.curr_tasks[t_id])
        
                for t_id, _ in actions_dqn_str["offload"]:
                    for job in dqn_env.curr_tasks[t_id]:
                        offload_task_energy[job.t_id,i]+=job.cons_energy
                        if job.deadline_missed:
                            offload_task_deadline_missed[job.t_id,i] += 1
                    offload_task_num_tasks[job.t_id,i] += len(dqn_env.curr_tasks[t_id])

                for jobs in dqn_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        dqn_task_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            dqn_task_deadline_missed[j.t_id,i] += 1
                    dqn_task_num_tasks[j.t_id,i] += len(jobs)

                if i ==len(task_sizes)-2:
                    tasks = normal_task_gen.step(target_cpu_load,task_sizes[i],max_task_load)
                else:
                    tasks = normal_task_gen.step(target_cpu_load,task_sizes[i+1],max_task_load)
                #print(task_size_val[2*i+1])
                next_state_dqn, _ = dqn_env.observe(copy.deepcopy(tasks))

                # Update current state
                dqn_state = next_state_dqn


    
    
    
        np.set_printoptions(suppress=True)
        dqn_avg_task_energy = dqn_task_energy / dqn_task_num_tasks
        big_avg_task_energy = big_task_energy / dqn_task_num_tasks
        little_avg_task_energy = little_task_energy / dqn_task_num_tasks
        offload_avg_task_energy = offload_task_energy / dqn_task_num_tasks



        dqn_percent_task_missed = dqn_task_deadline_missed / dqn_task_num_tasks * 100
        big_percent_task_missed = big_task_deadline_missed / dqn_task_num_tasks * 100
        little_percent_task_missed = little_task_deadline_missed / dqn_task_num_tasks * 100
        offload_percent_task_missed = offload_task_deadline_missed / dqn_task_num_tasks * 100



        avg_dqn_task_energy = np.sum(dqn_task_energy,axis=0) / np.sum(dqn_task_num_tasks,axis=0)
        avg_big_task_energy = np.sum(big_task_energy,axis=0) / np.sum(dqn_task_num_tasks,axis=0)
        avg_little_task_energy = np.sum(little_task_energy,axis=0) / np.sum(dqn_task_num_tasks,axis=0)
        avg_offload_task_energy = np.sum(offload_task_energy,axis=0) / np.sum(dqn_task_num_tasks,axis=0)



        total_dqn_missed_task = (
            np.sum(dqn_task_deadline_missed,axis=0) / np.sum(dqn_task_num_tasks,axis=0) * 100
        )

        total_big_missed_task = (
            np.sum(big_task_deadline_missed,axis=0) / np.sum(dqn_task_num_tasks,axis=0) * 100
        )

        total_little_missed_task = (
            np.sum(little_task_deadline_missed,axis=0) / np.sum(dqn_task_num_tasks,axis=0) * 100
        )

        total_offload_missed_task = (
            np.sum(offload_task_deadline_missed,axis=0) / np.sum(dqn_task_num_tasks,axis=0) * 100
        )


        results["task_energy"] = np.array(
            [
                
                avg_big_task_energy,
                avg_little_task_energy,
                avg_offload_task_energy,
                avg_dqn_task_energy
            ]
        )


        results["task_drop"] = np.array(
            [
                
                total_big_missed_task,
                total_little_missed_task,
                total_offload_missed_task,
                total_dqn_missed_task
            ]
        )



        

    if CN_eval:
        set_random_seed(42)
        max_task_load=3
        target_cpu_load = 0.35
        random_task_gen = RandomTaskGen(configs["task_set3"])

        dqn_env = BaseDQNEnv(configs, random_task_gen.get_wcet_bound(), random_task_gen.get_task_size_bound())
    
        dqn_dvfs = DQN_DVFS(state_dim=configs["dqn_state_dim"], act_space=dqn_env.get_action_space())
        dqn_dvfs.load_model("models/iDEAS_train")

        dqn_cn_energy = np.zeros((4,len(CNs)))
        dqn_cn_num_tasks = np.zeros((4,len(CNs)))
        big_cn_energy = np.zeros((4,len(CNs)))
        big_cn_num_tasks = np.zeros((4,len(CNs)))
        little_cn_energy = np.zeros((4,len(CNs)))
        little_cn_num_tasks = np.zeros((4,len(CNs)))
        offload_cn_energy = np.zeros((4,len(CNs)))
        offload_cn_num_tasks = np.zeros((4,len(CNs)))

        dqn_cn_deadline_missed =np.zeros((4,len(CNs)))
        big_cn_deadline_missed =np.zeros((4,len(CNs)))
        little_cn_deadline_missed =np.zeros((4,len(CNs)))
        offload_cn_deadline_missed =np.zeros((4,len(CNs)))


        for i in range(len(CNs)):

            dqn_env.w_inter.cn_setter(CNs[i])
            tasks = random_task_gen.step(target_cpu_load,max_task_load)
            dqn_state, _ = dqn_env.observe(copy.deepcopy(tasks))
            for _ in range(eval_itr):
                actions_dqn = dqn_dvfs.execute(dqn_state, eval_mode=True)
                actions_dqn_str = dqn_dvfs.conv_acts(actions_dqn)
            
                dqn_env.step(actions_dqn_str)

            
                for t_id, _ in actions_dqn_str["little"]:
                    for job in dqn_env.curr_tasks[t_id]:
                        little_cn_energy[job.t_id,i]+=job.cons_energy
                        if job.deadline_missed:
                            little_cn_deadline_missed[job.t_id,i] += 1
                    little_cn_num_tasks[job.t_id,i] += len(dqn_env.curr_tasks[t_id])

                for t_id, _ in actions_dqn_str["big"]:
                    for job in dqn_env.curr_tasks[t_id]:
                        big_cn_energy[job.t_id,i]+=job.cons_energy
                        if job.deadline_missed:
                            big_cn_deadline_missed[job.t_id,i] += 1
                    big_cn_num_tasks[job.t_id,i] += len(dqn_env.curr_tasks[t_id])
        
                for t_id, _ in actions_dqn_str["offload"]:
                    for job in dqn_env.curr_tasks[t_id]:
                        offload_cn_energy[job.t_id,i]+=job.cons_energy
                        if job.deadline_missed:
                            offload_cn_deadline_missed[job.t_id,i] += 1
                    offload_cn_num_tasks[job.t_id,i] += len(dqn_env.curr_tasks[t_id])

                for jobs in dqn_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        dqn_cn_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            dqn_cn_deadline_missed[j.t_id,i] += 1
                    dqn_cn_num_tasks[j.t_id,i] += len(jobs)


                tasks = random_task_gen.step(target_cpu_load,max_task_load)
                next_state_dqn, _ = dqn_env.observe(copy.deepcopy(tasks))


                # Update current state
                dqn_state = next_state_dqn
                #conference_state = next_state_conference


        np.set_printoptions(suppress=True)
        dqn_avg_cn_energy = dqn_cn_energy / dqn_cn_num_tasks
        big_avg_cn_energy = big_cn_energy / dqn_cn_num_tasks
        little_avg_cn_energy = little_cn_energy / dqn_cn_num_tasks
        offload_avg_cn_energy = offload_cn_energy / dqn_cn_num_tasks



        dqn_percent_cn_missed = dqn_cn_deadline_missed / dqn_cn_num_tasks * 100
        big_percent_cn_missed = big_cn_deadline_missed / dqn_cn_num_tasks * 100
        little_percent_cn_missed = little_cn_deadline_missed / dqn_cn_num_tasks * 100
        offload_percent_cn_missed = offload_cn_deadline_missed / dqn_cn_num_tasks * 100



        avg_dqn_cn_energy = np.sum(dqn_cn_energy,axis=0) / np.sum(dqn_cn_num_tasks,axis=0)
        avg_big_cn_energy = np.sum(big_cn_energy,axis=0) / np.sum(dqn_cn_num_tasks,axis=0)
        avg_little_cn_energy = np.sum(little_cn_energy,axis=0) / np.sum(dqn_cn_num_tasks,axis=0)
        avg_offload_cn_energy = np.sum(offload_cn_energy,axis=0) / np.sum(dqn_cn_num_tasks,axis=0)



        total_dqn_missed_cn = (
            np.sum(dqn_cn_deadline_missed,axis=0) / np.sum(dqn_cn_num_tasks,axis=0) * 100
        )
        total_big_missed_cn = (
            np.sum(big_cn_deadline_missed,axis=0) / np.sum(dqn_cn_num_tasks,axis=0) * 100
        )
        total_little_missed_cn = (
            np.sum(little_cn_deadline_missed,axis=0) / np.sum(dqn_cn_num_tasks,axis=0) * 100
        )
        total_offload_missed_cn = (
            np.sum(offload_cn_deadline_missed,axis=0) / np.sum(dqn_cn_num_tasks,axis=0) * 100
        )


        results["cn_energy"] = np.array(
            [
                
                avg_big_cn_energy,
                avg_little_cn_energy,
                avg_offload_cn_energy,
                avg_dqn_cn_energy
            ]
        )

        results["cn_drop"] = np.array(
            [
                
                total_big_missed_cn,
                total_little_missed_cn,
                total_offload_missed_cn,
                total_dqn_missed_cn
            ]
        )



    return results


def RRLO_evaluate(
        configs,
        cpu_loads,
        task_sizes, 
        CNs, 
        eval_itr=10000,
        taskset_eval=True,
        CPU_load_eval=True, 
        task_size_eval=True, 
        CN_eval= True
    ):


    results={ 
        "taskset_energy":[], "taskset_drop":[],
        "taskset_improvement":[],
        "cpu_energy":[], "cpu_drop":[],
        "task_energy":[], "task_drop":[],
        "cn_energy":[], "cn_drop":[]
    }

    dqn_energy = np.zeros((4,2))
    dqn_num_tasks = np.zeros((4,2))
    rrlo_energy = np.zeros((4,2))
    rrlo_num_tasks = np.zeros((4,2))
    local_energy = np.zeros((4,2))
    local_num_tasks = np.zeros((4,2))
    remote_energy = np.zeros((4,2))
    remote_num_tasks = np.zeros((4,2))
    random_energy = np.zeros((4,2))
    random_num_tasks = np.zeros((4,2))
            

    dqn_deadline_missed = np.zeros((4,2))
    rrlo_deadline_missed = np.zeros((4,2))
    local_deadline_missed = np.zeros((4,2))
    remote_deadline_missed = np.zeros((4,2))
    random_deadline_missed = np.zeros((4,2))

    if taskset_eval:
        for i in range(2):
            if i==0:
                task_gen = TaskGen(configs["task_set1"])
            if i==1:
                task_gen = TaskGen(configs["task_set2"])
            dqn_env = DQNEnv(configs, task_gen.get_wcet_bound(), task_gen.get_task_size_bound())
            local_env = DQNEnv(configs, task_gen.get_wcet_bound(), task_gen.get_task_size_bound())
            remote_env = DQNEnv(configs, task_gen.get_wcet_bound(), task_gen.get_task_size_bound())
            random_env = DQNEnv(configs, task_gen.get_wcet_bound(), task_gen.get_task_size_bound())
            rand_freq_list = random_env.cpu.freqs
            rand_powers_list = random_env.w_inter.powers
            rrlo_env = RRLOEnv(configs)

            dqn_dvfs = DQN_DVFS(state_dim=configs["dqn_state_dim"], act_space=dqn_env.get_action_space())
            rrlo_dvfs = RRLO_DVFS(
                state_bounds=rrlo_env.get_state_bounds(),
                num_w_inter_powers=len(rrlo_env.w_inter.powers),
                num_dvfs_algs=2,
                dvfs_algs=["cc", "la"],
                num_tasks=4,
            )

            dqn_dvfs.load_model("models/RRLO_train")
            rrlo_dvfs.load_model("models/RRLO_train")






            tasks = task_gen.step()
            dqn_state, _ = dqn_env.observe(copy.deepcopy(tasks))
            rrlo_state, _ = rrlo_env.observe(copy.deepcopy(tasks))
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


                dqn_env.step(actions_dqn_str)
                rrlo_env.step(actions_rrlo)
                local_env.step(actions_local_str)
                remote_env.step(actions_remote_str)
                random_env.step(actions_random_str)

                # Gather energy consumption values
                for jobs in local_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        local_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            local_deadline_missed[j.t_id,i] += 1
                    local_num_tasks[j.t_id,i] += len(jobs)

                for jobs in remote_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        remote_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            remote_deadline_missed[j.t_id,i] += 1
                    remote_num_tasks[j.t_id,i] += len(jobs)

                for jobs in random_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        random_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            random_deadline_missed[j.t_id,i] += 1
                    random_num_tasks[j.t_id,i] += len(jobs)

                for jobs in dqn_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        dqn_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            dqn_deadline_missed[j.t_id,i] += 1
                    dqn_num_tasks[j.t_id,i] += len(jobs)

                for jobs in rrlo_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        rrlo_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            rrlo_deadline_missed[j.t_id,i] += 1
                    rrlo_num_tasks[j.t_id,i] += len(jobs)


                tasks = task_gen.step()
                next_state_dqn, _ = dqn_env.observe(copy.deepcopy(tasks))
                next_state_rrlo, _ = rrlo_env.observe(copy.deepcopy(tasks))
                local_env.observe(copy.deepcopy(tasks))
                remote_env.observe(copy.deepcopy(tasks))
                random_env.observe(copy.deepcopy(tasks))

                # Update current state
                dqn_state = next_state_dqn
                rrlo_state = next_state_rrlo

        np.set_printoptions(suppress=True)
        dqn_avg_energy = dqn_energy / dqn_num_tasks
        local_avg_energy = local_energy / local_num_tasks
        remote_avg_energy = remote_energy / remote_num_tasks
        random_avg_energy = random_energy / random_num_tasks
        rrlo_avg_energy = rrlo_energy / rrlo_num_tasks

        dqn_percent_missed = dqn_deadline_missed / dqn_num_tasks * 100
        local_percent_missed = local_deadline_missed / local_num_tasks * 100
        remote_percent_missed = remote_deadline_missed / remote_num_tasks * 100
        random_percent_missed = random_deadline_missed / random_num_tasks * 100
        rrlo_percent_missed = rrlo_deadline_missed / rrlo_num_tasks * 100


        avg_dqn_energy = np.sum(dqn_energy,axis=0) / np.sum(dqn_num_tasks,axis=0)
        avg_local_energy = np.sum(local_energy,axis=0) / np.sum(local_num_tasks,axis=0)
        avg_remote_energy = np.sum(remote_energy,axis=0) / np.sum(remote_num_tasks,axis=0)
        avg_random_energy = np.sum(random_energy,axis=0) / np.sum(random_num_tasks,axis=0)
        avg_rrlo_energy = np.sum(rrlo_energy,axis=0) / np.sum(rrlo_num_tasks,axis=0)

        total_dqn_missed = (
            np.sum(dqn_deadline_missed,axis=0) / np.sum(dqn_num_tasks,axis=0) * 100
        )
        total_local_missed = (
            np.sum(local_deadline_missed,axis=0) / np.sum(local_num_tasks,axis=0) * 100
        )
        total_remote_missed = (
            np.sum(remote_deadline_missed,axis=0) / np.sum(remote_num_tasks,axis=0) * 100
        )
        total_random_missed = (
            np.sum(random_deadline_missed,axis=0) / np.sum(random_num_tasks,axis=0) * 100
        )
        total_rrlo_missed = (
            np.sum(rrlo_deadline_missed,axis=0) / np.sum(rrlo_num_tasks,axis=0) * 100
        )


        dqn_random_improvement = (
            (avg_dqn_energy - avg_random_energy) / (avg_random_energy) * 100
        )

        dqn_local_improvement = (
            (avg_dqn_energy - avg_local_energy) / (avg_local_energy) * 100
        )

        dqn_remote_improvement = (
            (avg_dqn_energy - avg_remote_energy) / (avg_remote_energy) * 100
        )


        dqn_rrlo_improvement = (
            (avg_dqn_energy - avg_rrlo_energy) / (avg_rrlo_energy) * 100
        )

        
        results["taskset_energy"] = np.array(
            [
                avg_random_energy,
                avg_rrlo_energy,
                avg_dqn_energy,
                avg_local_energy,
                avg_remote_energy
            ]
        )

        results["taskset_drop"] = np.array(
            [
                total_random_missed,
                total_rrlo_missed,
                total_dqn_missed,
                total_local_missed,
                total_remote_missed
            ]
        )

        results["taskset_improvement"] = np.array(
            [
                dqn_random_improvement,
                dqn_rrlo_improvement,
                dqn_local_improvement,
                dqn_remote_improvement
            ]
        )
    


    if CPU_load_eval:
        set_random_seed(42)
        max_task_load=2
        
        random_task_gen = RandomTaskGen(configs["task_set3"])
        dqn_env = DQNEnv(configs, random_task_gen.get_wcet_bound(), random_task_gen.get_task_size_bound())
        local_env = DQNEnv(configs, random_task_gen.get_wcet_bound(), random_task_gen.get_task_size_bound())
        remote_env = DQNEnv(configs, random_task_gen.get_wcet_bound(), random_task_gen.get_task_size_bound())
        random_env = DQNEnv(configs, random_task_gen.get_wcet_bound(), random_task_gen.get_task_size_bound())
        rand_freq_list = random_env.cpu.freqs
        rand_powers_list = random_env.w_inter.powers
        rrlo_env = RRLOEnv(configs)
    
        dqn_dvfs = DQN_DVFS(state_dim=configs["dqn_state_dim"], act_space=dqn_env.get_action_space())
        rrlo_dvfs = RRLO_DVFS(
            state_bounds=rrlo_env.get_state_bounds(),
            num_w_inter_powers=len(rrlo_env.w_inter.powers),
            num_dvfs_algs=2,
            dvfs_algs=["cc", "la"],
            num_tasks=4,
        )

        dqn_dvfs.load_model("models/RRLO_train")
        rrlo_dvfs.load_model("models/RRLO_train")

        dqn_cpu_energy = np.zeros((4,len(cpu_loads)))
        dqn_cpu_num_tasks = np.zeros((4,len(cpu_loads)))
        rrlo_cpu_energy = np.zeros((4,len(cpu_loads)))
        rrlo_cpu_num_tasks = np.zeros((4,len(cpu_loads)))

        local_cpu_energy = np.zeros((4,len(cpu_loads)))
        local_cpu_num_tasks = np.zeros((4,len(cpu_loads)))
        remote_cpu_energy = np.zeros((4,len(cpu_loads)))
        remote_cpu_num_tasks = np.zeros((4,len(cpu_loads)))
        random_cpu_energy = np.zeros((4,len(cpu_loads)))
        random_cpu_num_tasks = np.zeros((4,len(cpu_loads)))

        local_cpu_deadline_missed = np.zeros((4,len(cpu_loads)))
        random_cpu_deadline_missed = np.zeros((4,len(cpu_loads)))
        remote_cpu_deadline_missed = np.zeros((4,len(cpu_loads)))
        dqn_cpu_deadline_missed = np.zeros((4,len(cpu_loads)))
        rrlo_cpu_deadline_missed = np.zeros((4,len(cpu_loads)))
 
    
    
    
    
        for i in range(len(cpu_loads)):

            target_cpu_load=cpu_loads[i]
            tasks = random_task_gen.step(target_cpu_load,max_task_load)
            dqn_state, _ = dqn_env.observe(copy.deepcopy(tasks))
            rrlo_state, _ = rrlo_env.observe(copy.deepcopy(tasks))
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
            
                dqn_env.step(actions_dqn_str)
                rrlo_env.step(actions_rrlo)

                local_env.step(actions_local_str)
                remote_env.step(actions_remote_str)
                random_env.step(actions_random_str)

            
                # Gather energy consumption values
                for jobs in local_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        local_cpu_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            local_cpu_deadline_missed[j.t_id,i] += 1
                    local_cpu_num_tasks[j.t_id,i] += len(jobs)

                for jobs in remote_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        remote_cpu_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            remote_cpu_deadline_missed[j.t_id,i] += 1
                    remote_cpu_num_tasks[j.t_id,i] += len(jobs)

                for jobs in random_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        random_cpu_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            random_cpu_deadline_missed[j.t_id,i] += 1
                    random_cpu_num_tasks[j.t_id,i] += len(jobs)

                for jobs in dqn_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        dqn_cpu_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            dqn_cpu_deadline_missed[j.t_id,i] += 1
                    dqn_cpu_num_tasks[j.t_id,i] += len(jobs)

                for jobs in rrlo_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        rrlo_cpu_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            rrlo_cpu_deadline_missed[j.t_id,i] += 1
                    rrlo_cpu_num_tasks[j.t_id,i] += len(jobs)

                tasks = random_task_gen.step(target_cpu_load,max_task_load)
                next_state_dqn, _ = dqn_env.observe(copy.deepcopy(tasks))
                next_state_rrlo, _ = rrlo_env.observe(copy.deepcopy(tasks))
                #next_state_conference, _ = conference_env.observe(copy.deepcopy(tasks))
                local_env.observe(copy.deepcopy(tasks))
                remote_env.observe(copy.deepcopy(tasks))
                random_env.observe(copy.deepcopy(tasks))

                # Update current state
                dqn_state = next_state_dqn
                rrlo_state = next_state_rrlo
                #conference_state = next_state_conference


        np.set_printoptions(suppress=True)
        dqn_avg_cpu_energy = dqn_cpu_energy / dqn_cpu_num_tasks
        local_avg_cpu_energy = local_cpu_energy / local_cpu_num_tasks
        remote_avg_cpu_energy = remote_cpu_energy / remote_cpu_num_tasks
        random_avg_cpu_energy = random_cpu_energy / random_cpu_num_tasks
        rrlo_avg_cpu_energy = rrlo_cpu_energy / rrlo_cpu_num_tasks



        dqn_percent_missed_cpu = dqn_cpu_deadline_missed / dqn_cpu_num_tasks * 100
        local_percent_missed_cpu = local_cpu_deadline_missed / local_cpu_num_tasks * 100
        remote_percent_missed_cpu = remote_cpu_deadline_missed / remote_cpu_num_tasks * 100
        random_percent_missed_cpu = random_cpu_deadline_missed / random_cpu_num_tasks * 100
        rrlo_percent_missed_cpu = rrlo_cpu_deadline_missed / rrlo_cpu_num_tasks * 100



        avg_dqn_energy_cpu = np.sum(dqn_cpu_energy,axis=0) / np.sum(dqn_cpu_num_tasks,axis=0)
        avg_local_energy_cpu = np.sum(local_cpu_energy,axis=0) / np.sum(local_cpu_num_tasks,axis=0)
        avg_remote_energy_cpu = np.sum(remote_cpu_energy,axis=0) / np.sum(remote_cpu_num_tasks,axis=0)
        avg_random_energy_cpu = np.sum(random_cpu_energy,axis=0) / np.sum(random_cpu_num_tasks,axis=0)
        avg_rrlo_energy_cpu = np.sum(rrlo_cpu_energy,axis=0) / np.sum(rrlo_cpu_num_tasks,axis=0)



        total_dqn_missed_cpu = (
            np.sum(dqn_cpu_deadline_missed,axis=0) / np.sum(dqn_cpu_num_tasks,axis=0) * 100
        )
        total_local_missed_cpu = (
            np.sum(local_cpu_deadline_missed,axis=0) / np.sum(local_cpu_num_tasks,axis=0) * 100
        )
        total_remote_missed_cpu = (
            np.sum(remote_cpu_deadline_missed,axis=0) / np.sum(remote_cpu_num_tasks,axis=0) * 100
        )
        total_random_missed_cpu = (
            np.sum(random_cpu_deadline_missed,axis=0) / np.sum(random_cpu_num_tasks,axis=0) * 100
        )
        total_rrlo_missed_cpu = (
            np.sum(rrlo_cpu_deadline_missed,axis=0) / np.sum(rrlo_cpu_num_tasks,axis=0) * 100
        )


        results["cpu_energy"] = np.array(
            [
                avg_random_energy_cpu,
                avg_rrlo_energy_cpu,
                avg_dqn_energy_cpu,
                avg_local_energy_cpu,
                avg_remote_energy_cpu
            ]
        )

        results["cpu_drop"] = np.array(
            [
                total_random_missed_cpu,
                total_rrlo_missed_cpu,
                total_dqn_missed_cpu,
                total_local_missed_cpu,
                total_remote_missed_cpu
            ]
        )

    if task_size_eval:
        set_random_seed(42)

        target_cpu_load = 0.35
        max_task_load=2

        normal_task_gen = NormalTaskGen(configs["task_set3"])
        dqn_env = DQNEnv(configs, normal_task_gen.get_wcet_bound(), normal_task_gen.get_task_size_bound())
        local_env = DQNEnv(configs, normal_task_gen.get_wcet_bound(), normal_task_gen.get_task_size_bound())
        remote_env = DQNEnv(configs, normal_task_gen.get_wcet_bound(), normal_task_gen.get_task_size_bound())
        random_env = DQNEnv(configs, normal_task_gen.get_wcet_bound(), normal_task_gen.get_task_size_bound())
        rand_freq_list = random_env.cpu.freqs
        rand_powers_list = random_env.w_inter.powers
        rrlo_env = RRLOEnv(configs)

        dqn_dvfs = DQN_DVFS(state_dim=configs["dqn_state_dim"], act_space=dqn_env.get_action_space())
        rrlo_dvfs = RRLO_DVFS(
            state_bounds=rrlo_env.get_state_bounds(),
            num_w_inter_powers=len(rrlo_env.w_inter.powers),
            num_dvfs_algs=2,
            dvfs_algs=["cc", "la"],
            num_tasks=4,
        )

        dqn_dvfs.load_model("models/RRLO_train")
        rrlo_dvfs.load_model("models/RRLO_train")


        dqn_task_energy = np.zeros((4,len(task_sizes)-1))
        dqn_task_num_tasks = np.zeros((4,len(task_sizes)-1))
        rrlo_task_energy = np.zeros((4,len(task_sizes)-1))
        rrlo_task_num_tasks = np.zeros((4,len(task_sizes)-1))

        local_task_energy = np.zeros((4,len(task_sizes)-1))
        local_task_num_tasks = np.zeros((4,len(task_sizes)-1))
        remote_task_energy = np.zeros((4,len(task_sizes)-1))
        remote_task_num_tasks = np.zeros((4,len(task_sizes)-1))
        random_task_energy = np.zeros((4,len(task_sizes)-1))
        random_task_num_tasks = np.zeros((4,len(task_sizes)-1))

        local_task_deadline_missed = np.zeros((4,len(task_sizes)-1))
        random_task_deadline_missed = np.zeros((4,len(task_sizes)-1))
        remote_task_deadline_missed = np.zeros((4,len(task_sizes)-1))
        dqn_task_deadline_missed = np.zeros((4,len(task_sizes)-1))
        rrlo_task_deadline_missed = np.zeros((4,len(task_sizes)-1))
  
    
    
    
        tasks = normal_task_gen.step(target_cpu_load,task_sizes[0],max_task_load)
        for i in range(len(task_sizes)-1):

            dqn_state, _ = dqn_env.observe(copy.deepcopy(tasks))
            rrlo_state, _ = rrlo_env.observe(copy.deepcopy(tasks))
            #conference_state, _ = conference_env.observe(copy.deepcopy(tasks))
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
            
            
                dqn_env.step(actions_dqn_str)
                rrlo_env.step(actions_rrlo)
                
                local_env.step(actions_local_str)
                remote_env.step(actions_remote_str)
                random_env.step(actions_random_str)
                for jobs in local_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        local_task_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            local_task_deadline_missed[j.t_id,i] += 1
                    local_task_num_tasks[j.t_id,i] += len(jobs)

                for jobs in remote_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        remote_task_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            remote_task_deadline_missed[j.t_id,i] += 1
                    remote_task_num_tasks[j.t_id,i] += len(jobs)

                for jobs in random_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        random_task_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            random_task_deadline_missed[j.t_id,i] += 1
                    random_task_num_tasks[j.t_id,i] += len(jobs)

                for jobs in dqn_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        dqn_task_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            dqn_task_deadline_missed[j.t_id,i] += 1
                    dqn_task_num_tasks[j.t_id,i] += len(jobs)

                for jobs in rrlo_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        rrlo_task_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            rrlo_task_deadline_missed[j.t_id,i] += 1
                    rrlo_task_num_tasks[j.t_id,i] += len(jobs)

                if i ==len(task_sizes)-2:
                    tasks = normal_task_gen.step(target_cpu_load,task_sizes[i],max_task_load)
                else:
                    tasks = normal_task_gen.step(target_cpu_load,task_sizes[i+1],max_task_load)
          
                next_state_dqn, _ = dqn_env.observe(copy.deepcopy(tasks))
                next_state_rrlo, _ = rrlo_env.observe(copy.deepcopy(tasks))
                local_env.observe(copy.deepcopy(tasks))
                remote_env.observe(copy.deepcopy(tasks))
                random_env.observe(copy.deepcopy(tasks))

                # Update current state
                dqn_state = next_state_dqn
                rrlo_state = next_state_rrlo

    
    
    
        np.set_printoptions(suppress=True)
        dqn_avg_task_energy_cons = dqn_task_energy / dqn_task_num_tasks
        local_avg_task_energy_cons = local_task_energy / local_task_num_tasks
        remote_avg_task_energy_cons = remote_task_energy / remote_task_num_tasks
        random_avg_task_energy_cons = random_task_energy / random_task_num_tasks
        rrlo_avg_task_energy_cons = rrlo_task_energy / rrlo_task_num_tasks




        dqn_percent_task_missed = dqn_task_deadline_missed / dqn_task_num_tasks * 100
        local_percent_task_missed = local_task_deadline_missed / local_task_num_tasks * 100
        remote_percent_task_missed = remote_task_deadline_missed / remote_task_num_tasks * 100
        random_percent_task_missed = random_task_deadline_missed / random_task_num_tasks * 100
        rrlo_percent_task_missed = rrlo_task_deadline_missed / rrlo_task_num_tasks * 100



        avg_dqn_energy_task = np.sum(dqn_task_energy,axis=0) / np.sum(dqn_task_num_tasks,axis=0)
        avg_local_energy_task = np.sum(local_task_energy,axis=0) / np.sum(local_task_num_tasks,axis=0)
        avg_remote_energy_task = np.sum(remote_task_energy,axis=0) / np.sum(remote_task_num_tasks,axis=0)
        avg_random_energy_task = np.sum(random_task_energy,axis=0) / np.sum(random_task_num_tasks,axis=0)
        avg_rrlo_energy_task = np.sum(rrlo_task_energy,axis=0) / np.sum(rrlo_task_num_tasks,axis=0)


        total_dqn_missed_task = (
            np.sum(dqn_task_deadline_missed,axis=0) / np.sum(dqn_task_num_tasks,axis=0) * 100
        )
        total_local_missed_task = (
            np.sum(local_task_deadline_missed,axis=0) / np.sum(local_task_num_tasks,axis=0) * 100
        )
        total_remote_missed_task = (
            np.sum(remote_task_deadline_missed,axis=0) / np.sum(remote_task_num_tasks,axis=0) * 100
        )
        total_random_missed_task = (
            np.sum(random_task_deadline_missed,axis=0) / np.sum(random_task_num_tasks,axis=0) * 100
        )
        total_rrlo_missed_task = (
            np.sum(rrlo_task_deadline_missed,axis=0) / np.sum(rrlo_task_num_tasks,axis=0) * 100
        )


        results["task_energy"] = np.array(
            [
                avg_random_energy_task,
                avg_rrlo_energy_task,
                avg_dqn_energy_task,
                avg_local_energy_task,
                avg_remote_energy_task
            ]
        )


        results["task_drop"] = np.array(
            [
                total_random_missed_task,
                total_rrlo_missed_task,
                total_dqn_missed_task,
                total_local_missed_task,
                total_remote_missed_task
            ]
        )



    if CN_eval:
        set_random_seed(42)
        max_task_load=2
        target_cpu_load = 0.35

        random_task_gen = RandomTaskGen(configs["task_set3"])
        dqn_env = DQNEnv(configs, random_task_gen.get_wcet_bound(), random_task_gen.get_task_size_bound())
        local_env = DQNEnv(configs, random_task_gen.get_wcet_bound(), random_task_gen.get_task_size_bound())
        remote_env = DQNEnv(configs, random_task_gen.get_wcet_bound(), random_task_gen.get_task_size_bound())
        random_env = DQNEnv(configs, random_task_gen.get_wcet_bound(), random_task_gen.get_task_size_bound())
        rand_freq_list = random_env.cpu.freqs
        rand_powers_list = random_env.w_inter.powers
        rrlo_env = RRLOEnv(configs)
    
        dqn_dvfs = DQN_DVFS(state_dim=configs["dqn_state_dim"], act_space=dqn_env.get_action_space())
        rrlo_dvfs = RRLO_DVFS(
            state_bounds=rrlo_env.get_state_bounds(),
            num_w_inter_powers=len(rrlo_env.w_inter.powers),
            num_dvfs_algs=2,
            dvfs_algs=["cc", "la"],
            num_tasks=4,
        )

        dqn_dvfs.load_model("models/RRLO_train")
        rrlo_dvfs.load_model("models/RRLO_train")

        dqn_cn_energy = np.zeros((4,len(CNs)))
        dqn_cn_num_tasks = np.zeros((4,len(CNs)))
        rrlo_cn_energy = np.zeros((4,len(CNs)))
        rrlo_cn_num_tasks = np.zeros((4,len(CNs)))
        local_cn_energy = np.zeros((4,len(CNs)))
        local_cn_num_tasks = np.zeros((4,len(CNs)))
        remote_cn_energy = np.zeros((4,len(CNs)))
        remote_cn_num_tasks = np.zeros((4,len(CNs)))
        random_cn_energy = np.zeros((4,len(CNs)))
        random_cn_num_tasks = np.zeros((4,len(CNs)))

        dqn_cn_deadline_missed =np.zeros((4,len(CNs)))
        rrlo_cn_deadline_missed =np.zeros((4,len(CNs)))
        local_cn_deadline_missed =np.zeros((4,len(CNs)))
        remote_cn_deadline_missed =np.zeros((4,len(CNs)))
        random_cn_deadline_missed =np.zeros((4,len(CNs)))


        for i in range(len(CNs)):

            dqn_env.w_inter.cn_setter(CNs[i])
            rrlo_env.w_inter.cn_setter(CNs[i])
            local_env.w_inter.cn_setter(CNs[i])
            remote_env.w_inter.cn_setter(CNs[i])
            random_env.w_inter.cn_setter(CNs[i])

            tasks = random_task_gen.step(target_cpu_load,max_task_load)
            dqn_state, _ = dqn_env.observe(copy.deepcopy(tasks))
            rrlo_state, _ = rrlo_env.observe(copy.deepcopy(tasks))
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
            
            
                dqn_env.step(actions_dqn_str)
                rrlo_env.step(actions_rrlo)
                
                local_env.step(actions_local_str)
                remote_env.step(actions_remote_str)
                random_env.step(actions_random_str)

            
                for jobs in local_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        local_cn_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            local_cn_deadline_missed[j.t_id,i] += 1
                    local_cn_num_tasks[j.t_id,i] += len(jobs)

                for jobs in remote_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        remote_cn_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            remote_cn_deadline_missed[j.t_id,i] += 1
                    remote_cn_num_tasks[j.t_id,i] += len(jobs)

                for jobs in random_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        random_cn_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            random_cn_deadline_missed[j.t_id,i] += 1
                    random_cn_num_tasks[j.t_id,i] += len(jobs)

                for jobs in dqn_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        dqn_cn_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            dqn_cn_deadline_missed[j.t_id,i] += 1
                    dqn_cn_num_tasks[j.t_id,i] += len(jobs)

                for jobs in rrlo_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        rrlo_cn_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            rrlo_cn_deadline_missed[j.t_id,i] += 1
                    rrlo_cn_num_tasks[j.t_id,i] += len(jobs)


                tasks = random_task_gen.step(target_cpu_load,max_task_load)
                next_state_dqn, _ = dqn_env.observe(copy.deepcopy(tasks))
                next_state_rrlo, _ = rrlo_env.observe(copy.deepcopy(tasks))


                # Update current state
                dqn_state = next_state_dqn
                rrlo_state = next_state_rrlo
                #conference_state = next_state_conference


        np.set_printoptions(suppress=True)
        dqn_avg_cn_energy = dqn_cn_energy / dqn_cn_num_tasks
        rrlo_avg_cn_energy = rrlo_cn_energy / rrlo_cn_num_tasks
        local_avg_cn_energy = local_cn_energy / local_cn_num_tasks
        random_avg_cn_energy = random_cn_energy / random_cn_num_tasks
        remote_avg_cn_energy = remote_cn_energy / remote_cn_num_tasks




        dqn_percent_cn_missed = dqn_cn_deadline_missed / dqn_cn_num_tasks * 100
        local_percent_cn_missed = local_cn_deadline_missed / local_cn_num_tasks * 100
        rrlo_percent_cn_missed = rrlo_cn_deadline_missed / rrlo_cn_num_tasks * 100
        remote_percent_cn_missed = remote_cn_deadline_missed / remote_cn_num_tasks * 100
        random_percent_cn_missed = random_cn_deadline_missed / dqn_cn_num_tasks * 100



        avg_dqn_cn_energy = np.sum(dqn_cn_energy,axis=0) / np.sum(dqn_cn_num_tasks,axis=0)
        avg_rrlo_cn_energy = np.sum(rrlo_cn_energy,axis=0) / np.sum(rrlo_cn_num_tasks,axis=0)
        avg_local_cn_energy = np.sum(local_cn_energy,axis=0) / np.sum(local_cn_num_tasks,axis=0)
        avg_remote_cn_energy = np.sum(remote_cn_energy,axis=0) / np.sum(remote_cn_num_tasks,axis=0)
        avg_random_cn_energy = np.sum(random_cn_energy,axis=0) / np.sum(random_cn_num_tasks,axis=0)



        total_dqn_missed_cn = (
            np.sum(dqn_cn_deadline_missed,axis=0) / np.sum(dqn_cn_num_tasks,axis=0) * 100
        )
        total_rrlo_missed_cn = (
            np.sum(rrlo_cn_deadline_missed,axis=0) / np.sum(rrlo_cn_num_tasks,axis=0) * 100
        )
        total_local_missed_cn = (
            np.sum(local_cn_deadline_missed,axis=0) / np.sum(local_cn_num_tasks,axis=0) * 100
        )
        total_remote_missed_cn = (
            np.sum(remote_cn_deadline_missed,axis=0) / np.sum(remote_cn_num_tasks,axis=0) * 100
        )
        total_random_missed_cn = (
            np.sum(random_cn_deadline_missed,axis=0) / np.sum(random_cn_num_tasks,axis=0) * 100
        )


        results["cn_energy"] = np.array(
            [
                avg_random_cn_energy,
                avg_rrlo_cn_energy,
                avg_dqn_cn_energy,
                avg_local_cn_energy,
                avg_remote_cn_energy
            ]
        )

        results["cn_drop"] = np.array(
            [
                total_random_missed_cn,
                total_rrlo_missed_cn,
                total_dqn_missed_cn,
                total_local_missed_cn,
                total_remote_missed_cn
            ]
        )



    return results





def big_LITTLE_evaluate(
        configs,
        cpu_loads,
        task_sizes, 
        CNs, 
        eval_itr=10000,
        taskset_eval=True,
        CPU_load_eval=True, 
        task_size_eval=True, 
        CN_eval= True
    ):


    results={ 
        "taskset_energy":[], "taskset_drop":[],
        "taskset_improvement":[],
        "cpu_energy":[], "cpu_drop":[],
        "task_energy":[], "task_drop":[],
        "cn_energy":[], "cn_drop":[]
    }


    dqn_energy = np.zeros((4,2))
    dqn_num_tasks = np.zeros((4,2))
    local_energy = np.zeros((4,2))
    local_num_tasks = np.zeros((4,2))
    remote_energy = np.zeros((4,2))
    remote_num_tasks = np.zeros((4,2))
    random_energy = np.zeros((4,2))
    random_num_tasks = np.zeros((4,2))
            

    dqn_deadline_missed = np.zeros((4,2))
    local_deadline_missed = np.zeros((4,2))
    remote_deadline_missed = np.zeros((4,2))
    random_deadline_missed = np.zeros((4,2))

    if taskset_eval:
        for i in range(2):
            if i==0:
                task_gen = TaskGen(configs["task_set1"])
            if i==1:
                task_gen = TaskGen(configs["task_set2"])
            dqn_env = BaseDQNEnv(configs, task_gen.get_wcet_bound(), task_gen.get_task_size_bound())
            local_env = BaseDQNEnv(configs, task_gen.get_wcet_bound(), task_gen.get_task_size_bound())
            remote_env = BaseDQNEnv(configs, task_gen.get_wcet_bound(), task_gen.get_task_size_bound())
            random_env = BaseDQNEnv(configs, task_gen.get_wcet_bound(), task_gen.get_task_size_bound())
            rand_littlefreq_list = random_env.cpu_little.freqs
            rand_bigfreq_list = random_env.cpu_big.freqs
            rand_powers_list = random_env.w_inter.powers

            dqn_dvfs = DQN_DVFS(state_dim=configs["dqn_state_dim"], act_space=dqn_env.get_action_space())

            dqn_dvfs.load_model("models/iDEAS_train")






            tasks = task_gen.step()
            dqn_state, _ = dqn_env.observe(copy.deepcopy(tasks))
            local_env.observe(copy.deepcopy(tasks))
            remote_env.observe(copy.deepcopy(tasks))
            random_env.observe(copy.deepcopy(tasks))

            for _ in range(eval_itr):
                actions_dqn = dqn_dvfs.execute(dqn_state, eval_mode=True)
                actions_dqn_str = dqn_dvfs.conv_acts(actions_dqn)
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



                dqn_env.step(actions_dqn_str)
                local_env.step(actions_local_str)
                remote_env.step(actions_remote_str)
                random_env.step(actions_random_str)

                # Gather energy consumption values
                for jobs in local_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        local_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            local_deadline_missed[j.t_id,i] += 1
                    local_num_tasks[j.t_id,i] += len(jobs)

                for jobs in remote_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        remote_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            remote_deadline_missed[j.t_id,i] += 1
                    remote_num_tasks[j.t_id,i] += len(jobs)

                for jobs in random_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        random_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            random_deadline_missed[j.t_id,i] += 1
                    random_num_tasks[j.t_id,i] += len(jobs)

                for jobs in dqn_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        dqn_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            dqn_deadline_missed[j.t_id,i] += 1
                    dqn_num_tasks[j.t_id,i] += len(jobs)


                tasks = task_gen.step()
                next_state_dqn, _ = dqn_env.observe(copy.deepcopy(tasks))
                local_env.observe(copy.deepcopy(tasks))
                remote_env.observe(copy.deepcopy(tasks))
                random_env.observe(copy.deepcopy(tasks))

                # Update current state
                dqn_state = next_state_dqn

        np.set_printoptions(suppress=True)
        dqn_avg_energy = dqn_energy / dqn_num_tasks
        local_avg_energy = local_energy / local_num_tasks
        remote_avg_energy = remote_energy / remote_num_tasks
        random_avg_energy = random_energy / random_num_tasks

        dqn_percent_missed = dqn_deadline_missed / dqn_num_tasks * 100
        local_percent_missed = local_deadline_missed / local_num_tasks * 100
        remote_percent_missed = remote_deadline_missed / remote_num_tasks * 100
        random_percent_missed = random_deadline_missed / random_num_tasks * 100


        avg_dqn_energy = np.sum(dqn_energy,axis=0) / np.sum(dqn_num_tasks,axis=0)
        avg_local_energy = np.sum(local_energy,axis=0) / np.sum(local_num_tasks,axis=0)
        avg_remote_energy = np.sum(remote_energy,axis=0) / np.sum(remote_num_tasks,axis=0)
        avg_random_energy = np.sum(random_energy,axis=0) / np.sum(random_num_tasks,axis=0)

        total_dqn_missed = (
            np.sum(dqn_deadline_missed,axis=0) / np.sum(dqn_num_tasks,axis=0) * 100
        )
        total_local_missed = (
            np.sum(local_deadline_missed,axis=0) / np.sum(local_num_tasks,axis=0) * 100
        )
        total_remote_missed = (
            np.sum(remote_deadline_missed,axis=0) / np.sum(remote_num_tasks,axis=0) * 100
        )
        total_random_missed = (
            np.sum(random_deadline_missed,axis=0) / np.sum(random_num_tasks,axis=0) * 100
        )


        dqn_random_improvement = (
            (avg_dqn_energy - avg_random_energy) / (avg_random_energy) * 100
        )

        dqn_local_improvement = (
            (avg_dqn_energy - avg_local_energy) / (avg_local_energy) * 100
        )

        dqn_remote_improvement = (
            (avg_dqn_energy - avg_remote_energy) / (avg_remote_energy) * 100
        )



        
        results["taskset_energy"] = np.array(
            [
                avg_random_energy,
                avg_dqn_energy,
                avg_local_energy,
                avg_remote_energy
            ]
        )

        results["taskset_drop"] = np.array(
            [
                total_random_missed,
                total_dqn_missed,
                total_local_missed,
                total_remote_missed
            ]
        )

        results["taskset_improvement"] = np.array(
            [
                dqn_random_improvement,
                dqn_local_improvement,
                dqn_remote_improvement
            ]
        )
    


    if CPU_load_eval:
        set_random_seed(42)
        max_task_load=3
        
        random_task_gen = RandomTaskGen(configs["task_set3"])
        dqn_env = BaseDQNEnv(configs, random_task_gen.get_wcet_bound(), random_task_gen.get_task_size_bound())
        local_env = BaseDQNEnv(configs, random_task_gen.get_wcet_bound(), random_task_gen.get_task_size_bound())
        remote_env = BaseDQNEnv(configs, random_task_gen.get_wcet_bound(), random_task_gen.get_task_size_bound())
        random_env = BaseDQNEnv(configs, random_task_gen.get_wcet_bound(), random_task_gen.get_task_size_bound())
        rand_littlefreq_list = random_env.cpu_little.freqs
        rand_bigfreq_list = random_env.cpu_big.freqs
        rand_powers_list = random_env.w_inter.powers
    
        dqn_dvfs = DQN_DVFS(state_dim=configs["dqn_state_dim"], act_space=dqn_env.get_action_space())


        dqn_dvfs.load_model("models/iDEAS_train")

        dqn_cpu_energy = np.zeros((4,len(cpu_loads)))
        dqn_cpu_num_tasks = np.zeros((4,len(cpu_loads)))

        local_cpu_energy = np.zeros((4,len(cpu_loads)))
        local_cpu_num_tasks = np.zeros((4,len(cpu_loads)))
        remote_cpu_energy = np.zeros((4,len(cpu_loads)))
        remote_cpu_num_tasks = np.zeros((4,len(cpu_loads)))
        random_cpu_energy = np.zeros((4,len(cpu_loads)))
        random_cpu_num_tasks = np.zeros((4,len(cpu_loads)))

        local_cpu_deadline_missed = np.zeros((4,len(cpu_loads)))
        random_cpu_deadline_missed = np.zeros((4,len(cpu_loads)))
        remote_cpu_deadline_missed = np.zeros((4,len(cpu_loads)))
        dqn_cpu_deadline_missed = np.zeros((4,len(cpu_loads)))
 
    
    
    
    
        for i in range(len(cpu_loads)):

            target_cpu_load=cpu_loads[i]
            tasks = random_task_gen.step(target_cpu_load,max_task_load)
            dqn_state, _ = dqn_env.observe(copy.deepcopy(tasks))
            local_env.observe(copy.deepcopy(tasks))
            remote_env.observe(copy.deepcopy(tasks))
            random_env.observe(copy.deepcopy(tasks))
            for _ in range(eval_itr):
                actions_dqn = dqn_dvfs.execute(dqn_state, eval_mode=True)
                actions_dqn_str = dqn_dvfs.conv_acts(actions_dqn)
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
            
                dqn_env.step(actions_dqn_str)

                local_env.step(actions_local_str)
                remote_env.step(actions_remote_str)
                random_env.step(actions_random_str)

            
                # Gather energy consumption values
                for jobs in local_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        local_cpu_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            local_cpu_deadline_missed[j.t_id,i] += 1
                    local_cpu_num_tasks[j.t_id,i] += len(jobs)

                for jobs in remote_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        remote_cpu_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            remote_cpu_deadline_missed[j.t_id,i] += 1
                    remote_cpu_num_tasks[j.t_id,i] += len(jobs)

                for jobs in random_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        random_cpu_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            random_cpu_deadline_missed[j.t_id,i] += 1
                    random_cpu_num_tasks[j.t_id,i] += len(jobs)

                for jobs in dqn_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        dqn_cpu_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            dqn_cpu_deadline_missed[j.t_id,i] += 1
                    dqn_cpu_num_tasks[j.t_id,i] += len(jobs)


                tasks = random_task_gen.step(target_cpu_load,max_task_load)
                next_state_dqn, _ = dqn_env.observe(copy.deepcopy(tasks))
                #next_state_conference, _ = conference_env.observe(copy.deepcopy(tasks))
                local_env.observe(copy.deepcopy(tasks))
                remote_env.observe(copy.deepcopy(tasks))
                random_env.observe(copy.deepcopy(tasks))

                # Update current state
                dqn_state = next_state_dqn
                #conference_state = next_state_conference


        np.set_printoptions(suppress=True)
        dqn_avg_cpu_energy = dqn_cpu_energy / dqn_cpu_num_tasks
        local_avg_cpu_energy = local_cpu_energy / local_cpu_num_tasks
        remote_avg_cpu_energy = remote_cpu_energy / remote_cpu_num_tasks
        random_avg_cpu_energy = random_cpu_energy / random_cpu_num_tasks
    



        dqn_percent_missed_cpu = dqn_cpu_deadline_missed / dqn_cpu_num_tasks * 100
        local_percent_missed_cpu = local_cpu_deadline_missed / local_cpu_num_tasks * 100
        remote_percent_missed_cpu = remote_cpu_deadline_missed / remote_cpu_num_tasks * 100
        random_percent_missed_cpu = random_cpu_deadline_missed / random_cpu_num_tasks * 100



        avg_dqn_energy_cpu = np.sum(dqn_cpu_energy,axis=0) / np.sum(dqn_cpu_num_tasks,axis=0)
        avg_local_energy_cpu = np.sum(local_cpu_energy,axis=0) / np.sum(local_cpu_num_tasks,axis=0)
        avg_remote_energy_cpu = np.sum(remote_cpu_energy,axis=0) / np.sum(remote_cpu_num_tasks,axis=0)
        avg_random_energy_cpu = np.sum(random_cpu_energy,axis=0) / np.sum(random_cpu_num_tasks,axis=0)



        total_dqn_missed_cpu = (
            np.sum(dqn_cpu_deadline_missed,axis=0) / np.sum(dqn_cpu_num_tasks,axis=0) * 100
        )
        total_local_missed_cpu = (
            np.sum(local_cpu_deadline_missed,axis=0) / np.sum(local_cpu_num_tasks,axis=0) * 100
        )
        total_remote_missed_cpu = (
            np.sum(remote_cpu_deadline_missed,axis=0) / np.sum(remote_cpu_num_tasks,axis=0) * 100
        )
        total_random_missed_cpu = (
            np.sum(random_cpu_deadline_missed,axis=0) / np.sum(random_cpu_num_tasks,axis=0) * 100
        )


        results["cpu_energy"] = np.array(
            [
                avg_random_energy_cpu,
                avg_dqn_energy_cpu,
                avg_local_energy_cpu,
                avg_remote_energy_cpu
            ]
        )

        results["cpu_drop"] = np.array(
            [
                total_random_missed_cpu,
                total_dqn_missed_cpu,
                total_local_missed_cpu,
                total_remote_missed_cpu
            ]
        )

    if task_size_eval:
        set_random_seed(42)

        target_cpu_load = 0.35
        max_task_load=3

        normal_task_gen = NormalTaskGen(configs["task_set3"])
        dqn_env = BaseDQNEnv(configs, normal_task_gen.get_wcet_bound(), normal_task_gen.get_task_size_bound())
        local_env = BaseDQNEnv(configs, normal_task_gen.get_wcet_bound(), normal_task_gen.get_task_size_bound())
        remote_env = BaseDQNEnv(configs, normal_task_gen.get_wcet_bound(), normal_task_gen.get_task_size_bound())
        random_env = BaseDQNEnv(configs, normal_task_gen.get_wcet_bound(), normal_task_gen.get_task_size_bound())
        rand_littlefreq_list = random_env.cpu_little.freqs
        rand_bigfreq_list = random_env.cpu_big.freqs
        rand_powers_list = random_env.w_inter.powers

        dqn_dvfs = DQN_DVFS(state_dim=configs["dqn_state_dim"], act_space=dqn_env.get_action_space())

        dqn_dvfs.load_model("models/iDEAS_train")


        dqn_task_energy = np.zeros((4,len(task_sizes)-1))
        dqn_task_num_tasks = np.zeros((4,len(task_sizes)-1))

        local_task_energy = np.zeros((4,len(task_sizes)-1))
        local_task_num_tasks = np.zeros((4,len(task_sizes)-1))
        remote_task_energy = np.zeros((4,len(task_sizes)-1))
        remote_task_num_tasks = np.zeros((4,len(task_sizes)-1))
        random_task_energy = np.zeros((4,len(task_sizes)-1))
        random_task_num_tasks = np.zeros((4,len(task_sizes)-1))

        local_task_deadline_missed = np.zeros((4,len(task_sizes)-1))
        random_task_deadline_missed = np.zeros((4,len(task_sizes)-1))
        remote_task_deadline_missed = np.zeros((4,len(task_sizes)-1))
        dqn_task_deadline_missed = np.zeros((4,len(task_sizes)-1))
  
    
    
    
        tasks = normal_task_gen.step(target_cpu_load,task_sizes[0],max_task_load)
        for i in range(len(task_sizes)-1):

            dqn_state, _ = dqn_env.observe(copy.deepcopy(tasks))
            #conference_state, _ = conference_env.observe(copy.deepcopy(tasks))
            local_env.observe(copy.deepcopy(tasks))
            remote_env.observe(copy.deepcopy(tasks))
            random_env.observe(copy.deepcopy(tasks))
            for _ in range(eval_itr):
                actions_dqn = dqn_dvfs.execute(dqn_state, eval_mode=True)
                actions_dqn_str = dqn_dvfs.conv_acts(actions_dqn)
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
            
            
                dqn_env.step(actions_dqn_str)
                
                local_env.step(actions_local_str)
                remote_env.step(actions_remote_str)
                random_env.step(actions_random_str)
                for jobs in local_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        local_task_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            local_task_deadline_missed[j.t_id,i] += 1
                    local_task_num_tasks[j.t_id,i] += len(jobs)

                for jobs in remote_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        remote_task_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            remote_task_deadline_missed[j.t_id,i] += 1
                    remote_task_num_tasks[j.t_id,i] += len(jobs)

                for jobs in random_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        random_task_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            random_task_deadline_missed[j.t_id,i] += 1
                    random_task_num_tasks[j.t_id,i] += len(jobs)

                for jobs in dqn_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        dqn_task_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            dqn_task_deadline_missed[j.t_id,i] += 1
                    dqn_task_num_tasks[j.t_id,i] += len(jobs)


                if i ==len(task_sizes)-2:
                    tasks = normal_task_gen.step(target_cpu_load,task_sizes[i],max_task_load)
                else:
                    tasks = normal_task_gen.step(target_cpu_load,task_sizes[i+1],max_task_load)
          
                next_state_dqn, _ = dqn_env.observe(copy.deepcopy(tasks))
                local_env.observe(copy.deepcopy(tasks))
                remote_env.observe(copy.deepcopy(tasks))
                random_env.observe(copy.deepcopy(tasks))

                # Update current state
                dqn_state = next_state_dqn

    
    
    
        np.set_printoptions(suppress=True)
        dqn_avg_task_energy_cons = dqn_task_energy / dqn_task_num_tasks
        local_avg_task_energy_cons = local_task_energy / local_task_num_tasks
        remote_avg_task_energy_cons = remote_task_energy / remote_task_num_tasks
        random_avg_task_energy_cons = random_task_energy / random_task_num_tasks




        dqn_percent_task_missed = dqn_task_deadline_missed / dqn_task_num_tasks * 100
        local_percent_task_missed = local_task_deadline_missed / local_task_num_tasks * 100
        remote_percent_task_missed = remote_task_deadline_missed / remote_task_num_tasks * 100
        random_percent_task_missed = random_task_deadline_missed / random_task_num_tasks * 100



        avg_dqn_energy_task = np.sum(dqn_task_energy,axis=0) / np.sum(dqn_task_num_tasks,axis=0)
        avg_local_energy_task = np.sum(local_task_energy,axis=0) / np.sum(local_task_num_tasks,axis=0)
        avg_remote_energy_task = np.sum(remote_task_energy,axis=0) / np.sum(remote_task_num_tasks,axis=0)
        avg_random_energy_task = np.sum(random_task_energy,axis=0) / np.sum(random_task_num_tasks,axis=0)


        total_dqn_missed_task = (
            np.sum(dqn_task_deadline_missed,axis=0) / np.sum(dqn_task_num_tasks,axis=0) * 100
        )
        total_local_missed_task = (
            np.sum(local_task_deadline_missed,axis=0) / np.sum(local_task_num_tasks,axis=0) * 100
        )
        total_remote_missed_task = (
            np.sum(remote_task_deadline_missed,axis=0) / np.sum(remote_task_num_tasks,axis=0) * 100
        )
        total_random_missed_task = (
            np.sum(random_task_deadline_missed,axis=0) / np.sum(random_task_num_tasks,axis=0) * 100
        )



        results["task_energy"] = np.array(
            [
                avg_random_energy_task,
                avg_dqn_energy_task,
                avg_local_energy_task,
                avg_remote_energy_task
            ]
        )


        results["task_drop"] = np.array(
            [
                total_random_missed_task,
                total_dqn_missed_task,
                total_local_missed_task,
                total_remote_missed_task
            ]
        )



    if CN_eval:
        set_random_seed(42)
        max_task_load=3
        target_cpu_load = 0.35

        random_task_gen = RandomTaskGen(configs["task_set3"])
        dqn_env = BaseDQNEnv(configs, random_task_gen.get_wcet_bound(), random_task_gen.get_task_size_bound())
        local_env = BaseDQNEnv(configs, random_task_gen.get_wcet_bound(), random_task_gen.get_task_size_bound())
        remote_env = BaseDQNEnv(configs, random_task_gen.get_wcet_bound(), random_task_gen.get_task_size_bound())
        random_env = BaseDQNEnv(configs, random_task_gen.get_wcet_bound(), random_task_gen.get_task_size_bound())
        rand_littlefreq_list = random_env.cpu_little.freqs
        rand_bigfreq_list = random_env.cpu_big.freqs
        rand_powers_list = random_env.w_inter.powers
    
        dqn_dvfs = DQN_DVFS(state_dim=configs["dqn_state_dim"], act_space=dqn_env.get_action_space())

        dqn_dvfs.load_model("models/iDEAS_train")

        dqn_cn_energy = np.zeros((4,len(CNs)))
        dqn_cn_num_tasks = np.zeros((4,len(CNs)))
        local_cn_energy = np.zeros((4,len(CNs)))
        local_cn_num_tasks = np.zeros((4,len(CNs)))
        remote_cn_energy = np.zeros((4,len(CNs)))
        remote_cn_num_tasks = np.zeros((4,len(CNs)))
        random_cn_energy = np.zeros((4,len(CNs)))
        random_cn_num_tasks = np.zeros((4,len(CNs)))

        dqn_cn_deadline_missed =np.zeros((4,len(CNs)))
        local_cn_deadline_missed =np.zeros((4,len(CNs)))
        remote_cn_deadline_missed =np.zeros((4,len(CNs)))
        random_cn_deadline_missed =np.zeros((4,len(CNs)))


        for i in range(len(CNs)):

            dqn_env.w_inter.cn_setter(CNs[i])
            local_env.w_inter.cn_setter(CNs[i])
            remote_env.w_inter.cn_setter(CNs[i])
            random_env.w_inter.cn_setter(CNs[i])

            tasks = random_task_gen.step(target_cpu_load,max_task_load)
            dqn_state, _ = dqn_env.observe(copy.deepcopy(tasks))
            local_env.observe(copy.deepcopy(tasks))
            remote_env.observe(copy.deepcopy(tasks))
            random_env.observe(copy.deepcopy(tasks))

            for _ in range(eval_itr):
                actions_dqn = dqn_dvfs.execute(dqn_state, eval_mode=True)
                actions_dqn_str = dqn_dvfs.conv_acts(actions_dqn)
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
            
            
                dqn_env.step(actions_dqn_str)
                
                local_env.step(actions_local_str)
                remote_env.step(actions_remote_str)
                random_env.step(actions_random_str)

            
                for jobs in local_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        local_cn_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            local_cn_deadline_missed[j.t_id,i] += 1
                    local_cn_num_tasks[j.t_id,i] += len(jobs)

                for jobs in remote_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        remote_cn_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            remote_cn_deadline_missed[j.t_id,i] += 1
                    remote_cn_num_tasks[j.t_id,i] += len(jobs)

                for jobs in random_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        random_cn_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            random_cn_deadline_missed[j.t_id,i] += 1
                    random_cn_num_tasks[j.t_id,i] += len(jobs)

                for jobs in dqn_env.curr_tasks.values():
                    for j in jobs:
                        # print(j)
                        dqn_cn_energy[j.t_id,i] += j.cons_energy
                        if j.deadline_missed:
                            dqn_cn_deadline_missed[j.t_id,i] += 1
                    dqn_cn_num_tasks[j.t_id,i] += len(jobs)


                tasks = random_task_gen.step(target_cpu_load,max_task_load)
                next_state_dqn, _ = dqn_env.observe(copy.deepcopy(tasks))


                # Update current state
                dqn_state = next_state_dqn
                #conference_state = next_state_conference


        np.set_printoptions(suppress=True)
        dqn_avg_cn_energy = dqn_cn_energy / dqn_cn_num_tasks
        local_avg_cn_energy = local_cn_energy / local_cn_num_tasks
        random_avg_cn_energy = random_cn_energy / random_cn_num_tasks
        remote_avg_cn_energy = remote_cn_energy / remote_cn_num_tasks




        dqn_percent_cn_missed = dqn_cn_deadline_missed / dqn_cn_num_tasks * 100
        local_percent_cn_missed = local_cn_deadline_missed / local_cn_num_tasks * 100
        remote_percent_cn_missed = remote_cn_deadline_missed / remote_cn_num_tasks * 100
        random_percent_cn_missed = random_cn_deadline_missed / dqn_cn_num_tasks * 100



        avg_dqn_cn_energy = np.sum(dqn_cn_energy,axis=0) / np.sum(dqn_cn_num_tasks,axis=0)
        avg_local_cn_energy = np.sum(local_cn_energy,axis=0) / np.sum(local_cn_num_tasks,axis=0)
        avg_remote_cn_energy = np.sum(remote_cn_energy,axis=0) / np.sum(remote_cn_num_tasks,axis=0)
        avg_random_cn_energy = np.sum(random_cn_energy,axis=0) / np.sum(random_cn_num_tasks,axis=0)



        total_dqn_missed_cn = (
            np.sum(dqn_cn_deadline_missed,axis=0) / np.sum(dqn_cn_num_tasks,axis=0) * 100
        )
        total_local_missed_cn = (
            np.sum(local_cn_deadline_missed,axis=0) / np.sum(local_cn_num_tasks,axis=0) * 100
        )
        total_remote_missed_cn = (
            np.sum(remote_cn_deadline_missed,axis=0) / np.sum(remote_cn_num_tasks,axis=0) * 100
        )
        total_random_missed_cn = (
            np.sum(random_cn_deadline_missed,axis=0) / np.sum(random_cn_num_tasks,axis=0) * 100
        )


        results["cn_energy"] = np.array(
            [
                avg_random_cn_energy,
                avg_dqn_cn_energy,
                avg_local_cn_energy,
                avg_remote_cn_energy
            ]
        )

        results["cn_drop"] = np.array(
            [
                total_random_missed_cn,
                total_dqn_missed_cn,
                total_local_missed_cn,
                total_remote_missed_cn
            ]
        )

    return results


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

if __name__ == "__main__":
    rrlo_scenario_configs = {
        "task_set1": "configs/task_set_eval.json",
        "task_set2": "configs/task_set_eval2.json",
        "task_set3": "configs/task_set_train.json",
        "cpu_local": "configs/cpu_local.json",
        "w_inter": "configs/wireless_interface.json",
        "dqn_state_dim": 4,
    }


    dqn_scenario_configs = {
        "task_set1": "configs/task_set_eval.json",
        "task_set2": "configs/task_set_eval2.json",
        "task_set3": "configs/task_set_train.json",
        "cpu_local": "configs/cpu_local.json",
        "cpu_little": "configs/cpu_little.json",
        "cpu_big": "configs/cpu_big.json",
        "w_inter": "configs/wireless_interface.json",
        "dqn_state_dim": 5,
    }

    cpuloads= np.arange(0.05, 1.01, 0.05)
    tasksizes= np.round(np.linspace(110, 490, 11))
    cns = cn_values=np.linspace(2e-11, 2e-6, 10)
    iDEAS_evaluate(dqn_scenario_configs,cpu_loads=cpuloads,task_sizes=tasksizes,CNs=cns,eval_itr=5000)
    RRLO_evaluate(rrlo_scenario_configs, cpu_loads=cpuloads,task_sizes=tasksizes,CNs=cns,eval_itr=5000)
    big_LITTLE_evaluate(dqn_scenario_configs, cpu_loads=cpuloads,task_sizes=tasksizes,CNs=cns,eval_itr=5000)
