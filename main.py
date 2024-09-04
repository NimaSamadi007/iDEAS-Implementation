from train import train_rrlo_scenario, train_dqn_scenario
from evaluate import evaluate_rrlo_scenario, evaluate_dqn_scenario,evaluate_cpu_load_scenario, evaluate_task_size_scenario
from utils.utils import plot_res,print_improvement,plot_loss_function,moving_avg,line_plot_res
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def compare_dqn_rrlo(eval_itr=10000, iter=3,Train=False):
    DQN_STATE_DIM = 4
    train_configs = {
        "task_set": "configs/task_set_train.json",
        "cpu_local": "configs/cpu_local.json",
        "w_inter": "configs/wireless_interface.json",
        "dqn_state_dim": DQN_STATE_DIM,
    }

    eval_configs = {
        "task_set": "configs/task_set_eval.json",
        "cpu_local": "configs/cpu_local.json",
        "w_inter": "configs/wireless_interface.json",
        "dqn_state_dim": DQN_STATE_DIM,
    }
    test_configs = {
        "task_set": "configs/task_set_eval2.json",
        "cpu_local": "configs/cpu_local.json",
        "w_inter": "configs/wireless_interface.json",
        "dqn_state_dim": DQN_STATE_DIM,
    }

    if Train:
        dqn_loss=train_rrlo_scenario(train_configs)
        #print(dqn_loss.shape)

        plot_loss_function(dqn_loss, "DQN", "iterations", "loss","DQN_Loss_RRLO_Scenario")


    all_energy_eval1 = np.empty((0, 5))
    all_energy_eval2 = np.empty((0, 5))
    all_deadline_eval1 = np.empty((0, 5))
    all_deadline_eval2 = np.empty((0, 5))


    all_energy_improvement1 = np.empty((0, 4))
    all_energy_improvement2 = np.empty((0, 4))


    for j in tqdm(range(iter)):
        energy_consumption_eval1, missed_deadline_eval1, energy_improvement1 = evaluate_rrlo_scenario(
        eval_configs, eval_itr
        )
        all_energy_eval1=np.vstack((all_energy_eval1,energy_consumption_eval1))
        all_deadline_eval1=np.vstack((all_deadline_eval1,missed_deadline_eval1))
        all_energy_improvement1=np.vstack((all_energy_improvement1,energy_improvement1))

        energy_consumption_eval2, missed_deadline_eval2, energy_improvement2 = evaluate_rrlo_scenario(
        test_configs, eval_itr
        )
        all_energy_eval2=np.vstack((all_energy_eval2,energy_consumption_eval2))
        all_deadline_eval2=np.vstack((all_deadline_eval2,missed_deadline_eval2))
        all_energy_improvement2=np.vstack((all_energy_improvement2,energy_improvement2))

    # Calculate mean and standard deviation column-wise
    mean_energy_consumption_eval1 = np.mean(all_energy_eval1, axis=0)
    std_energy_consumption_eval1 = np.std(10*np.log10(all_energy_eval1/1e-3), axis=0)

    mean_energy_consumption_improvement1 = np.mean(all_energy_improvement1, axis=0)

    mean_missed_deadline_eval1 = np.mean(all_deadline_eval1, axis=0)
    std_missed_deadline_eval1 = np.std(all_deadline_eval1, axis=0)


    # Repeat the same process for eval2
    mean_energy_consumption_eval2 = np.mean(all_energy_eval2, axis=0)
    std_energy_consumption_eval2 = np.std(10*np.log10(all_energy_eval2/1e-3), axis=0)

    mean_energy_consumption_improvement2 = np.mean(all_energy_improvement2, axis=0)

    mean_missed_deadline_eval2 = np.mean(all_deadline_eval2, axis=0)
    std_missed_deadline_eval2 = np.std(all_deadline_eval2, axis=0)







    #print("*"*20)
    #print(mean_energy_consumption_eval1)
    #print(mean_energy_consumption_eval2)
    #print(mean_missed_deadline_eval1)
    #print(mean_missed_deadline_eval2)
    #print("*"*20)
    #print(std_energy_consumption_eval1)
    #print(std_energy_consumption_eval2)
    #print(std_missed_deadline_eval1)
    #print(std_missed_deadline_eval2)
    alg_set = ["Random", "Local", "Remote", "RRLO", "DQN"]
    plot_res(
        alg_set,
        mean_energy_consumption_eval1,
        mean_energy_consumption_eval2,
        std_energy_consumption_eval1,
        std_energy_consumption_eval2,
        "Algorithm",
        "Energy Consumption (mJ)",
        "Energy Consumption of Different Algorithms in RRLO Scenario",
        "Energy_Consumption_RRLO",
        ylog=True
    )
    plot_res(
        alg_set,
        mean_missed_deadline_eval1,
        mean_missed_deadline_eval2,
        std_missed_deadline_eval1,
        std_missed_deadline_eval2,
        "Algorithm",
        "Dropped Tasks (%) ",
        "Dropped Tasks of Different Algorithms in RRLO Scenario",
        "Missed_Deadline_RRLO",
    )
    

    print(" Energy Consumption Improvements.......")
    print("")
    print("")
    print(
        print_improvement(
            alg_set,
            mean_energy_consumption_improvement1,
            mean_energy_consumption_improvement2,
            4,
            4
        )
    )
   # print("")
    #print("")
    #print(" Missed Deadline Improvements.......")
    #print(
     #   print_improvement(
      #      alg_set,
       #     mean_missed_deadline_improvement1,
        #    mean_missed_deadline_improvement2,
         #   4,
          #  4
        #)
    #)
def compare_dqn_base(eval_itr=10000,iter=3, Train= False):
    DQN_STATE_DIM = 5
    train_configs = {
        "task_set": "configs/task_set_train.json",
        "cpu_little": "configs/cpu_little.json",
        "cpu_big": "configs/cpu_big.json",
        "cpu_local": "configs/cpu_local.json",
        "w_inter": "configs/wireless_interface.json",
        "dqn_state_dim": DQN_STATE_DIM,
    }

    eval_configs = {
        "task_set": "configs/task_set_eval.json",
        "cpu_little": "configs/cpu_little.json",
        "cpu_big": "configs/cpu_big.json",
        "cpu_local": "configs/cpu_local.json",
        "w_inter": "configs/wireless_interface.json",
        "dqn_state_dim": DQN_STATE_DIM,
    }

    test_configs = {
        "task_set": "configs/task_set_eval2.json",
        "cpu_little": "configs/cpu_little.json",
        "cpu_big": "configs/cpu_big.json",
        "cpu_local": "configs/cpu_local.json",
        "w_inter": "configs/wireless_interface.json",
        "dqn_state_dim": DQN_STATE_DIM,
    }

    if Train:
        dqn_loss=train_dqn_scenario(train_configs)

        plot_loss_function(dqn_loss, "DQN", "iterations", "loss","DQN_Loss_DQN Scenario")


    all_energy_eval1 = np.empty((0, 4))
    all_energy_eval2 = np.empty((0, 4))
    all_deadline_eval1 = np.empty((0, 4))
    all_deadline_eval2 = np.empty((0, 4))

    all_energy_improvement1 = np.empty((0, 3))
    all_energy_improvement2 = np.empty((0, 3))

    for j in tqdm(range(iter)):
        energy_consumption_eval1, missed_deadline_eval1, energy_improvement1 = evaluate_dqn_scenario(
        eval_configs, eval_itr
        )
        all_energy_eval1=np.vstack((all_energy_eval1,energy_consumption_eval1))
        all_deadline_eval1=np.vstack((all_deadline_eval1,missed_deadline_eval1))
        all_energy_improvement1=np.vstack((all_energy_improvement1,energy_improvement1))
        energy_consumption_eval2, missed_deadline_eval2,energy_improvement2= evaluate_dqn_scenario(
        test_configs, eval_itr
        )
        all_energy_eval2=np.vstack((all_energy_eval2,energy_consumption_eval2))
        all_deadline_eval2=np.vstack((all_deadline_eval2,missed_deadline_eval2))
        all_energy_improvement2=np.vstack((all_energy_improvement2,energy_improvement2))
    # Calculate mean and standard deviation column-wise
    mean_energy_consumption_eval1 = np.mean(all_energy_eval1, axis=0)
    std_energy_consumption_eval1 = np.std(10*np.log10(all_energy_eval1/1e-3), axis=0)
    mean_energy_consumption_improvement1 = np.mean(all_energy_improvement1, axis=0)

    mean_missed_deadline_eval1 = np.mean(all_deadline_eval1, axis=0)
    std_missed_deadline_eval1 = np.std(all_deadline_eval1, axis=0)


    # Repeat the same process for eval2
    mean_energy_consumption_eval2 = np.mean(all_energy_eval2, axis=0)
    std_energy_consumption_eval2 = np.std(10*np.log10(all_energy_eval2/1e-3), axis=0)
    mean_energy_consumption_improvement2 = np.mean(all_energy_improvement2, axis=0)

    mean_missed_deadline_eval2 = np.mean(all_deadline_eval2, axis=0)
    std_missed_deadline_eval2 = np.std(all_deadline_eval2, axis=0)


    #print("*"*20)
    #print(mean_energy_consumption_eval1)
    #print(mean_energy_consumption_eval2)
    #print(mean_missed_deadline_eval1)
    #print(mean_missed_deadline_eval2)
    #print("*"*20)
    #print(std_energy_consumption_eval1)
    #print(std_energy_consumption_eval2)
    #print(std_missed_deadline_eval1)
    #print(std_missed_deadline_eval2)
    alg_set = ["Random", "Local", "Remote","DQN"]
    plot_res(
        alg_set,
        mean_energy_consumption_eval1,
        mean_energy_consumption_eval2,
        std_energy_consumption_eval1,
        std_energy_consumption_eval2,
        "Algorithm",
        "Energy Consumption (mJ)",
        "Energy Consumption of Different Algorithms in DQN Scenario",
        "Energy_Consumption_DQN",
        ylog=True
    )
    plot_res(
        alg_set,
        mean_missed_deadline_eval1,
        mean_missed_deadline_eval2,
        std_missed_deadline_eval1,
        std_missed_deadline_eval2,
        "Algorithm",
        "Dropped Tasks (%) ",
        "Dropped Tasks of Different Algorithms in DQN Scenario",
        "Missed_Deadline_DQN",
    )
    print(" Energy Consumption Improvements.......")
    print("")
    print("")
    print(
        print_improvement(
            alg_set,
            mean_energy_consumption_improvement1,
            mean_energy_consumption_improvement2,
            3,
            3
        )
    )
    print("")
    print("")
    #print(" Missed Deadline Improvements.......")
    #print(
     #   print_improvement(
      #      alg_set,
       #     mean_missed_deadline_improvement1,
        #    mean_missed_deadline_improvement2,
         #   3,
          #  3
        #)
    #)
def compare_cpu_load(eval_itr=10000, Train=False, mean_iter=3):
    DQN_STATE_DIM = 4
    train_configs = {
        "task_set": "configs/task_set_train.json",
        "cpu_local": "configs/cpu_local.json",
        "w_inter": "configs/wireless_interface.json",
        "dqn_state_dim": DQN_STATE_DIM,
    }

    #eval_configs = {
     #   "task_set": "configs/task_set_eval.json",
      #  "cpu_local": "configs/cpu_local.json",
       # "w_inter": "configs/wireless_interface.json",
        #"dqn_state_dim": DQN_STATE_DIM,
    #}
    #test_configs = {
     #   "task_set": "configs/task_set_eval2.json",
      #  "cpu_local": "configs/cpu_local.json",
       # "w_inter": "configs/wireless_interface.json",
        #"dqn_state_dim": DQN_STATE_DIM,
    #}

    if Train:
        dqn_loss=train_rrlo_scenario(train_configs)
        #print(dqn_loss.shape)
        plot_loss_function(dqn_loss, "DQN", "iterations", "loss","DQN_Loss_CPU_Load_Scenario")

    cpu_load_val = np.arange(0.05, 1.00, 0.05)


    all_energy_eval1 = np.empty((0, 5, len(cpu_load_val)))
    all_deadline_eval1 = np.empty((0,  5, len(cpu_load_val)))
    all_improvement_eval1 = np.empty((0,  4, len(cpu_load_val)))

    for i in tqdm(range(mean_iter)):
        energy_consumption_eval1, missed_deadline_eval1, energy_improvement1 = evaluate_cpu_load_scenario(
            train_configs,cpu_load_val, eval_itr
        )
        all_energy_eval1 = np.append(all_energy_eval1, [energy_consumption_eval1], axis=0)
        all_deadline_eval1 =  np.append(all_deadline_eval1, [missed_deadline_eval1], axis=0)
        all_improvement_eval1 =  np.append(all_improvement_eval1, [energy_improvement1], axis=0)
    mean_energy_eval1 = np.mean(all_energy_eval1, axis=0)
    mean_deadline_eval1 = np.mean(all_deadline_eval1, axis=0)
    mean_improvement_eval1 = np.mean(all_improvement_eval1, axis=0)




    alg_set = ["Random", "Local", "Remote", "RRLO", "DQN"]

    line_plot_res(
        alg_set,
        mean_energy_eval1,
        cpu_load_val,
        "CPU Load (Utilization)",
        "Energy Consumption (mJ)",
        "Energy Consumption of Different Algorithms With Respect to Various CPU Load ",
        "Energy_Consumption_CPU",
        ylog=True
    )
    line_plot_res(
        alg_set,
        mean_deadline_eval1,
        cpu_load_val,
        "CPU Load (Utilization)",
        "Dropped Tasks (%) ",
        "Dropped Tasks of Different Algorithms With Respect to Various CPU Load",
        "Missed_Deadline_CPU"
    )
    alg_set2 = ["Random", "Local", "Remote", "RRLO"]
    
    line_plot_res(
        alg_set2,
        mean_improvement_eval1*-1,
        cpu_load_val,
        "CPU Load (Utilization)",
        "Energy Improvement (%)",
        "Energy Improvement of DQN compared to Different Algorithms With Respect to Various CPU Load",
        "Energy_Improvement_CPU"
    )
def compare_task_size(eval_itr=10000, Train=False, mean_iter=3):
    DQN_STATE_DIM = 4
    train_configs = {
        "task_set": "configs/task_set_train.json",
        "cpu_local": "configs/cpu_local.json",
        "w_inter": "configs/wireless_interface.json",
        "dqn_state_dim": DQN_STATE_DIM,
    }

    #eval_configs = {
     #   "task_set": "configs/task_set_eval.json",
      #  "cpu_local": "configs/cpu_local.json",
       # "w_inter": "configs/wireless_interface.json",
        #"dqn_state_dim": DQN_STATE_DIM,
    #}
    #test_configs = {
     #   "task_set": "configs/task_set_eval2.json",
      #  "cpu_local": "configs/cpu_local.json",
       # "w_inter": "configs/wireless_interface.json",
        #"dqn_state_dim": DQN_STATE_DIM,
    #}

    if Train:
        dqn_loss=train_rrlo_scenario(train_configs)
        #print(dqn_loss.shape)
        plot_loss_function(dqn_loss, "DQN", "iterations", "loss","DQN_Loss_task_size_Scenario")

    task_size_val = np.round(np.linspace(110, 490, 11))

    #print(task_size_val)


    all_energy_eval1 = np.empty((0, 5, len(task_size_val)-1))
    all_deadline_eval1 = np.empty((0,  5, len(task_size_val)-1))
    all_improvement_eval1 = np.empty((0,  4, len(task_size_val)-1))

    for i in tqdm(range(mean_iter)):
        energy_consumption_eval1, missed_deadline_eval1, energy_improvement1 = evaluate_task_size_scenario(
            train_configs,task_size_val, eval_itr
        )
        all_energy_eval1 = np.append(all_energy_eval1, [energy_consumption_eval1], axis=0)
        all_deadline_eval1 =  np.append(all_deadline_eval1, [missed_deadline_eval1], axis=0)
        all_improvement_eval1 =  np.append(all_improvement_eval1, [energy_improvement1], axis=0)
    mean_energy_eval1 = np.mean(all_energy_eval1, axis=0)
    mean_deadline_eval1 = np.mean(all_deadline_eval1, axis=0)
    mean_improvement_eval1 = np.mean(all_improvement_eval1, axis=0)




    alg_set = ["Random", "Local", "Remote", "RRLO", "DQN"]

    line_plot_res(
        alg_set,
        mean_energy_eval1,
        task_size_val[:-1],
        "Task Size (bits)",
        "Energy Consumption (mJ)",
        "Energy Consumption of Different Algorithms With Respect to Various Task Size ",
        "Energy_Consumption_Task",
        ylog=True
    )
    line_plot_res(
        alg_set,
        mean_deadline_eval1,
        task_size_val[:-1],
        "Task Size (bits)",
        "Dropped Tasks (%) ",
        "Dropped Tasks of Different Algorithms With Respect to Various Task Size",
        "Missed_Deadline_Task"
    )
    alg_set2 = ["Random", "Local", "Remote", "RRLO"]
    
    line_plot_res(
        alg_set2,
        mean_improvement_eval1*-1,
        task_size_val[:-1],
        "Task Size (bits)",
        "Energy Improvement (%)",
        "Energy Improvement of DQN compared to Different Algorithms With Respect to Various Task Size",
        "Energy_Improvement_Task"
    )

 
if __name__ == "__main__":
    #compare_dqn_rrlo(Train=True)
    #compare_dqn_base(Train=True)
    compare_cpu_load(Train=False)
    compare_task_size(Train=False)
