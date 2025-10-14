import random
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns

from sklearn.cluster import KMeans
import math
import matplotlib.pyplot as plt
import ast
def get_task_specific_features(dataset, ModelName):
    DataPath = f"../Dataset/{dataset.upper()}/"
    ResultPath = '../Results/Pairwise/'
    if dataset == 'Chemical' and ModelName == 'SVM':
        task_info = pd.read_csv(f'{DataPath}Task_Information_{dataset}_SVM.csv')
    else:
        task_info = pd.read_csv(f'{DataPath}Task_Information_{dataset}.csv')

    pair_results = pd.read_csv(f'{ResultPath}{ModelName}/{dataset}_Results_from_Pairwise_Training_ALL_{ModelName}.csv')
    single_results = pd.read_csv(f'../Results/STL/STL_{dataset}_{ModelName}.csv')

    if dataset == 'Chemical':
        ChemicalData = pd.read_csv(f'{DataPath}ChemicalData_All.csv', low_memory=False)
        TASKS = list(ChemicalData['180'].unique())
    if dataset == 'School':
        TASKS = [i for i in range(1, 140)]
    if dataset == 'Landmine':
        TASKS = [i for i in range(0, 29)]
    if dataset == 'Parkinsons':
        TASKS = [i for i in range(1, 43)]

    task_len = {}
    variance_dict = {}
    std_dev_dict = {}
    euclidean_dist_dict = {}
    manhattan_dist_dict = {}
    hamming_dist_dict = {}
    euclidean_dist_dict_scaled = {}
    manhattan_dist_dict_scaled = {}
    hamming_dist_dict_scaled = {}
    Single_res_dict = {}
    loss_dict = {}

    for Selected_Task in TASKS:

        if dataset == 'Chemical':
            task_data = task_info[task_info.Molecule == Selected_Task].reset_index()
        else:
            task_data = task_info[task_info.Task_Name == Selected_Task].reset_index()

        task_len.update({Selected_Task: task_data.Dataset_Size[0]})
        variance_dict.update({Selected_Task: task_data.Variance[0]})
        std_dev_dict.update({Selected_Task: task_data.Std_Dev[0]})

        if dataset == 'Chemical':
            hamming_dist_dict.update({Selected_Task: task_data.Average_Hamming_Distance_within_Task[0]})
            euclidean_dist_dict.update({Selected_Task: task_data.Average_Euclidian_Distance_within_Task[0]})
            manhattan_dist_dict.update({Selected_Task: task_data.Average_Manhattan_Distance_within_Task[0]})
        else:
            euclidean_dist_dict.update({Selected_Task: task_data.Average_Euclidian_Distance_within_Task[0]})
            manhattan_dist_dict.update({Selected_Task: task_data.Average_Manhattan_Distance_within_Task[0]})

            euclidean_dist_dict_scaled.update(
                {Selected_Task: task_data.Average_Euclidian_Distance_within_Task_after_Scaling[0]})
            manhattan_dist_dict_scaled.update(
                {Selected_Task: task_data.Average_Manhattan_Distance_within_Task_after_Scaling[0]})
        # hamming_dist_dict_scaled.update(
        #     {Selected_Task: task_data.Average_Hamming_Distance_within_Task_after_Scaling[0]})
        single_res = single_results[single_results.Task == Selected_Task].reset_index()
        Single_res_dict.update({Selected_Task: single_res.LOSS[0]})

    Task_1 = []
    Task_2 = []
    Dataset_Task1 = []
    Dataset_Task2 = []
    Variance_Task1 = []
    Variance_Task2 = []
    StdDev_Task1 = []
    StdDev_Task2 = []
    Loss_Task1 = []
    Loss_Task2 = []
    Distance_Task1 = []
    Distance_Task2 = []

    task_combo = []
    count = 0
    TASKS = TASKS

    for i in range(len(TASKS)):
        for j in range(len(TASKS)):
            if TASKS[i] != TASKS[j]:
                task1 = TASKS[i]
                task2 = TASKS[j]
                Task_1.append(task1)
                Task_2.append(task2)
                if TASKS[i] == 83:
                    count += 1
                task_combo.append([TASKS[i], TASKS[j]])

                Dataset_Task1.append(task_len[task1])
                Dataset_Task2.append(task_len[task2])
                Variance_Task1.append(variance_dict[task1])
                Variance_Task2.append(variance_dict[task2])
                StdDev_Task1.append(std_dev_dict[task1])
                StdDev_Task2.append(std_dev_dict[task2])
                Loss_Task1.append(Single_res_dict[task1])
                Loss_Task2.append(Single_res_dict[task2])

                if dataset != 'Chemical':
                    Distance_Task1.append(euclidean_dist_dict[task1])
                    Distance_Task2.append(euclidean_dist_dict[task2])
                else:
                    Distance_Task1.append(hamming_dist_dict[task1])
                    Distance_Task2.append(hamming_dist_dict[task2])

                # for c in range(10, 100, 10):
                #     Loss_dict_Task1[f'lc_task1_{c}'].append(loss_dict[(task1, c)])
                #     Loss_dict_Task2[f'lc_task2_{c}'].append(loss_dict[(task2, c)])
                #
                # ys_task1 = []
                # ys_task2 = []
                # for c in range(10, 100, 10):
                #     Loss_dict_Task1[f'lc_task1_{c}'].append(loss_dict[(task1, c)])
                #     ys_task1.append(loss_dict[(task1, c)])
                #     Loss_dict_Task2[f'lc_task2_{c}'].append(loss_dict[(task2, c)])
                #     ys_task2.append(loss_dict[(task2, c)])
                #
                # fit = fit_log_curve(ys_task1)
                # Fitted_param_a_Task1.append(fit[0, 0])
                # Fitted_param_b_Task1.append(fit[1, 0])
                #
                # fit = fit_log_curve(ys_task2)
                # Fitted_param_a_Task2.append(fit[0, 0])
                # Fitted_param_b_Task2.append(fit[1, 0])

    # Loss_dict_Task1_X = {}
    # Loss_dict_Task2_X = {}
    #
    # for i in range(10, 100, 10):
    #     Loss_dict_Task1_X.update({f'lc_task1_{i}': []})
    #     Loss_dict_Task2_X.update({f'lc_task2_{i}': []})
    for i in range(len(TASKS)):
        for j in range(len(TASKS)):
            if TASKS[j] != TASKS[i]:
                task1 = TASKS[i]
                task2 = TASKS[j]
                Task_1.append(task1)
                Task_2.append(task2)
                if TASKS[i] == 83:
                    count += 1
                task_combo.append([TASKS[i], TASKS[j]])

                Dataset_Task1.append(task_len[task1])
                Dataset_Task2.append(task_len[task2])
                Variance_Task1.append(variance_dict[task1])
                Variance_Task2.append(variance_dict[task2])
                StdDev_Task1.append(std_dev_dict[task1])
                StdDev_Task2.append(std_dev_dict[task2])
                Loss_Task1.append(Single_res_dict[task1])
                Loss_Task2.append(Single_res_dict[task2])
                # Distance_Task1.append(euclidean_dist_dict[task1])
                # Distance_Task2.append(euclidean_dist_dict[task2])

                if dataset != 'Chemical':
                    Distance_Task1.append(euclidean_dist_dict[task1])
                    Distance_Task2.append(euclidean_dist_dict[task2])
                else:
                    Distance_Task1.append(hamming_dist_dict[task1])
                    Distance_Task2.append(hamming_dist_dict[task2])

                # for c in range(10, 100, 10):
                #     Loss_dict_Task1_X[f'lc_task1_{c}'].append(loss_dict[(task1, c)])
                #     Loss_dict_Task2_X[f'lc_task2_{c}'].append(loss_dict[(task2, c)])

                # ys_task1 = []
                # ys_task2 = []
                # for c in range(10, 100, 10):
                #     Loss_dict_Task1_X[f'lc_task1_{c}'].append(loss_dict[(task1, c)])
                #     ys_task1.append(loss_dict[(task1, c)])
                #     Loss_dict_Task2_X[f'lc_task2_{c}'].append(loss_dict[(task2, c)])
                #     ys_task2.append(loss_dict[(task2, c)])
                #
                # # ys_task1 = []
                # # ys_task2 = []
                # # for c in range(10, 100, 10):
                # #     Loss_dict_Task1_X[f'lc_task1_{c}'].append(loss_dict[(task1, c)])
                # #     ys_task1.append(loss_dict[(task1, c)])
                # #     Loss_dict_Task2_X[f'lc_task2_{c}'].append(loss_dict[(task2, c)])
                # #     ys_task2.append(loss_dict[(task2, c)])
                #
                # fit = fit_log_curve(ys_task1)
                # Fitted_param_a_Task1.append(fit[0, 0])
                # Fitted_param_b_Task1.append(fit[1, 0])
                #
                # fit = fit_log_curve(ys_task2)
                # Fitted_param_a_Task2.append(fit[0, 0])
                # Fitted_param_b_Task2.append(fit[1, 0])

    paired_improvement = []
    Weight = []
    InterTaskAffinity = []
    for pair in task_combo:

        stl_loss = 0
        stl_loss += Single_res_dict[pair[0]]
        stl_loss += Single_res_dict[pair[1]]
        pair_specific = pair_results[
            (pair_results.Task_1 == pair[0]) & (pair_results.Task_2 == pair[1])].reset_index()
        if len(pair_specific) == 0:
            pair_specific = pair_results[
                (pair_results.Task_1 == pair[1]) & (pair_results.Task_2 == pair[0])].reset_index()
        change = (stl_loss - pair_specific.Total_Loss[0]) / stl_loss
        # if change<-1:
        #     print(pair, change, stl_loss, pair_specific.Total_Loss[0])
        paired_improvement.append((stl_loss - pair_specific.Total_Loss[0]) / stl_loss)




    # '''*************'''
    # print('\n\n\n\n*********\n\n\n')
    # print(len(Dataset_Task1))
    # print(len(Dataset_Task2))
    # print(len(Variance_Task1))
    # print(len(Variance_Task2))
    # print(len(StdDev_Task1))
    # print(len(StdDev_Task2))
    # print(len(Loss_Task1))
    # print(len(Loss_Task2))
    # print(len(Distance_Task1))
    # print(len(Distance_Task2))
    # print(len(Loss_dict_Task2))
    # print(len(Loss_dict_Task1))
    # print(len(Loss_dict_Task1_X))
    # print(len(Loss_dict_Task2_X))
    # print(len(Fitted_param_a_Task1), len(Fitted_param_a_Task2))

    df = pd.DataFrame({
        'Task1': Task_1,
        'Task2': Task_2,
        'Dataset_Task1': Dataset_Task1,
        'Dataset_Task2': Dataset_Task2,
        'Variance_Task1': Variance_Task1,
        'Variance_Task2': Variance_Task2,
        'StdDev_Task1': StdDev_Task1,
        'StdDev_Task2': StdDev_Task2,
        'Distance_Task1': Distance_Task1,
        'Distance_Task2': Distance_Task2,
        'Loss_Task1': Loss_Task1,
        'Loss_Task2': Loss_Task2,
        # 'Fitted_param_a_Task1': Fitted_param_a_Task1,
        # 'Fitted_param_b_Task1': Fitted_param_b_Task1,
        # 'Fitted_param_a_Task2': Fitted_param_a_Task2,
        # 'Fitted_param_b_Task2': Fitted_param_b_Task2,
    })
    df['Change'] = paired_improvement

    print(len(df))

    df.to_csv(f'{ResultPath}{ModelName}/Pairwise_Task_Specific_Features_{dataset}_{ModelName}.csv', index=False)

for dataset in ['Parkinsons']:
    DataPath = f'../Dataset/{dataset.upper()}/'
    ResultPath = '../Results/Pairwise/'
    ModelName = 'xgBoost'

    feature_type = 'Task_Specific'

    get_task_specific_features(dataset,ModelName)
    # continue

    task_specific = pd.read_csv(f'{ResultPath}{ModelName}/Pairwise_{feature_type}_Features_{dataset}_{ModelName}.csv', low_memory=False)
    Change = list(task_specific['Change'].unique())


    if dataset == 'Chemical':
        ChemicalData = pd.read_csv(f'{DataPath}ChemicalData_All.csv', low_memory=False)
        TASKS = list(ChemicalData['180'].unique())
        TASKS = sorted(TASKS)
    if dataset == 'School':
        TASKS = [i for i in range(1,140)]
    if dataset == 'Landmine':
        TASKS = [i for i in range(0, 29)]
    if dataset == 'Parkinsons':
        TASKS = [i for i in range(1, 43)]



    tasks_dict = {}
    key = 0
    for task in TASKS:
        tasks_dict[key] = task
        key += 1

    print(f'tasks_dict = {tasks_dict}')

    affinity_matrix = np.zeros((len(TASKS), len(TASKS)))

    # tasks_dict = {}
    # key = 0
    # for task in TASKS:
    #     tasks_dict[task] = key
    #     key += 1
    # print(f'tasks_dict = {tasks_dict}')

    tasks_dict_X = {}
    key = 0
    for task in TASKS:
        tasks_dict_X[key] = task
        key += 1
    print(f'tasks_dict_X = {tasks_dict_X}')

    for type in ['Exponential', 'NonNegative']:

        if type == 'Exponential':
            Change = sorted(Change)
            affinity = [math.exp(-i) for i in Change]


            '''Similarity Matrix with Exponetial Affinity'''
            for task_1 in range(0, len(TASKS)):
                for task_2 in range(0, len(TASKS)):
                    if task_1 != task_2:
                        task_df = task_specific[task_specific.Task1 == tasks_dict[task_1]]
                        task_df = task_df[task_df.Task2 == tasks_dict[task_2]]
                        change = list(task_df.Change.unique())[0]

                        affinity_matrix[task_1 - 1][task_2 - 1] = math.exp(-change)
                        affinity_matrix[task_2 - 1][task_1 - 1] = math.exp(-change)

        elif type == 'NonNegative':
            '''Similarity Matrix with Scaled Integers for Affinity'''
            Change = sorted(Change)
            val = len(Change) - 1
            affinity_dict = {}
            for change in Change:
                affinity_dict[change] = val
                val -= 1

            for task_1 in range(0, len(TASKS)):
                for task_2 in range(0, len(TASKS)):
                    if task_1 != task_2:
                        task_df = task_specific[task_specific.Task1 == tasks_dict[task_1]]
                        task_df = task_df[task_df.Task2 == tasks_dict[task_2]]
                        change = list(task_df.Change.unique())[0]
                        affinity_matrix[task_1 - 1][task_2 - 1] = affinity_dict[change]
                        affinity_matrix[task_2 - 1][task_1 - 1] = affinity_dict[change]
        # continue
        '''Clustering'''


        print(np.shape(affinity_matrix))



        # Sum_of_squared_distances = []
        # MAX_Tasks = min(len(TASKS), 100)
        # K = range(2,MAX_Tasks)
        # for k in K:
        #     km = KMeans(n_clusters=k)
        #     km = km.fit(affinity_matrix)
        #     Sum_of_squared_distances.append(km.inertia_)
        # #
        # plt.plot(K, Sum_of_squared_distances, 'bx-')
        # plt.xlabel('Number of Clusters')
        # plt.ylabel('Sum_of_squared_distances')
        # plt.title(f'Elbow Method For Optimal k = {dataset}')
        # plt.show()
        #
        # continue
        TASK_Group = []
        Number_of_Clusters = []
        if dataset=='School':
            Min_range = 5
            Max_range = 30
        elif dataset == 'Chemical':
            Min_range = 3
            Max_range = 18
        elif dataset== 'Landmine':
            Min_range = 3
            Max_range = 14
        else:
            Min_range = 4
            Max_range = 20



        for k in range(Min_range, Max_range):
            Number_of_Clusters.append(k)
            print('-----------------------------------------------------')
            kmeans = KMeans(n_clusters=k, random_state=0,n_init=10, max_iter=300,tol=1e-04)
            kmeans.fit(affinity_matrix)
            label_df = pd.DataFrame({'Tasks': TASKS, 'Cluster': kmeans.fit_predict(affinity_matrix)})
            label_dict = {}
            u_labels = np.unique(kmeans.labels_)
            for i in u_labels:
                filtered_label = label_df[label_df.Cluster == int(i)]
                idx = list(filtered_label.index)
                task_list = [tasks_dict_X[key] for key in idx]
                label_dict[i] = task_list
                # print(f'Cluster {i} : {idx}')
            print(f'label_dict = {label_dict}')

            TASK_Group.append(label_dict.copy())
            print('-----------------------------------------------------')

        CLUSTERS = pd.DataFrame({"Number_of_Clusters":Number_of_Clusters,
                                 "TASK_Group":TASK_Group})
        print(CLUSTERS)
        CLUSTERS.to_csv(f'../Results/Clusters/{ModelName}/{dataset}_Clusters_{type}_{ModelName}_KMeans.csv',index=False)


