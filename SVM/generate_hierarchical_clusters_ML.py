import math
import random
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import ast

def get_task_specific_features(dataset, ModelName):
    DataPath = f"../Dataset/{dataset.upper()}/"
    ResultPath = '../Results/Pairwise/'
    if ModelName == 'SVM':
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
                # if TASKS[i] == 83:
                #     count += 1
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


    for i in range(len(TASKS)):
        for j in range(len(TASKS)):
            if TASKS[j] != TASKS[i]:
                task1 = TASKS[i]
                task2 = TASKS[j]
                Task_1.append(task1)
                Task_2.append(task2)

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

    paired_improvement = []
    for pair in task_combo:
        stl_loss = 0
        stl_loss += Single_res_dict[pair[0]]
        stl_loss += Single_res_dict[pair[1]]
        pair_specific = pair_results[
            (pair_results.Task_1 == pair[0]) & (pair_results.Task_2 == pair[1])].reset_index()
        # print(pair_specific)
        if len(pair_specific) == 0:
            pair_specific = pair_results[
                (pair_results.Task_1 == pair[1]) & (pair_results.Task_2 == pair[0])].reset_index()

        change = (stl_loss - pair_specific.Total_Loss[0]) / stl_loss
        # if change == -math.inf:
        #     print(f'change = {change}, pair = {pair}, STL loss = {stl_loss}, pair = {pair_specific.Total_Loss[0]}')
        #     exit(0)
        paired_improvement.append(change)




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

    df.to_csv(f'{ResultPath}/Pairwise_Task_Specific_Features_{dataset}_{ModelName}.csv', index=False)

def get_clusters(affinity_matrix, tasks_dict,min_cluster_size,max_cluster_size):
    Linkage = []
    Affinity = []
    Number_of_Clusters = []
    TASK_Group = []

    for linkage in ['average', 'complete']:
        plt.title(f'linkage = {linkage}')
        AFFINITY_VAL = ['euclidean', 'manhattan', 'cosine', 'precomputed']
        for affinity in AFFINITY_VAL:
            print(f'*******linkage = {linkage}\taffinity = {affinity}*******')
            for number_cluster in range(min_cluster_size,max_cluster_size):
                Linkage.append(linkage)
                Affinity.append(affinity)
                Number_of_Clusters.append(number_cluster)

                model = AgglomerativeClustering(linkage=linkage,
                                                affinity=affinity,
                                                n_clusters=number_cluster)
                clustering = model.fit(affinity_matrix)

                tot = 0

                for val in set(clustering.labels_):
                    count_val = list(clustering.labels_).count(val)
                    tot += count_val

                df = pd.DataFrame(affinity_matrix)
                df['label'] = clustering.labels_
                u_labels = np.unique(clustering.labels_)
                task_group = {}
                for t in u_labels:
                    filtered_label0 = df[df.label == int(t)]
                    idx = list(filtered_label0.index)
                    task_cluster = [tasks_dict[key] for key in idx]
                    task_group[f'{t}'] = task_cluster

                TASK_Group.append(task_group.copy())

    print(len(Linkage), len(Affinity), len(Number_of_Clusters), len(TASK_Group))
    CLUSTERS = pd.DataFrame({"Linkage": Linkage,
                             "Affinity": Affinity,
                             "Number_of_Clusters": Number_of_Clusters,
                             "TASK_Group": TASK_Group})
    return CLUSTERS



def silhouette_scores_plot(affinity_matrix):
    '''Silhouette Scores to find Optimal number of Groups'''
    for linkage in ['average', 'single', 'complete']:
        fig, axs = plt.subplots(nrows=1, ncols=4)
        for affinity in ['euclidean', 'manhattan', 'cosine', 'precomputed']:
            silhouette_scores = []
            number_of_Clusters = [i for i in range(2, 29)]
            for k in number_of_Clusters:
                model = AgglomerativeClustering(linkage=linkage,
                                                affinity=affinity,
                                                n_clusters=k)

                silhouette_scores.append(
                    silhouette_score(affinity_matrix, model.fit_predict(affinity_matrix)))

            plt_idx = ['euclidean', 'manhattan', 'cosine', 'precomputed'].index(affinity)
            # Plotting a bar graph to compare the results
            axs[plt_idx].bar(number_of_Clusters, silhouette_scores)
            axs[plt_idx].set_xlabel('Number of clusters')
            axs[plt_idx].set_ylabel('Silhouette Scores')
            axs[plt_idx].set_title(f'Linkage = {linkage}\t affinity = {affinity}')

        plt.grid()
        plt.show()
        print('**********************************************************************************************')
        for i in range(len(number_of_Clusters)):
            if silhouette_scores[i]>0.2:
                print(f'Number of Clusters = {number_of_Clusters[i]}\tSilhouette Score = {silhouette_scores[i]}')

for dataset in ['School','Chemical','Landmine','Parkinsons']:
    DataPath = f'../Dataset/{dataset.upper()}/'
    ResultPath = '../Results/Pairwise/'
    ModelName = 'xgBoost'

    feature_type = 'Task_Specific'

    # get_task_specific_features(dataset,ModelName)
    # continue

    task_specific = pd.read_csv(f'{ResultPath}Pairwise_{feature_type}_Features_{dataset}_{ModelName}.csv', low_memory=False)
    task_specific = task_specific.dropna()
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

    pair_results = pd.read_csv(
        f'{ResultPath}{ModelName}/{dataset}_Results_from_Pairwise_Training_ALL_{ModelName}.csv')
    single_results = pd.read_csv(f'../Results/STL/STL_{dataset}_{ModelName}.csv')



    tasks_dict = {}
    key = 0
    for task in TASKS:
        tasks_dict[key] = task
        key += 1

    print(f'tasks_dict = {tasks_dict}')

    affinity_matrix = np.zeros((len(TASKS), len(TASKS)))

    for type in ['Exponential','NonNegative']:

        if dataset == 'School':
            min_cluster_size = 4
            max_cluster_size = 21

        if dataset == 'Chemical':
            min_cluster_size = 2
            max_cluster_size = 16

        if dataset == 'Landmine':
            min_cluster_size = 2
            max_cluster_size = 16

        if dataset == 'Parkinsons':
            min_cluster_size = 2
            max_cluster_size = 21

        if type == 'Exponential':
            Change = sorted(Change)
            affinity = [math.exp(-i) for i in Change]

            # plt.plot(Change,label = 'Relative Improvement')
            # plt.plot(affinity,label = 'Affinity value for clustering')
            # plt.legend()
            # plt.grid()
            # plt.xlabel('Task Pairs')
            # plt.ylabel('Values')
            # plt.title(f'math.exp(-i)')
            # plt.show()


            '''Similarity Matrix with Exponetial Affinity'''
            for task_1 in range(0,len(TASKS)):
                for task_2 in range(0,len(TASKS)):
                    if task_1!=task_2:
                        task_df = task_specific[task_specific.Task1 == tasks_dict[task_1]]
                        task_df = task_df[task_df.Task2 == tasks_dict[task_2]]
                        change = list(task_df.Change.unique())[0]
                        if change == math.inf:
                            print(f'change = {change} for {task_1} and {task_2}')
                        if change == -math.inf:
                            stl_1 = single_results[single_results.Task == task_1].reset_index()
                            stl_1 = stl_1.LOSS[0]
                            stl_2 = single_results[single_results.Task == task_2].reset_index()
                            stl_2 = stl_2.LOSS[0]
                            pair = [task_1, task_2]
                            pair_specific = pair_results[
                                (pair_results.Task_1 == pair[0]) & (pair_results.Task_2 == pair[1])].reset_index()
                            if len(pair_specific) == 0:
                                pair_specific = pair_results[
                                    (pair_results.Task_1 == pair[1]) & (pair_results.Task_2 == pair[0])].reset_index()
                            pair_res = pair_specific.Total_Loss[0]
                            print(f'STL for {task_1} and {task_2} == {stl_1} and {stl_2}')
                            print(f'pairwise for {task_1} and {task_2} == {pair_res} so change = {(stl_1+stl_2 - pair_res)/(stl_1+stl_2)}')
                            print(f'neg change = {change} for {task_1} and {task_2}')

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

            # plt.plot(Change, label='Relative Improvement')
            # plt.plot(list(affinity_dict.values()), label='Affinity value for clustering')
            # plt.legend()
            # plt.grid()
            # plt.xlabel('Task Pairs')
            # plt.ylabel('Values')
            # plt.title(f'Affinity = len(Change) - 1')
            # plt.show()

            for task_1 in range(0, len(TASKS)):
                for task_2 in range(0, len(TASKS)):
                    if task_1 != task_2:
                        task_df = task_specific[task_specific.Task1 == tasks_dict[task_1]]
                        task_df = task_df[task_df.Task2 == tasks_dict[task_2]]
                        change = list(task_df.Change.unique())[0]
                        affinity_matrix[task_1 - 1][task_2 - 1] = affinity_dict[change]
                        affinity_matrix[task_2 - 1][task_1 - 1] = affinity_dict[change]
        # silhouette_scores_plot(affinity_matrix)
        # continue
        # continue
        '''Clustering'''
        CLUSTERS = get_clusters(affinity_matrix, tasks_dict, min_cluster_size, max_cluster_size)
        CLUSTERS.to_csv(f'../Results/Clusters/{ModelName}/{dataset}_Clusters_{type}_{ModelName}.csv', index=False)

