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



def silhouette_scores(affinity_matrix):
    '''Silhouette Scores to find Optimal number of Groups'''
    for linkage in ['average', 'single', 'complete']:
        fig, axs = plt.subplots(nrows=1, ncols=4)
        for affinity in ['euclidean', 'manhattan', 'cosine', 'precomputed']:
            silhouette_scores = []
            number_of_Clusters = [i for i in range(2, 15)]
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

for dataset in ['Chemical', 'School', 'Landmine','Parkinsons']:
    DataPath = f'../Dataset/{dataset.upper()}/'
    ResultPath = '../Results/Pairwise/'

    feature_type = 'Task_Specific'
    task_specific = pd.read_csv(f'{ResultPath}Pairwise_{feature_type}_Features_{dataset}.csv', low_memory=False)
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


        '''Clustering'''
        CLUSTERS = get_clusters(affinity_matrix, tasks_dict, min_cluster_size, max_cluster_size)
        CLUSTERS.to_csv(f'../baseline/clustering/data/{dataset}_Clusters_{type}.csv', index=False)

