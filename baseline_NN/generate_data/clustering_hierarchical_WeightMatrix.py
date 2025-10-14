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


def get_clusters_weightMatrix(affinity_matrix, tasks_dict_reverse, min_cluster_size, max_cluster_size):
    Linkage = []
    Affinity = []
    Number_of_Clusters = []
    TASK_Group = []

    for linkage in ['average', 'single', 'complete']:
        AFFINITY_VAL = ['euclidean', 'manhattan', 'cosine', 'precomputed']
        for affinity in AFFINITY_VAL:
            print(f'*******linkage = {linkage}\taffinity = {affinity}*******')
            for number_cluster in range(min_cluster_size, max_cluster_size):
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
                    task_cluster = [tasks_dict_reverse[key] for key in idx]

                    task_group[f'{t}'] = task_cluster

                TASK_Group.append(task_group.copy())

    print(len(Linkage), len(Affinity), len(Number_of_Clusters), len(TASK_Group))
    CLUSTERS = pd.DataFrame({"Linkage": Linkage,
                             "Affinity": Affinity,
                             "Number_of_Clusters": Number_of_Clusters,
                             "TASK_Group": TASK_Group})
    return CLUSTERS


for dataset in ['Chemical', 'School', 'Landmine','Parkinsons']:
    DataPath = f'../../Dataset/{dataset.upper()}/'
    ResultPath = '../../Results/'

    # Load the data
    Weight_Matrix = pd.read_csv(f'{ResultPath}Weight_Affinity_{dataset}.csv', low_memory=False)


    if dataset == 'Chemical':
        ChemicalData = pd.read_csv(f'{DataPath}ChemicalData_All.csv', low_memory=False)
        TASKS = list(ChemicalData['180'].unique())

        min_cluster_size = 2
        max_cluster_size = 21

    if dataset == 'School':
        TASKS = [i for i in range(1, 140)]
        min_cluster_size = 2
        max_cluster_size = 30
    if dataset == 'Parkinsons':
        TASKS = [i for i in range(1, 43)]

        min_cluster_size = 2
        max_cluster_size = 21
    if dataset == 'Landmine':
        TASKS = [i for i in range(0, 29)]

        min_cluster_size = 2
        max_cluster_size = 16


    tasks_dict = {}
    key = 0
    for task in TASKS:
        tasks_dict[task] = key
        key += 1
    print(tasks_dict)

    tasks_dict_X = {}
    key = 0
    for task in TASKS:
        tasks_dict_X[key] = task
        key += 1
    print(tasks_dict_X)


    Pairs = list(Weight_Matrix['Pairs'])
    Change = list(Weight_Matrix['Change'])
    affinity = list(Weight_Matrix['Weight'])


    Task_1 = []
    Task_2 = []

    affinity_matrix = np.zeros( (len(TASKS), len(TASKS)) )

    print(np.shape(affinity_matrix))
    '''Similarity Matrix with Dot Products of Weight Matrix'''

    for i in range(len(Weight_Matrix)):
        pair = ast.literal_eval(Pairs[i])
        task_1 = int(pair[0])
        task_2 = int(pair[1])
        task_1_index = tasks_dict[task_1]
        task_2_index = tasks_dict[task_2]

        affinity_matrix[task_1_index][task_2_index] = Weight_Matrix['Weight'][i]
        affinity_matrix[task_2_index][task_1_index] = Weight_Matrix['Weight'][i]


    print(np.shape(affinity_matrix))

    CLUSTERS = get_clusters_weightMatrix(affinity_matrix, tasks_dict_X, min_cluster_size, max_cluster_size)

    CLUSTERS.to_csv(f'../clustering/data/{dataset}_Hierarchical_Clusters_WeightMatrix.csv',index=False)


