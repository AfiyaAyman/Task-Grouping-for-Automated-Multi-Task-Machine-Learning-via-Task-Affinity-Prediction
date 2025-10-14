import random
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import ast



for dataset in ['Chemical', 'School', 'Landmine','Parkinsons']:
    DataPath = f'../../Dataset/{dataset.upper()}/'
    ResultPath = '../../Results/'

    # Load the data
    Weight_Matrix = pd.read_csv(f'{ResultPath}Weight_Affinity_{dataset}.csv', low_memory=False)

    if dataset == 'Chemical':
        ChemicalData = pd.read_csv(f'{DataPath}ChemicalData_All.csv', low_memory=False)
        TASKS = list(ChemicalData['180'].unique())

        min_cluster = 2
        max_cluster = 21

    if dataset == 'School':
        TASKS = [i for i in range(1, 140)]
        min_cluster = 2
        max_cluster = 26
    if dataset == 'Parkinsons':
        TASKS = [i for i in range(1, 43)]
        min_cluster = 2
        max_cluster = 21
    if dataset == 'Landmine':
        TASKS = [i for i in range(0, 29)]
        min_cluster = 2
        max_cluster = 16


    tasks_dict = {}
    key = 0
    for task in TASKS:
        tasks_dict[task] = key
        key += 1

    tasks_dict_X = {}
    key = 0
    for task in TASKS:
        tasks_dict_X[key] = task
        key += 1

    Pairs = list(Weight_Matrix['Pairs'])
    Change = list(Weight_Matrix['Change'])
    affinity = list(Weight_Matrix['Weight'])
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

    Sum_of_squared_distances = []
    K = range(1,30)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(affinity_matrix)
        Sum_of_squared_distances.append(km.inertia_)

    TASK_Group = []
    Number_of_Clusters = []
    for k in range(min_cluster, max_cluster):
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


        TASK_Group.append(label_dict.copy())
        print('-----------------------------------------------------')

    CLUSTERS = pd.DataFrame({"Number_of_Clusters":Number_of_Clusters,
                             "TASK_Group":TASK_Group})
    CLUSTERS.to_csv(f'../clustering/data/{dataset}_KMeans_Clusters_WeightMatrix.csv',index=False)
