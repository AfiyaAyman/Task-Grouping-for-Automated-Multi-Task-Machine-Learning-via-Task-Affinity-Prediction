import ast

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean,hamming,cityblock
from sklearn import preprocessing
import itertools
import tqdm

def convert_to_csv():
    datafile = f'../Dataset/{dataset.upper()}/parkinsons_updrs.data'
    file1 = open(datafile, 'r')
    Lines = file1.readlines()

    ColumnHeader = Lines[0]
    Column_dict = {}
    COLUMNS = []
    ColumnHeader = ColumnHeader.split(',')
    for col in ColumnHeader:
        col = col.replace('\n','')
        Column_dict.update({col:[]})
        COLUMNS.append(col)

    for line in Lines[1:]:
        line = line.split(',')
        for elem in range(len(line)):
            val = ast.literal_eval(line[elem])
            Column_dict[COLUMNS[elem]].append(val)

    Parkinsons = pd.DataFrame.from_dict(Column_dict)
    Parkinsons.to_csv(f'{DataPath}parkinsons_updrs.csv',index=False)


def task_specific_dataset():
    parkinsons = pd.read_csv(f'{DataPath}parkinsons_updrs.csv')
    print(parkinsons.columns)
    print(len(parkinsons['subject#'].unique()))
    subjects = list(parkinsons['subject#'].unique())

    Tasks = []
    Length = []
    Variance = []
    StdDev = []

    Average_Euclidian_Distance_within_Task = []
    Average_Manhattan_Distance_within_Task = []
    Average_Euclidian_Distance_within_Task_after_Scaling = []
    Average_Manhattan_Distance_within_Task_after_Scaling = []

    parkinsons = parkinsons[['subject#', 'age', 'sex', 'test_time',
       'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
       'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
       'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE',
                             'motor_UPDRS', 'total_UPDRS']]

    X = parkinsons.describe()
    X.to_csv(f'{DataPath}parkinsons_EDA.csv')


    for subj in subjects:
        task_specific = parkinsons[parkinsons['subject#']==subj].reset_index()
        task_specific = task_specific[['age', 'sex', 'test_time', 'Jitter(%)',
         'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP', 'Shimmer',
         'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11',
         'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE', 'motor_UPDRS',
         'total_UPDRS']]

        Tasks.append(subj)
        Length.append(len(task_specific))
        var = (np.var(task_specific.total_UPDRS) +np.var(task_specific.motor_UPDRS))/2
        stddev = (np.std(task_specific.total_UPDRS) + np.std(task_specific.motor_UPDRS)) / 2
        Variance.append(var)
        StdDev.append(stddev)
        print(np.var(task_specific.total_UPDRS),np.var(task_specific.motor_UPDRS))
        print(var)
        print(np.var(task_specific[['total_UPDRS','motor_UPDRS']]))



        DataSet = np.array(task_specific, dtype=float)
        dist = []
        mdist = []
        for i in range(0, len(DataSet) - 1):
            dist.append(euclidean(DataSet[i], DataSet[i + 1]))
            mdist.append(cityblock(DataSet[i], DataSet[i + 1]))
        Average_Euclidian_Distance_within_Task.append(np.mean(dist))
        Average_Manhattan_Distance_within_Task.append(np.mean(mdist))

        '''Perform Scaling'''
        x = task_specific.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        DataSet_Scaled = pd.DataFrame(x_scaled)
        DataSet_Scaled = np.array(DataSet_Scaled, dtype=float)

        dist = []
        mdist = []
        for i in range(0, len(DataSet_Scaled) - 1):
            dist.append(euclidean(DataSet_Scaled[i], DataSet_Scaled[i + 1]))
            mdist.append(cityblock(DataSet_Scaled[i], DataSet_Scaled[i + 1]))
        Average_Euclidian_Distance_within_Task_after_Scaling.append(np.mean(dist))
        Average_Manhattan_Distance_within_Task_after_Scaling.append(np.mean(mdist))

        task_specific.to_csv(f'{DataPath}parkinsons_subject_{subj}.csv', index=False)

    Task_Information = pd.DataFrame({'Task_Name': Tasks,
                                     'Dataset_Size': Length,
                                     'Variance': Variance,
                                     'Std_Dev': StdDev,
                                     'Average_Euclidian_Distance_within_Task': Average_Euclidian_Distance_within_Task,
                                     'Average_Manhattan_Distance_within_Task': Average_Manhattan_Distance_within_Task,
                                     'Average_Euclidian_Distance_within_Task_after_Scaling': Average_Euclidian_Distance_within_Task_after_Scaling,
                                     'Average_Manhattan_Distance_within_Task_after_Scaling': Average_Manhattan_Distance_within_Task_after_Scaling})
    Task_Information.to_csv(f'{DataPath}Task_Information_{dataset}.csv', index=False)





def prep_data_for_mtl():
    parkinsons = pd.read_csv(f'{DataPath}parkinsons_updrs.csv')
    print(parkinsons.columns)
    print(len(parkinsons['subject#'].unique()))
    subjects = list(parkinsons['subject#'].unique())
    Task_Information = pd.read_csv(f'{DataPath}Task_Information_{dataset}.csv')
    maxlen = max(list(Task_Information.Dataset_Size))
    print(maxlen)
    for subj in subjects:
        csv = (f"{DataPath}/parkinsons_subject_{subj}.csv")
        df = pd.read_csv(csv, low_memory=False)
        samples_to_be_repeated = maxlen-len(df)
        if samples_to_be_repeated>0:
            df2 = df.sample(n=samples_to_be_repeated,replace=True)
            df = df.append(df2, ignore_index=True)
        else:
            df = df.sample(n=maxlen)
        new_csv = (f"{DataPath}DATA/parkinsons_subject_{subj}_for_MTL.csv")
        df.to_csv(new_csv,index=False)


def task_distance_Calc():
    parkinsons = pd.read_csv(f'{DataPath}parkinsons_updrs.csv')
    subjects = list(parkinsons['subject#'].unique())

    Average_Euclidean_Distance = []
    Average_Manhattan_Distance = []
    Average_Euclidean_Distance_after_Scaling = []
    Average_Manhattan_Distance_after_Scaling = []
    Task_1 = []
    Task_2 = []

    Subjects = [i for i in range(1, 43)]
    pairwise_tasks = list(itertools.combinations(Subjects, 2))

    # calculate manhattan distance
    def manhattan_distance(a, b):
        return sum(abs(e1 - e2) for e1, e2 in zip(a, b))

    for task_pair in tqdm.tqdm(pairwise_tasks):
        task = list(task_pair)
        Task_1.append(task[0])
        Task_2.append(task[1])

        sch_id = task_pair[0]
        csv = (f'{DataPath}parkinsons_subject_{sch_id}.csv')

        df = pd.read_csv(csv, low_memory=False)
        dataset_1 = df[['Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP',
       'Jitter:PPQ5', 'Jitter:DDP', 'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3',
       'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE',
       'DFA', 'PPE', 'motor_UPDRS', 'total_UPDRS']]

        x = dataset_1.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        DataSet_1_Normalized = pd.DataFrame(x_scaled)
        DataSet_1_Normalized = np.array(DataSet_1_Normalized, dtype=float)

        DataSet_1 = np.array(dataset_1, dtype=float)
        sch_id = task_pair[1]
        csv = (f'{DataPath}parkinsons_subject_{sch_id}.csv')

        df = pd.read_csv(csv, low_memory=False)
        dataset_2 = df[['Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP',
                               'Jitter:PPQ5', 'Jitter:DDP', 'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3',
                               'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE',
                               'DFA', 'PPE', 'motor_UPDRS', 'total_UPDRS']]
        # print(df)
        y = dataset_2.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        y_scaled = min_max_scaler.fit_transform(y)
        DataSet_2_Normalized = pd.DataFrame(y_scaled)

        DataSet_2 = np.array(dataset_2, dtype=float)
        DataSet_2_Normalized = np.array(DataSet_2_Normalized, dtype=float)
        # print(len(DataSet_1),len(DataSet_2))

        euclidean_dist = []
        manhattan_dist = []
        for i in range(0, len(DataSet_1)):
            for j in range(0, len(DataSet_2)):
                euclidean_dist.append(euclidean(DataSet_1[i], DataSet_2[j]))
                manhattan_dist.append(cityblock(DataSet_1[i], DataSet_2[j]))
        Average_Euclidean_Distance.append(np.mean(euclidean_dist))
        Average_Manhattan_Distance.append(np.mean(manhattan_dist))

        euclidean_dist = []
        manhattan_dist = []
        # print(DataSet[0],DataSet[len(DataSet)-1])
        for i in range(0, len(DataSet_1_Normalized)):
            for j in range(0, len(DataSet_2_Normalized)):
                euclidean_dist.append(euclidean(DataSet_1_Normalized[i], DataSet_2_Normalized[j]))
                manhattan_dist.append(cityblock(DataSet_1_Normalized[i], DataSet_2_Normalized[j]))

        Average_Euclidean_Distance_after_Scaling.append(np.mean(euclidean_dist))
        Average_Manhattan_Distance_after_Scaling.append(np.mean(manhattan_dist))

    Task_Information = pd.DataFrame({'Task_1': Task_1,
                                     'Task_2': Task_2,
                                     'Average_Euclidean_Distance': Average_Euclidean_Distance,
                                     'Average_Manhattan_Distance': Average_Manhattan_Distance,
                                     'Average_Euclidean_Distance_after_Scaling': Average_Euclidean_Distance_after_Scaling,
                                     'Average_Manhattan_Distance_after_Scaling': Average_Manhattan_Distance_after_Scaling})
    Task_Information.to_csv(f'{DataPath}/Task_Distance_{dataset}.csv', index=False)



dataset = 'Parkinsons'
DataPath = f'../Dataset/{dataset.upper()}/'
# convert_to_csv()
# task_specific_dataset()
# prep_data_for_mtl()
task_distance_Calc()