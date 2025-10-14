# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import ast
import tqdm
import itertools
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn import metrics
import keras
from keras.callbacks import ModelCheckpoint
from keras import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class TGPDataPreparation:

    def __init__(self, dataset_name, modelName):
        self.dataset_name = dataset_name
        self.DataPath = f'../Dataset/{self.dataset_name.upper()}/'
        self.ResultPath = f'../Results/'
        self.dataresultPath = '../Results'

        self.task_len = {}
        self.variance_dict = {}
        self.std_dev_dict = {}
        self.dist_dict = {}
        self.single_res_dict = {}
        self.FULL_TRAIN = True

        if self.dataset_name == 'School':
            self.TASKS = [i for i in range(1, 140)]
        if self.dataset_name == 'Landmine':
            self.TASKS = [i for i in range(0, 29)]
        if self.dataset_name == 'Parkinsons':
            self.TASKS = [i for i in range(1, 43)]
        if self.dataset_name == 'Chemical':
            chemical_data = pd.read_csv(f'{self.DataPath}ChemicalData_All.csv', low_memory=False)
            self.TASKS = list(chemical_data['180'].unique())

        self.task_info = pd.read_csv(f'{self.DataPath}Task_Information_{self.dataset_name}.csv')
        self.task_distance_info = pd.read_csv(f'{self.DataPath}Task_Distance_{self.dataset_name}.csv')

        '''Model Specific Data'''
        self.single_results = pd.read_csv(f'{self.dataresultPath}/STL/STL_{self.dataset_name}_{modelName}.csv')
        self.pair_results = pd.read_csv(
                f'{self.dataresultPath}/Pairwise/NN/{self.dataset_name}_Results_from_Pairwise_Training_ALL_{modelName}.csv')

        # self.ITA_data = pd.read_csv(f'{self.dataresultPath}/InterTask_Affinity/Pairwise_ITA_{self.dataset_name}.csv', low_memory=False)

        self.Weight_Matrix = pd.read_csv(f'{self.dataresultPath}/Weight_Matrix/Weight_Affinity_{self.dataset_name}.csv',
                                    low_memory=False)



        for selected_task in self.TASKS:
            if self.dataset_name == 'Chemical':
                task_data = self.task_info[self.task_info.Molecule == selected_task].reset_index()
            else:
                task_data = self.task_info[self.task_info.Task_Name == selected_task].reset_index()
            self.task_len.update({selected_task: task_data.Dataset_Size[0]})
            self.variance_dict.update({selected_task: task_data.Variance[0]})
            self.std_dev_dict.update({selected_task: task_data.Std_Dev[0]})
            self.dist_dict.update({selected_task: task_data.Average_Euclidian_Distance_within_Task[0]})
            single_res = self.single_results[self.single_results.Task == selected_task].reset_index()
            self.single_res_dict.update({selected_task: single_res.LOSS[0]})

    def predictor_data_prep(self, task_grouping_results):
        single_task_loss = []
        group_info = []
        group_loss = []
        group_dataset_size = []
        group_variance = []
        group_stddev = []
        group_distance = []
        number_of_tasks = []

        pairwise_improvement_average = []
        pairwise_improvement_variance = []
        pairwise_improvement_stddev = []

        # pairwise_ITA_average = []
        pairwise_Weight_average = []


        change = []



        # count = 0
        for group in tqdm.tqdm(range(len(task_grouping_results))):

            Task_Group = ast.literal_eval(task_grouping_results.Task_group[group])
            Individual_Group_Score = ast.literal_eval(task_grouping_results.Individual_Group_Score[group])

            for group_no, tasks in Task_Group.items():
                sample_size = 0
                avg_var = []
                avg_stddev = []
                avg_dist = []
                sum_loss_single_task = 0
                if len(tasks)<=1:
                    continue
                task_combo = list(itertools.combinations(sorted(tasks), 2))
                for task_pair in task_combo:
                    task_dist_data = self.task_distance_info[(self.task_distance_info.Task_1 == task_pair[0]) & (
                            self.task_distance_info.Task_2 == task_pair[1])].reset_index()
                    if len(task_dist_data) == 0:
                        task_dist_data = self.task_distance_info[(self.task_distance_info.Task_1 == task_pair[1]) & (
                                self.task_distance_info.Task_2 == task_pair[0])].reset_index()

                    if self.dataset_name == 'Chemical':
                        avg_dist.append(task_dist_data.Average_Hamming_Distance[0])
                    else:
                        avg_dist.append(task_dist_data.Average_Euclidean_Distance[0])


                paired_improvement = []
                paired_ITA = []
                paired_weight = []
                for pair in task_combo:
                    stl_loss = 0
                    stl_loss += self.single_res_dict[pair[0]]
                    stl_loss += self.single_res_dict[pair[1]]
                    pair_specific = self.pair_results[
                        (self.pair_results.Task_1 == pair[0]) & (self.pair_results.Task_2 == pair[1])].reset_index()
                    if len(pair_specific) == 0:
                        pair_specific = self.pair_results[
                            (self.pair_results.Task_1 == pair[1]) & (self.pair_results.Task_2 == pair[0])].reset_index()
                    paired_improvement.append((stl_loss - pair_specific.Total_Loss[0]) / stl_loss)

                    p = tuple(sorted([pair[0], pair[1]]))
                    # pair_idx = list(self.ITA_data['Pairs']).index(str(p))
                    # paired_ITA.append(list(self.ITA_data.Pairwise_ITA)[pair_idx])

                    pair_idx = list(self.Weight_Matrix['Pairs']).index(str(p))
                    paired_weight.append(list(self.Weight_Matrix.Weight)[pair_idx])

                # pairwise_ITA_average.append(np.mean(paired_ITA))
                pairwise_Weight_average.append(np.mean(paired_weight))


                pairwise_improvement_average.append(np.mean(paired_improvement))
                pairwise_improvement_variance.append(np.var(paired_improvement))
                pairwise_improvement_stddev.append(np.std(paired_improvement))
                for t in tasks:
                    sample_size += self.task_len[t]
                    avg_var.append(self.variance_dict[t])
                    avg_stddev.append(self.std_dev_dict[t])
                    sum_loss_single_task += self.single_res_dict[t]
                group_info.append(tasks)
                group_loss.append(Individual_Group_Score[group_no])
                number_of_tasks.append(len(tasks))
                group_dataset_size.append(sample_size / len(tasks))  # individual length

                group_distance.append(np.mean(avg_dist))
                group_variance.append(np.mean(avg_var))
                group_stddev.append(np.mean(avg_stddev))

                change.append((sum_loss_single_task - Individual_Group_Score[group_no]) / sum_loss_single_task)

        predictor_data = pd.DataFrame({
            'group_variance': group_variance,
            'group_stddev': group_stddev,
            'group_distance': group_distance,
            'number_of_tasks': number_of_tasks,
            'group_dataset_size': group_dataset_size,
            'pairwise_improvement_average': pairwise_improvement_average,
            'pairwise_improvement_variance': pairwise_improvement_variance,
            'pairwise_improvement_stddev': pairwise_improvement_stddev,
            # 'pairwise_ITA_average': pairwise_ITA_average,
            'pairwise_Weight_average': pairwise_Weight_average,
            'change': change
        })
        predictor_data = predictor_data[predictor_data.number_of_tasks > 2]
        print(f'total group samples = {len(predictor_data)}, shape = {predictor_data.shape}')

        predictor_data.to_csv(f'{self.ResultPath}partition_sample/group_sample/Data_for_GroupPredictor_{self.dataset_name}_{modelName}.csv', index=False)




if __name__ == '__main__':
    '''Predictor Functions'''
    modelName = 'NN'
    for dataset in ['School', 'Chemical', 'Landmine', 'Parkinsons']:
        DataPath = f"../Dataset/{dataset.upper()}/"
        ResultPath = '../Results/'
        if dataset == 'Chemical':
            ChemicalData = pd.read_csv(f'{DataPath}ChemicalData_All.csv', low_memory=False)
            TASKS = list(ChemicalData['180'].unique())
        if dataset == 'School':
            TASKS = [i for i in range(1, 140)]
        if dataset == 'Landmine':
            TASKS = [i for i in range(0, 29)]
        if dataset == 'Parkinsons':
            TASKS = [i for i in range(1, 43)]

        partition_sample = pd.read_csv(f'../Results/partition_sample/{dataset}_partition_sample_MTL.csv')


        obj = TGPDataPreparation(dataset_name=dataset,modelName=modelName)
        TGPDataPreparation.predictor_data_prep(obj, partition_sample)

