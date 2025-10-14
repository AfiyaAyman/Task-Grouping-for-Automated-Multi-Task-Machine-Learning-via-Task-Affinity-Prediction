import sys

import pandas as pd
import numpy as np
import random
import time, os
import ast
import copy
import tqdm
import math
import multiprocessing as mp
from multiprocessing.pool import ThreadPool

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import itertools

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.layers import *
# print(f'version = {tf.__version__}')
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model, backend
# from tensorflow.keras.utils.layer_utils import count_params
from multiprocessing.pool import ThreadPool
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# gpus = tf.config.list_physical_devices('GPU')
# gpu_device = gpus[3]
# core_config = tf.config.experimental.set_visible_devices(gpu_device, 'GPU')
# tf.config.experimental.set_memory_growth(gpu_device, True)
# tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=core_config))

'''Task Grouping Predictor Functions'''


def Data_Prep(school_data):
    school_data_arr = np.array(school_data, dtype=float)
    Number_of_Records = np.shape(school_data_arr)[0]
    Number_of_Features = np.shape(school_data_arr)[1]

    Input_Features = school_data.columns[:Number_of_Features - 1]
    Target_Features = school_data.columns[Number_of_Features - 1:]

    Sample_Inputs = np.zeros((Number_of_Records, len(Input_Features)))
    for t in range(Number_of_Records):
        Sample_Inputs[t] = school_data_arr[t, :len(Input_Features)]
    Sample_Label = np.zeros((Number_of_Records, len(Target_Features)))
    for t in range(Number_of_Records):
        Sample_Label[t] = school_data_arr[t, Number_of_Features - len(Target_Features):]
    # print(Sample_Label[0])

    return Sample_Inputs, Sample_Label, len(Input_Features)


def readData(TASKS):
    data_param_dictionary = {}
    for sch_id in TASKS:

        csv = (f"{DataPath}{sch_id}_School_Data.csv")
        school_data = pd.read_csv(csv, low_memory=False)
        # print(len(df))

        school_data = school_data[[
            '1985', '1986', '1987',
            'ESWI', 'African', 'Arab', 'Bangladeshi', 'Caribbean', 'Greek', 'Indian', 'Pakistani', 'SE_Asian',
            'Turkish', 'Other',
            'VR_Band', 'Gender',
            'FSM', 'VR_BAND_Student', 'School_Gender', 'Maintained', 'Church', 'Roman_Cath',
            'ExamScore',
        ]]

        Sample_Inputs, Sample_Label, Number_of_Features_FF = Data_Prep(school_data)

        data_param_dictionary.update({f'School_{sch_id}_FF_Inputs': Sample_Inputs})
        data_param_dictionary.update({f'School_{sch_id}_Labels': Sample_Label})

        '''*********************************'''
    return data_param_dictionary, Number_of_Features_FF


def SplitLabels(Target):
    label_data = np.zeros((len(Target), 1))
    for t in range(len(Target)):
        label_data[t] = Target[t][0]
    return label_data


def Splitting_Values(Labels):
    Predicted = []
    for i in Labels:
        for j in i:
            Predicted.append(j)
    return Predicted


'''MTL for Task Groups Functions'''


def kFold_validation(current_task_specific_architecture, current_shared_architecture, task, group_no, random_seed):

    data_param_dictionary, Number_of_Features_FF = readData(task)

    data_param_dict_for_specific_task = {}
    if datasetName == 'School':
        max_size = 252
        train_set_size = math.floor(max_size*(1-num_folds/100))
        test_set_size = math.ceil(max_size*(num_folds/100))
    # print(data_param_dictionary.keys())
    # print(len(data_param_dictionary.keys()))

    for f in range(num_folds):
        data_param_dict_for_specific_task[f] = {}

    for sch_id in task:
        Sample_Inputs = data_param_dictionary[f'School_{sch_id}_FF_Inputs']
        Sample_Label = data_param_dictionary[f'School_{sch_id}_Labels']

        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)

        fold = 0
        ALL_FOLDS = []

        for train, test in kfold.split(Sample_Inputs):
            # data_param_dict_for_specific_task.update({fold: {}})
            X_train = Sample_Inputs[train]
            X_test = Sample_Inputs[test]
            y_train = Sample_Label[train]
            y_test = Sample_Label[test]

            samples_to_be_repeated = train_set_size - len(X_train)
            # print(f'samples_to_be_repeated = {samples_to_be_repeated}')
            if samples_to_be_repeated > 0:
                random_indices = np.random.choice(X_train.shape[0], samples_to_be_repeated)
                X_train = np.concatenate((X_train, X_train[random_indices]), axis=0)
                y_train = np.concatenate((y_train, y_train[random_indices]), axis=0)

            samples_to_be_repeated = test_set_size - len(X_test)
            # print(f'samples_to_be_repeated = {samples_to_be_repeated}')
            if samples_to_be_repeated > 0:
                random_indices = np.random.choice(X_test.shape[0], samples_to_be_repeated)
                X_test = np.concatenate((X_test, X_test[random_indices]), axis=0)
                y_test = np.concatenate((y_test, y_test[random_indices]), axis=0)

            y_train = SplitLabels(y_train)
            y_test = SplitLabels(y_test)

            data_param_dict_for_specific_task[fold][f'School_{sch_id}_fold_{fold}_X_train'] = X_train
            data_param_dict_for_specific_task[fold][f'School_{sch_id}_fold_{fold}_X_test'] = X_test

            data_param_dict_for_specific_task[fold][f'School_{sch_id}_fold_{fold}_y_train'] = y_train
            data_param_dict_for_specific_task[fold][f'School_{sch_id}_fold_{fold}_y_test'] = y_test
            # data_param_dict_for_specific_task[fold][f'School_{sch_id}_fold_{fold}_Actual_Exam_Score'] = Actual_Exam_Score

            tmp = (
                current_task_specific_architecture, current_shared_architecture, task,
                data_param_dict_for_specific_task,
                Number_of_Features_FF,
                fold, group_no)

            ALL_FOLDS.append(tmp)

            fold += 1

    number_of_models = 5
    current_idx = random.sample(range(len(ALL_FOLDS)), number_of_models)
    args = [ALL_FOLDS[index] for index in sorted(current_idx)]
    return args

def sort_Results(all_scores,task):
    scores = []
    for i in range(len(all_scores)):
        scores.append(all_scores[i][0])
    # print(f'scores = {scores}')

    score_param_per_task_group_per_fold = {}
    if len(task) < 2:
        for sch_id in task:
            score_param_per_task_group_per_fold.update({f'sch_{sch_id}': scores[0]})
    else:
        for sch_id in task:
            score_param_per_task_group_per_fold.update({f'sch_{sch_id}': []})

        for i in range(0, len(scores)):
            # print(f'i = {i, scores[i]}')
            for j in range(1, len(scores[i])):
                # print(f'i = {i,scores[i]}, j = {j,scores[i][j]}')
                score_param_per_task_group_per_fold[f'sch_{task[j - 1]}'].append(scores[i][j])

    total_loss_per_task_group_per_fold = 0
    for t, MSE_list in score_param_per_task_group_per_fold.items():
        total_loss_per_task_group_per_fold += np.mean(MSE_list)
    # '''
    # print(f'Group {group_no} - Total Loss = {total_loss_per_task_group_per_fold}')
    return total_loss_per_task_group_per_fold


def final_model(task_hyperparameters, shared_hyperparameters, task_group_list, data_param_dict_for_specific_task,
                Number_of_Features, fold, group_no):
    data_param = copy.deepcopy(data_param_dict_for_specific_task[fold])

    if ClusterType == 'Hierarchical':
        filepath = f'SavedModels/{ClusterType}_{Type}_clusters_School_Group_{group_no}_{fold}_{which_half}.h5'
    else:
        filepath = f'SavedModels/{ClusterType}_{Type}_clusters_School_Group_{group_no}_{fold}.h5'

    MTL_model_param = {}
    shared_module_param_FF = {}

    input_layers = []

    train_data = []
    train_label = []
    test_data = []
    test_label = []

    for sch_id in task_group_list:
        train_data.append(data_param[f'School_{sch_id}_fold_{fold}_X_train'])
        train_label.append(data_param[f'School_{sch_id}_fold_{fold}_y_train'])
        test_data.append(data_param[f'School_{sch_id}_fold_{fold}_X_test'])
        test_label.append(data_param[f'School_{sch_id}_fold_{fold}_y_test'])

        # input = tf.constant(data_param[f'School_{sch_id}_fold_{fold}_X_train'])
        hyperparameters = copy.deepcopy(task_hyperparameters[sch_id])
        Input_FF = tf.keras.layers.Input(shape=(Number_of_Features,))
        input_layers.append(Input_FF)

        MTL_model_param[f'sch_{sch_id}_fold_{fold}_Input_FF'] = Input_FF
        if hyperparameters['preprocessing_FF_layers'] > 0:
            hidden_ff = Dense(hyperparameters['preprocessing_FF_Neurons'][0], activation='sigmoid')(
                Input_FF)
            for h in range(1, hyperparameters['preprocessing_FF_layers']):
                hidden_ff = Dense(hyperparameters['preprocessing_FF_Neurons'][h], activation='sigmoid')(hidden_ff)

            MTL_model_param[f'sch_{sch_id}_fold_{fold}_ff_preprocessing_model'] = hidden_ff

    for h in range(0, shared_hyperparameters['shared_FF_Layers']):
        shared_ff = tf.keras.layers.Dense(shared_hyperparameters['shared_FF_Neurons'][h], activation='sigmoid')
        shared_module_param_FF[f'FF_{h}_fold_{fold}'] = shared_ff

    # for sch_id in task_group_list:
    #     preprocessed_ff = MTL_model_param[f'sch_{sch_id}_fold_{fold}_ff_preprocessing_model']
    #
    #     shared_FF = shared_module_param_FF[f'FF_0_fold_{fold}'](preprocessed_ff)
    #
    #     for h in range(1, shared_hyperparameters['shared_FF_Layers']):
    #         shared_FF = shared_module_param_FF[f'FF_{h}_fold_{fold}'](shared_FF)
    #
    #     MTL_model_param[f'sch_{sch_id}_fold_{fold}_last_hidden_layer'] = shared_FF

    for sch_id in task_group_list:
        hyperparameters = copy.deepcopy(task_hyperparameters[sch_id])
        if hyperparameters['preprocessing_FF_layers'] > 0:
            preprocessed_ff = MTL_model_param[f'sch_{sch_id}_fold_{fold}_ff_preprocessing_model']
            adaptive_ff = Dense(shared_hyperparameters['adaptive_FF_neurons'],
                                activation='sigmoid')(
                preprocessed_ff)
        else:
            Input_FF = MTL_model_param[f'sch_{sch_id}_fold_{fold}_Input_FF']
            adaptive_ff = Dense(shared_hyperparameters['adaptive_FF_neurons'],
                                activation='sigmoid')(
                Input_FF)

        shared_FF = shared_module_param_FF[f'FF_0_fold_{fold}'](adaptive_ff)

        for h in range(1, shared_hyperparameters['shared_FF_Layers']):
            shared_FF = shared_module_param_FF[f'FF_{h}_fold_{fold}'](shared_FF)

        MTL_model_param[f'sch_{sch_id}_fold_{fold}_last_hidden_layer'] = shared_FF

    # output Neurons
    output_layers = []

    for sch_id in task_group_list:
        shared_model = Model(inputs=MTL_model_param[f'sch_{sch_id}_fold_{fold}_Input_FF'],
                             outputs=MTL_model_param[f'sch_{sch_id}_fold_{fold}_last_hidden_layer'])
        # combined_input = concatenate([MTL_model_param[f'sch_{sch_id}_fold_{fold}_Input_FF'], shared_model.output])
        outputlayer = Dense(1, activation='linear', name=f'ExamScore_{sch_id}')(shared_model.output)
        output_layers.append(outputlayer)

    finalModel = Model(inputs=input_layers, outputs=output_layers)


    # print(finalModel.summary())
    # tf.keras.utils.plot_model(finalModel, f"MTL_School_{group_no}.png", show_shapes=True)
    # exit(0)

    # from keras.utils.layer_utils import count_params
    # trainable_count = count_params(finalModel.trainable_weights)
    # print(f'fold = {fold}\tMTL = {len(MTL_model_param.keys())}\t trainable_count = {trainable_count}')

    opt = tf.keras.optimizers.Adam(learning_rate=shared_hyperparameters['learning_rate'])
    finalModel.compile(optimizer=opt, loss='mse')

    checkpoint = ModelCheckpoint(filepath, verbose=0, monitor='val_loss', save_best_only=True, mode='auto')
    number_of_epoch = 500
    if len(task_group_list)>10:
        batch_size = 64
        number_of_epoch = 600
    else:
        batch_size = 32
        number_of_epoch = 400
    history = finalModel.fit(x=train_data,
                             y=train_label,
                             # shuffle=True,
                             epochs=number_of_epoch,
                             batch_size=batch_size,
                             validation_data=(test_data,
                                              test_label),
                             callbacks=checkpoint,
                             verbose=0)

    finalModel = tf.keras.models.load_model(filepath)
    scores = finalModel.evaluate(test_data, test_label, verbose=0)

    if os.path.exists(filepath):
        os.remove(os.path.join(filepath))
    # print(f'done')
    return scores, 999


def random_task_grouping(task_Set, min_task_groups):
    task_group = {}
    total_dataset = len(task_Set)

    group = 0
    while total_dataset > 0 and min_task_groups > 0:
        team = random.sample(task_Set, int(total_dataset / min_task_groups))
        task_group.update({group: team})
        group += 1
        for x in team:
            task_Set.remove(x)
        total_dataset -= int(total_dataset / min_task_groups)
        min_task_groups -= 1

    return task_group


def prep_task_specific_arch(current_task_group):
    TASK_Specific_Arch = {}
    for group_no in current_task_group.keys():
        Number_of_Tasks = current_task_group[group_no]
        initial_task_specific_architecture = {}
        for n in Number_of_Tasks:
            initial_task_specific_architecture.update({n: {'preprocessing_FF_layers': 1,
                                                           'preprocessing_FF_Neurons': [5],
                                                           'postprocessing_FF_layers': 0,
                                                           'postprocessing_FF_Neurons': [0]
                                                           }})

        TASK_Specific_Arch.update({group_no: initial_task_specific_architecture})
    return TASK_Specific_Arch


if __name__ == '__main__':

    ClusterType = str(sys.argv[1])
    Type = str(sys.argv[2])
    run = int(sys.argv[3])
    # which_half = str(sys.argv[4])
    num_folds = 10
    # initial_shared_architecture = {'adaptive_FF_neurons': 4, 'shared_FF_Layers': 1, 'shared_FF_Neurons': [10],
                                   # 'learning_rate': 0.00779959}
    initial_shared_architecture = {'adaptive_FF_neurons': 4, 'shared_FF_Layers': 1, 'shared_FF_Neurons': [10],
                                 'learning_rate': 0.033806674289462206,}

    '''Global Files and Data for Task-Grouping Predictor'''
    task_len = {}
    variance_dict = {}
    std_dev_dict = {}
    dist_dict = {}
    Single_res_dict = {}
    Pairwise_res_dict = {}

    datasetName = 'School'
    DataPath = f'../Dataset/{datasetName.upper()}/'
    ResultPath = '../Results'

    task_info = pd.read_csv(f'{DataPath}Task_Information_{datasetName}.csv')
    task_distance_info = pd.read_csv(f'{DataPath}Task_Distance_{datasetName}.csv')
    single_results = pd.read_csv(f'{ResultPath}/STL/STL_{datasetName}_NN.csv')
    # pair_results = pd.read_csv(f'{ResultPath}/School_Results_from_Pairwise_Training_ALL_Final.csv')
    pair_results = pd.read_csv(f'../Results/new_runs/{datasetName}_Results_from_Pairwise_Training_ALL_NN.csv')

    TASKS = [i for i in range(1, 140)]
    for Selected_Task in TASKS:
        task_data = task_info[task_info.Task_Name == Selected_Task].reset_index()
        task_len.update({Selected_Task: task_data.Dataset_Size[0]})
        variance_dict.update({Selected_Task: task_data.Variance[0]})
        std_dev_dict.update({Selected_Task: task_data.Std_Dev[0]})
        dist_dict.update({Selected_Task: task_data.Average_Euclidian_Distance_within_Task[0]})
        single_res = single_results[single_results.Task == Selected_Task].reset_index()
        Single_res_dict.update({Selected_Task: single_res.Loss_MSE[0]})

    Task1 = list(pair_results.Task_1)
    Task2 = list(pair_results.Task_2)
    Pairs = [(Task1[i], Task2[i]) for i in range(len(Task1))]
    print(len(Pairs))
    print(Pairs[:10])
    for p in Pairs:
        task1 = p[0]
        task2 = p[1]
        pair_res = pair_results[(pair_results.Task_1 == task1) & (pair_results.Task_2 == task2)].reset_index()
        Pairwise_res_dict.update({p: pair_res.Total_Loss[0]})

    print(len(Single_res_dict), len(Pairwise_res_dict))
    ''''''
    switch_architecture = ['ACCEPT', 'REJECT']

    '''Predictor Functions'''


    Task_group = []
    Total_Loss = []
    Individual_Group_Score = []
    Number_of_Groups = []

    # Type = 'Exponential'
    # Type = 'NonNegative'


    if Type != 'WeightMatrix':
        TASK_Clusters = pd.read_csv(f'../Results/Cluster_CSV/NN/{datasetName}_Clusters_{Type}_NN.csv')
    else:
        TASK_Clusters = pd.read_csv(f'../Results/Cluster_CSV/NN/{datasetName}_{ClusterType}_Clusters_{Type}_NN.csv')
    TASK_Group = list(TASK_Clusters.TASK_Group)


    if ClusterType == 'Hierarchical':
        which_half = str(sys.argv[4])
        if which_half == 'first':
            TASK_Group = TASK_Group[:len(TASK_Group)//2]
        else:
            TASK_Group = TASK_Group[len(TASK_Group)//2:]

    Prev_Groups = {}

    for count in range(len(TASK_Group)):
        print(f'Initial Training for {datasetName}-partition {count}')

        task_group = TASK_Group[count]
        task_group = ast.literal_eval(task_group)
        print(f'task_group = {task_group}')

        TASK_Specific_Arch = prep_task_specific_arch(task_group)
        random_seed = random.randint(0, 100)

        args_tasks = []
        group_score = {}
        tot_loss = 0

        for group_no, task in task_group.items():
            group_score.update({group_no: 0})
            grouping_sorted = tuple(sorted(task))
            if grouping_sorted not in Prev_Groups.keys():
                if len(task) == 1:
                    loss = Single_res_dict[task[0]]
                elif len(task) == 2:
                    loss = Pairwise_res_dict[grouping_sorted]
                else:
                    tmp = (TASK_Specific_Arch[group_no], initial_shared_architecture, task, group_no, random_seed)
                    args_tasks.append(tmp)

                    args = kFold_validation(*tmp)

                    number_of_pools = len(args)+5
                    timeStart = time.time()
                    pool = mp.Pool(number_of_pools)
                    all_scores = pool.starmap(final_model, args)
                    pool.close()
                    print(f'Time required = {(time.time() - timeStart) / 60} minutes')
                    loss = sort_Results(all_scores, task)

                tot_loss += loss
                group_score[group_no] = loss
                Prev_Groups[grouping_sorted] = loss
            else:
                loss = Prev_Groups[grouping_sorted]
                if group_no not in group_score.keys():
                    group_score.update({group_no: loss})
                else:
                    group_score[group_no] = loss
                tot_loss += loss
            print(f'group_no = {group_no}\tloss = {loss}')

        print(f'tot_loss = {tot_loss}')
        print(f'group_score = {group_score}')
        Task_group.append(task_group)
        Number_of_Groups.append(len(task_group))
        Total_Loss.append(tot_loss)
        Individual_Group_Score.append(group_score.copy())
        #print(Individual_Group_Score)
        # print(f'tot_loss = {tot_loss}')
        if len(Total_Loss)%5 == 0:
            temp_res= pd.DataFrame({'Total_Loss': Total_Loss,
                                    'Number_of_Groups': Number_of_Groups,
                                    'Task_group': Task_group,
                                    'Individual_Group_Score': Individual_Group_Score,
                                    # 'Individual_Error_Rate': Individual_Error_Rate,
                                    # 'Individual_AP': Individual_AP
                                    })
            if ClusterType == 'Hierarchical':
                temp_res.to_csv(f'../Results/Cluster_CSV/{datasetName}_MTL_Results_for_{ClusterType}_{Type}_{len(Total_Loss)}_run_{run}_{which_half}.csv', index=False)
            else:
                temp_res.to_csv(f'../Results/Cluster_CSV/{datasetName}_MTL_Results_for_{ClusterType}_{Type}_{len(Total_Loss)}_run_{run}.csv', index=False)

            if len(Total_Loss)>5:
                if ClusterType == 'Hierarchical':
                    old_file = f'../Results/Cluster_CSV/{datasetName}_MTL_Results_for_{ClusterType}_{Type}_{len(Total_Loss)-5}_run_{run}_{which_half}.csv'
                else:
                    old_file = f'../Results/Cluster_CSV/{datasetName}_MTL_Results_for_{ClusterType}_{Type}_{len(Total_Loss)-5}_run_{run}.csv'
                if os.path.exists(old_file):
                    os.remove(os.path.join(old_file))

    print(len(Total_Loss), len(Number_of_Groups), len(Task_group), len(Individual_Group_Score))
    initial_results = pd.DataFrame({'Total_Loss': Total_Loss,
                                    'Number_of_Groups': Number_of_Groups,
                                    'Task_group': Task_group,
                                    'Individual_Group_Score': Individual_Group_Score})
    
    #old_df = pd.read_csv(f'../Results/Cluster_CSV/{datasetName}_MTL_Results_for_{ClusterType}_{Type}_15_run_{run}.csv')
    #initial_results = pd.concat([old_df,initial_results])
    #print(len(initial_results))
    if ClusterType == 'Hierarchical':
        initial_results.to_csv(f'../Results/Cluster_CSV/{datasetName}_MTL_Results_for_{ClusterType}_Clusters_{Type}_run_{run}_{which_half}.csv',
                           index=False)
    else:
        initial_results.to_csv(f'../Results/Cluster_CSV/{datasetName}_MTL_Results_for_{ClusterType}_Clusters_{Type}_run_{run}.csv',
                           index=False)
