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
USE_GPU = False
if USE_GPU:
    device_idx = 0
    gpus = tf.config.list_physical_devices('GPU')
    gpu_device = gpus[device_idx]
    core_config = tf.config.experimental.set_visible_devices(gpu_device, 'GPU')
    tf.config.experimental.set_memory_growth(gpu_device, True)
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=core_config))
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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

    return Sample_Inputs, Sample_Label, len(Input_Features), len(Sample_Inputs)


def readData(TASKS):
    data_param_dictionary = {}
    training_sample_size = []
    for sch_id in TASKS:

        # csv = (f"{DataPath}DATA/{sch_id}_School_Data_for_MTL.csv")
        csv = (f"{DataPath}{sch_id}_School_Data.csv")
        school_data = pd.read_csv(csv, low_memory=False)
        # print(f'sch_id = {sch_id}, school_data = {school_data.shape}')

        school_data = school_data[[
            '1985', '1986', '1987',
            'ESWI', 'African', 'Arab', 'Bangladeshi', 'Caribbean', 'Greek', 'Indian', 'Pakistani', 'SE_Asian',
            'Turkish', 'Other',
            'VR_Band', 'Gender',
            'FSM', 'VR_BAND_Student', 'School_Gender', 'Maintained', 'Church', 'Roman_Cath',
            'ExamScore',
        ]]

        Sample_Inputs, Sample_Label, Number_of_Features_FF, datasize = Data_Prep(school_data)

        data_param_dictionary.update({f'School_{sch_id}_FF_Inputs': Sample_Inputs})
        data_param_dictionary.update({f'School_{sch_id}_Labels': Sample_Label})
        training_sample_size.append(datasize)

        '''*********************************'''
    return data_param_dictionary, Number_of_Features_FF,training_sample_size


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

    data_param_dictionary, Number_of_Features_FF, training_sample_size = readData(task)

    data_param_dict_for_specific_task = {}
    if datasetName == 'School':
        max_size = 252
        train_set_size = math.floor(max_size*(1-num_folds/100))
        test_set_size = math.ceil(max_size*(num_folds/100))

    # print(f'train_set_size = {train_set_size}, test_set_size = {test_set_size}')
    # max_size = max(training_sample_size)

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

            # print(f'X_train = {X_train.shape}, X_test = {X_test.shape}, y_train = {y_train.shape}, y_test = {y_test.shape}')
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
    # print(f'score_param_per_task_group_per_fold = {score_param_per_task_group_per_fold}')

    task_specific_scores = {}
    for key in score_param_per_task_group_per_fold.keys():
        task_specific_scores.update({key: np.mean(score_param_per_task_group_per_fold[key])})
    # '''
    # print(f'Group {group_no} - Total Loss = {total_loss_per_task_group_per_fold}')
    return total_loss_per_task_group_per_fold, task_specific_scores


def final_model(task_hyperparameters, shared_hyperparameters, task_group_list, data_param_dict_for_specific_task,
                Number_of_Features, fold, group_no):
    data_param = copy.deepcopy(data_param_dict_for_specific_task[fold])

    filepath = f'SavedModels/RS_{datasetName}_v_{v}_Group_{group_no}_{fold}.h5'

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

    opt = tf.keras.optimizers.legacy.Adam(learning_rate=shared_hyperparameters['learning_rate'])
    finalModel.compile(optimizer=opt, loss='mse')

    checkpoint = ModelCheckpoint(filepath, verbose=0, monitor='val_loss', save_best_only=True, mode='auto')
    number_of_epoch = 400
    if len(task_group_list)>10:
        batch_size = 128
        number_of_epoch = 800
    elif len(task_group_list)<10 and len(task_group_list)>5:
        batch_size = 64
        number_of_epoch = 300
    else:
        batch_size = 32
        number_of_epoch = 250
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

def mtls_for_clusters():
    Task_group = []
    Total_Loss = []
    Individual_Group_Score = []
    Number_of_Groups = []

    Prev_Groups = {}
    min_task_groups = 28
    for count in range(0, 15):
        print(f'Initial Training for {datasetName}-partition {count}')
        task_Set = copy.deepcopy(TASKS)

        task_group = random_task_grouping(task_Set, min_task_groups)


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

                    number_of_pools = len(args)
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
        print(Individual_Group_Score)
        # print(f'tot_loss = {tot_loss}')

    print(len(Total_Loss),len(Number_of_Groups), len(Task_group), len(Individual_Group_Score))
    initial_results = pd.DataFrame({'Total_Loss': Total_Loss,
                                    'Number_of_Groups': Number_of_Groups,
                                    'Task_group': Task_group,
                                    'Individual_Group_Score': Individual_Group_Score})
    initial_results.to_csv(f'{ResultPath}/{datasetName}_MTL_Results_for_{ClusterType}_Clusters_{min_task_groups}_Groups.csv',
                           index=False)

if __name__ == '__main__':
    v = sys.argv[1]
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
    single_results = pd.read_csv(f'{ResultPath}/SingleTaskTraining_{datasetName}.csv')
    pair_results = pd.read_csv(f'{ResultPath}/School_Results_from_Pairwise_Training_ALL_Final.csv')
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
    Pairwise_res_dict_indiv = {}

    for p in Pairs:
        task1 = p[0]
        task2 = p[1]
        pair_res = pair_results[(pair_results.Task_1 == task1) & (pair_results.Task_2 == task2)].reset_index()
        Pairwise_res_dict.update({p: pair_res.Total_Loss[0]})

        if task1 == pair_res.Task_1[0]:
            Pairwise_res_dict_indiv.update({p: {f'sch_{task1}':pair_res.Individual_loss_Task_1[0],
                                                f'sch_{task2}':pair_res.Individual_loss_Task_2[0]}})
        else:
            Pairwise_res_dict_indiv.update({p: {f'sch_{task1}':pair_res.Individual_loss_Task_2[0],
                                                f'sch_{task2}':pair_res.Individual_loss_Task_1[0]}})


    print(len(Single_res_dict), len(Pairwise_res_dict))

    previously_trained = pd.read_csv(f'{datasetName}_previously_trained_groups.csv')
    previously_trained_groups = previously_trained['previously_trained_groups'].tolist()
    previously_trained_groups_loss = previously_trained['previously_trained_groups_loss'].tolist()
    for i, group in enumerate(previously_trained_groups):
        previously_trained_groups[i] = ast.literal_eval(group)

    ''''''
    switch_architecture = ['ACCEPT', 'REJECT']
    '''Predictor Functions'''

    # mtls_for_clusters()
    # exit(0)
    Task_group = []
    Total_Loss = []
    Individual_Group_Score = []
    Individual_Task_Score = []
    Number_of_Groups = []
    run = 2

    partition_data = pd.read_csv(f'{datasetName}_Random_Task_Groups_New.csv')
    # partition_data = pd.read_csv(f'{datasetName}_initial_groups_for_predictor_MTL.csv')
    TASK_Group = list(partition_data.Task_Groups)
    # TASK_Group = [{0:TASKS}]
    print(len(TASK_Group))

    Prev_Groups = {}
    v = int(v)
    for count in range(v,v+100):
        print(f'Initial Training for {datasetName}-partition {count}')
        task_group = TASK_Group[count]
        task_group = ast.literal_eval(task_group)
        print(f'task_group = {task_group}')

        TASK_Specific_Arch = prep_task_specific_arch(task_group)
        random_seed = random.randint(0, 100)

        args_tasks = []
        group_score = {}
        tmp_task_score = []
        tot_loss = 0

        for group_no, task in task_group.items():
            group_score.update({group_no: 0})
            grouping_sorted = tuple(sorted(task))
            if grouping_sorted not in Prev_Groups.keys():
                if len(task) == 1:
                    loss = Single_res_dict[task[0]]
                    task_scores = {f'sch_{task[0]}':Single_res_dict[task[0]]}
                elif len(task) == 2:
                    loss = Pairwise_res_dict[grouping_sorted]
                    task_scores = copy.deepcopy(Pairwise_res_dict_indiv[grouping_sorted])
                else:
                    tmp = (TASK_Specific_Arch[group_no], initial_shared_architecture, task, group_no, random_seed)
                    args_tasks.append(tmp)

                    args = kFold_validation(*tmp)

                    number_of_pools = len(args)+10
                    timeStart = time.time()

                    pool = mp.Pool(number_of_pools)
                    all_scores = pool.starmap(final_model, args)
                    pool.close()

                    # with ThreadPool(15) as tp:
                    #     all_scores = tp.starmap(final_model, args)
                    # tp.join()
                    # print(f'Time required = {(time.time() - timeStart) / 60} minutes')
                    loss, task_scores = sort_Results(all_scores, task)

                tot_loss += loss
                group_score[group_no] = loss
                tmp_task_score.append(copy.deepcopy(task_scores))
                Prev_Groups[grouping_sorted] = [loss, copy.deepcopy(task_scores)]
            else:
                loss = Prev_Groups[grouping_sorted][0]
                task_scores = Prev_Groups[grouping_sorted][1]
                if group_no not in group_score.keys():
                    group_score.update({group_no: loss})
                else:
                    group_score[group_no] = loss
                tmp_task_score.append(copy.deepcopy(task_scores))
                tot_loss += loss
            print(f'group_no = {group_no}\tloss = {loss}, task_scores = {task_scores}')

        print(f'tot_loss = {tot_loss}')
        # print(f'group_score = {group_score}')
        Task_group.append(task_group)
        Number_of_Groups.append(len(task_group))
        Total_Loss.append(tot_loss)
        Individual_Group_Score.append(group_score.copy())
        Individual_Task_Score.append(tmp_task_score.copy())
        print(
            f'len(Total_Loss) = {len(Total_Loss)}, len(Task_group) = {len(Task_group)}, len(Individual_Group_Score) = {len(Individual_Group_Score)}, len(Individual_Task_Score) = {len(Individual_Task_Score)}')

        #print(Individual_Group_Score)
        # print(f'tot_loss = {tot_loss}')
        if len(Total_Loss)%5 == 0:
            temp_res= pd.DataFrame({'Total_Loss': Total_Loss,
                                    'Number_of_Groups': Number_of_Groups,
                                    'Task_group': Task_group,
                                    'Individual_Group_Score': Individual_Group_Score,
                                    'Individual_Task_Score': Individual_Task_Score,
                                    # 'Individual_Error_Rate': Individual_Error_Rate,
                                    # 'Individual_AP': Individual_AP
                                    })
            temp_res.to_csv(f'../Groupwise_Affinity/{datasetName}_Random_Search_{v+len(Total_Loss)}_MTL_run_{run}_v_{v}.csv', index=False)

            if len(Total_Loss)>5:
                old_file = f'../Groupwise_Affinity/{datasetName}_Random_Search_{v+len(Total_Loss)-5}_MTL_run_{run}_v_{v}.csv'
                if os.path.exists(old_file):
                    os.remove(os.path.join(old_file))

    print(len(Total_Loss), len(Number_of_Groups), len(Task_group), len(Individual_Group_Score))
    initial_results = pd.DataFrame({'Total_Loss': Total_Loss,
                                    'Number_of_Groups': Number_of_Groups,
                                    'Task_group': Task_group,
                                    'Individual_Group_Score': Individual_Group_Score,
                                    'Individual_Task_Score': Individual_Task_Score})
    initial_results.to_csv(f'{ResultPath}/{datasetName}_Random_Search_{len(Total_Loss)}_MTL_run_{run}_v_{v}.csv',
                           index=False)
    # initial_results.to_csv(f'{ResultPath}/{datasetName}_SimpleMTL.csv',
    #                        index=False)
