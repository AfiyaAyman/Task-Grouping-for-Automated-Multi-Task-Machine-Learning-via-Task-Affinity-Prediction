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

# from School_Settings import *
# from School_Predictor import *
# from School_Grouping_Predictor import *

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import itertools

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
# import keras
from tensorflow.keras.layers import *
print(f'version = {tf.__version__}')
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model

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

# from Quick_Reject_Class import TaskGroupingPredictor



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
        csv = (f"{DataPath}/{sch_id}_School_Data.csv")
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
        # data_param_dictionary.update({f'route_{route}_{d}_RNN_Inputs': RNN_features_reshape})

        '''*********************************'''
    # return data_param_dictionary,Number_of_Features_FF,Number_of_Features_RNN
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
        for train, test in kfold.split(Sample_Inputs):
            # data_param_dict_for_specific_task.update({fold: {}})
            X_train = Sample_Inputs[train]
            X_test = Sample_Inputs[test]
            y_train = Sample_Label[train]
            y_test = Sample_Label[test]
            samples_to_be_repeated = train_set_size - len(X_train)
            # print(
            #     f'sch_id = {sch_id}, X_train, X_test = {len(X_train), len(X_test)}, samples_to_be_repeated = {samples_to_be_repeated}')
            if samples_to_be_repeated > 0:
                random_indices = np.random.choice(X_train.shape[0], samples_to_be_repeated)
                X_train = np.concatenate((X_train, X_train[random_indices]), axis=0)
                y_train = np.concatenate((y_train, y_train[random_indices]), axis=0)

            samples_to_be_repeated = test_set_size - len(X_test)
            if samples_to_be_repeated > 0:
                random_indices = np.random.choice(X_test.shape[0], samples_to_be_repeated)
                X_test = np.concatenate((X_test, X_test[random_indices]), axis=0)
                y_test = np.concatenate((y_test, y_test[random_indices]), axis=0)
            # print(
            #     f'fold = {fold} sch_id = {sch_id}, X_train = {X_train.shape}, X_test = {X_test.shape}, y_train = {y_train.shape}, y_test = {y_test.shape}')

            y_train = SplitLabels(y_train)
            y_test = SplitLabels(y_test)

            data_param_dict_for_specific_task[fold][f'School_{sch_id}_fold_{fold}_X_train'] = X_train
            data_param_dict_for_specific_task[fold][f'School_{sch_id}_fold_{fold}_X_test'] = X_test

            data_param_dict_for_specific_task[fold][f'School_{sch_id}_fold_{fold}_y_train'] = y_train
            data_param_dict_for_specific_task[fold][f'School_{sch_id}_fold_{fold}_y_test'] = y_test

            fold+=1
            # data_param_dict_for_specific_task[fold][f'School_{sch_id}_fold_{fold}_Actual_Exam_Score'] = Actual_Exam_Score

    ALL_FOLDS = []
    for fold in range(0, 10):
        tmp = (
            current_task_specific_architecture, current_shared_architecture, task,
            data_param_dict_for_specific_task,
            Number_of_Features_FF,
            fold, group_no,run)

        ALL_FOLDS.append(tmp)

    number_of_models = 5
    current_idx = random.sample(range(len(ALL_FOLDS)), number_of_models)
    args = [ALL_FOLDS[index] for index in sorted(current_idx)]

    return args


def sort_Results(all_scores,task):
    scores = []
    for i in range(len(all_scores)):
        scores.append(all_scores[i][0])

    score_param_per_task_group_per_fold = {}
    if len(task) < 2:
        for sch_id in task:
            score_param_per_task_group_per_fold.update({f'sch_{sch_id}': scores[0]})
    else:
        for sch_id in task:
            score_param_per_task_group_per_fold.update({f'sch_{sch_id}': []})

        for i in range(0, len(scores)):
            for j in range(1, len(scores[i])):
                score_param_per_task_group_per_fold[f'sch_{task[j - 1]}'].append(scores[i][j])

    total_loss_per_task_group_per_fold = 0
    for t, MSE_list in score_param_per_task_group_per_fold.items():
        total_loss_per_task_group_per_fold += np.mean(MSE_list)

    return total_loss_per_task_group_per_fold


def final_model(task_hyperparameters, shared_hyperparameters, task_group_list, data_param_dict_for_specific_task,
                Number_of_Features, fold, group_no, run):
    data_param = copy.deepcopy(data_param_dict_for_specific_task[fold])

    filepath = f'SavedModels/School_QR_Run_{run}_Group_{group_no}_{fold}.h5'

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

    for sch_id in task_group_list:
        preprocessed_ff = MTL_model_param[f'sch_{sch_id}_fold_{fold}_ff_preprocessing_model']
        shared_FF = shared_module_param_FF[f'FF_0_fold_{fold}'](preprocessed_ff)

        for h in range(1, shared_hyperparameters['shared_FF_Layers']):
            shared_FF = shared_module_param_FF[f'FF_{h}_fold_{fold}'](shared_FF)

        MTL_model_param[f'sch_{sch_id}_fold_{fold}_last_hidden_layer'] = shared_FF

    # output Neurons
    output_layers = []

    for sch_id in task_group_list:
        shared_model = Model(inputs=MTL_model_param[f'sch_{sch_id}_fold_{fold}_Input_FF'],
                             outputs=MTL_model_param[f'sch_{sch_id}_fold_{fold}_last_hidden_layer'])
        combined_input = concatenate([MTL_model_param[f'sch_{sch_id}_fold_{fold}_Input_FF'], shared_model.output])
        outputlayer = Dense(1, activation='linear', name=f'ExamScore_{sch_id}')(shared_model.output)
        output_layers.append(outputlayer)

    finalModel = Model(inputs=input_layers, outputs=output_layers)


    opt = tf.keras.optimizers.Adam(learning_rate=shared_hyperparameters['learning_rate'])
    finalModel.compile(optimizer=opt, loss='mse')

    checkpoint = ModelCheckpoint(filepath, verbose=0, monitor='val_loss', save_best_only=True, mode='auto')
    number_of_epoch = 400
    history = finalModel.fit(x=train_data,
                             y=train_label,
                             # shuffle=True,
                             epochs=number_of_epoch,
                             batch_size=32,
                             validation_data=(test_data,
                                              test_label),
                             callbacks=checkpoint,
                             verbose=0)

    finalModel = tf.keras.models.load_model(filepath)
    scores = finalModel.evaluate(test_data, test_label, verbose=0)

    if os.path.exists(filepath):
        os.remove(os.path.join(filepath))
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
                                                           'postprocessing_FF_Neurons': []
                                                           }})

        TASK_Specific_Arch.update({group_no: initial_task_specific_architecture})
    return TASK_Specific_Arch

def mutate_groups(new_task_group,new_group_score):

    task_rand = random.sample(TASKS, 1)
    task_rand = task_rand[0]
    changed_group = []

    # print(f'task_rand = {task_rand}\nTask-Group = {new_task_group.keys()}\nGroup-Score = {new_group_score.keys()}')

    # find out old group
    for key, task_list in new_task_group.items():
        if task_rand in task_list:
            g_old = key

    # check if old group is empty->delete the old group and assign task to new group
    # print(f'g_old = {g_old}')

    if len(new_task_group[g_old]) == 1:
        del new_task_group[g_old]
        del new_group_score[g_old]
        g_new = random.choice(list(new_task_group.keys()))
        new_task_group[g_new].append(task_rand)
        changed_group.append(g_new)

    else:
        new_task_group[g_old].remove(task_rand)
        g_new = random.choice([g for g in range(len(new_task_group) + 1) if g != g_old])

        if g_new not in new_task_group.keys():
            new_task_group.update({g_new: list([task_rand])})
        else:
            new_task_group[g_new].append(task_rand)
        changed_group = [g_old, g_new]

    return changed_group, task_rand
def predictor_network(x_train, y_train):

    network_architecture = {'FF_Layers': 4, 'FF_Neurons': [88, 30, 44, 90], 'learning_rate': 0.0012340955805592241,
            'activation_function': 'tanh', 'output_activation': 'linear',
            'Features': ['pairwise_improvement_average', 'group_dataset_size', 'pairwise_improvement_variance',
                         'group_variance', 'number_of_tasks', 'group_stddev', 'pairwise_improvement_stddev',
                         'group_distance', 'pairwise_Weight_average']}

    number_of_epoch = 200
    filepath = f'{run_results}/SavedModels/{datasetName}_TG_predictor_Best.h5'

    number_of_features = np.shape(x_train)[1]

    Input_FF = tf.keras.layers.Input(shape=(number_of_features,))
    hidden_FF = tf.keras.layers.Dense(network_architecture['FF_Neurons'][0],
                                      activation=network_architecture['activation_function'])(Input_FF)
    for h in range(1, network_architecture['FF_Layers']):
        hidden_FF = tf.keras.layers.Dense(network_architecture['FF_Neurons'][h],
                                              activation=network_architecture['activation_function'])(hidden_FF)

    output = tf.keras.layers.Dense(1, activation=network_architecture['output_activation'])(hidden_FF)
    # output = tf.keras.layers.Dense(1, activation='linear')(hidden_FF)

    finalModel = Model(inputs=Input_FF, outputs=output)
    opt = tf.keras.optimizers.Adam(learning_rate=network_architecture['learning_rate'])
    finalModel.compile(optimizer=opt, loss='mse')
    # print(finalModel.summary())

    checkpoint = ModelCheckpoint(filepath, verbose=0, monitor='val_loss', save_best_only=True, mode='auto')

    history = finalModel.fit(x=x_train,
                 y=y_train,
                 shuffle=True,
                 epochs=number_of_epoch,
                 batch_size=16,
                 callbacks=checkpoint,
                validation_split=0.2,
                 verbose=0)


def predict_performance_of_new_group(tasks):
    if len(tasks) > 1:
        number_of_tasks = []
        pairwise_improvement_average = []
        pairwise_improvement_variance = []
        pairwise_improvement_stddev = []
        group_variance = []
        group_stddev = []
        group_distance = []
        group_dataset_size = []

        pairwise_ITA_average = []
        pairwise_Weight_average = []
        number_of_tasks.append(len(tasks))
        sample_size = 0
        avg_var = []
        avg_stddev = []
        avg_dist = []
        task_combo = list(itertools.combinations(sorted(tasks), 2))
        for task_pair in task_combo:
            task_dist_data = task_distance_info[(task_distance_info.Task_1 == task_pair[0]) & (
                    task_distance_info.Task_2 == task_pair[1])].reset_index()
            if len(task_dist_data) == 0:
                task_dist_data = task_distance_info[(task_distance_info.Task_1 == task_pair[1]) & (
                        task_distance_info.Task_2 == task_pair[0])].reset_index()

            if datasetName == 'Chemical':
                avg_dist.append(task_dist_data.Average_Hamming_Distance[0])
            else:
                avg_dist.append(task_dist_data.Average_Euclidean_Distance[0])

        group_distance.append(np.mean(avg_dist))
        paired_improvement = []

        paired_weight = []
        for pair in task_combo:
            stl_loss = 0
            stl_loss += Single_res_dict[pair[0]]
            stl_loss += Single_res_dict[pair[1]]
            pair_specific = pair_results[
                (pair_results.Task_1 == pair[0]) & (pair_results.Task_2 == pair[1])].reset_index()
            if len(pair_specific) == 0:
                pair_specific = pair_results[
                    (pair_results.Task_1 == pair[1]) & (pair_results.Task_2 == pair[0])].reset_index()
            paired_improvement.append((stl_loss - pair_specific.Total_Loss[0]) / stl_loss)
            p = tuple(sorted([pair[0], pair[1]]))
            # pair_idx = list(ITA_data['Pairs']).index(str(p))
            # paired_ITA.append(list(ITA_data.Pairwise_ITA)[pair_idx])

            pair_idx = list(Weight_Matrix['Pairs']).index(str(p))
            paired_weight.append(list(Weight_Matrix.Weight)[pair_idx])

        pairwise_improvement_average.append(np.mean(paired_improvement))
        pairwise_improvement_variance.append(np.var(paired_improvement))
        pairwise_improvement_stddev.append(np.std(paired_improvement))
        # pairwise_ITA_average.append(np.mean(paired_ITA))
        pairwise_Weight_average.append(np.mean(paired_weight))

        single_task_total_loss = 0
        for t in tasks:
            single_task_total_loss += Single_res_dict[t]
            sample_size += task_len[t]
            avg_var.append(variance_dict[t])
            avg_stddev.append(std_dev_dict[t])

        group_variance.append(np.mean(avg_var))
        group_stddev.append(np.mean(avg_stddev))
        group_dataset_size.append(sample_size / len(TASKS))

        new_groups = pd.DataFrame({
            'group_dataset_size': group_dataset_size,
            'group_variance': group_variance,
            'group_stddev': group_stddev,
            'group_distance': group_distance,
            'number_of_tasks': number_of_tasks,
            'pairwise_improvement_average': pairwise_improvement_average,
            'pairwise_improvement_variance': pairwise_improvement_variance,
            'pairwise_improvement_stddev': pairwise_improvement_stddev,
            # 'pairwise_ITA_average': pairwise_ITA_average,
            'pairwise_Weight_average': pairwise_Weight_average
        })

        # affinity_pred_arch = {'FF_Layers': 4, 'FF_Neurons': [88, 30, 44, 90], 'learning_rate': 0.0012340955805592241,
        #                       'activation_function': 'tanh', 'output_activation': 'linear',
        #                       'Features': ['pairwise_improvement_average', 'group_dataset_size',
        #                                    'pairwise_improvement_variance',
        #                                    'group_variance', 'number_of_tasks', 'group_stddev',
        #                                    'pairwise_improvement_stddev',
        #                                    'group_distance', 'pairwise_Weight_average']}

        affinity_pred_arch = {'FF_Layers': 4, 'FF_Neurons': [57, 23, 18, 53], 'learning_rate': 0.0016620815899787528, 'activation_function': 'sigmoid',
                              'output_activation': 'linear',
                              'Features': ['pairwise_improvement_average', 'group_dataset_size', 'pairwise_improvement_variance', 'group_variance', 'number_of_tasks']}

        pred_features = affinity_pred_arch['Features']
        new_groups = new_groups[pred_features]
        # print(f'new_groups.columns = {new_groups.columns}')

        x = np.array(new_groups, dtype='float32')
        filepath = f'{run_results}/SavedModels/{datasetName}_TG_predictor_Best.h5'
        finalModel = tf.keras.models.load_model(filepath)
        final_score = finalModel.predict(x, verbose=0)
        return final_score[0][0], single_task_total_loss
    else:
        return 0, Single_res_dict[tasks[0]]

def retrain_predictor(datasetName):
    predictor_data = pd.read_csv(f'{run_results}/Data_for_Predictor_{datasetName}_updated.csv')
    # affinity_pred_arch = {'FF_Layers': 4, 'FF_Neurons': [88, 30, 44, 90], 'learning_rate': 0.0012340955805592241,
    #         'activation_function': 'tanh', 'output_activation': 'linear',
    #         'Features': ['pairwise_improvement_average', 'group_dataset_size', 'pairwise_improvement_variance',
    #                      'group_variance', 'number_of_tasks', 'group_stddev', 'pairwise_improvement_stddev',
    #                      'group_distance', 'pairwise_Weight_average']}
    affinity_pred_arch = {'FF_Layers': 4, 'FF_Neurons': [57, 23, 18, 53], 'learning_rate': 0.0016620815899787528,
                          'activation_function': 'sigmoid',
                          'output_activation': 'linear',
                          'Features': ['pairwise_improvement_average', 'group_dataset_size',
                                       'pairwise_improvement_variance', 'group_variance', 'number_of_tasks']}

    pred_features = affinity_pred_arch['Features']

    print(f'\n\n******* Training Samples = {len(predictor_data)} *******\n\n')
    predictor_data.dropna(inplace=True)
    # predictor_data = predictor_data[[
    #     # 'group_variance', 'group_stddev', 'group_distance',
    #     'number_of_tasks',
    #            # 'group_dataset_size',
    #     'pairwise_improvement_average',
    #            'pairwise_improvement_variance', 'pairwise_improvement_stddev',
    #     # 'pairwise_ITA_average',
    #     'pairwise_Weight_average',
    #            'change']]



    Sample_Label = np.array(list(predictor_data.change), dtype=float)
    predictor_data = predictor_data[pred_features]
    # print(f'predictor_data.columns = {predictor_data.columns}')

    # DataSet = np.array(predictor_data, dtype=float)
    Sample_Inputs = np.array(predictor_data, dtype=float)


    predictor_network(Sample_Inputs,Sample_Label)

def predictor_data_prep(task_grouping_results, counter, run, dataset_name):
    task_info = pd.read_csv(f'{DataPath}Task_Information_{datasetName}.csv')
    task_distance_info = pd.read_csv(f'{DataPath}Task_Distance_{datasetName}.csv')
    single_results = pd.read_csv(f'{ResultPath}/STL/STL_{datasetName}_{modelname}.csv')
    pair_results = pd.read_csv(f'../Results/new_runs/{dataset_name}_Results_from_Pairwise_Training_ALL_NN.csv')


    Weight_Matrix = pd.read_csv(f'{ResultPath}/Weight_Matrix/Weight_Affinity_{datasetName}.csv',
                                low_memory=False)
    TASKS = [i for i in range(1, 140)]
    for Selected_Task in TASKS:
        task_data = task_info[task_info.Task_Name == Selected_Task].reset_index()
        task_len.update({Selected_Task: task_data.Dataset_Size[0]})
        variance_dict.update({Selected_Task: task_data.Variance[0]})
        std_dev_dict.update({Selected_Task: task_data.Std_Dev[0]})
        dist_dict.update({Selected_Task: task_data.Average_Euclidian_Distance_within_Task[0]})
        single_res = single_results[single_results.Task == Selected_Task].reset_index()
        Single_res_dict.update({Selected_Task: single_res.Loss_MSE[0]})

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
        if counter!=0 and counter != 'test_obj':

            Changed_Groups = ast.literal_eval(task_grouping_results.Changed_Groups[group])

            if Changed_Groups != None:
                for gr in Changed_Groups:
                    sample_size = 0
                    avg_var = []
                    avg_stddev = []
                    avg_dist = []
                    sum_loss_single_task = 0

                    if len(Task_Group[gr])<=1:
                        continue

                    task_combo = list(itertools.combinations(sorted(Task_Group[gr]), 2))
                    for task_pair in task_combo:
                        task_dist_data = task_distance_info[(task_distance_info.Task_1 == task_pair[0]) & (
                                task_distance_info.Task_2 == task_pair[1])].reset_index()

                        if len(task_dist_data) == 0:
                            task_dist_data = task_distance_info[(task_distance_info.Task_1 == task_pair[1]) & (
                                    task_distance_info.Task_2 == task_pair[0])].reset_index()

                        if dataset_name == 'Chemical':
                            avg_dist.append(task_dist_data.Average_Hamming_Distance[0])
                        else:
                            avg_dist.append(task_dist_data.Average_Euclidean_Distance[0])
                    # group_distance.append(np.mean(avg_dist))

                    paired_improvement = []
                    paired_ITA = []
                    paired_weight = []
                    for pair in task_combo:
                        stl_loss = 0
                        stl_loss += Single_res_dict[pair[0]]
                        stl_loss += Single_res_dict[pair[1]]
                        pair_specific = pair_results[
                            (pair_results.Task_1 == pair[0]) & (pair_results.Task_2 == pair[1])].reset_index()
                        if len(pair_specific) == 0:
                            pair_specific = pair_results[
                                (pair_results.Task_1 == pair[1]) & (pair_results.Task_2 == pair[0])].reset_index()
                        paired_improvement.append((stl_loss - pair_specific.Total_Loss[0]) / stl_loss)

                        p = tuple(sorted([pair[0], pair[1]]))
                        # pair_idx = list(ITA_data['Pairs']).index(str(p))
                        # paired_ITA.append(list(ITA_data.Pairwise_ITA)[pair_idx])

                        pair_idx = list(Weight_Matrix['Pairs']).index(str(p))
                        paired_weight.append(list(Weight_Matrix.Weight)[pair_idx])


                    pairwise_improvement_average.append(np.mean(paired_improvement))
                    pairwise_improvement_variance.append(np.var(paired_improvement))
                    pairwise_improvement_stddev.append(np.std(paired_improvement))





                    # pairwise_ITA_average.append(np.mean(paired_ITA))
                    pairwise_Weight_average.append(np.mean(paired_weight))

                    for t in Task_Group[gr]:
                        sample_size += task_len[t]
                        avg_var.append(variance_dict[t])
                        avg_stddev.append(std_dev_dict[t])
                        sum_loss_single_task += Single_res_dict[t]
                    group_info.append(Task_Group[gr])
                    group_loss.append(Individual_Group_Score[gr])
                    number_of_tasks.append(len(Task_Group[gr]))
                    group_dataset_size.append(sample_size / len(Task_Group[gr]))  # individual length

                    group_distance.append(np.mean(avg_dist))
                    group_variance.append(np.mean(avg_var))
                    group_stddev.append(np.mean(avg_stddev))

                    # group_distance.append(np.mean(avg_dist))
                    single_task_loss.append(sum_loss_single_task)

                    change.append((sum_loss_single_task - Individual_Group_Score[gr]) / sum_loss_single_task)
        else:
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
                    task_dist_data = task_distance_info[(task_distance_info.Task_1 == task_pair[0]) & (
                            task_distance_info.Task_2 == task_pair[1])].reset_index()
                    if len(task_dist_data) == 0:
                        task_dist_data = task_distance_info[(task_distance_info.Task_1 == task_pair[1]) & (
                                task_distance_info.Task_2 == task_pair[0])].reset_index()

                    if dataset_name == 'Chemical':
                        avg_dist.append(task_dist_data.Average_Hamming_Distance[0])
                    else:
                        avg_dist.append(task_dist_data.Average_Euclidean_Distance[0])


                paired_improvement = []
                paired_ITA = []
                paired_weight = []
                for pair in task_combo:
                    stl_loss = 0
                    stl_loss += Single_res_dict[pair[0]]
                    stl_loss += Single_res_dict[pair[1]]
                    pair_specific = pair_results[
                        (pair_results.Task_1 == pair[0]) & (pair_results.Task_2 == pair[1])].reset_index()
                    if len(pair_specific) == 0:
                        pair_specific = pair_results[
                            (pair_results.Task_1 == pair[1]) & (pair_results.Task_2 == pair[0])].reset_index()
                    paired_improvement.append((stl_loss - pair_specific.Total_Loss[0]) / stl_loss)

                    p = tuple(sorted([pair[0], pair[1]]))
                    # pair_idx = list(ITA_data['Pairs']).index(str(p))
                    # paired_ITA.append(list(ITA_data.Pairwise_ITA)[pair_idx])

                    pair_idx = list(Weight_Matrix['Pairs']).index(str(p))
                    paired_weight.append(list(Weight_Matrix.Weight)[pair_idx])

                # pairwise_ITA_average.append(np.mean(paired_ITA))
                pairwise_Weight_average.append(np.mean(paired_weight))


                pairwise_improvement_average.append(np.mean(paired_improvement))
                pairwise_improvement_variance.append(np.var(paired_improvement))
                pairwise_improvement_stddev.append(np.std(paired_improvement))
                for t in tasks:
                    sample_size += task_len[t]
                    avg_var.append(variance_dict[t])
                    avg_stddev.append(std_dev_dict[t])
                    sum_loss_single_task += Single_res_dict[t]
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

    predictor_data.to_csv(f'{run_results}/Data_for_Predictor_{dataset_name}_new_updated.csv', index=False)

    '''Save the data for the first time'''
    if counter == 0:
        predictor_data.to_csv(f'{run_results}/Data_for_Predictor_{dataset_name}_first.csv', index=False)
        predictor_data.to_csv(f'{run_results}/Data_for_Predictor_{dataset_name}_updated.csv', index=False)

    elif counter == 'rerun':
        old_file = f'{run_results}/Data_for_Predictor_{dataset_name}_first.csv'
        if os.path.exists(old_file):
            df_1 = pd.read_csv(f'{run_results}/Data_for_Predictor_{dataset_name}_first.csv')
            df_2 = pd.read_csv(f'{run_results}/Data_for_Predictor_{dataset_name}_new_updated.csv')
            frames = [df_1, df_2]

            result = pd.concat(frames)
            result.to_csv(f'{run_results}/Data_for_Predictor_{dataset_name}_updated.csv', index=False)

    else:
        old_file = f'{run_results}/Data_for_Predictor_{dataset_name}_updated.csv'
        if os.path.exists(old_file):
            df_1 = pd.read_csv(f'{run_results}/Data_for_Predictor_{dataset_name}_updated.csv')
            df_2 = pd.read_csv(f'{run_results}/Data_for_Predictor_{dataset_name}_new_updated.csv')
            frames = [df_1, df_2]

            result = pd.concat(frames)
            result.to_csv(f'{run_results}/Data_for_Predictor_{dataset_name}_updated.csv', index=False)

if __name__ == '__main__':

    import sys
    run = int(sys.argv[1])
    rerun = int(sys.argv[2])
    rerun_counter = int(sys.argv[3])
    # run = 1
    # rerun = 0
    num_folds = 10
    # initial_shared_architecture = {'adaptive_FF_neurons': 4, 'shared_FF_Layers': 1, 'shared_FF_Neurons': [1],
    #                                'learning_rate': 0.00779959}
    initial_shared_architecture = {'adaptive_FF_neurons': 4, 'shared_FF_Layers': 2, 'shared_FF_Neurons': [3, 2],
                                   'learning_rate': 0.033806674289462206,
                                   # 'activation_func' :'tanh'
                                   }

    '''Global Files and Data for Task-Grouping Predictor'''
    task_len = {}
    variance_dict = {}
    std_dev_dict = {}
    dist_dict = {}
    Single_res_dict = {}

    datasetName = 'School'
    DataPath = f'../Dataset/{datasetName.upper()}/'
    ResultPath = '../Results'
    run_results = f'../Results/Run_{run}'
    if not os.path.exists(run_results):
        os.mkdir(run_results)

    modelname = 'NN'
    task_info = pd.read_csv(f'{DataPath}Task_Information_{datasetName}.csv')
    task_distance_info = pd.read_csv(f'{DataPath}Task_Distance_{datasetName}.csv')
    single_results = pd.read_csv(f'{ResultPath}/STL/STL_{datasetName}_{modelname}.csv')
    # pair_results = pd.read_csv(f'{ResultPath}/Pairwise/{modelname}/{datasetName}_Results_from_Pairwise_Training_ALL_{modelname}.csv')

    pair_results = pd.read_csv(f'../Results/new_runs/{datasetName}_Results_from_Pairwise_Training_ALL_NN.csv')

    Weight_Matrix = pd.read_csv(f'{ResultPath}/Weight_Matrix/Weight_Affinity_{datasetName}.csv',
                                low_memory=False)
    TASKS = [i for i in range(1, 140)]
    for Selected_Task in TASKS:
        task_data = task_info[task_info.Task_Name == Selected_Task].reset_index()
        task_len.update({Selected_Task: task_data.Dataset_Size[0]})
        variance_dict.update({Selected_Task: task_data.Variance[0]})
        std_dev_dict.update({Selected_Task: task_data.Std_Dev[0]})
        dist_dict.update({Selected_Task: task_data.Average_Euclidian_Distance_within_Task[0]})
        single_res = single_results[single_results.Task == Selected_Task].reset_index()
        Single_res_dict.update({Selected_Task: single_res.Loss_MSE[0]})

    Pairwise_res_dict = {}
    Task1 = list(pair_results.Task_1)
    Task2 = list(pair_results.Task_2)
    Pairs = [(Task1[i], Task2[i]) for i in range(len(Task1))]
    for p in Pairs:
        task1 = p[0]
        task2 = p[1]
        pair_res = pair_results[(pair_results.Task_1 == task1) & (pair_results.Task_2 == task2)].reset_index()
        Pairwise_res_dict.update({p: pair_res.Total_Loss[0]})

    print(len(Single_res_dict), len(Pairwise_res_dict))
    ''''''
    switch_architecture = ['ACCEPT', 'REJECT']

    Iteration_MTL = []
    Task_group = []

    Total_Loss = []
    Group_No = []
    Trainable_Param = []
    Individual_Group_Score = []
    Individual_Task_Score = []
    Group_Score = []
    Prev_Solution = []
    Prev_iter = []
    Switch = []
    Changed_Group = []
    Number_of_Groups = []
    Random_Task = []
    Predicted_Score = []
    STL_Score = []
    TRAIN_Model = []
    Fast_Reject = []
    Training_Time = []
    Number_of_MTL_Training = []
    Predicted_Improvement = []
    actual_improvement_values = []
    predicted_improvement_values = []
    Prev_Groups = {}

    '''Predictor Functions'''
    if rerun!=1:
        initial_results =    pd.read_csv(f'../Results/partition_sample/{modelname}/{datasetName}_partition_sample_MTL_Final.csv',low_memory=False)

        initial_results = initial_results.sort_values(by=['Total_Loss'], ascending=True).reset_index(drop=True)
        old_file = f'{run_results}/Data_for_Predictor_{datasetName}_first.csv'
        new_file = f'{run_results}/Data_for_Predictor_{datasetName}_updated.csv'
        if os.path.exists(old_file):
            os.remove(os.path.join(old_file))
        if os.path.exists(new_file):
            os.remove(os.path.join(new_file))

        predictor_data_prep(initial_results, 0, run, datasetName)
    else:

        initial_results = pd.read_csv(
            f'{run_results}/{datasetName}_Task_Grouping_Results_{rerun_counter}_run_{run}_{modelname}.csv',
            low_memory=False)
        predictor_data_prep(initial_results, 'rerun', run, datasetName)



    time_pred_Start = time.time()
    retrain_predictor(datasetName)
    print(f'Time to train predictor = {(time.time()-time_pred_Start)/60} min')





    for best_group_index in range(len(initial_results)):
        task_group = ast.literal_eval(initial_results.Task_group[best_group_index])
        group_score = ast.literal_eval(initial_results.Individual_Group_Score[best_group_index])

        for group_no, task in task_group.items():
            grouping_sorted = tuple(sorted(task))
            if grouping_sorted not in Prev_Groups.keys():
                Prev_Groups.update({grouping_sorted: 0})

        for group_no, score in group_score.items():
            t = tuple(sorted(task_group[group_no]))
            Prev_Groups[t]=score

    # print(f'Prev Groups = {len(Prev_Groups)}')

    if rerun != 1:
        best_group_index = int(run)-1
    else:
        group_score = list(initial_results.Total_Loss)
        best_group_index = group_score.index(min(group_score))

    task_group = ast.literal_eval(initial_results.Task_group[best_group_index])
    group_score = ast.literal_eval(initial_results.Individual_Group_Score[best_group_index])
    prev_solution = sum(list(group_score.values()))
    print(f'Task Group = {task_group}')


    if rerun==1:
        Switch = list(initial_results.ACCEPT)
        Changed_Group = list(initial_results.Changed_Groups)
        Prev_Solution = list(initial_results.Prev_Solution)
        Prev_iter = list(initial_results.Last_switch_at_iter)
        Training_Time = list(initial_results.Training_Time)
        Iteration_MTL = list(initial_results.Iteration_MTL)
        Random_Task = list(initial_results.Random_Task)
        Total_Loss = list(initial_results.Total_Loss)
        Task_group = list(initial_results.Task_group)
        Number_of_Groups = list(initial_results.Number_of_Groups)
        Predicted_Score = list(initial_results.Predicted_Score)
        Fast_Reject = list(initial_results.Fast_Reject)
        Individual_Group_Score = list(initial_results.Individual_Group_Score)
        Number_of_MTL_Training = list(initial_results.Number_of_MTL_Training)
        Predicted_Improvement = list(initial_results.Predicted_Improvement)
        actual_improvement_values= list(initial_results.Actual_Improvement)
        predicted_improvement_values= list(initial_results.predicted_improvement_values)
        print(len(actual_improvement_values), len(predicted_improvement_values),len(Switch))


    else:
        Switch.append('None')
        Changed_Group.append('None')
        Prev_Solution.append(prev_solution)
        Prev_iter.append(0)
        Training_Time.append('None')
        Iteration_MTL.append(0)
        Random_Task.append('None')
        Total_Loss.append(prev_solution)
        Task_group.append(task_group)
        Number_of_Groups.append(len(task_group))
        Predicted_Score.append('None')
        Fast_Reject.append('None')
        Predicted_Improvement.append('None')
        Individual_Group_Score.append(group_score)
        Number_of_MTL_Training.append(len(group_score))
        actual_improvement_values.append(0)
        predicted_improvement_values.append(0)


    print(len(actual_improvement_values), len(predicted_improvement_values),len(Switch),len(Total_Loss))

    iter_max = 10000000
    last_iter = 0

    current_best_group = copy.deepcopy(task_group)
    current_best_group_score = copy.deepcopy(group_score)
    current_best_prev_solution = prev_solution

    print(f'Initial Solution = {prev_solution}, Number of Groups = {len(task_group)}')

    timeStamp = time.time()
    x = 0

    if rerun==1:
        init_range = Iteration_MTL[-1]
        old_length = len(initial_results)

    else:
        init_range = 1
        old_length = 0

    for iteration_MTL in range(init_range, iter_max):
        predicted_performance = {}
        single_loss_dict = {}

        new_task_group = copy.deepcopy(task_group)
        new_group_score = copy.deepcopy(group_score)
        predicted_group_score = copy.deepcopy(group_score)

        changed_group,task_rand = mutate_groups(new_task_group, new_group_score)

        if len(changed_group) == 0:
            x += 1
            continue

        changed_group = sorted(changed_group)
        time_query_Start = time.time()
        tmp_predicted_improvement = {}
        tmp_stl_score = {}
        for group_no in changed_group:
            prediction_of_performance = predict_performance_of_new_group(new_task_group[group_no])
            # print(f'Prediction of performance = {prediction_of_performance}')

            predicted_performance[group_no] = prediction_of_performance[0]
            single_loss_dict[group_no] = prediction_of_performance[1]
            tmp_predicted_improvement[group_no] = prediction_of_performance[0]
            tmp_stl_score[group_no] = prediction_of_performance[1]

        for group_no, task in new_task_group.items():
            if group_no in changed_group:
                total_loss_per_task_group_per_fold = single_loss_dict[group_no] * (
                        1 - predicted_performance[group_no])
                predicted_group_score[group_no] = total_loss_per_task_group_per_fold
                # print(f'{1 - predicted_performance[group_no]}')
                # print(f'Predicted Group Score for group {group_no} = {predicted_group_score[group_no]}')
        # print(f'Time to query predictor = {(time.time() - time_query_Start) / 60} min')


        predicted_solution = sum(list(predicted_group_score.values()))


        K = 5

        if predicted_solution<prev_solution:
            PROBABILITIES_ACCEPT = [0.99, 0.01]
        else:
            PROBABILITIES_ACCEPT = [0, 1]

        fast_reject = np.random.choice(switch_architecture, 1, p=PROBABILITIES_ACCEPT)
        if fast_reject[0] == 'ACCEPT':
            train_model = 1
        else:
            if len(Switch) < 50:
                train_model = np.random.choice([1, 0], 1, p=[0.1, 0.9])
            elif 50 < len(Switch) < 100:
                train_model = np.random.choice([1, 0], 1, p=[0.05, 0.95])
            else:
                train_model = np.random.choice([1, 0], 1, p=[0.005, 0.995])
            train_model = train_model[0]

        if train_model == 1:
            print(f'predicted_improvement_values = {tmp_predicted_improvement}')
            # print(f'predicted_solution = {predicted_solution}')
            Train_MTL_Model = True
            TASK_Specific_Arch = prep_task_specific_arch(new_task_group)
            random_seed = random.randint(0, 100)

            count = 0
            timeStart = time.time()
            actual_improvement = {}
            for group_no, task in new_task_group.items():
                if group_no in changed_group:
                    grouping_sorted = tuple(sorted(task))
                    if grouping_sorted not in Prev_Groups.keys():
                        if len(task) == 1:
                            total_loss_per_task_group_per_fold = Single_res_dict[task[0]]

                        elif len(task) == 2:
                            total_loss_per_task_group_per_fold = Pairwise_res_dict[grouping_sorted]
                        else:
                            count += 1
                            print(f'changed group = {len(new_task_group[group_no])}')
                            args = kFold_validation(TASK_Specific_Arch[group_no],
                                                    initial_shared_architecture,
                                                    new_task_group[group_no], group_no,
                                                    random_seed)
                            with ThreadPool(20) as tp:
                                all_scores = tp.starmap(final_model, args)
                            tp.join()
                            # number_of_pools = len(args)+5
                            # pool = mp.Pool(number_of_pools)
                            # all_scores = pool.starmap(final_model, args)
                            # pool.close()

                            total_loss_per_task_group_per_fold = sort_Results(all_scores, new_task_group[group_no])
                            print(f'Loss from MTL for group {group_no} = {total_loss_per_task_group_per_fold}')
                            actual_improvement[group_no] = (tmp_stl_score[group_no] - total_loss_per_task_group_per_fold)/tmp_stl_score[group_no]

                        if group_no not in new_group_score.keys():
                            new_group_score.update({group_no: total_loss_per_task_group_per_fold})
                        else:
                            new_group_score[group_no] = total_loss_per_task_group_per_fold
                        Prev_Groups[grouping_sorted] = total_loss_per_task_group_per_fold
                    else:
                        loss = Prev_Groups[grouping_sorted]
                        if group_no not in new_group_score.keys():
                            new_group_score.update({group_no: loss})
                        else:
                            new_group_score[group_no] = loss

            final_time = time.time() - timeStart
            print(f'actual_improvement = {actual_improvement}')
            # print(f'time to train mtl = {final_time/60}')

            mtl_solution = sum(list(new_group_score.values()))

            Accept_Probability = min(1, math.exp((prev_solution - mtl_solution) * K))
            PROBABILITIES_ACCEPT = [Accept_Probability, max(0, 1 - Accept_Probability)]
            final_accept = np.random.choice(switch_architecture, 1, p=PROBABILITIES_ACCEPT)

            print(f'iteration = {iteration_MTL}\t mtl_solution = {mtl_solution}\t final_accept = {final_accept[0]}\t predicted_solution = {predicted_solution}\t predictor_dcssn = {fast_reject[0]}')
            for g, s in new_group_score.items():
                if s == 0:
                    print(f'new_group_score = {new_group_score}')
                    print(f'Exiting because of error')
                    exit(0)

            if len(new_task_group.keys()) != len(new_group_score.keys()):
                print(f'new_group_score = {new_group_score}\nnew_task_group = {new_task_group}')
                print(f'Exiting because of key-mismatch error')
                exit(0)

            if final_accept[0] == 'ACCEPT':
                group_score = copy.deepcopy(new_group_score)
                task_group = copy.deepcopy(new_task_group)

                Switch.append('yes')
                prev_solution = mtl_solution
                Prev_Solution.append(prev_solution)
                Prev_iter.append(iteration_MTL)
                last_iter = iteration_MTL

            else:
                Switch.append('no')
                Prev_Solution.append(prev_solution)
                Prev_iter.append(last_iter)

            Training_Time.append(final_time)
            Iteration_MTL.append(iteration_MTL)
            Random_Task.append(task_rand)
            Total_Loss.append(mtl_solution)
            Task_group.append(new_task_group)
            Number_of_Groups.append(len(new_task_group))
            Changed_Group.append(changed_group)
            Predicted_Score.append(predicted_solution)
            Fast_Reject.append(fast_reject[0])
            Predicted_Improvement.append(predicted_performance.values())
            Individual_Group_Score.append(new_group_score)
            Number_of_MTL_Training.append(count)
            actual_improvement_values.append(list(actual_improvement.values()))
            predicted_improvement_values.append(list(tmp_predicted_improvement.values()))

            length = len(Switch)
            print(f'Length = {length}')

        length = len(Switch)
        tail_pointer = 5
        if length > old_length and length % tail_pointer == 0:
            print(len(Switch), len(Iteration_MTL), len(Prev_Solution), len(Changed_Group), len(Total_Loss),
                  len(Number_of_Groups), len(Prev_iter), len(Task_group), len(Individual_Group_Score),
                  len(Individual_Task_Score), len(Predicted_Score), len(Predicted_Improvement),len(TRAIN_Model), len(Fast_Reject),
                  len(Training_Time),len(Number_of_MTL_Training), len(predicted_improvement_values), len(actual_improvement_values))
            results = pd.DataFrame({'Iteration_MTL': Iteration_MTL,
                                    'Random_Task': Random_Task,
                                    'ACCEPT': Switch,
                                    'Changed_Groups': Changed_Group,
                                    'Total_Loss': Total_Loss,
                                    'Prev_Solution': Prev_Solution,
                                    'Training_Time': Training_Time,
                                    'Number_of_Groups': Number_of_Groups,
                                    'Number_of_MTL_Training': Number_of_MTL_Training,
                                    'Fast_Reject': Fast_Reject,
                                    'Task_group': Task_group,
                                    'Individual_Group_Score': Individual_Group_Score,
                                    'Predicted_Score': Predicted_Score,
                                    'Predicted_Improvement': Predicted_Improvement,
                                    "Last_switch_at_iter": Prev_iter,
                                    'Actual_Improvement': actual_improvement_values,
                                    'predicted_improvement_values': predicted_improvement_values,})
            results.to_csv(f'{run_results}/{datasetName}_Task_Grouping_Results_{length}_run_{run}_{modelname}.csv',
                           index=False)
            old_length = len(Switch)

            new_results = pd.read_csv(f'{run_results}/{datasetName}_Task_Grouping_Results_{length}_run_{run}_{modelname}.csv',
                                      low_memory=False)

            new_results = new_results.tail(tail_pointer).reset_index(drop=True)

            # new_object = TaskGroupingPredictor(datasetName,1,run)
            # TaskGroupingPredictor.predictor_data_prep(new_object, new_results)
            predictor_data_prep(new_results, 1, run, datasetName)
            retrain_predictor(datasetName)

            # if length > 5:
            length = length - tail_pointer
            old_file = f'{run_results}/{datasetName}_Task_Grouping_Results_{length}_run_{run}_{modelname}.csv'
            if os.path.exists(old_file):
                os.remove(os.path.join(f'{run_results}/{datasetName}_Task_Grouping_Results_{length}_run_{run}_{modelname}.csv'))

        if np.sum(Number_of_MTL_Training) >= 1000:
            break
    print(len(Switch), len(Random_Task), len(Iteration_MTL), len(Prev_Solution), len(Changed_Group), len(Total_Loss),
          len(Number_of_Groups), len(Prev_iter), len(Task_group), len(Individual_Group_Score),
          len(Individual_Task_Score))

    results = pd.DataFrame({'Iteration_MTL': Iteration_MTL,
                            'Random_Task': Random_Task,
                            'ACCEPT': Switch,
                            'Changed_Groups': Changed_Group,
                            'Total_Loss': Total_Loss,
                            'Prev_Solution': Prev_Solution,
                            'Training_Time': Training_Time,
                            'Number_of_Groups': Number_of_Groups,
                            'Number_of_MTL_Training': Number_of_MTL_Training,
                            'Fast_Reject': Fast_Reject,
                            'Task_group': Task_group,
                            'Individual_Group_Score': Individual_Group_Score,
                            'Predicted_Score': Predicted_Score,
                            'Predicted_Improvement': Predicted_Improvement,
                            "Last_switch_at_iter": Prev_iter,
                            'Actual_Improvement': actual_improvement_values,
                            'predicted_improvement_values': predicted_improvement_values,
                            })
    results.to_csv(f'{run_results}/{datasetName}_Task_Grouping_Results_run_{run}_{modelname}.csv',
                   index=False)

