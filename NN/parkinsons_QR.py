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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.layers import *
# print(f'version = {tf.__version__}')
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model, backend
from sklearn.metrics import explained_variance_score, r2_score

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


def readData(TASKS):
    data_param_dictionary = {}
    for subj_id in TASKS:
        # csv = (f"{DataPath}DATA/parkinsons_subject_{subj_id}_for_MTL.csv")
        csv = (f"{DataPath}parkinsons_subject_{subj_id}.csv")

        df = pd.read_csv(csv, low_memory=False)
        # print(df.columns)
        # exit(0)

        df = df[[
            # 'age', 'sex', 'test_time',
            'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP', 'Shimmer',
            'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11',
            'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE',
            'motor_UPDRS', 'total_UPDRS']]


        target1 = list(df.pop('motor_UPDRS'))
        target2 = list(df.pop('total_UPDRS'))

        # df = (df - df.min()) / (df.max() - df.min())
        Sample_Label = []
        for t1, t2 in zip(target1, target2):
            Sample_Label.append([t1, t2])

        Sample_Label = np.array(Sample_Label)

        Sample_Label_target_1 = np.array(target1)
        Sample_Label_target_2 = np.array(target2)

        Sample_Inputs = np.array(df)
        Number_of_Features = np.shape(Sample_Inputs)[1]

        data_param_dictionary.update({f'Subject_{subj_id}_FF_Inputs': Sample_Inputs})
        data_param_dictionary.update({f'Subject_{subj_id}_Label': Sample_Label})
        data_param_dictionary.update({f'Subject_{subj_id}_Label_1': Sample_Label_target_1})
        data_param_dictionary.update({f'Subject_{subj_id}_Label_2': Sample_Label_target_2})
        '''*********************************'''
    return data_param_dictionary,Number_of_Features


'''MTL for Task Groups Functions'''


def kFold_validation(current_task_specific_architecture, current_shared_architecture, task, group_no, random_seed):

    data_param_dictionary, Number_of_Features_FF = readData(task)

    data_param_dict_for_specific_task = {}

    max_size = 170
    train_set_size = math.floor(max_size * (1 - num_folds / 100))
    test_set_size = math.ceil(max_size * (num_folds / 100))

    for f in range(num_folds):
        data_param_dict_for_specific_task[f] = {}

    for subj_id in task:
        Sample_Inputs = data_param_dictionary[f'Subject_{subj_id}_FF_Inputs']
        Sample_Label = data_param_dictionary[f'Subject_{subj_id}_Label']
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)

        fold = 0
        for train, test in kfold.split(Sample_Inputs):
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

            data_param_dict_for_specific_task[f'Subject_{subj_id}_fold_{fold}_X_train'] = X_train
            data_param_dict_for_specific_task[f'Subject_{subj_id}_fold_{fold}_X_test'] = X_test

            data_param_dict_for_specific_task[f'Subject_{subj_id}_fold_{fold}_y_train'] = y_train
            data_param_dict_for_specific_task[f'Subject_{subj_id}_fold_{fold}_y_test'] = y_test

            fold += 1

    ALL_FOLDS = []
    for fold in range(0, 10):
        tmp = (
            current_task_specific_architecture, current_shared_architecture, task,
            data_param_dict_for_specific_task,
            Number_of_Features_FF,
            fold, group_no, run)

        ALL_FOLDS.append(tmp)

    number_of_models = 5
    current_idx = random.sample(range(len(ALL_FOLDS)), number_of_models)
    args = [ALL_FOLDS[index] for index in sorted(current_idx)]

    return args

def sort_results(all_scores):

    scores = []
    for i in range(len(all_scores)):
        scores.append(all_scores[i][0])

    score_param_per_task_group_per_fold = {}
    explained_var = {}
    r_square = {}
    if len(task) < 2:
        for subj_id in task:
            score_param_per_task_group_per_fold.update({f'sub_{subj_id}': scores[0]})
            explained_var.update({f'sub_{subj_id}': scores[1]})
            r_square.update({f'sub_{subj_id}': scores[2]})
    else:
        for subj_id in task:
            score_param_per_task_group_per_fold.update({f'sub_{subj_id}': []})
            explained_var.update({f'sub_{subj_id}': []})
            r_square.update({f'sub_{subj_id}': []})

        for i in range(0, len(scores)):
            for j in range(1, len(scores[i])):
                score_param_per_task_group_per_fold[f'sub_{task[j - 1]}'].append(scores[i][j])

    explained_var = []
    r_square = []
    for c in range(len(all_scores)):
        explained_var.append(all_scores[c][1])
        r_square.append(all_scores[c][2])

    total_loss_per_task_group_per_fold = 0
    for t, MSE_list in score_param_per_task_group_per_fold.items():
        total_loss_per_task_group_per_fold += np.mean(MSE_list)

    return total_loss_per_task_group_per_fold, np.mean(explained_var)




def final_model(task_hyperparameters, shared_hyperparameters, task_group_list, data_param_dict_for_specific_task,
                Number_of_Features, fold, group_no, run):
    data_param = copy.deepcopy(data_param_dict_for_specific_task)

    filepath = f'SavedModels/Parkinsons_group_{group_no}_{fold}_run_{run}.h5'

    MTL_model_param = {}
    shared_module_param_FF = {}

    input_layers = []

    train_data = []
    train_label = []
    test_data = []
    test_label = []

    activation_func = shared_hyperparameters['activation']

    for subj_id in task_group_list:
        train_data.append(data_param[f'Subject_{subj_id}_fold_{fold}_X_train'])
        train_label.append(data_param[f'Subject_{subj_id}_fold_{fold}_y_train'])
        test_data.append(data_param[f'Subject_{subj_id}_fold_{fold}_X_test'])
        test_label.append(data_param[f'Subject_{subj_id}_fold_{fold}_y_test'])

        # input = tf.constant(data_param[f'Subject_{subj_id}_fold_{fold}_X_train'])
        hyperparameters = copy.deepcopy(task_hyperparameters[subj_id])
        Input_FF = tf.keras.layers.Input(shape=(Number_of_Features,))
        input_layers.append(Input_FF)

        MTL_model_param[f'sub_{subj_id}_fold_{fold}_Input_FF'] = Input_FF
        if hyperparameters['preprocessing_FF_layers'] > 0:
            hidden_ff = Dense(hyperparameters['preprocessing_FF_Neurons'][0], activation=activation_func)(
                Input_FF)
            for h in range(1, hyperparameters['preprocessing_FF_layers']):
                hidden_ff = Dense(hyperparameters['preprocessing_FF_Neurons'][h], activation=activation_func)(hidden_ff)

            MTL_model_param[f'sub_{subj_id}_fold_{fold}_ff_preprocessing_model'] = hidden_ff

    for h in range(0, shared_hyperparameters['shared_FF_Layers']):
        shared_ff = tf.keras.layers.Dense(shared_hyperparameters['shared_FF_Neurons'][h], activation=activation_func)
        shared_module_param_FF[f'FF_{h}_fold_{fold}'] = shared_ff

    for subj_id in task_group_list:
        preprocessed_ff = MTL_model_param[f'sub_{subj_id}_fold_{fold}_ff_preprocessing_model']
        shared_FF = shared_module_param_FF[f'FF_0_fold_{fold}'](preprocessed_ff)

        for h in range(1, shared_hyperparameters['shared_FF_Layers']):
            shared_FF = shared_module_param_FF[f'FF_{h}_fold_{fold}'](shared_FF)

        MTL_model_param[f'sub_{subj_id}_fold_{fold}_last_hidden_layer'] = shared_FF

    # output Neurons
    output_layers = []

    for subj_id in task_group_list:
        shared_model = Model(inputs=MTL_model_param[f'sub_{subj_id}_fold_{fold}_Input_FF'],
                             outputs=MTL_model_param[f'sub_{subj_id}_fold_{fold}_last_hidden_layer'])

        outputlayer = Dense(2, activation='linear', name=f'Score_{subj_id}')(shared_model.output)
        output_layers.append(outputlayer)

    finalModel = Model(inputs=input_layers, outputs=output_layers)

    # from keras.utils.layer_utils import count_params
    # trainable_count = count_params(finalModel.trainable_weights)
    # print(f'fold = {fold}\tMTL = {len(MTL_model_param.keys())}\t trainable_count = {trainable_count}')

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
    # print(f'done')

    y_pred = finalModel.predict(test_data, verbose=0)
    tmp_explained_var = []
    tmp_R_square = []
    for idx in range(len(task_group_list)):
        tmp_explained_var.append(explained_variance_score(test_label[idx], y_pred[idx], multioutput='raw_values'))
        tmp_R_square.append(r2_score(test_label[idx], y_pred[idx]))

    explained_var = np.mean(tmp_explained_var)
    r_square = np.mean(tmp_R_square)

    # print(f'fold = {fold}\t explained_var = {explained_var}\t r_square = {r_square}')

    if os.path.exists(filepath):
        os.remove(os.path.join(filepath))

    return scores, explained_var, r_square


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


def mutate_groups(new_task_group, new_group_score):
    task_rand = random.sample(TASKS, 1)
    task_rand = task_rand[0]
    changed_group = []

    # find out old group
    for key, task_list in new_task_group.items():
        if task_rand in task_list:
            g_old = key

    # check if old group is empty->delete the old group and assign task to new group
    print(f'task_rand = {task_rand}\tg_old = {g_old}\tTask-Group = {new_task_group}')

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

        # new_grouping_sorted = tuple(sorted(new_task_group[g_new]))
        # old_grouping_sorted = tuple(sorted(new_task_group[g_old]))
        #
        # if old_grouping_sorted not in Prev_Groups.keys():
        #     changed_group.append(g_old)
        #     if g_old not in new_group_score.keys():
        #         new_group_score.update({g_old: 0})
        #
        # if new_grouping_sorted not in Prev_Groups.keys():
        #     changed_group.append(g_new)
        #     if g_new not in new_group_score.keys():
        #         new_group_score.update({g_new: 0})

    return changed_group, task_rand

def predictor_network(x_train, y_train):

    network_architecture = {'FF_Layers': 4, 'FF_Neurons': [39, 14, 57, 40], 'learning_rate': 0.0015235747908138567,
                              'activation_function': 'tanh', 'output_activation': 'tanh',
                              'Features': ['pairwise_improvement_average', 'pairwise_improvement_stddev',
                                           'group_variance',
                                           'group_distance', 'group_stddev']}

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

        # pairwise_ITA_average = []
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

        affinity_pred_arch = {'FF_Layers': 4, 'FF_Neurons': [39, 14, 57, 40], 'learning_rate': 0.0015235747908138567,
                              'activation_function': 'tanh', 'output_activation': 'tanh',
                              'Features': ['pairwise_improvement_average', 'pairwise_improvement_stddev',
                                           'group_variance',
                                           'group_distance', 'group_stddev']}

        pred_features = affinity_pred_arch['Features']
        new_groups = new_groups[pred_features]
        print(f'new_groups.columns = {new_groups.columns}')

        x = np.array(new_groups, dtype='float32')
        filepath = f'{run_results}/SavedModels/{datasetName}_TG_predictor_Best.h5'
        finalModel = tf.keras.models.load_model(filepath)
        final_score = finalModel.predict(x, verbose=0)
        return final_score[0][0], single_task_total_loss
    else:
        return 0, Single_res_dict[tasks[0]]


def retrain_predictor(datasetName):
    predictor_data = pd.read_csv(f'{run_results}/Data_for_Predictor_{datasetName}_updated.csv')
    affinity_pred_arch = {'FF_Layers': 4, 'FF_Neurons': [39, 14, 57, 40], 'learning_rate': 0.0015235747908138567,
            'activation_function': 'tanh', 'output_activation': 'tanh',
            'Features': ['pairwise_improvement_average', 'pairwise_improvement_stddev', 'group_variance',
                         'group_distance', 'group_stddev']}
    pred_features = affinity_pred_arch['Features']
    print(f'\n\n******* Training Samples = {len(predictor_data)} *******\n\n')
    predictor_data.dropna(inplace=True)
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
    # pair_results = pd.read_csv(
    #     f'{ResultPath}/Pairwise/{modelname}/{datasetName}_Results_from_Pairwise_Training_ALL_{modelname}.csv')

    pair_results = pd.read_csv(f'../Results/new_runs/{datasetName}_Results_from_Pairwise_Training_ALL_NN.csv')

    Weight_Matrix = pd.read_csv(f'{ResultPath}/Weight_Matrix/Weight_Affinity_{datasetName}.csv',
                                low_memory=False)
    TASKS = [i for i in range(1, 43)]
    for Selected_Task in TASKS:
        task_data = task_info[task_info.Task_Name == Selected_Task].reset_index()
        task_len.update({Selected_Task: task_data.Dataset_Size[0]})
        variance_dict.update({Selected_Task: task_data.Variance[0]})
        std_dev_dict.update({Selected_Task: task_data.Std_Dev[0]})
        dist_dict.update({Selected_Task: task_data.Average_Euclidian_Distance_within_Task[0]})
        single_res = single_results[single_results.Task == Selected_Task].reset_index()
        Single_res_dict.update({Selected_Task: single_res.LOSS[0]})

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
        if counter!=0:

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

    if rerun == 1:
        rerun = True
    else:
        rerun = False
    # run = 1


    num_folds = 10
    initial_shared_architecture = {'adaptive_FF_neurons': 4, 'shared_FF_Layers': 1, 'shared_FF_Neurons': [12],
                                   'learning_rate': 0.00779959, 'activation': 'sigmoid'}

    '''Global Files and Data for Task-Grouping Predictor'''
    task_len = {}
    variance_dict = {}
    std_dev_dict = {}
    dist_dict = {}
    Single_res_dict = {}
    STL_EV = {}
    STL_Rsq = {}

    datasetName = 'Parkinsons'
    DataPath = f'../Dataset/{datasetName.upper()}/'
    ResultPath = '../Results/'
    run_results = f'../Results/Run_{run}'
    if not os.path.exists(run_results):
        os.mkdir(run_results)

    modelname = 'NN'
    task_info = pd.read_csv(f'{DataPath}Task_Information_{datasetName}.csv')
    task_distance_info = pd.read_csv(f'{DataPath}Task_Distance_{datasetName}.csv')
    single_results = pd.read_csv(f'{ResultPath}/STL/STL_{datasetName}_{modelname}.csv')
    # pair_results = pd.read_csv(
    #     f'{ResultPath}/Pairwise/{modelname}/{datasetName}_Results_from_Pairwise_Training_ALL_{modelname}.csv')

    pair_results = pd.read_csv(f'../Results/new_runs/{datasetName}_Results_from_Pairwise_Training_ALL_NN.csv')

    Weight_Matrix = pd.read_csv(f'{ResultPath}/Weight_Matrix/Weight_Affinity_{datasetName}.csv',
                                low_memory=False)

    TASKS = [i for i in range(1, 43)]

    for Selected_Task in TASKS:
        task_data = task_info[task_info.Task_Name == Selected_Task].reset_index()
        task_len.update({Selected_Task: task_data.Dataset_Size[0]})
        variance_dict.update({Selected_Task: task_data.Variance[0]})
        std_dev_dict.update({Selected_Task: task_data.Std_Dev[0]})
        dist_dict.update({Selected_Task: task_data.Average_Euclidian_Distance_within_Task[0]})
        single_res = single_results[single_results.Task == Selected_Task].reset_index()
        Single_res_dict.update({Selected_Task: single_res.LOSS[0]})
        STL_EV.update({Selected_Task: single_res.Explained_Variance[0]})
        # STL_Rsq.update({Selected_Task: single_res.R_Square[0]})

    Pairwise_res_dict = {}
    PTL_EV = {}
    PTL_Rsq = {}
    Task1 = list(pair_results.Task_1)
    Task2 = list(pair_results.Task_2)
    Pairs = [(Task1[i], Task2[i]) for i in range(len(Task1))]
    for p in Pairs:
        task1 = p[0]
        task2 = p[1]
        pair_res = pair_results[(pair_results.Task_1 == task1) & (pair_results.Task_2 == task2)].reset_index()
        Pairwise_res_dict.update({p: pair_res.Total_Loss[0]})
        PTL_EV.update({p: pair_res.Explained_Var[0]})
        PTL_Rsq.update({p:pair_res.R_Square[0]})

    print(len(Single_res_dict), len(Pairwise_res_dict), len(PTL_EV),len(PTL_Rsq), len(STL_EV), len(STL_Rsq))

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
    # STL_Score = []
    # TRAIN_Model = []
    Fast_Reject = []
    Training_Time = []
    Number_of_MTL_Training = []
    # Explained_Variance = []
    R_Square = []


    actual_improvement_values = []
    predicted_improvement_values = []
    Prev_Groups = {}

    '''Predictor Functions'''
    if not rerun:
        # initial_results = pd.read_csv(f'../Results/partition_sample/{modelname}/{datasetName}_partition_sample_MTL.csv',
        #                               low_memory=False)
        initial_results = pd.read_csv(
            f'../Results/partition_sample/{modelname}/{datasetName}_partition_sample_MTL_Final.csv', low_memory=False)
        print(f'initial_results = {initial_results.columns}')

        old_file = f'{DataPath}Data_for_Predictor_{datasetName}_first.csv'
        new_file = f'{DataPath}Data_for_Predictor_{datasetName}_updated.csv'
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

    if not rerun:
        initial_results = initial_results.sort_values(by=['Total_Loss'], ascending=True).reset_index(drop=True)
        best_group_index = int(run) - 1
    else:
        for idx in range(len(initial_results),0,-1):
            if initial_results.ACCEPT[idx-1] == 'yes':
                best_group_index = idx-1
                break

        # score = list(initial_results.Total_Loss)
        # best_group_index = score.index(min(score))

    # for best_group_index in range(len(initial_results)):
    #     task_group = ast.literal_eval(initial_results.Task_group[best_group_index])
    #     group_score = ast.literal_eval(initial_results.Individual_Group_Score[best_group_index])
    #
    #     for group_no, task in task_group.items():
    #         grouping_sorted = tuple(sorted(task))
    #         if grouping_sorted not in Prev_Groups.keys():
    #             Prev_Groups.update({grouping_sorted: 0})
    #
    #     for group_no, score in group_score.items():
    #         t = tuple(sorted(task_group[group_no]))
    #         Prev_Groups[t]=score


    # best_group_index = int(run) - 1
    task_group = ast.literal_eval(initial_results.Task_group[best_group_index])
    group_score = ast.literal_eval(initial_results.Individual_Group_Score[best_group_index])
    prev_solution = sum(list(group_score.values()))
    # print(initial_results.Individual_Explained_Variance[best_group_index])
    # best_ev_idx = 0
    # best_ev = -math.inf


    # for group_no, task in task_group.items():
    #     grouping_sorted = tuple(sorted(task))
    #     if grouping_sorted not in Prev_Groups.keys():
    #         Prev_Groups.update({grouping_sorted: 0})

    for group_no, score in group_score.items():
        t = tuple(sorted(task_group[group_no]))
        Prev_Groups[t] = group_score[group_no]

    # print(f'Prev Groups = {len(Prev_Groups)}')
    # print(f'prev_solution = {prev_solution}')
    # print(f'group_score = {group_score}')
    print(sum(list(group_score.values())))
    print(f'Task Group = {task_group}')

    if not rerun:
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
        # Explained_Variance.append(group_ev)
        Predicted_Score.append('None')
        Fast_Reject.append('None')
        Individual_Group_Score.append(group_score)
        Number_of_MTL_Training.append(len(group_score))

        actual_improvement_values.append(0)
        predicted_improvement_values.append(0)

    else:
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
        # Explained_Variance = list(initial_results.Explained_Variance)
        actual_improvement_values = list(initial_results.Actual_Improvement)
        predicted_improvement_values = list(initial_results.predicted_improvement_values)

    iter_max = 10000000
    last_iter = 0

    current_best_group = copy.deepcopy(task_group)
    current_best_group_score = copy.deepcopy(group_score)
    current_best_prev_solution = prev_solution
    # current_best_ev = copy.deepcopy(group_ev)
    timeStamp = time.time()
    if rerun:
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
        # new_ev = copy.deepcopy(current_best_ev)

        changed_group, task_rand = mutate_groups(new_task_group, new_group_score)

        if len(changed_group) == 0:
            x += 1
            continue

        changed_group = sorted(changed_group)
        time_query_Start = time.time()
        tmp_predicted_improvement = {}
        tmp_stl_score = {}
        for group_no in changed_group:
            prediction_of_performance = predict_performance_of_new_group(new_task_group[group_no])
            predicted_performance[group_no] = prediction_of_performance[0]
            single_loss_dict[group_no] = prediction_of_performance[1]
            tmp_predicted_improvement[group_no] = prediction_of_performance[0]
            tmp_stl_score[group_no] = prediction_of_performance[1]

        for group_no, task in new_task_group.items():
            if group_no in changed_group:
                total_loss_per_task_group_per_fold = single_loss_dict[group_no] * (
                        1 - predicted_performance[group_no])
                predicted_group_score[group_no] = total_loss_per_task_group_per_fold
        print(f'Time to query predictor = {(time.time() - time_query_Start) / 60} min')

        predicted_solution = sum(list(predicted_group_score.values()))

        # K = 0.99

        if predicted_solution < prev_solution:
            PROBABILITIES_ACCEPT = [1, 0]
        else:
            PROBABILITIES_ACCEPT = [0, 1]

        fast_reject = np.random.choice(switch_architecture, 1, p=PROBABILITIES_ACCEPT)
        if fast_reject[0] == 'ACCEPT':
            train_model = 1
        else:
            if len(Switch) < 100:
                train_model = np.random.choice([1, 0], 1, p=[0.15, 0.85])
            elif len(Switch) < 500:
                train_model = np.random.choice([1, 0], 1, p=[0.1, 0.9])
            else:
                train_model = np.random.choice([1, 0], 1, p=[0.02, 0.98])

            train_model = train_model[0]
        if train_model == 1:

            TASK_Specific_Arch = prep_task_specific_arch(new_task_group)
            random_seed = random.randint(0, 100)
            timeStart = time.time()

            actual_improvement = {}
            count = 0
            for group_no, task in new_task_group.items():
                if group_no in changed_group:
                    grouping_sorted = tuple(sorted(task))
                    if grouping_sorted not in Prev_Groups.keys():
                        if len(task) == 1:
                            total_loss_per_task_group_per_fold = Single_res_dict[task[0]]
                            # ev = STL_EV[task[0]]

                        elif len(task) == 2:
                            total_loss_per_task_group_per_fold = Pairwise_res_dict[grouping_sorted]
                            # ev = PTL_EV[grouping_sorted]

                        else:
                            count += 1
                            args = kFold_validation(TASK_Specific_Arch[group_no],initial_shared_architecture,
                                                                                  new_task_group[group_no], group_no,
                                                                                  random_seed)
                            timeStart = time.time()
                            # number_of_pools = len(args)
                            # pool = mp.Pool(number_of_pools)
                            # all_scores = np.array(pool.starmap(final_model, args))
                            # pool.close()
                            with ThreadPool(20) as tp:
                                all_scores = tp.starmap(final_model, args)
                            tp.join()
                            print(f'Time required = {(time.time() - timeStart) / 60} minutes')
                            total_loss_per_task_group_per_fold,ev = sort_results(all_scores)
                            actual_improvement[group_no] = (tmp_stl_score[group_no] - total_loss_per_task_group_per_fold)/tmp_stl_score[group_no]

                        if group_no not in new_group_score.keys():
                            new_group_score.update({group_no: total_loss_per_task_group_per_fold})
                            # new_ev.update({group_no: ev})
                        else:
                            new_group_score[group_no] = total_loss_per_task_group_per_fold
                            # new_ev[group_no] = ev

                        Prev_Groups[grouping_sorted] = total_loss_per_task_group_per_fold

                    else:
                        loss = Prev_Groups[grouping_sorted]
                        if group_no not in new_group_score.keys():
                            new_group_score.update({group_no: loss})
                            # new_ev.update({group_no: ev})
                        else:
                            new_group_score[group_no] = loss
                            # new_ev[group_no] = ev

            final_time = time.time() - timeStart

            print(f'actual_improvement = {actual_improvement}')
            mtl_solution = sum(list(new_group_score.values()))
            # avg_ev = np.mean(list(new_ev.values()))
            K = 0.99
            Accept_Probability = min(1, math.exp((prev_solution - mtl_solution) * K))
            PROBABILITIES_ACCEPT = [Accept_Probability, max(0, 1 - Accept_Probability)]
            final_accept = np.random.choice(switch_architecture, 1, p=PROBABILITIES_ACCEPT)
            print(f'iteration_MTL = {iteration_MTL}\tmtl_solution = {mtl_solution}\tfinal_accept = {final_accept[0]}\t')

            for g, s in new_group_score.items():
                if s == 0:
                    print(f'new_group_score = {new_group_score}')
                    print(f'Exiting because of error')
                    exit(0)
                    break

            if len(new_task_group.keys()) != len(new_group_score.keys()):
                print(f'new_group_score = {new_group_score}\nnew_task_group = {new_task_group}')
                print(f'Exiting because of key-mismatch error')
                exit(0)
                break

            if final_accept[0] == 'ACCEPT':
                group_score = copy.deepcopy(new_group_score)
                task_group = copy.deepcopy(new_task_group)
                # current_best_ev = copy.deepcopy(new_ev)

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
            Individual_Group_Score.append(new_group_score)
            Number_of_MTL_Training.append(count)
            # Explained_Variance.append(new_ev)

            actual_improvement_values.append(list(actual_improvement.values()))
            predicted_improvement_values.append(list(tmp_predicted_improvement.values()))

        length = len(Switch)
        print(f'Length = {length}')
        tail_pointer = 3
        if length > old_length and length % tail_pointer == 0:
            print(len(Switch),
                  # len(Explained_Variance),
                  len(Iteration_MTL), len(Prev_Solution), len(Changed_Group), len(Total_Loss),
                  len(Number_of_Groups), len(Prev_iter), len(Task_group), len(Individual_Group_Score),
                  len(Individual_Task_Score), len(Training_Time))
            results = pd.DataFrame({'Iteration_MTL': Iteration_MTL,
                                    'Random_Task': Random_Task,
                                    'ACCEPT': Switch,
                                    'Changed_Groups': Changed_Group,
                                    'Total_Loss': Total_Loss,
                                    'Prev_Solution': Prev_Solution,
                                    # "Explained_Variance": Explained_Variance,
                                    'Training_Time': Training_Time,
                                    'Number_of_Groups': Number_of_Groups,
                                    'Number_of_MTL_Training': Number_of_MTL_Training,
                                    'Fast_Reject':Fast_Reject,
                                    'Task_group': Task_group,
                                    'Individual_Group_Score': Individual_Group_Score,
                                    'Predicted_Score': Predicted_Score,
                                    "Last_switch_at_iter": Prev_iter,
                                    'Actual_Improvement': actual_improvement_values,
                                    'predicted_improvement_values': predicted_improvement_values,})
            results.to_csv(f'{run_results}/{datasetName}_Task_Grouping_Results_{length}_run_{run}_{modelname}.csv',
                           index=False)

            old_length = len(Switch)
            new_results = pd.read_csv(f'{run_results}/{datasetName}_Task_Grouping_Results_{length}_run_{run}_{modelname}.csv',
                                      low_memory=False)
            new_results = new_results.tail(tail_pointer).reset_index(drop=True)

            # new_object = TaskGroupingPredictor(datasetName, 1, run)
            # TaskGroupingPredictor.predictor_data_prep(new_object, new_results)
            predictor_data_prep(new_results, 1, run, datasetName)
            retrain_predictor(datasetName)


            length = length - tail_pointer
            old_file = f'{run_results}/{datasetName}_Task_Grouping_Results_{length}_run_{run}_{modelname}.csv'
            if os.path.exists(old_file):
                os.remove(os.path.join(
                    f'{run_results}/{datasetName}_Task_Grouping_Results_{length}_run_{run}_{modelname}.csv'))

        if np.sum(Number_of_MTL_Training) >= 1000:
            break

    print(len(Switch),
          len(Explained_Variance),
          len(Random_Task), len(Iteration_MTL), len(Prev_Solution), len(Changed_Group), len(Total_Loss),
          len(Number_of_Groups), len(Prev_iter), len(Task_group), len(Individual_Group_Score),
          len(Individual_Task_Score))
    results = pd.DataFrame({'Iteration_MTL': Iteration_MTL,
                            'Random_Task': Random_Task,
                            'ACCEPT': Switch,
                            'Changed_Groups': Changed_Group,
                            'Total_Loss': Total_Loss,
                            'Prev_Solution': Prev_Solution,
                            # "Explained_Variance": Explained_Variance,
                            'Training_Time': Training_Time,
                            'Number_of_Groups': Number_of_Groups,
                            'Number_of_MTL_Training': Number_of_MTL_Training,
                            'Fast_Reject': Fast_Reject,
                            'Task_group': Task_group,
                            'Individual_Group_Score': Individual_Group_Score,
                            'Predicted_Score': Predicted_Score,
                            "Last_switch_at_iter": Prev_iter,
                            'Actual_Improvement': actual_improvement_values,
                            'predicted_improvement_values': predicted_improvement_values,
                            })
    results.to_csv(f'{run_results}/{datasetName}_Task_Grouping_Results_run_{run}_{modelname}.csv',
                   index=False)

