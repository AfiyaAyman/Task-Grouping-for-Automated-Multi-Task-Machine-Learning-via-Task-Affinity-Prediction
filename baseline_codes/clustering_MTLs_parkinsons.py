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
from sklearn.metrics import explained_variance_score, r2_score
# from tensorflow.keras.utils.layer_utils import count_params

import subprocess
from subprocess import PIPE
# gpus = tf.config.list_physical_devices('GPU')
# gpu_device = gpus[0]
# core_config = tf.config.experimental.set_visible_devices(gpu_device, 'GPU')
# tf.config.experimental.set_memory_growth(gpu_device, True)
# tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=core_config))

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


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

        print(np.shape(df))

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
    return data_param_dictionary, Number_of_Features


'''MTL for Task Groups Functions'''


def kFold_validation(current_task_specific_architecture, current_shared_architecture, task, group_no, random_seed):
    # print(f'task = {task}')

    data_param_dictionary, Number_of_Features_FF = readData(task)

    data_param_dict_for_specific_task = {}
    if datasetName == 'Parkinsons':
        max_size = 170
        train_set_size = math.floor(max_size * (1 - num_folds / 100))
        test_set_size = math.ceil(max_size * (num_folds / 100))



    for f in range(num_folds):
        data_param_dict_for_specific_task[f] = {}

    for subj_id in task:
        Sample_Inputs = data_param_dictionary[f'Subject_{subj_id}_FF_Inputs']
        Sample_Label = data_param_dictionary[f'Subject_{subj_id}_Label']
        Sample_Label_target_1 = data_param_dictionary[f'Subject_{subj_id}_Label_1']
        Sample_Label_target_2 = data_param_dictionary[f'Subject_{subj_id}_Label_2']
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
            fold, group_no)

        ALL_FOLDS.append(tmp)

    number_of_models = 5
    current_idx = random.sample(range(len(ALL_FOLDS)), number_of_models)
    args = [ALL_FOLDS[index] for index in sorted(current_idx)]
    number_of_pools = len(args)
    timeStart = time.time()

    # pool = mp.Pool(number_of_pools)
    # all_scores = np.array(pool.starmap(final_model, args))
    # pool.close()
    with ThreadPool(20) as tp:
        all_scores = tp.starmap(final_model, args)
    tp.join()

    print(f'Time required = {(time.time() - timeStart) / 60} minutes')
    print(f'all_scores = {all_scores}')

    scores = []
    for i in range(len(all_scores)):
        scores.append(all_scores[i][0])
    # print(f'scores = {scores}')

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

    return total_loss_per_task_group_per_fold, np.mean(explained_var), np.mean(
        r_square)


def final_model(task_hyperparameters, shared_hyperparameters, task_group_list, data_param_dict_for_specific_task,
                Number_of_Features, fold, group_no):
    data_param = copy.deepcopy(data_param_dict_for_specific_task)
    print(f'task_group_list = {task_group_list}')

    #filepath = f'SavedModels/Parkinsons_group_{group_no}_{fold}.h5'
    if ClusterType == 'Hierarchical' and Type == 'WeightMatrix':
        filepath = f'SavedModels/{ClusterType}_{Type}_clusters_{datasetName}_Group_{group_no}_{fold}_{which_half}.h5'
    else:
        filepath = f'SavedModels/{ClusterType}_{Type}_clusters_{datasetName}_Group_{group_no}_{fold}.h5'
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

    checkpoint = ModelCheckpoint(filepath, verbose=2, monitor='val_loss', save_best_only=True, mode='auto')
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

    print(f'fold = {fold}\t explained_var = {explained_var}\t r_square = {r_square}')

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



if __name__ == '__main__':
    import sys

    ClusterType = str(sys.argv[1])
    Type = str(sys.argv[2])
    run = int(sys.argv[3])

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

    #gpus = tf.config.list_physical_devices('GPU')
    #gpu_device = gpus[1]
    #core_config = tf.config.experimental.set_visible_devices(gpu_device, 'GPU')
    #tf.config.experimental.set_memory_growth(gpu_device, True)
    #tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=core_config))

    task_info = pd.read_csv(f'{DataPath}Task_Information_{datasetName}.csv')
    single_results = pd.read_csv(f'{ResultPath}/STL/STL_{datasetName}_NN.csv')
    pair_results = pd.read_csv(f'../Results/new_runs/{datasetName}_Results_from_Pairwise_Training_ALL_NN.csv')

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
        STL_Rsq.update({Selected_Task: single_res.R_Square[0]})
        
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
        PTL_Rsq.update({p: pair_res.R_Square[0]})
    ''''''
    switch_architecture = ['ACCEPT', 'REJECT']

    Task_group = []
    Total_Loss = []
    Individual_Group_Score = []
    Number_of_Groups = []
    Individual_Explained_Variance = []
    Individual_RSquare = []

    # temp_res = pd.read_csv(
    #     f'../Groupwise_Affinity/{datasetName}_MTL_Results_for_Hierarchical_Clusters_NonNegative_40.csv')

    # Total_Loss = list(temp_res['Total_Loss'])
    # Number_of_Groups = list(temp_res['Number_of_Groups'])
    # Individual_Group_Score = list(temp_res['Individual_Group_Score'])
    # Individual_Explained_Variance = list(temp_res['Individual_Explained_Variance'])
    # Individual_RSquare = list(temp_res['Individual_RSquare'])

    if Type != 'WeightMatrix':
        TASK_Clusters = pd.read_csv(f'../Results/Cluster_CSV/NN/{datasetName}_Clusters_{Type}_NN.csv')
    else:
        TASK_Clusters = pd.read_csv(f'../Results/Cluster_CSV/NN/{datasetName}_{ClusterType}_Clusters_{Type}_NN.csv')

    TASK_Group = list(TASK_Clusters.TASK_Group)
    if ClusterType == 'Hierarchical' and Type == 'WeightMatrix':
        which_half = str(sys.argv[4])
        if which_half == 'first':
            TASK_Group = TASK_Group[:len(TASK_Group) // 2]
        else:
            TASK_Group = TASK_Group[len(TASK_Group) // 2:]

    Prev_Groups = {}
    # for count in range(0,len(temp_res)):
    #     task_group = TASK_Group[count]
    #     task_group = ast.literal_eval(task_group)
    #     Individual_Group_Score[count] = ast.literal_eval(Individual_Group_Score[count])
    #     Individual_Explained_Variance[count] = ast.literal_eval(Individual_Explained_Variance[count])
    #     Individual_RSquare[count] = ast.literal_eval(Individual_RSquare[count])
    #     for group_no, task in task_group.items():
    #         grouping_sorted = tuple(sorted(task))
    #         loss = Individual_Group_Score[count][group_no]
    #         EV = Individual_Explained_Variance[count][group_no]
    #         Rsq = Individual_RSquare[count][group_no]
    #         Prev_Groups[grouping_sorted] = (loss, EV, Rsq)
    #     Task_group.append(task_group)
    #
    # print(len(TASK_Group), len(Total_Loss), len(Number_of_Groups), len(Task_group), len(Individual_Group_Score))

    for count in range(len(TASK_Group)):
        print(f'Initial Training for {datasetName}-partition {count}')
        # task_Set = copy.deepcopy(TASKS)

        # task_group = random_task_grouping(task_Set, min_task_groups)
        task_group = TASK_Group[count]
        task_group = ast.literal_eval(task_group)

        TASK_Specific_Arch = prep_task_specific_arch(task_group)
        random_seed = random.randint(0, 100)

        group_score = {}
        group_avg_EV = {}
        group_avg_R2 = {}
        tot_loss = 0
        for group_no, task in task_group.items():
            group_score.update({group_no: 0})
            group_avg_EV.update({group_no: 0})
            group_avg_R2.update({group_no: 0})
            grouping_sorted = tuple(sorted(task))
            if grouping_sorted not in Prev_Groups.keys():
                if len(task) == 1:
                    loss = Single_res_dict[task[0]]
                    EV = STL_EV[task[0]]
                    Rsq = STL_Rsq[task[0]]
                elif len(task) == 2:
                    loss = Pairwise_res_dict[grouping_sorted]
                    EV = PTL_EV[grouping_sorted]
                    Rsq = PTL_Rsq[grouping_sorted]

                else:
                    tmp = (TASK_Specific_Arch[group_no], initial_shared_architecture, task, group_no, random_seed)
                    loss, EV, Rsq = kFold_validation(*tmp)
                tot_loss += loss
                group_score[group_no] = loss
                group_avg_EV[group_no] = EV
                group_avg_R2[group_no] = Rsq
                Prev_Groups[grouping_sorted] = (loss, EV, Rsq)

            else:
                loss, EV, Rsq = Prev_Groups[grouping_sorted]
                if group_no not in group_score.keys():
                    group_score.update({group_no: loss})
                    group_avg_EV.update({group_no: EV})
                    group_avg_R2.update({group_no: Rsq})

                else:
                    group_score[group_no] = loss
                    group_avg_EV[group_no] = EV
                    group_avg_R2[group_no] = Rsq

                tot_loss += loss

        Task_group.append(task_group)
        Number_of_Groups.append(len(task_group))
        Total_Loss.append(tot_loss)
        Individual_Group_Score.append(group_score.copy())
        Individual_Explained_Variance.append(group_avg_EV.copy())
        Individual_RSquare.append(group_avg_R2.copy())
        # print(Individual_Group_Score)
        print(len(TASK_Group), len(Total_Loss), len(Number_of_Groups), len(Task_group), len(Individual_Group_Score))
        if len(Total_Loss) % 10 == 0:
            temp_res = pd.DataFrame({'Total_Loss': Total_Loss,
                                     'Number_of_Groups': Number_of_Groups,
                                     'Individual_Group_Score': Individual_Group_Score,
                                     'Individual_Explained_Variance': Individual_Explained_Variance,
                                     'Individual_RSquare': Individual_RSquare})

            if ClusterType == 'Hierarchical' and Type == 'WeightMatrix':
                temp_res.to_csv(
                    f'../Results/Cluster_CSV/{datasetName}_MTL_Results_for_{ClusterType}_{Type}_{len(Total_Loss)}_run_{run}_{which_half}.csv',
                    index=False)
            else:
                temp_res.to_csv(
                f'../Results/Cluster_CSV/{datasetName}_MTL_Results_for_{ClusterType}_{Type}_{len(Total_Loss)}_run_{run}.csv',
                index=False)

            if len(Total_Loss) > 10:
                if ClusterType == 'Hierarchical' and Type == 'WeightMatrix':
                    old_file = f'../Results/Cluster_CSV/{datasetName}_MTL_Results_for_{ClusterType}_{Type}_{len(Total_Loss) - 10}_run_{run}_{which_half}.csv'
                else:
                    old_file = f'../Results/Cluster_CSV/{datasetName}_MTL_Results_for_{ClusterType}_{Type}_{len(Total_Loss) - 10}_run_{run}.csv'
                if os.path.exists(old_file):
                    os.remove(os.path.join(old_file))

    print(len(Total_Loss), len(Number_of_Groups), len(Task_group), len(Individual_Group_Score))
    initial_results = pd.DataFrame({'Total_Loss': Total_Loss,
                                    'Number_of_Groups': Number_of_Groups,
                                    'Task_group': Task_group,
                                    'Individual_Group_Score': Individual_Group_Score,
                                    'Individual_Explained_Variance': Individual_Explained_Variance,
                                    'Individual_RSquare': Individual_RSquare})
    
    
    #old_df = pd.read_csv(f'../Results/Cluster_CSV/{datasetName}_MTL_Results_for_{ClusterType}_{Type}_10_run_{run}.csv')
    #initial_results = pd.concat([old_df,initial_results])
    #print(len(initial_results))
    if ClusterType == 'Hierarchical' and Type == 'WeightMatrix':
        initial_results.to_csv(
            f'../Results/Cluster_CSV/{datasetName}_MTL_Results_for_{ClusterType}_Clusters_{Type}_run_{run}_{which_half}.csv',
            index=False)
    else:
        initial_results.to_csv(
            f'../Results/Cluster_CSV/{datasetName}_MTL_Results_for_{ClusterType}_Clusters_{Type}_run_{run}.csv',
            index=False)

