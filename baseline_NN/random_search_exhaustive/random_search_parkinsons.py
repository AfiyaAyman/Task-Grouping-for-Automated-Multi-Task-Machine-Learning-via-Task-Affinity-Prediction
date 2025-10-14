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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model, backend
from sklearn.metrics import explained_variance_score, r2_score

USE_GPU = True
if USE_GPU:
    device_idx = 0
    gpus = tf.config.list_physical_devices('GPU')
    gpu_device = gpus[device_idx]
    core_config = tf.config.experimental.set_visible_devices(gpu_device, 'GPU')
    tf.config.experimental.set_memory_growth(gpu_device, True)
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=core_config))
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

    return Sample_Inputs, Sample_Label, len(Input_Features)


def readData(TASKS):
    data_param_dictionary = {}
    for subj_id in TASKS:
        csv = (f"{DataPath}DATA/parkinsons_subject_{subj_id}_for_MTL.csv")
        df = pd.read_csv(csv, low_memory=False)


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
    return data_param_dictionary, Number_of_Features

def kFold_validation(current_task_specific_architecture, current_shared_architecture, task, group_no, random_seed):
    data_param_dictionary, Number_of_Features_FF = readData(task)

    data_param_dict_for_specific_task = {}

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

    timeStart = time.time()
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

    return total_loss_per_task_group_per_fold


def final_model(task_hyperparameters, shared_hyperparameters, task_group_list, data_param_dict_for_specific_task,
                Number_of_Features, fold, group_no):
    data_param = copy.deepcopy(data_param_dict_for_specific_task)

    filepath = f'SavedModels/RS_Parkinsons_Group_{group_no}_{fold}.h5'
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

    if os.path.exists(filepath):
        os.remove(os.path.join(filepath))

    return scores, explained_var, r_square


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


if __name__ == '__main__':

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
    DataPath = f'../../Dataset/{datasetName.upper()}/'
    ResultPath = '../../Results/'

    task_info = pd.read_csv(f'{DataPath}Task_Information_{datasetName}.csv')
    single_results = pd.read_csv(f'{ResultPath}STL/STL_{datasetName}_NN.csv')
    pair_results = pd.read_csv(f'{ResultPath}Pairwise/NN/{datasetName}_Results_from_Pairwise_Training_ALL_NN.csv')

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


    run = 1
    Task_group = []
    Total_Loss = []
    Individual_Group_Score = []
    Number_of_Groups = []
    Individual_Explained_Variance = []
    Individual_RSquare = []


    partition_data = pd.read_csv(f'{datasetName}_OnlyGroups_URS.csv')
    TASK_Group = list(partition_data.Task_group)

    Prev_Groups = {}

    for count in range(len(TASK_Group)):
        print(f'Initial Training for {datasetName}-partition {count}')
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
                elif len(task) == 2:
                    loss = Pairwise_res_dict[grouping_sorted]
                else:
                    tmp = (TASK_Specific_Arch[group_no], initial_shared_architecture, task, group_no, random_seed)
                    loss = kFold_validation(*tmp)
                tot_loss += loss
                group_score[group_no] = loss
                Prev_Groups[grouping_sorted] = loss

            else:
                loss = Prev_Groups[grouping_sorted]
                if group_no not in group_score.keys():
                    group_score.update({group_no: loss})

                else:
                    group_score[group_no] = loss
                    # group_avg_EV[group_no] = EV
                    # group_avg_R2[group_no] = Rsq

                tot_loss += loss

        Task_group.append(task_group)
        Number_of_Groups.append(len(task_group))
        Total_Loss.append(tot_loss)
        Individual_Group_Score.append(group_score.copy())
        print(len(TASK_Group), len(Total_Loss), len(Number_of_Groups), len(Task_group), len(Individual_Group_Score))
        if np.sum(Number_of_Groups)>=1000:
            break
        if len(Total_Loss) % 10 == 0:
            temp_res = pd.DataFrame({'Total_Loss': Total_Loss,
                                     'Number_of_Groups': Number_of_Groups,
                                     'Individual_Group_Score': Individual_Group_Score,
                                     })
            temp_res.to_csv(f'{datasetName}_Random_Search_{len(Total_Loss)}_run_{run}.csv',
                            index=False)

            if len(Total_Loss) > 10:
                old_file = f'{datasetName}_Random_Search_{len(Total_Loss)-10}_run_{run}.csv'
                if os.path.exists(old_file):
                    os.remove(os.path.join(old_file))

    print(len(Total_Loss), len(Number_of_Groups), len(Task_group), len(Individual_Group_Score))
    initial_results = pd.DataFrame({'Total_Loss': Total_Loss,
                                    'Number_of_Groups': Number_of_Groups,
                                    'Task_group': Task_group,
                                    'Individual_Group_Score': Individual_Group_Score,
                                    })
    initial_results.to_csv(f'{datasetName}_Random_Search_{len(Total_Loss)}_run_{run}.csv',
                           index=False)
