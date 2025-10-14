import pandas as pd
import numpy as np
import random
import time, os
import copy
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
from sklearn.metrics import explained_variance_score,r2_score

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


def kFold_validation(current_task_specific_architecture, current_shared_architecture, task, random_seed):

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

            fold+=1

    ALL_FOLDS = []
    for fold in range(0, 10):
        tmp = (
            current_task_specific_architecture, current_shared_architecture, task,
            data_param_dict_for_specific_task,
            Number_of_Features_FF,
            fold)
        ALL_FOLDS.append(tmp)

    number_of_models = 10
    current_idx = random.sample(range(len(ALL_FOLDS)), number_of_models)
    args = [ALL_FOLDS[index] for index in sorted(current_idx)]
    number_of_pools = len(args)
    timeStart = time.time()

    with ThreadPool(number_of_pools) as tp:
        all_scores = tp.starmap(final_model, args)
    tp.join()
    print(f'Time required = {(time.time() - timeStart) / 60} minutes')
    print(f'all_scores = {all_scores}')


    scores = []
    weight_dict = {}
    for i in range(len(all_scores)):
        scores.append(all_scores[i][0])
        for task_name, weight in all_scores[i][3].items():
            if task_name not in weight_dict.keys():
                weight_dict[task_name] = []
            weight_dict[task_name].append(weight)
    print(f'scores = {scores}')

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

    print(f'total_loss_per_task_group_per_fold = {total_loss_per_task_group_per_fold}')
    print(f'explained_var = {explained_var}')
    print(f'explained_var = {np.mean(explained_var)}')
    print(f'r_square = {r_square}')

    file = open("weight_dict_Parkinsons.txt", "w")
    file.write("weight_dict = " + repr(weight_dict) + "\n")
    file.close()

    return total_loss_per_task_group_per_fold,np.mean(explained_var),np.mean(r_square)


def final_model(task_hyperparameters, shared_hyperparameters, task_group_list, data_param_dict_for_specific_task,
                Number_of_Features, fold):
    data_param = copy.deepcopy(data_param_dict_for_specific_task)

    filepath = f'SavedModels/All_Tasks_Parkinsons_fold_{fold}.h5'

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

    for subj_id in task_group_list:
        hyperparameters = copy.deepcopy(task_hyperparameters[subj_id])
        last_shared = MTL_model_param[f'sub_{subj_id}_fold_{fold}_last_hidden_layer']
        if hyperparameters['postprocessing_FF_layers'] > 0:

            hidden_ff = Dense(hyperparameters['postprocessing_FF_Neurons'][0], activation=activation_func)(
                last_shared)
            for h in range(1, hyperparameters['postprocessing_FF_layers']):
                hidden_ff = Dense(hyperparameters['postprocessing_FF_Neurons'][h], activation=activation_func)(hidden_ff)

            MTL_model_param[f'sub_{subj_id}_fold_{fold}_ff_postprocessing_model'] = hidden_ff



    # output Neurons
    output_layers = []

    for subj_id in task_group_list:
        hyperparameters = copy.deepcopy(task_hyperparameters[subj_id])
        if hyperparameters['postprocessing_FF_layers'] > 0:

            shared_model = Model(inputs=MTL_model_param[f'sub_{subj_id}_fold_{fold}_Input_FF'],
                                 outputs=MTL_model_param[f'sub_{subj_id}_fold_{fold}_ff_postprocessing_model'])

        else:
            shared_model = Model(inputs=MTL_model_param[f'sub_{subj_id}_fold_{fold}_Input_FF'],
                                 outputs=MTL_model_param[f'sub_{subj_id}_fold_{fold}_last_hidden_layer'])
        outputlayer = Dense(2, activation='linear', name=f'Score_{subj_id}')(shared_model.output)
        output_layers.append(outputlayer)

    finalModel = Model(inputs=input_layers, outputs=output_layers)

    from keras.utils.layer_utils import count_params
    trainable_count = count_params(finalModel.trainable_weights)

    opt = tf.keras.optimizers.Adam(learning_rate=shared_hyperparameters['learning_rate'])
    finalModel.compile(optimizer=opt, loss='mse')

    checkpoint = ModelCheckpoint(filepath, verbose=2, monitor='val_loss', save_best_only=True, mode='auto')
    number_of_epoch = 1000
    history = finalModel.fit(x=train_data,
                             y=train_label,
                             # shuffle=True,
                             epochs=number_of_epoch,
                             batch_size=132,
                             validation_data=(test_data,
                                              test_label),
                             callbacks=checkpoint,
                             verbose=0)

    finalModel = tf.keras.models.load_model(filepath)
    print("Weights and biases of the layers after training the model: \n")
    tmp_weight = {}
    for layer in finalModel.layers:
        config = layer.get_config()

        if f'Score_' in config['name']:
            task_name = config['name']

            weight = layer.get_weights()[0]
            bias = layer.get_weights()[1]
            # print(f'task_name = {task_name}, weights = {np.shape(weight)}, bias = {np.shape(bias)}, weight = {weight}, bias = {bias}')

            weight_and_bias = []
            for w in weight:
                weight_and_bias.append(w[0])
            weight_and_bias.append(bias[0])

            if task_name not in tmp_weight:
                tmp_weight[task_name] = weight_and_bias
            # tmp_weight[task_name].append(weight_and_bias)

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

    return scores, explained_var, r_square,tmp_weight


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


def all_task_MTL():
    Task_group = []
    Total_Loss = []
    Number_of_Groups = []

    task_group = {0:TASKS}
    print(f'task_group = {task_group}')


    TASK_Specific_Arch = prep_task_specific_arch(task_group)

    random_seed = random.randint(0, 100)

    args_tasks = []
    group_score = {}
    tot_loss = 0
    group_no = 0
    task = task_group[group_no]
    group_score.update({group_no: 0})
    tmp = (TASK_Specific_Arch[group_no], initial_shared_architecture, task, random_seed)
    args_tasks.append(tmp)

    loss,ev,r_square = kFold_validation(*tmp)
    tot_loss+=loss
    group_score[group_no] = loss


    Task_group.append(task_group)
    Number_of_Groups.append(len(task_group))
    Total_Loss.append(tot_loss)

    print(f'tot_loss = {tot_loss}')
    print(f'Explained Variance = {ev}')
    # print(f'R-Square = {r_square}')

if __name__ == '__main__':


    num_folds = 10
    initial_shared_architecture = {'adaptive_FF_neurons': 4, 'shared_FF_Layers': 2, 'shared_FF_Neurons': [20,10],
                                   'learning_rate': 0.00779959, 'activation': 'relu'}

    '''Global Files and Data for Task-Grouping Predictor'''
    task_len = {}
    variance_dict = {}
    std_dev_dict = {}
    dist_dict = {}
    Single_res_dict = {}

    datasetName = 'Parkinsons'
    DataPath = f'../../Dataset/{datasetName.upper()}/'
    ResultPath = '../../Results'

    TASKS = [i for i in range(1, 43)]
    number_of_epoch = 100

    all_task_MTL()



