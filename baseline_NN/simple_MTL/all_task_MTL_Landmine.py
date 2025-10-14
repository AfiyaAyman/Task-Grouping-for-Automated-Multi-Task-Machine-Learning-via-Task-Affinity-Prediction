import pandas as pd
import copy
import numpy as np
import math
import os
import random
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import ast
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.layers import *
from sklearn.metrics import average_precision_score,roc_auc_score
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# print(f'version = {tf.__version__}')
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

def readData(landmine_list):
    data_param_dictionary = {}
    for landmine in landmine_list:
        csv = (f"{DataPath}DATA/LandmineData_{landmine}_for_MTL.csv")
        df = pd.read_csv(csv, low_memory=False)


        DataSet = np.array(df, dtype=float)
        Number_of_Records = np.shape(DataSet)[0]
        Number_of_Features = np.shape(DataSet)[1]

        # print(df.columns)

        Input_Features = df.columns[:Number_of_Features - 1]
        Target_Features = df.columns[Number_of_Features - 1:]

        Sample_Inputs = np.zeros((Number_of_Records, len(Input_Features)))
        for t in range(Number_of_Records):
            Sample_Inputs[t] = DataSet[t, :len(Input_Features)]

        Sample_Label = np.zeros((Number_of_Records, len(Target_Features)))
        for t in range(Number_of_Records):
            Sample_Label[t] = DataSet[t, Number_of_Features - len(Target_Features):]

        Number_of_Features = len(Input_Features)
        data_param_dictionary.update({f'Landmine_{landmine}_FF_Inputs': Sample_Inputs})
        data_param_dictionary.update({f'Landmine_{landmine}_Labels': Sample_Label})

        '''*********************************'''

    return data_param_dictionary, Number_of_Features

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

def kFold_validation(current_task_specific_architecture, current_shared_architecture, landmine_list, group_no,
                     random_seed):

    data_param_dictionary, Number_of_Features = readData(landmine_list)

    data_param_dict_for_specific_task = {}

    for landmine in landmine_list:

        Sample_Inputs = data_param_dictionary[f'Landmine_{landmine}_FF_Inputs']
        Sample_Label = data_param_dictionary[f'Landmine_{landmine}_Labels']

        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)

        fold = 0
        ALL_FOLDS = []

        for train, test in kfold.split(Sample_Inputs):
            X_train = Sample_Inputs[train]
            X_test = Sample_Inputs[test]
            y_train = Sample_Label[train]
            y_test = Sample_Label[test]

            y_train = SplitLabels(y_train)
            y_test = SplitLabels(y_test)

            data_param_dict_for_specific_task.update({f'Landmine_{landmine}_fold_{fold}_X_train': X_train})
            data_param_dict_for_specific_task.update({f'Landmine_{landmine}_fold_{fold}_X_test': X_test})

            data_param_dict_for_specific_task.update({f'Landmine_{landmine}_fold_{fold}_y_train': y_train})
            data_param_dict_for_specific_task.update({f'Landmine_{landmine}_fold_{fold}_y_test': y_test})

            tmp = (current_task_specific_architecture, current_shared_architecture, landmine_list,
                   data_param_dict_for_specific_task,
                   Number_of_Features, fold, group_no)

            ALL_FOLDS.append(tmp)

            fold += 1

    number_of_models = 5
    current_idx = random.sample(range(len(ALL_FOLDS)), number_of_models)
    args = [ALL_FOLDS[index] for index in sorted(current_idx)]

    number_of_pools = len(args)

    with ThreadPool(number_of_pools) as tp:
        all_scores = tp.starmap(final_model, args)
    tp.join()
    score_param_per_task_group_per_fold = {}
    error_param_per_task_group_per_fold = {}
    AP_param_per_task_group_per_fold = {}
    for landmine in landmine_list:
        score_param_per_task_group_per_fold.update({f'landmine_{landmine}': []})
        error_param_per_task_group_per_fold.update({f'landmine_{landmine}': []})
        AP_param_per_task_group_per_fold.update({f'landmine_{landmine}': []})

    if len(landmine_list) == 1:
        for c in range(len(all_scores)):
            score_param_per_task_group_per_fold[f'landmine_{landmine_list[0]}'].append(all_scores[c][0])


    else:
        scores = []
        for c in range(len(all_scores)):
            scores.append(all_scores[c][0])
        for c in scores:
            for j in range(1, len(c)):
                score_param_per_task_group_per_fold[f'landmine_{landmine_list[j - 1]}'].append(c[j])


    ap = []
    for c in range(len(all_scores)):
        ap.append(all_scores[c][2])

    auc = []
    for c in range(len(all_scores)):
        auc.append(all_scores[c][3])

    total_loss_per_task_group_per_fold = 0
    for t, loss_val in score_param_per_task_group_per_fold.items():
        total_loss_per_task_group_per_fold += np.mean(loss_val)

    weight_dict = {}
    for c in range(len(all_scores)):
        for task_name, weight in all_scores[c][4].items():
            if task_name not in weight_dict.keys():
                weight_dict[task_name] = []
                # print(f'weight = {weight}')
            weight_dict[task_name].append(weight[0])

    print(f'ap = {ap}')
    print(f'auc = {auc}\n {np.mean(auc)}')
    print(f'Total Loss = {total_loss_per_task_group_per_fold}')

    AP = np.mean(ap)
    AUC = np.mean(auc)
    file = open("weight_dict_Landmine.txt", "w")
    file.write("" + repr(weight_dict) + "")
    file.close()

    return total_loss_per_task_group_per_fold,AP,AUC

def postprocessing_feedforward(hyperparameters, last_hidden):
    hidden_ff = Dense(hyperparameters['postprocessing_FF_Neurons'][0], activation='relu')(last_hidden)
    for h in range(1, hyperparameters['postprocessing_FF_layers']):
        hidden_ff = Dense(hyperparameters['postprocessing_FF_Neurons'][h], activation='relu')(hidden_ff)

    return hidden_ff
def final_model(task_hyperparameters, shared_hyperparameters, landmine_list, data_param_dict_for_specific_task,
                Number_of_Features, fold, group_no):


    filepath = f'SavedModels/Landmine_Group_{group_no}_{fold}.h5'
    MTL_model_param = {}
    input_layers = []

    train_data = []
    train_label = []
    test_data = []
    test_label = []

    for landmine in landmine_list:

        train_data.append(data_param_dict_for_specific_task[f'Landmine_{landmine}_fold_{fold}_X_train'])
        train_label.append(data_param_dict_for_specific_task[f'Landmine_{landmine}_fold_{fold}_y_train'])

        test_data.append(data_param_dict_for_specific_task[f'Landmine_{landmine}_fold_{fold}_X_test'])
        test_label.append(data_param_dict_for_specific_task[f'Landmine_{landmine}_fold_{fold}_y_test'])

        hyperparameters = copy.deepcopy(task_hyperparameters[landmine])
        Input_FF = tf.keras.layers.Input(shape=(Number_of_Features,))
        input_layers.append(Input_FF)

        MTL_model_param.update({f'landmine_{landmine}_Input_FF': Input_FF})

        if hyperparameters['preprocessing_FF_layers'] > 0:
            hidden_ff = Dense(hyperparameters['preprocessing_FF_Neurons'][0], activation='relu')(Input_FF)
            for h in range(1, hyperparameters['preprocessing_FF_layers']):
                hidden_ff = Dense(hyperparameters['preprocessing_FF_Neurons'][h], activation='relu')(hidden_ff)
            MTL_model_param.update({f'landmine_{landmine}_ff_preprocessing_model': hidden_ff})

    SHARED_module_param_FF = {}

    for h in range(0, shared_hyperparameters['shared_FF_Layers']):
        shared_ff = tf.keras.layers.Dense(shared_hyperparameters['shared_FF_Neurons'][h], activation='relu')
        SHARED_module_param_FF.update({f'FF_{h}': shared_ff})

    for landmine in landmine_list:
        Input_FF = MTL_model_param[f'landmine_{landmine}_Input_FF']
        shared_FF = SHARED_module_param_FF['FF_0'](Input_FF)
        for h in range(1, shared_hyperparameters['shared_FF_Layers']):
            shared_FF = SHARED_module_param_FF[f'FF_{h}'](shared_FF)

        # ff_model = Model(inputs=Input_FF, outputs=shared_FF)
        MTL_model_param.update({f'landmine_{landmine}_last_hidden_layer': shared_FF})

    # MTL_model_param = combined_layers(shared_hyperparameters, MTL_model_param,landmine_list)
    for landmine in landmine_list:
        hyperparameters = copy.deepcopy(task_hyperparameters[landmine])
        if hyperparameters['postprocessing_FF_layers'] > 0:
            ff_postprocessing_model = postprocessing_feedforward(hyperparameters,
                                                                 MTL_model_param[
                                                                     f'landmine_{landmine}_last_hidden_layer'])
            MTL_model_param.update({f'landmine_{landmine}_ff_postprocessing_model': ff_postprocessing_model})

    output_layers = []
    for landmine in landmine_list:
        outputLayer = Dense(1, activation='sigmoid', name=f'Landmine_{landmine}')
        output = outputLayer(MTL_model_param[f'landmine_{landmine}_last_hidden_layer'])

        output_layers.append(output)

    finalModel = Model(inputs=input_layers, outputs=output_layers)

    # Compile model

    opt = tf.keras.optimizers.Adam(learning_rate=shared_hyperparameters['learning_rate'])
    finalModel.compile(optimizer=opt, loss='binary_crossentropy')

    checkpoint = ModelCheckpoint(filepath, verbose=2, monitor='val_loss', save_best_only=True, mode='auto')
    number_of_epoch = 700
    history = finalModel.fit(x=train_data,
                             y=tuple(train_label),
                             shuffle=True,
                             epochs=number_of_epoch,
                             batch_size=32,
                             validation_data=(test_data,
                                              tuple(test_label)),
                             callbacks=[checkpoint],
                             verbose=0)
    finalModel = tf.keras.models.load_model(filepath)
    scores = finalModel.evaluate(tuple(test_data), tuple(test_label), verbose=0)

    tmp_weight = {}
    for layer in finalModel.layers:
        config = layer.get_config()
        if 'Landmine_' in config['name']:
            task_name = config['name']
            task_name = layer.name

            weight_and_bias = []

            weight = layer.get_weights()[0]
            bias = layer.get_weights()[1]

            for w in weight:
                weight_and_bias.append(w[0])
            weight_and_bias.append(bias[0])

            if task_name not in tmp_weight:
                tmp_weight[task_name] = []
            tmp_weight[task_name].append(weight_and_bias)



    y_pred = finalModel.predict(test_data)
    y_pred = Splitting_Values(y_pred)
    y_test = Splitting_Values(test_label)

    ap = average_precision_score(y_test, y_pred)
    auc = 0
    try:
        auc = roc_auc_score(y_test, y_pred)
    except ValueError:
        pass
    from keras.utils.layer_utils import count_params
    trainable_count = count_params(finalModel.trainable_weights)

    return scores, trainable_count, ap,auc,tmp_weight


def prep_task_specific_arch(current_task_group):
    TASK_Specific_Arch = {}
    for group_no in current_task_group.keys():
        Number_of_Tasks = current_task_group[group_no]
        initial_task_specific_architecture = {}
        for n in Number_of_Tasks:
            initial_task_specific_architecture.update({n: {'preprocessing_FF_layers': 0,
                                                           'preprocessing_FF_Neurons': [],
                                                           'postprocessing_FF_layers': 0,
                                                           'postprocessing_FF_Neurons': []
                                                           }})

        TASK_Specific_Arch.update({group_no: initial_task_specific_architecture})
    return TASK_Specific_Arch


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

def all_task_MTL():
    Task_group = []
    Total_Loss = []
    Individual_Group_Score = []
    Number_of_Groups = []


    task_group = {0: TASKS}
    print(f'task_group = {task_group}')

    TASK_Specific_Arch = prep_task_specific_arch(task_group)
    random_seed = random.randint(0, 100)

    args_tasks = []
    group_score = {}
    tot_loss = 0

    AP = []
    AUC = []
    for group_no, task in task_group.items():
        print(f'processing group {group_no}')
        group_score.update({group_no: 0})
        tmp = (TASK_Specific_Arch[group_no], initial_shared_architecture, task, group_no, random_seed)
        args_tasks.append(tmp)

        loss, ap,auc = kFold_validation(*tmp)
        tot_loss += loss
        group_score[group_no] = loss
        AP.append(ap)
        AUC.append(auc)

        print(f'loss = {loss}\tauc = {auc}\t')

    Task_group.append(task_group)
    Number_of_Groups.append(len(task_group))
    Total_Loss.append(tot_loss)
    Individual_Group_Score.append(group_score.copy())
    print(f'Individual_Group_Score = {Individual_Group_Score}')
    print(f'tot_loss = {tot_loss}')
    print(f'AP = {AP}\n MEAN = {np.mean(AP)}')
    print(f'AUC = {AUC}\n MEAN = {np.mean(AUC)}')


if __name__ == "__main__":
    datasetName = 'Landmine'
    DataPath = f'../../Dataset/{datasetName.upper()}/'
    ResultPath = '../../Results'


    TASKS = [i for i in range(0, 29)]
    print(len(TASKS))

    number_of_epoch = 400
    min_task_groups = 5
    num_folds = 10

    iter_max = 100000
    Initial_MTL_Train = 50

    initial_shared_architecture = {'adaptive_FF_neurons': 6, 'shared_FF_Layers': 2, 'shared_FF_Neurons': [6, 11],
                                   'learning_rate': 0.00020960514116261997}

    switch_architecture = ['ACCEPT', 'REJECT']

    all_task_MTL()




