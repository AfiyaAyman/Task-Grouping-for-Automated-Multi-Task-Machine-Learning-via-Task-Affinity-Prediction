import pandas as pd
import copy
import numpy as np
import os
import random
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from sklearn.metrics import roc_auc_score,average_precision_score
import tensorflow as tf
# print(f'version = {tf.__version__}')
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model, backend
import ast
import math

from sklearn.model_selection import KFold,StratifiedKFold
# from sklearn.model_selection import StratifiedKFold

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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



import subprocess
from subprocess import PIPE


class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


# sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(mp.pool.Pool):
    Process = NoDaemonProcess


def readData(landmine_list):
    data_param_dictionary = {}
    for landmine in landmine_list:
        csv = (f"{DataPath}LandmineData_{landmine}.csv")
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

    if datasetName == 'Landmine':
        max_size = 690
        train_set_size = math.floor(max_size * (1 - num_folds / 100))
        test_set_size = math.ceil(max_size * (num_folds / 100))


    for landmine in landmine_list:

        Sample_Inputs = data_param_dictionary[f'Landmine_{landmine}_FF_Inputs']
        Sample_Label = data_param_dictionary[f'Landmine_{landmine}_Labels']


        kfold = StratifiedKFold(n_splits=num_folds, random_state=random_seed, shuffle=True)

        fold = 0
        ALL_FOLDS = []

        for train, test in kfold.split(Sample_Inputs, Sample_Label):
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

            print(
                f'fold = {fold} landmine = {landmine}, '
                f'X_train = {X_train.shape}, X_test = {X_test.shape}, y_train = {y_train.shape}, y_test = {y_test.shape}')

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


    number_of_models = 10
    current_idx = random.sample(range(len(ALL_FOLDS)), number_of_models)
    args = [ALL_FOLDS[index] for index in sorted(current_idx)]

    number_of_pools = len(args)+10
    # pool = mp.Pool(number_of_pools)
    # all_scores = np.array(pool.starmap(final_model, args))
    # pool.close()
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


    scores = []
    for c in range(len(all_scores)):
        scores.append(all_scores[c][0])

    for score in scores:
        idx = 1
        for landmine in landmine_list:
            score_param_per_task_group_per_fold[f'landmine_{landmine}'].append(score[idx])
            idx += 1
            if idx == len(landmine_list) + 1:
                break
        for landmine in landmine_list:
            error_param_per_task_group_per_fold[f'landmine_{landmine}'].append(1 - score[idx])
            idx = idx + 1

    # errorRate = []
    # for t, err_val in error_param_per_task_group_per_fold.items():
    #     errorRate.append(np.mean(err_val))
    #
    # ap = []
    # for c in range(len(all_scores)):
    #     ap.append(all_scores[c][3])

    weight_dict = {}
    for c in range(len(all_scores)):
        for task_name, weight in all_scores[c][3].items():
            if task_name not in weight_dict.keys():
                weight_dict[task_name] = []
                # print(f'weight = {weight}')
            weight_dict[task_name].append(weight[0])
    file = open("../weight_dict_Landmine.txt", "w")
    file.write("" + repr(weight_dict) + "")
    file.close()

    auc = []
    for c in range(len(all_scores)):
        auc.append(all_scores[c][2])

    total_loss_per_task_group_per_fold = 0
    for t, loss_val in score_param_per_task_group_per_fold.items():
        total_loss_per_task_group_per_fold += np.mean(loss_val)

    task_specific_scores = {}
    for key in score_param_per_task_group_per_fold.keys():
        task_specific_scores.update({key: np.mean(score_param_per_task_group_per_fold[key])})

    # print(f'error_param_per_task_group_per_fold = {error_param_per_task_group_per_fold}')
    # print(f'ap = {ap}')
    # print(f'ap = {ap}')

    # avg_error = np.mean(errorRate)
    # AP = np.mean(ap)
    AUC = np.mean(auc)

    print(
        f'total_loss_per_task_group_per_fold = {total_loss_per_task_group_per_fold}\tAUC = {AUC}')

    return total_loss_per_task_group_per_fold, task_specific_scores, AUC



def postprocessing_feedforward(hyperparameters, last_hidden):

    hidden_ff = Dense(hyperparameters['postprocessing_FF_Neurons'][0], activation='relu')(last_hidden)
    for h in range(1, hyperparameters['postprocessing_FF_layers']):
        hidden_ff = Dense(hyperparameters['postprocessing_FF_Neurons'][h], activation='relu')(hidden_ff)

    return hidden_ff


def final_model(task_hyperparameters, shared_hyperparameters, landmine_list, data_param_dict_for_specific_task,
                Number_of_Features, fold, group_no):


    #filepath = f'SavedModels/{ClusterType}_Landmine_Group_{group_no}_{fold}.h5'
    filepath = f'SavedModels/RS_{datasetName}_Group_{group_no}_{fold}.h5'
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

        # shared_model = Model(inputs=MTL_model_param[f'landmine_{landmine}_Input_FF'],
        #                      outputs=MTL_model_param[f'landmine_{landmine}_last_hidden_layer'])

        # combinedInput = concatenate([MTL_model_param[f'landmine_{landmine}_Input_FF'], shared_model.output])
        output = outputLayer(MTL_model_param[f'landmine_{landmine}_last_hidden_layer'])
        # output = outputLayer(combinedInput)

        output_layers.append(output)

    finalModel = Model(inputs=input_layers, outputs=output_layers)

    # from tensorflow import keras
    # keras.utils.plot_model(finalModel, f"multitask_model_Landmine_{group_no}.png", show_shapes=True)

    # Compile model

    opt = tf.keras.optimizers.Adam(learning_rate=shared_hyperparameters['learning_rate'])
    finalModel.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    # print(model.summary)
    checkpoint = ModelCheckpoint(filepath, verbose=0, monitor='val_loss', save_best_only=True, mode='auto')
    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, min_delta=0.1, patience=10)
    number_of_epoch = 400
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
        # if 'input' not in config['name'] and 'concatenate' not in config['name']:
        if 'Landmine_' in config['name']:
            task_name = config['name']
            task_name = layer.name

            # task_name = task_name.split('_')
            weight_and_bias = []

            weight = layer.get_weights()[0]
            bias = layer.get_weights()[1]

            print(weight)
            print(bias)

            for w in weight:
                weight_and_bias.append(w[0])
            weight_and_bias.append(bias[0])

            if task_name not in tmp_weight:
                tmp_weight[task_name] = []
            tmp_weight[task_name].append(weight_and_bias)

    y_pred = finalModel.predict(test_data, verbose=0)
    y_pred = Splitting_Values(y_pred)
    y_test = Splitting_Values(test_label)

    # print(f'y_pred = {y_pred[:5]}')
    # print(f'y_test = {y_test[:5]}')

    # predicted_val = []
    # for i in y_pred:
    #     if i < 0.75:
    #         predicted_val.append(0)
    #     else:
    #         predicted_val.append(1)
    #
    # Error = [abs(a_i - b_i) for a_i, b_i in zip(y_test, predicted_val)]
    # wrong_pred = Error.count(1)
    # errorRate = wrong_pred / len(y_test)

    # ap = average_precision_score(y_test, y_pred)


    auc = 0
    try:
        auc = roc_auc_score(y_test, y_pred)
    except ValueError:
        pass
    if os.path.exists(filepath):
        os.remove(os.path.join(filepath))
    trainable_count = 1111
    # return scores, (auc1, auc2), trainable_count
    return scores, trainable_count, auc, tmp_weight


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


if __name__ == "__main__":
    import sys

    datasetName = 'Landmine'
    DataPath = f'../Dataset/{datasetName.upper()}/'

    ResultPath = '../Results'

    task_len = {}
    variance_dict = {}
    std_dev_dict = {}
    dist_dict = {}
    Single_res_dict = {}
    STL_AUC = {}
    loss_dict = {}
    task_info = pd.read_csv(f'{DataPath}Task_Information_{datasetName}.csv')
    task_distance_info = pd.read_csv(f'{DataPath}Task_Distance_{datasetName}.csv')
    single_results = pd.read_csv(f'{ResultPath}/SingleTaskTraining_{datasetName}.csv')
    pair_results = pd.read_csv(f'{ResultPath}/{datasetName}_Results_from_Pairwise_Training_ALL_NN.csv')

    TASKS = [i for i in range(0, 29)]
    print(len(TASKS))
    for Selected_Task in TASKS:
        task_data = task_info[task_info.Task_Name == Selected_Task].reset_index()
        task_len.update({Selected_Task: task_data.Dataset_Size[0]})
        variance_dict.update({Selected_Task: task_data.Variance[0]})
        std_dev_dict.update({Selected_Task: task_data.Std_Dev[0]})
        dist_dict.update({Selected_Task: task_data.Average_Euclidian_Distance_within_Task[0]})
        single_res = single_results[single_results.Task == Selected_Task].reset_index()
        Single_res_dict.update({Selected_Task: single_res.LOSS[0]})
        STL_AUC.update({Selected_Task: single_res.AUC[0]})

    Pairwise_res_dict = {}
    Pairwise_res_dict_indiv = {}
    PTL_AUC = {}
    Task1 = list(pair_results.Task_1)
    Task2 = list(pair_results.Task_2)
    Pairs = [(Task1[i], Task2[i]) for i in range(len(Task1))]
    print(len(Pairs))

    for p in Pairs:
        task1 = p[0]
        task2 = p[1]
        pair_res = pair_results[(pair_results.Task_1 == task1) & (pair_results.Task_2 == task2)].reset_index()

        if task1 == pair_res.Task_1[0]:
            Pairwise_res_dict_indiv.update({p: {f'landmine_{task1}': pair_res.Individual_loss_Task_1[0],
                                                f'landmine_{task2}': pair_res.Individual_loss_Task_2[0]}})
        else:
            Pairwise_res_dict_indiv.update({p: {f'landmine_{task1}': pair_res.Individual_loss_Task_2[0],
                                                f'landmine_{task2}': pair_res.Individual_loss_Task_1[0]}})

        Pairwise_res_dict.update({p: pair_res.Total_Loss[0]})
        avg_auc = (pair_res.Individual_Auc_task1[0] + pair_res.Individual_Auc_task2[0]) / 2
        PTL_AUC.update({p: avg_auc})

    print(len(Single_res_dict), len(Pairwise_res_dict), len(PTL_AUC))

    previously_trained = pd.read_csv(f'{datasetName}_previously_trained_groups.csv')
    previously_trained_groups = previously_trained['previously_trained_groups'].tolist()
    previously_trained_groups_loss = previously_trained['previously_trained_groups_loss'].tolist()
    prev_Average_AUC = previously_trained['prev_Average_AUC'].tolist()

    number_of_epoch = 400
    num_folds = 10

    iter_max = 100000
    Initial_MTL_Train = 50

    initial_shared_architecture = {'adaptive_FF_neurons': 6, 'shared_FF_Layers': 2, 'shared_FF_Neurons': [6, 11],
                                   'learning_rate': 0.00020960514116261997}

    switch_architecture = ['ACCEPT', 'REJECT']

    '''Initial Training Functions'''
    # gpus = tf.config.list_physical_devices('GPU')
    # gpu_device = gpus[0]
    # core_config = tf.config.experimental.set_visible_devices(gpu_device, 'GPU')
    # tf.config.experimental.set_memory_growth(gpu_device, True)
    # tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=core_config))
    # Type = 'WeightMatrix'
    # # ClusterType = 'Hierarchical'
    # ClusterType = 'KMeans'
    # mtls_for_clusters()
    # rs_for_partitions()

    Task_group = []
    Total_Loss = []
    Individual_Group_Score = []
    Number_of_Groups = []
    Individual_AUC = []
    Individual_Task_Score = []

    run = 1

    # partition_data = pd.read_csv(f'{datasetName}_Random_Task_Groups_New.csv')
    # TASK_Group = list(partition_data.Task_Groups)

    Prev_Groups = {}
    TASK_Group = [{0: TASKS}]
    # for count in range(V, V+300):
    for count in range(len(TASK_Group)):
        print(f'Initial Training for {datasetName}-partition {count}')
        # task_Set = copy.deepcopy(TASKS)

        # task_group = random_task_grouping(task_Set, min_task_groups)
        task_group = TASK_Group[count]
        # task_group = ast.literal_eval(task_group)

        TASK_Specific_Arch = prep_task_specific_arch(task_group)
        random_seed = random.randint(0, 100)

        group_score = {}
        group_avg_AUC = {}
        tmp_task_score = []
        tot_loss = 0
        for group_no, task in task_group.items():
            group_score.update({group_no: 0})
            group_avg_AUC.update({group_no: 0})
            grouping_sorted = tuple(sorted(task))
            if grouping_sorted not in Prev_Groups.keys():
                if len(task) == 1:
                    loss = Single_res_dict[task[0]]
                    AUC = STL_AUC[task[0]]
                    task_scores = {f'landmine_{task[0]}': Single_res_dict[task[0]]}

                elif len(task) == 2:
                    loss = Pairwise_res_dict[grouping_sorted]
                    AUC = PTL_AUC[grouping_sorted]
                    task_scores = copy.deepcopy(Pairwise_res_dict_indiv[grouping_sorted])

                else:
                    tmp = (TASK_Specific_Arch[group_no], initial_shared_architecture, task, group_no, random_seed)
                    loss, task_scores, AUC = kFold_validation(*tmp)
                tot_loss += loss
                group_score[group_no] = loss
                group_avg_AUC[group_no] = AUC
                Prev_Groups[grouping_sorted] = (loss, AUC,copy.deepcopy(task_scores))
                tmp_task_score.append(copy.deepcopy(task_scores))

            else:
                loss, AUC, task_scores = Prev_Groups[grouping_sorted]
                if group_no not in group_score.keys():
                    group_score.update({group_no: loss})
                    group_avg_AUC.update({group_no: AUC})

                else:
                    group_score[group_no] = loss
                    group_avg_AUC[group_no] = AUC

                tot_loss += loss
                tmp_task_score.append(copy.deepcopy(task_scores))

            print(f'group_no = {group_no}\tloss = {loss}, task_scores = {task_scores}')

        print(f'tot_loss = {tot_loss}')
        Task_group.append(task_group)
        Number_of_Groups.append(len(task_group))
        Total_Loss.append(tot_loss)
        Individual_Group_Score.append(group_score.copy())
        Individual_AUC.append(group_avg_AUC.copy())
        Individual_Task_Score.append(tmp_task_score.copy())
        # print(Individual_Group_Score)
        print(len(TASK_Group), len(Total_Loss), len(Number_of_Groups), len(Task_group), len(Individual_Group_Score), len(Individual_AUC), len(Individual_Task_Score))
        cut_off = 2
        if len(Total_Loss) % cut_off == 0:
            temp_res = pd.DataFrame({'Total_Loss': Total_Loss,
                                     'Number_of_Groups': Number_of_Groups,
                                        'Task_group': Task_group,
                                        'Individual_Task_Score': Individual_Task_Score,
                                     'Individual_Group_Score': Individual_Group_Score,
                                     'Individual_AUC': Individual_AUC})
            temp_res.to_csv(f'../Groupwise_Affinity/{datasetName}_Random_Search_{V+len(Total_Loss)}_run_{run}_v_{V}.csv',
                            index=False)

            if len(Total_Loss) > cut_off:
                old_file = f'../Groupwise_Affinity/{datasetName}_Random_Search_{V+len(Total_Loss)-cut_off}_run_{run}_v_{V}.csv'
                if os.path.exists(old_file):
                    os.remove(os.path.join(old_file))

    print(len(Total_Loss), len(Number_of_Groups), len(Task_group), len(Individual_Group_Score))
    initial_results = pd.DataFrame({'Total_Loss': Total_Loss,
                                    'Number_of_Groups': Number_of_Groups,
                                    'Task_group': Task_group,
                                    'Individual_Task_Score': Individual_Task_Score,
                                    'Individual_Group_Score': Individual_Group_Score,
                                    'Individual_AUC': Individual_AUC})

    initial_results.to_csv(f'{ResultPath}/{datasetName}_SimpleMTL.csv',
                           index=False)
