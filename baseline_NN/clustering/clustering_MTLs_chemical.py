import pandas as pd
import copy
import numpy as np
import math
import os
import random
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import ast
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# print(f'version = {tf.__version__}')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def readData(molecule_list):
    data_param_dictionary = {}
    for molecule in molecule_list:
        csv = (f"{DataPath}DATA/{molecule}_Chemical_Data_for_MTL.csv")
        chem_molecule_data = pd.read_csv(csv, low_memory=False)
        chem_molecule_data.loc[chem_molecule_data['181'] < 0, '181'] = 0

        DataSet = np.array(chem_molecule_data, dtype=float)
        Number_of_Records = np.shape(DataSet)[0]
        Number_of_Features = np.shape(DataSet)[1]

        Input_Features = chem_molecule_data.columns[:Number_of_Features - 1]
        Target_Features = chem_molecule_data.columns[Number_of_Features - 1:]

        Sample_Inputs = np.zeros((Number_of_Records, len(Input_Features)))
        for t in range(Number_of_Records):
            Sample_Inputs[t] = DataSet[t, :len(Input_Features)]
        # print(Sample_Inputs[0])
        Sample_Label = np.zeros((Number_of_Records, len(Target_Features)))
        for t in range(Number_of_Records):
            Sample_Label[t] = DataSet[t, Number_of_Features - len(Target_Features):]

        Number_of_Features = len(Input_Features)
        data_param_dictionary.update({f'Molecule_{molecule}_FF_Inputs': Sample_Inputs})
        data_param_dictionary.update({f'Molecule_{molecule}_Labels': Sample_Label})

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

def kFold_validation(current_task_specific_architecture, current_shared_architecture, molecule_list, group_no,
                     random_seed):
    data_param_dictionary, Number_of_Features = readData(molecule_list)

    data_param_dict_for_specific_task = {}

    for molecule in molecule_list:

        Sample_Inputs = data_param_dictionary[f'Molecule_{molecule}_FF_Inputs']
        Sample_Label = data_param_dictionary[f'Molecule_{molecule}_Labels']

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

            data_param_dict_for_specific_task.update({f'Molecule_{molecule}_fold_{fold}_X_train': X_train})
            data_param_dict_for_specific_task.update({f'Molecule_{molecule}_fold_{fold}_X_test': X_test})

            data_param_dict_for_specific_task.update({f'Molecule_{molecule}_fold_{fold}_y_train': y_train})
            data_param_dict_for_specific_task.update({f'Molecule_{molecule}_fold_{fold}_y_test': y_test})

            tmp = (current_task_specific_architecture, current_shared_architecture, molecule_list,
                   data_param_dict_for_specific_task,
                   Number_of_Features, fold, group_no)

            ALL_FOLDS.append(tmp)

            fold += 1

    number_of_models = 5
    current_idx = random.sample(range(len(ALL_FOLDS)), number_of_models)
    args = [ALL_FOLDS[index] for index in sorted(current_idx)]

    # print(len(args))
    number_of_pools = len(args)
    # pool = mp.Pool(number_of_pools)
    # all_scores = pool.starmap(final_model, args)
    # pool.close()cd ..

    with ThreadPool(number_of_pools) as tp:
        all_scores = tp.starmap(final_model, args)
    tp.join()

    score_param_per_task_group_per_fold = {}
    error_param_per_task_group_per_fold = {}
    AP_param_per_task_group_per_fold = {}
    for molecule in molecule_list:
        score_param_per_task_group_per_fold.update({f'molecule_{molecule}': []})
        error_param_per_task_group_per_fold.update({f'molecule_{molecule}': []})
        AP_param_per_task_group_per_fold.update({f'molecule_{molecule}': []})

    scores = []
    for c in range(len(all_scores)):
        scores.append(all_scores[c][0])
    # print(f'scores = {scores}')
    # print(len(molecule_list))
    for score in scores:
        idx = 1
        for molecule in molecule_list:
            score_param_per_task_group_per_fold[f'molecule_{molecule}'].append(score[idx])
            idx += 1
            if idx == len(molecule_list) + 1:
                break
        for molecule in molecule_list:
            error_param_per_task_group_per_fold[f'molecule_{molecule}'].append(1 - score[idx])
            idx = idx + 1

    errorRate = []
    # for t, err_val in error_param_per_task_group_per_fold.items():
    #     errorRate.append(np.mean(err_val))

    for c in range(len(all_scores)):
        errorRate.append(all_scores[c][2])

    ap = []
    for c in range(len(all_scores)):
        ap.append(all_scores[c][3])

    total_loss_per_task_group_per_fold = 0
    for t, loss_val in score_param_per_task_group_per_fold.items():
        total_loss_per_task_group_per_fold += np.mean(loss_val)

    # print(f'error_param_per_task_group_per_fold = {error_param_per_task_group_per_fold}')
    # print(f'ap = {ap}')
    # print(f'ap = {ap}')

    avg_error = np.mean(errorRate)
    AP = np.mean(ap)

    print(
        f'total_loss_per_task_group_per_fold = {total_loss_per_task_group_per_fold}\tavg_error = {avg_error}\tAP = {AP}')

    return total_loss_per_task_group_per_fold, avg_error, AP

def final_model(task_hyperparameters, shared_hyperparameters, molecule_list, data_param_dict_for_specific_task,
                Number_of_Features, fold, group_no):

        filepath = f'SavedModels/clusters_{datasetName}_Group_{group_no}_{fold}.h5'
        MTL_model_param = {}
        input_layers = []

        train_data = []
        train_label = []
        test_data = []
        test_label = []

        for molecule in molecule_list:

            train_data.append(data_param_dict_for_specific_task[f'Molecule_{molecule}_fold_{fold}_X_train'])
            train_label.append(data_param_dict_for_specific_task[f'Molecule_{molecule}_fold_{fold}_y_train'])

            test_data.append(data_param_dict_for_specific_task[f'Molecule_{molecule}_fold_{fold}_X_test'])
            test_label.append(data_param_dict_for_specific_task[f'Molecule_{molecule}_fold_{fold}_y_test'])

            hyperparameters = copy.deepcopy(task_hyperparameters[molecule])
            Input_FF = tf.keras.layers.Input(shape=(Number_of_Features,))
            input_layers.append(Input_FF)

            MTL_model_param.update({f'molecule_{molecule}_Input_FF': Input_FF})

            if hyperparameters['preprocessing_FF_layers'] > 0:
                hidden_ff = Dense(hyperparameters['preprocessing_FF_Neurons'][0], activation='relu')(Input_FF)
                for h in range(1, hyperparameters['preprocessing_FF_layers']):
                    hidden_ff = Dense(hyperparameters['preprocessing_FF_Neurons'][h], activation='relu')(hidden_ff)
                MTL_model_param.update({f'molecule_{molecule}_ff_preprocessing_model': hidden_ff})

        SHARED_module_param_FF = {}

        for h in range(0, shared_hyperparameters['shared_FF_Layers']):
            shared_ff = tf.keras.layers.Dense(shared_hyperparameters['shared_FF_Neurons'][h], activation='relu')
            SHARED_module_param_FF.update({f'FF_{h}': shared_ff})

        for molecule in molecule_list:
            shared_FF = SHARED_module_param_FF['FF_0'](MTL_model_param[f'molecule_{molecule}_ff_preprocessing_model'])
            for h in range(1, shared_hyperparameters['shared_FF_Layers']):
                shared_FF = SHARED_module_param_FF[f'FF_{h}'](shared_FF)

            MTL_model_param.update({f'molecule_{molecule}_last_hidden_layer': shared_FF})

        output_layers = []

        for molecule in molecule_list:
            outputLayer = Dense(1, activation='sigmoid', name=f'Molecule_{molecule}')

            shared_model = Model(inputs=MTL_model_param[f'molecule_{molecule}_Input_FF'],
                                 outputs=MTL_model_param[f'molecule_{molecule}_last_hidden_layer'])

            combinedInput = concatenate([MTL_model_param[f'molecule_{molecule}_Input_FF'], shared_model.output])
            output = outputLayer(combinedInput)

            output_layers.append(output)

        # print(output_layers)

        finalModel = Model(inputs=input_layers, outputs=output_layers)

        # Compile model

        opt = tf.keras.optimizers.Adam(learning_rate=shared_hyperparameters['learning_rate'])
        finalModel.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        checkpoint = ModelCheckpoint(filepath, verbose=0, monitor='val_loss', save_best_only=True, mode='auto')
        number_of_epoch = 400
        history = finalModel.fit(x=train_data,
                                 y=tuple(train_label),
                                 shuffle=True,
                                 epochs=number_of_epoch,
                                 batch_size=264,
                                 validation_data=(test_data,
                                                  tuple(test_label)),
                                 callbacks=[checkpoint],
                                 verbose=0)

        finalModel = tf.keras.models.load_model(filepath)
        scores = finalModel.evaluate(tuple(test_data), tuple(test_label), verbose=0)

        y_pred = finalModel.predict(test_data)
        y_pred = Splitting_Values(y_pred)
        y_test = Splitting_Values(test_label)

        # print(f'y_pred = {y_pred[:5]}')
        # print(f'y_test = {y_test[:5]}')

        predicted_val = []
        for i in y_pred:
            if i < 0.75:
                predicted_val.append(0)
            else:
                predicted_val.append(1)

        Error = [abs(a_i - b_i) for a_i, b_i in zip(y_test, predicted_val)]
        wrong_pred = Error.count(1)
        errorRate = wrong_pred / len(y_test)

        ap = average_precision_score(y_test, y_pred)

        if os.path.exists(filepath):
            os.remove(os.path.join(filepath))
        trainable_count = 1111

        # print(f'Score = {scores}')
        # print(f'errorRate = {errorRate}')
        # print(f'ap = {ap}')
        return scores, trainable_count, errorRate, ap


def prep_task_specific_arch(current_task_group):
    TASK_Specific_Arch = {}
    for group_no in current_task_group.keys():
        Number_of_Tasks = current_task_group[group_no]
        initial_task_specific_architecture = {}
        for n in Number_of_Tasks:
            initial_task_specific_architecture.update({n: {'preprocessing_FF_layers': 1,
                                                           'preprocessing_FF_Neurons': [5],
                                                           # 'postprocessing_FF_layers': 0,
                                                           # 'postprocessing_FF_Neurons': [0]
                                                           }})

        TASK_Specific_Arch.update({group_no: initial_task_specific_architecture})
    return TASK_Specific_Arch

if __name__ == "__main__":
    datasetName = 'Chemical'
    DataPath = f'../../Dataset/{datasetName.upper()}/'
    ResultPath = '../../Results/'

    ChemicalData = pd.read_csv(f'{DataPath}ChemicalData_All.csv', low_memory=False)
    TASKS = list(ChemicalData['180'].unique())

    task_len = {}
    variance_dict = {}
    std_dev_dict = {}
    dist_dict = {}
    Single_res_dict = {}
    STL_error = {}
    STL_AP = {}
    task_info = pd.read_csv(f'{DataPath}Task_Information_{datasetName}.csv')
    task_distance_info = pd.read_csv(f'{DataPath}Task_Distance_{datasetName}.csv')
    single_results = pd.read_csv(f'{ResultPath}/STL/STL_{datasetName}_NN.csv')
    pair_results = pd.read_csv(f'{ResultPath}/Pairwise/{datasetName}_Results_from_Pairwise_Training_ALL_NN.csv')

    for Selected_Task in TASKS:
        task_data = task_info[task_info.Molecule == Selected_Task].reset_index()
        task_len.update({Selected_Task: task_data.Dataset_Size[0]})
        variance_dict.update({Selected_Task: task_data.Variance[0]})
        std_dev_dict.update({Selected_Task: task_data.Std_Dev[0]})
        dist_dict.update({Selected_Task: task_data.Average_Hamming_Distance_within_Task[0]})
        single_res = single_results[single_results.Task == Selected_Task].reset_index()
        Single_res_dict.update({Selected_Task: single_res.LOSS[0]})
        STL_error.update({Selected_Task: single_res.Error_Rate[0]})
        STL_AP.update({Selected_Task: single_res.Avg_Precision[0]})

    Pairwise_res_dict = {}
    PTL_error = {}
    PTL_AP = {}
    Task1 = list(pair_results.Task_1)
    Task2 = list(pair_results.Task_2)
    Pairs = [tuple(sorted([Task1[i], Task2[i]])) for i in range(len(Task1))]
    for p in Pairs:
        task1 = p[0]
        task2 = p[1]
        pair_res = pair_results[(pair_results.Task_1 == task1) & (pair_results.Task_2 == task2)].reset_index()
        if len(pair_res) == 0:
            pair_res = pair_results[(pair_results.Task_1 == task2) & (pair_results.Task_2 == task1)].reset_index()
        Pairwise_res_dict.update({p: pair_res.Total_Loss[0]})
        PTL_error.update({p: 1 - pair_res.Avg_Accuracy[0]})
        avg_ap = (pair_res.Individual_AP_task1[0]+pair_res.Individual_AP_task2[0])/2
        PTL_AP.update({p: avg_ap})

    print(len(Single_res_dict), len(Pairwise_res_dict),len(STL_error), len(STL_AP), len(PTL_error), len(PTL_AP))

    number_of_epoch = 400
    min_task_groups = 5
    num_folds = 10
    initial_shared_architecture = {'adaptive_FF_neurons': 5, 'shared_FF_Layers': 1, 'shared_FF_Neurons': [6],
                                   'learning_rate': 0.00029055748415145487}

    Task_group = []
    Total_Loss = []
    Individual_Group_Score = []
    Individual_Error_Rate = []
    Individual_AP = []
    Number_of_Groups = []

    Prev_Groups = {}
    run = 1
    for Type in ['Exponential', 'NonNegative', 'WeightMatrix']:
        if Type == 'Exponential' or Type == 'NonNegative':
            cluster_algorithms = ['Hierarchical']
        else:
            cluster_algorithms = ['Hierarchical', 'KMeans']

        for ClusterType in cluster_algorithms:
            if Type == 'WeightMatrix':
                TASK_Clusters = pd.read_csv(f'data/{datasetName}_{ClusterType}_Clusters_{Type}.csv')
            else:
                TASK_Clusters = pd.read_csv(f'data/{datasetName}_Clusters_{Type}.csv')

            TASK_Group = list(TASK_Clusters.TASK_Group)

            for count in range(0, len(TASK_Group)):
                print(f'Initial Training for {datasetName}-partition {count}')
                task_group = TASK_Group[count]
                task_group = ast.literal_eval(task_group)

                TASK_Specific_Arch = prep_task_specific_arch(task_group)
                random_seed = random.randint(0, 100)

                args_tasks = []
                group_score = {}
                group_avg_err = {}
                group_avg_AP = {}
                tot_loss = 0
                for group_no, task in task_group.items():
                    group_score.update({group_no: 0})
                    group_avg_err.update({group_no: 0})
                    group_avg_AP.update({group_no: 0})
                    grouping_sorted = tuple(sorted(task))
                    if grouping_sorted not in Prev_Groups.keys():
                        if len(task) == 1:
                            loss = Single_res_dict[task[0]]
                            avg_error = STL_error[task[0]]
                            AP = STL_AP[task[0]]

                        elif len(task) == 2:
                            loss = Pairwise_res_dict[grouping_sorted]
                            avg_error = PTL_error[grouping_sorted]
                            AP = PTL_AP[grouping_sorted]

                        else:
                            tmp = (TASK_Specific_Arch[group_no], initial_shared_architecture, task, group_no, random_seed)
                            args_tasks.append(tmp)

                            loss, avg_error, AP = kFold_validation(*tmp)

                        tot_loss += loss
                        group_score[group_no] = loss
                        group_avg_err[group_no] = avg_error
                        group_avg_AP[group_no] = AP
                        Prev_Groups[grouping_sorted] = (loss, avg_error, AP)

                    else:
                        loss, avg_error, AP = Prev_Groups[grouping_sorted]
                        if group_no not in group_score.keys():
                            group_score.update({group_no: loss})
                            group_avg_err.update({group_no: avg_error})
                            group_avg_AP.update({group_no: AP})
                        else:
                            group_score[group_no] = loss
                            group_avg_err[group_no] = avg_error
                            group_avg_AP[group_no] = AP

                        tot_loss += loss

                print(f'tot_loss = {tot_loss}')
                print(f'group_score = {group_score}')
                Task_group.append(task_group)
                Number_of_Groups.append(len(task_group))
                Total_Loss.append(tot_loss)
                Individual_Group_Score.append(group_score.copy())
                Individual_Error_Rate.append(group_avg_err.copy())
                Individual_AP.append(group_avg_AP.copy())
                print(Individual_Group_Score)

                print(len(TASK_Group), len(Total_Loss), len(Number_of_Groups), len(Task_group), len(Individual_Group_Score))
                if len(Total_Loss)%10 == 0:
                    temp_res= pd.DataFrame({'Total_Loss': Total_Loss,
                                            'Number_of_Groups': Number_of_Groups,
                                            'Task_group': Task_group,
                                            'Individual_Group_Score': Individual_Group_Score,
                                            'Individual_Error_Rate': Individual_Error_Rate,
                                            'Individual_AP': Individual_AP})
                    temp_res.to_csv(f'{datasetName}_MTL_Results_for_{ClusterType}_Clusters_{Type}_{len(Total_Loss)}_run_{run}.csv', index=False)

                    if len(Total_Loss)>10:
                        old_file = f'{datasetName}_MTL_Results_for_{ClusterType}_Clusters_{Type}_{len(Total_Loss)-10}_run_{run}.csv'
                        if os.path.exists(old_file):
                            os.remove(os.path.join(old_file))

            print(len(Total_Loss), len(Number_of_Groups), len(Task_group), len(Individual_Group_Score))
            initial_results = pd.DataFrame({'Total_Loss': Total_Loss,
                                            'Number_of_Groups': Number_of_Groups,
                                            'Task_group': Task_group,
                                            'Individual_Group_Score': Individual_Group_Score,
                                            'Individual_Error_Rate': Individual_Error_Rate,
                                            'Individual_AP': Individual_AP})
            initial_results.to_csv(
                f'{datasetName}_MTL_Results_for_{ClusterType}_Clusters_{Type}_run_{run}.csv',
                index=False)

