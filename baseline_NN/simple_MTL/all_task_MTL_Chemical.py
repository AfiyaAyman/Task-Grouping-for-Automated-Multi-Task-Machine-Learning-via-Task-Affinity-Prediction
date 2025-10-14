import matplotlib.pyplot as plt
import pandas as pd
import copy
import numpy as np
import os
import random
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.layers import *
from sklearn.metrics import average_precision_score
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from multiprocessing.pool import ThreadPool
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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

    number_of_pools = len(ALL_FOLDS)
    with ThreadPool(number_of_pools) as tp:
        all_scores = tp.starmap(final_model, ALL_FOLDS)
    tp.join()

    print(f'all_scores = {all_scores}')


    score_param_per_task_group_per_fold = {}
    error_param_per_task_group_per_fold = {}
    AP_param_per_task_group_per_fold={}
    for molecule in molecule_list:
        score_param_per_task_group_per_fold.update({f'molecule_{molecule}': []})
        error_param_per_task_group_per_fold.update({f'molecule_{molecule}': []})
        AP_param_per_task_group_per_fold.update({f'molecule_{molecule}': []})

    scores = []

    for c in range(len(all_scores)):
        scores.append(all_scores[c][0])

    for c in scores:
        for j in range(1, len(c)):
            score_param_per_task_group_per_fold[f'molecule_{molecule_list[j - 1]}'].append(c[j])

    errorRate = []
    for c in range(len(all_scores)):
        errorRate.append(all_scores[c][2])

    ap = []
    for c in range(len(all_scores)):
        ap.append(all_scores[c][5])

    total_loss_per_task_group_per_fold = 0
    for t, loss_val in score_param_per_task_group_per_fold.items():
        total_loss_per_task_group_per_fold += np.mean(loss_val)

    weight_dict = {}
    for c in range(len(all_scores)):
        for task_name, weight in all_scores[c][6].items():
            if task_name not in weight_dict.keys():
                weight_dict[task_name] = []
                # print(f'weight = {weight}')
            weight_dict[task_name].append(weight[0])


    print(f'errorRate = {errorRate}')
    print(f'ap = {ap}')
    # tot_error = []
    # for t, err in error_param_per_task_group_per_fold.items():
    #     tot_error.append(np.mean(err))
    avg_error = np.mean(errorRate)

    # tot_ap = []
    # for t, ap in AP_param_per_task_group_per_fold.items():
    #     tot_ap.append(np.mean(ap))
    AP = np.mean(ap)
    file = open("weight_dict_Chemical.txt", "w")
    file.write("" + repr(weight_dict) + "")
    file.close()

    return total_loss_per_task_group_per_fold,avg_error,AP

def final_model(task_hyperparameters, shared_hyperparameters, molecule_list, data_param_dict_for_specific_task,
                Number_of_Features, fold, group_no):
    filepath = f'SavedModels/Chemical_Group_{group_no}_{fold}.h5'
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
            hidden_ff = Dense(hyperparameters['preprocessing_FF_Neurons'][0], activation='relu',name=f'TaskSpecific_{molecule}')(Input_FF)
            for h in range(1, hyperparameters['preprocessing_FF_layers']):
                hidden_ff = Dense(hyperparameters['preprocessing_FF_Neurons'][h], activation='relu')(hidden_ff)
            MTL_model_param.update({f'molecule_{molecule}_ff_preprocessing_model': hidden_ff})

    SHARED_module_param_FF = {}

    for h in range(0, shared_hyperparameters['shared_FF_Layers']):
        shared_ff = tf.keras.layers.Dense(shared_hyperparameters['shared_FF_Neurons'][h], activation='relu',name=f'SharedLayer')
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

        # combinedInput = concatenate([MTL_model_param[f'molecule_{molecule}_Input_FF'], shared_model.output])
        # output = outputLayer(combinedInput)
        output = outputLayer(shared_model.output)
        output_layers.append(output)




    finalModel = Model(inputs=input_layers, outputs=output_layers)

    number_of_epoch = 700
    opt = tf.keras.optimizers.Adam(learning_rate=shared_hyperparameters['learning_rate'])
    finalModel.compile(optimizer=opt, loss='binary_crossentropy')


    checkpoint = ModelCheckpoint(filepath, verbose=2, monitor='val_loss', save_best_only=True, mode='auto')
    history = finalModel.fit(x=train_data,
                             y=tuple(train_label),
                             shuffle=True,
                             epochs=number_of_epoch,
                             batch_size=64,
                             validation_data=(test_data,
                                              tuple(test_label)),
                             callbacks=[checkpoint],
                             verbose=0)

    finalModel = tf.keras.models.load_model(filepath)

    tmp_weight = {}
    for layer in finalModel.layers:
        config = layer.get_config()
        if 'Molecule_' in config['name']:
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


    scores = finalModel.evaluate(tuple(test_data), tuple(test_label), verbose=0)



    y_pred = finalModel.predict(test_data)
    y_pred = Splitting_Values(y_pred)
    y_test = Splitting_Values(test_label)


    predicted_val = []
    for i in y_pred:
        if i < 0.75:
            predicted_val.append(0)
        else:
            predicted_val.append(1)

    Error = [abs(a_i - b_i) for a_i, b_i in zip(y_test, predicted_val)]
    wrong_pred = Error.count(1)
    errorRate = wrong_pred / len(y_test)

    tp = []
    fp = []
    fn = []
    tn = []

    for a_i, b_i in zip(predicted_val, y_test):
        if a_i == 1 and b_i == 1:
            tp.append(1)
        elif a_i == 1 and b_i == 0:
            fp.append(1)
        elif a_i == 0 and b_i == 1:
            fn.append(1)
        elif a_i == 0 and b_i == 0:
            tn.append(1)

    if sum(tp) + sum(fp) == 0:
        Precision = 0
    else:
        Precision = sum(tp) / (sum(tp) + sum(fp))
    if sum(tp) + sum(fn) == 0:
        Recall = 0
    else:
        Recall = sum(tp) / (sum(tp) + sum(fn))

    ap = average_precision_score(y_test, y_pred)

    if os.path.exists(filepath):
        os.remove(os.path.join(filepath))
    from keras.utils.layer_utils import count_params
    trainable_count = count_params(finalModel.trainable_weights)
    return scores, trainable_count, errorRate, Precision, Recall, ap,tmp_weight


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
    tmp = (TASK_Specific_Arch[group_no], initial_shared_architecture, task, group_no, random_seed)
    args_tasks.append(tmp)

    loss, errorRate, AP = kFold_validation(*tmp)
    tot_loss+=loss
    group_score[group_no] = loss

    Task_group.append(task_group)
    Number_of_Groups.append(len(task_group))
    Total_Loss.append(tot_loss)
    Individual_Group_Score.append(group_score.copy())
    print(f'Individual_Group_Score = {Individual_Group_Score}')
    print(f'tot_loss = {tot_loss}')
    print(f'errorRate = {errorRate}')
    print(f'AP = {AP}')


if __name__ == "__main__":
    datasetName = 'Chemical'
    DataPath = f'../../Dataset/{datasetName.upper()}/'

    ResultPath = '../../Results/'

    ChemicalData = pd.read_csv(f'{DataPath}ChemicalData_All.csv', low_memory=False)
    TASKS = list(ChemicalData['180'].unique())
    print(TASKS)
    # TASKS = TASKS[:10]


    num_folds = 10
    initial_shared_architecture = {'adaptive_FF_neurons': 5, 'shared_FF_Layers': 1, 'shared_FF_Neurons': [6],
                                   'learning_rate': 0.00029055748415145487}


    all_task_MTL()




