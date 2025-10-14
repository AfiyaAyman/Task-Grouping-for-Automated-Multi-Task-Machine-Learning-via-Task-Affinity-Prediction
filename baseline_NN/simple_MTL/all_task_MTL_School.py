import pandas as pd
import copy
import numpy as np
import os
import random
from multiprocessing.pool import ThreadPool
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
import time
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

        csv = (f"{DataPath}DATA/{sch_id}_School_Data_for_MTL.csv")
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

        '''*********************************'''
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


def kFold_validation(current_task_specific_architecture, current_shared_architecture, task, group_no, random_seed):

    data_param_dictionary, Number_of_Features_FF = readData(task)

    data_param_dict_for_specific_task = {}

    for f in range(num_folds):
        data_param_dict_for_specific_task[f] = {}


    for sch_id in task:
        Sample_Inputs = data_param_dictionary[f'School_{sch_id}_FF_Inputs']
        Sample_Label = data_param_dictionary[f'School_{sch_id}_Labels']

        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)

        fold = 0

        for train, test in kfold.split(Sample_Inputs):
            X_train = Sample_Inputs[train]
            X_test = Sample_Inputs[test]
            y_train = Sample_Label[train]
            y_test = Sample_Label[test]

            y_train = SplitLabels(y_train)
            y_test = SplitLabels(y_test)

            data_param_dict_for_specific_task[fold][f'School_{sch_id}_fold_{fold}_X_train'] = X_train
            data_param_dict_for_specific_task[fold][f'School_{sch_id}_fold_{fold}_X_test'] = X_test

            data_param_dict_for_specific_task[fold][f'School_{sch_id}_fold_{fold}_y_train'] = y_train
            data_param_dict_for_specific_task[fold][f'School_{sch_id}_fold_{fold}_y_test'] = y_test

            fold+=1

    ALL_FOLDS = []
    for fold in range(num_folds):
        tmp = (current_task_specific_architecture, current_shared_architecture, task,
            data_param_dict_for_specific_task,
            Number_of_Features_FF,
            fold, group_no)

        ALL_FOLDS.append(tmp)

    number_of_pools = len(ALL_FOLDS)
    timeStart = time.time()

    with ThreadPool(number_of_pools) as tp:
        all_scores = tp.starmap(final_model, ALL_FOLDS)
    tp.join()
    print(f'Time required = {(time.time() - timeStart) / 60} minutes')


    scores = []
    weight_dict = {}
    for i in range(len(all_scores)):
        scores.append(all_scores[i][0])
        for task_name,weight in all_scores[i][2].items():
            if task_name not in weight_dict.keys():
                weight_dict[task_name] = []
            weight_dict[task_name].append(weight)


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

    print(f'total_loss_per_task_group_per_fold = {total_loss_per_task_group_per_fold}')

    file = open("weight_dict_School.txt", "w")
    file.write("weight_dict = " + repr(weight_dict) + "\n")
    file.close()

    return total_loss_per_task_group_per_fold


def final_model(task_hyperparameters, shared_hyperparameters, task_group_list, data_param_dict_for_specific_task,
                Number_of_Features, fold, group_no):

    data_param = copy.deepcopy(data_param_dict_for_specific_task[fold])

    filepath = f'SavedModels/Initial_School_Group_{group_no}_{fold}.h5'

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
            hidden_ff = Dense(hyperparameters['preprocessing_FF_Neurons'][0], activation='sigmoid',name=f'TaskSpecific_{sch_id}')(
                Input_FF)
            for h in range(1, hyperparameters['preprocessing_FF_layers']):
                hidden_ff = Dense(hyperparameters['preprocessing_FF_Neurons'][h], activation='sigmoid')(hidden_ff)

            MTL_model_param[f'sch_{sch_id}_fold_{fold}_ff_preprocessing_model'] = hidden_ff

    for h in range(0, shared_hyperparameters['shared_FF_Layers']):
        shared_ff = tf.keras.layers.Dense(shared_hyperparameters['shared_FF_Neurons'][h], activation='sigmoid')
        shared_module_param_FF[f'FF_{h}_fold_{fold}'] = shared_ff

    for sch_id in task_group_list:
        hyperparameters = copy.deepcopy(task_hyperparameters[sch_id])
        if hyperparameters['preprocessing_FF_layers'] > 0:
            preprocessed_ff = MTL_model_param[f'sch_{sch_id}_fold_{fold}_ff_preprocessing_model']
            shared_FF = shared_module_param_FF[f'FF_0_fold_{fold}'](preprocessed_ff)
        else:
            Input_FF = MTL_model_param[f'sch_{sch_id}_fold_{fold}_Input_FF']
            shared_FF = shared_module_param_FF[f'FF_0_fold_{fold}'](Input_FF)

        for h in range(1, shared_hyperparameters['shared_FF_Layers']):
            shared_FF = shared_module_param_FF[f'FF_{h}_fold_{fold}'](shared_FF)

        MTL_model_param[f'sch_{sch_id}_fold_{fold}_last_hidden_layer'] = shared_FF

    # output Neurons
    output_layers = []

    for sch_id in task_group_list:
        shared_model = Model(inputs=MTL_model_param[f'sch_{sch_id}_fold_{fold}_Input_FF'],
                             outputs=MTL_model_param[f'sch_{sch_id}_fold_{fold}_last_hidden_layer'])
        # combined_input = concatenate([MTL_model_param[f'sch_{sch_id}_fold_{fold}_Input_FF'], shared_model.output])
        # outputlayer = Dense(1, activation='linear', name=f'ExamScore_{sch_id}')(combined_input)
        outputlayer = Dense(1, activation='linear', name=f'ExamScore_{sch_id}')(shared_model.output)
        output_layers.append(outputlayer)

    finalModel = Model(inputs=input_layers, outputs=output_layers)

    from keras.utils.layer_utils import count_params
    trainable_count = count_params(finalModel.trainable_weights)

    opt = tf.keras.optimizers.Adam(learning_rate=shared_hyperparameters['learning_rate'])
    finalModel.compile(optimizer=opt, loss='mse')




    checkpoint = ModelCheckpoint(filepath, verbose=2, monitor='val_loss', save_best_only=True, mode='auto')
    batch_size = 32
    number_of_epoch = 1000
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
    print("Weights and biases of the layers after training the model: \n")
    tmp_weight = {}
    for layer in finalModel.layers:
        config = layer.get_config()

        if f'ExamScore_' in config['name']:
            task_name = config['name']

            weight = layer.get_weights()[0]
            bias = layer.get_weights()[1]

            weight_and_bias = []
            for w in weight:
                weight_and_bias.append(w[0])
            weight_and_bias.append(bias[0])

            if task_name not in tmp_weight:
                tmp_weight[task_name] = weight_and_bias

    scores = finalModel.evaluate(test_data, test_label, verbose=0)

    return scores, trainable_count,tmp_weight


def prep_task_specific_arch(current_task_group):
    TASK_Specific_Arch = {}
    for group_no in current_task_group.keys():
        Number_of_Tasks = current_task_group[group_no]
        initial_task_specific_architecture = {}
        for n in Number_of_Tasks:
            initial_task_specific_architecture.update({n: {'preprocessing_FF_layers': 1,
                                                           'preprocessing_FF_Neurons': [3],
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

    task_group = {0:TASKS}
    print(f'task_group = {task_group}')


    TASK_Specific_Arch = prep_task_specific_arch(task_group)
    print(TASK_Specific_Arch)
    random_seed = random.randint(0, 100)

    args_tasks = []
    group_score = {}
    tot_loss = 0
    group_no = 0
    task = task_group[group_no]
    group_score.update({group_no: 0})
    tmp = (TASK_Specific_Arch[group_no], initial_shared_architecture, task, group_no, random_seed)
    args_tasks.append(tmp)

    loss = kFold_validation(*tmp)
    tot_loss+=loss
    group_score[group_no] = loss

    Task_group.append(task_group)
    Number_of_Groups.append(len(task_group))
    Total_Loss.append(tot_loss)
    Individual_Group_Score.append(group_score.copy())
    print(f'Individual_Group_Score = {Individual_Group_Score}')
    print(f'tot_loss = {tot_loss}')


if __name__ == "__main__":
    num_folds = 10
    initial_shared_architecture = {'adaptive_FF_neurons': 4, 'shared_FF_Layers': 2, 'shared_FF_Neurons': [3, 2],
                                   'learning_rate': 0.033806674289462206,
    #                                'activation_func' :'tanh'
                                   }

    '''Global Files and Data for Task-Grouping Predictor'''
    task_len = {}
    variance_dict = {}
    std_dev_dict = {}
    dist_dict = {}
    Single_res_dict = {}

    datasetName = 'School'
    DataPath = f'../../Dataset/{datasetName.upper()}/'
    ResultPath = '../../Results'

    TASKS = [i for i in range(1, 140)]


    all_task_MTL()




