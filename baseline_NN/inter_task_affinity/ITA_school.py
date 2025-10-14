import matplotlib.pyplot as plt
import pandas as pd
import copy
import numpy as np
import os
import random
import multiprocessing as mp
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
import time
import tqdm
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

            fold += 1

    ALL_FOLDS = []
    for fold in range(num_folds):
        tmp = (current_task_specific_architecture, current_shared_architecture, task,
               data_param_dict_for_specific_task,
               Number_of_Features_FF,
               fold, group_no)

        ALL_FOLDS.append(tmp)

    number_of_pools = len(ALL_FOLDS)
    pool = mp.Pool(number_of_pools)
    all_scores = pool.starmap(final_model, ALL_FOLDS)
    pool.close()
    print(f'all_scores = {all_scores}')
    print(f'final score = {np.mean(all_scores)}')

def train_step(X,y, model, optimizer,filepath):

    with tf.GradientTape() as model_tape:
        _y_train_pred = model((X), training=True)
        _y_train_predicted_values = tf.convert_to_tensor(_y_train_pred)
        mse_loss = tf.math.reduce_mean(tf.math.square(y - _y_train_predicted_values))

    gradients = model_tape.gradient(mse_loss, model.trainable_variables)
    '''Saving for current epoch'''
    model.save(filepath)
    '''apply full gradient for regular training'''
    model = tf.keras.models.load_model(filepath)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return mse_loss,model,gradients

def final_model(task_hyperparameters, shared_hyperparameters, task_group_list, data_param_dict_for_specific_task,
                Number_of_Features, fold, group_no):
    gpus = tf.config.list_physical_devices('GPU')
    gpu_device = gpus[0]
    core_config = tf.config.experimental.set_visible_devices(gpu_device, 'GPU')
    tf.config.experimental.set_memory_growth(gpu_device, True)
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=core_config))

    data_param = copy.deepcopy(data_param_dict_for_specific_task[fold])

    filepath = f'SavedModels/fifty_School_Group_{fold}.h5'

    MTL_model_param = {}
    shared_module_param_FF = {}

    input_layers = []

    train_data = []
    train_label = []
    test_data = []
    test_label = []

    '''Separate Input and PreProcessing'''
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
            hidden_ff = Dense(hyperparameters['preprocessing_FF_Neurons'][0], activation='sigmoid',name=f'TaskSpecific_{sch_id}')(
                Input_FF)
            for h in range(1, hyperparameters['preprocessing_FF_layers']):
                hidden_ff = Dense(hyperparameters['preprocessing_FF_Neurons'][h], activation='sigmoid')(hidden_ff)

            MTL_model_param[f'sch_{sch_id}_fold_{fold}_ff_preprocessing_model'] = hidden_ff

    '''Create Shared module'''
    for h in range(0, shared_hyperparameters['shared_FF_Layers']):
        shared_ff = tf.keras.layers.Dense(shared_hyperparameters['shared_FF_Neurons'][h], activation='sigmoid')
        shared_module_param_FF[f'FF_{h}_fold_{fold}'] = shared_ff

    '''Connect Shared module with Task Specific module'''
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

    '''Output Neurons'''
    output_layers = []
    for sch_id in task_group_list:
        shared_model = Model(inputs=MTL_model_param[f'sch_{sch_id}_fold_{fold}_Input_FF'],
                             outputs=MTL_model_param[f'sch_{sch_id}_fold_{fold}_last_hidden_layer'])

        outputlayer = Dense(1, activation='linear', name=f'ExamScore_{sch_id}')(shared_model.output)
        output_layers.append(outputlayer)

    finalModel = Model(inputs=input_layers, outputs=output_layers)



    '''Loss Function and Optimizer'''
    opt = tf.keras.optimizers.Adam(learning_rate=shared_hyperparameters['learning_rate'])
    finalModel.compile(optimizer=opt, loss='mse')
    # tf.keras.utils.plot_model(finalModel, f"MTL_School_{group_no}.png", show_shapes=True)
    # # print(finalModel.summary())


    checkpoint = ModelCheckpoint(filepath, verbose=2, monitor='val_loss', save_best_only=True, mode='auto')
    batch_size = 32
    number_of_epoch = 1500

    task_loss_before_applying_gradient = {}
    task_loss_after_applying_gradient = {}
    for t in range(0, len(task_group_list)):
        task_loss_before_applying_gradient[f'Task_{task_group_list[t]}'] = []
        task_loss_after_applying_gradient[f'Task_{task_group_list[t]}_Gradient'] = {}
        for tt in range(0, len(task_group_list)):
            task_loss_after_applying_gradient[f'Task_{task_group_list[t]}_Gradient'][f'Task_{task_group_list[tt]}'] = []

    MSE = []
    Val_MSE = []
    for epoch in tqdm.tqdm(range(0,number_of_epoch,1)):

        filepath = f'SavedModels/fifty_School_epoch_{epoch}_fold_{fold}.h5'

        '''Calculate Inter task affinity using task-specific gradient onto the model from prev_epoch'''
        if (epoch>0 and epoch<=100) and epoch%10 == 0:
            updated_shared_gradient = {}
            model_from_prev_epoch = f'SavedModels/fifty_School_epoch_{epoch - 1}_fold_{fold}.h5'
            PrevModel = tf.keras.models.load_model(model_from_prev_epoch)

            # print(f'\n\n!!!!!!!!!!!!!\n')
            for t in range(0, len(task_group_list)):
                with tf.GradientTape() as model_tape:
                    _y_train_pred = PrevModel((train_data), training=True)
                    _y_train_predicted_values = tf.convert_to_tensor(_y_train_pred)
                    task_mse = tf.math.reduce_mean(tf.math.square(train_label[t] - _y_train_predicted_values[t]))

                task_loss_before_applying_gradient[f'Task_{task_group_list[t]}'].append(tf.keras.backend.get_value(task_mse))
                task_gradients = model_tape.gradient(task_mse, PrevModel.trainable_variables)

                '''Store weight and bias for the shared layer'''
                for var, g in zip(PrevModel.trainable_variables, task_gradients):
                    if f'dense' in var.name:
                        # print(f'var.name: {var.name}\tg.shape: {g.shape}')
                        layer_name = var.name.split('/')[0]
                        if f'Task_{task_group_list[t]}_{layer_name}' not in updated_shared_gradient.keys():
                            updated_shared_gradient[f'Task_{task_group_list[t]}_{layer_name}'] = []
                        if f'kernel' in var.name:
                            updated_shared_gradient[f'Task_{task_group_list[t]}_{layer_name}'].append(g)
                        if f'bias' in var.name:
                            updated_shared_gradient[f'Task_{task_group_list[t]}_{layer_name}'].append(g)


            '''apply gradient(calculated for each task) to the shared layer'''
            # print(f'\n\n*************\n')
            for t in range(0, len(task_group_list)):
                model_from_prev_epoch = f'SavedModels/fifty_School_epoch_{epoch-1}_fold_{fold}.h5'
                PrevModel = tf.keras.models.load_model(model_from_prev_epoch)
                for grad_idx in range(len(prev_gradient)):
                    var = PrevModel.trainable_variables[grad_idx]

                    '''Only update gradients for shared module - based on one task'''
                    new_layer_name = var.name.split('/')
                    if f'dense' in var.name:
                        if f'kernel' in new_layer_name[1]:
                            curr_kernel = updated_shared_gradient[f'Task_{task_group_list[t]}_{new_layer_name[0]}'][0]
                            prev_gradient[grad_idx] = curr_kernel
                        if f'bias' in new_layer_name[1]:
                            curr_bias = updated_shared_gradient[f'Task_{task_group_list[t]}_{new_layer_name[0]}'][1]
                            prev_gradient[grad_idx] = curr_bias

                opt.apply_gradients(zip(prev_gradient, PrevModel.trainable_variables))
                '''Get loss for each task after applying gradient'''
                _y_train_pred = PrevModel((train_data), training=False)
                _y_train_predicted_values = tf.convert_to_tensor(_y_train_pred)
                for tt in range(0, len(task_group_list)):
                    task_mse = tf.math.reduce_mean(tf.math.square(train_label[tt] - _y_train_predicted_values[tt]))
                    task_loss_after_applying_gradient[f'Task_{task_group_list[t]}_Gradient'][f'Task_{task_group_list[tt]}'].append(tf.keras.backend.get_value(task_mse))

        if epoch==110:
            before_loss = pd.DataFrame.from_dict(task_loss_before_applying_gradient)
            after_loss = pd.DataFrame.from_dict(task_loss_after_applying_gradient)
            datasetName = 'School'
            before_loss.to_csv(f'before_loss_{datasetName}_fold_{fold}_{len(task_group_list)}_Tasks.csv')
            after_loss.to_csv(f'after_loss_{datasetName}_fold_{fold}_{len(task_group_list)}_Tasks.csv')

        '''main training loop'''
        batch_size = 32
        loss_per_batch = []
        for bid in range(int(len(train_data[1]) / batch_size)):
            X = [train_data[t][bid * batch_size: (bid + 1) * batch_size] for t in range(len(task_group_list))]
            y = [train_label[t][bid * batch_size: (bid + 1) * batch_size] for t in range(len(task_group_list))]
            loss, finalModel, gradients = train_step(X, y, finalModel, opt, filepath)
            loss_per_batch.append(loss)

        mse_loss = np.mean(loss_per_batch)
        MSE.append(mse_loss)

        _y_validation_values = finalModel(test_data, training=False)
        _y_validation_values = tf.convert_to_tensor(_y_validation_values)
        validation_mse_loss = tf.math.reduce_mean(tf.math.square(test_label - _y_validation_values))
        Val_MSE.append(validation_mse_loss)

        prev_gradient = gradients

        if (epoch % 100 == 0 or epoch == number_of_epoch - 1):
            print(f"Epoch = {epoch}\tTrain MSE: {mse_loss}\tValidation MSE: {validation_mse_loss}")
            scores = finalModel.evaluate(tuple(test_data), tuple(test_label), verbose=0)
            print(f"Epoch = {epoch}\tTest MSE: {scores[0]}\t {np.sum(scores[1:])} {np.mean(scores[1:])}")

    print("*" * 500)
    print(f"FINAL: Train MSE: {mse_loss}\tValidation MSE: {validation_mse_loss}\t {validation_mse_loss*len(task_group_list)}")

    best_epoch = MSE.index(min(MSE))
    print(f'fold = {fold}, Best Score {min(MSE)} at epoch: {best_epoch}')
    filepath = f'SavedModels/fifty_School_epoch_{best_epoch}_fold_{fold}.h5'
    finalModel = tf.keras.models.load_model(filepath)
    scores = finalModel.evaluate(tuple(test_data), tuple(test_label), verbose=0)
    print(f"fold = {fold}\t scores = {scores[0]}")
    for e in range(0,number_of_epoch):
        if e != best_epoch:
            old_models = f'SavedModels/fifty_School_epoch_{e}_fold_{fold}.h5'
            if os.path.exists(old_models):
                os.remove(os.path.join(old_models))

    return scores[0]


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


def get_ITA():
    task_group = {0:TASKS}
    print(f'task_group = {task_group}')


    TASK_Specific_Arch = prep_task_specific_arch(task_group)
    print(TASK_Specific_Arch)
    random_seed = random.randint(0, 100)

    group_no = 0
    task = task_group[group_no]
    tmp = (TASK_Specific_Arch[group_no], initial_shared_architecture, task, group_no, random_seed)

    kFold_validation(*tmp)


if __name__ == "__main__":
    num_folds = 10
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
    DataPath = f'../../Dataset/{datasetName.upper()}/'
    ResultPath = '../../Results'

    task_info = pd.read_csv(f'{DataPath}Task_Information_{datasetName}.csv')
    task_distance_info = pd.read_csv(f'{DataPath}Task_Distance_{datasetName}.csv')
    single_results = pd.read_csv(f'{ResultPath}STL/STL_{datasetName}_NN.csv')
    pair_results = pd.read_csv(f'{ResultPath}Pairwise/NN/{datasetName}_Results_from_Pairwise_Training_ALL_NN.csv')

    # TASKS = [i for i in range(1, 140)]
    TASKS = [i for i in range(1, 10)]
    for Selected_Task in TASKS:
        task_data = task_info[task_info.Task_Name == Selected_Task].reset_index()
        task_len.update({Selected_Task: task_data.Dataset_Size[0]})
        variance_dict.update({Selected_Task: task_data.Variance[0]})
        std_dev_dict.update({Selected_Task: task_data.Std_Dev[0]})
        dist_dict.update({Selected_Task: task_data.Average_Euclidian_Distance_within_Task[0]})
        single_res = single_results[single_results.Task == Selected_Task].reset_index()
        Single_res_dict.update({Selected_Task: single_res.Loss_MSE[0]})

    get_ITA()




