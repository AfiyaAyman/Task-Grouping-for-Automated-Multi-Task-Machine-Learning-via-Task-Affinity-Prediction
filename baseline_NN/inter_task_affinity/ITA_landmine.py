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
from sklearn.metrics import roc_auc_score
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
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



            fold += 1
    ALL_FOLDS = []
    for fold in range(num_folds):
        tmp = (current_task_specific_architecture, current_shared_architecture, landmine_list,
               data_param_dict_for_specific_task,
               Number_of_Features, fold, group_no)

        ALL_FOLDS.append(tmp)

    number_of_pools = len(ALL_FOLDS)
    with ThreadPool(number_of_pools) as tp:
        all_scores = tp.starmap(final_model, ALL_FOLDS)
    tp.join()
    print(f'all_scores = {all_scores}')
    print(f'score = {np.mean(all_scores)}')
    

def postprocessing_feedforward(hyperparameters, last_hidden):
    hidden_ff = Dense(hyperparameters['postprocessing_FF_Neurons'][0], activation='relu')(last_hidden)
    for h in range(1, hyperparameters['postprocessing_FF_layers']):
        hidden_ff = Dense(hyperparameters['postprocessing_FF_Neurons'][h], activation='relu')(hidden_ff)

    return hidden_ff

def train_step(X,y, model, optimizer,loss_object,filepath):
    # print(f'X = {np.shape(X)}')
    with tf.GradientTape() as model_tape:
        _y_train_pred = model((X), training=True)
        _y_train_predicted_values = tf.convert_to_tensor(_y_train_pred)
        log_loss = loss_object(y, _y_train_predicted_values)

    gradients = model_tape.gradient(log_loss, model.trainable_variables)
    '''Saving for current epoch'''
    model.save(filepath)
    '''apply full gradient for regular training'''
    model = tf.keras.models.load_model(filepath)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return log_loss, model, gradients

def final_model(task_hyperparameters, shared_hyperparameters, landmine_list, data_param_dict_for_specific_task,
                Number_of_Features, fold, group_no):
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
        shared_ff = tf.keras.layers.Dense(shared_hyperparameters['shared_FF_Neurons'][h], activation='relu',name=f'SharedLayer_{h}')
        SHARED_module_param_FF.update({f'FF_{h}': shared_ff})

    for landmine in landmine_list:
        # Input_FF = MTL_model_param[f'landmine_{landmine}_Input_FF']
        preprocessed = MTL_model_param[f'landmine_{landmine}_ff_preprocessing_model']
        shared_FF = SHARED_module_param_FF['FF_0'](preprocessed)
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
    # if fold ==0:
    #     finalModel.summary()
    #     from tensorflow import keras
    #     keras.utils.plot_model(finalModel, f"multitask_model_Landmine_{group_no}.png", show_shapes=True)


    # from tensorflow import keras
    # keras.utils.plot_model(finalModel, f"multitask_model_Landmine_{group_no}.png", show_shapes=True)

    # Compile model

    opt = tf.keras.optimizers.Adam(learning_rate=shared_hyperparameters['learning_rate'])
    # finalModel.compile(optimizer=opt, loss='binary_crossentropy')

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    finalModel.compile(optimizer=opt, loss=loss_fn)

    # print(model.summary)
    # checkpoint = ModelCheckpoint(filepath, verbose=2, monitor='val_loss', save_best_only=True, mode='auto')
    # # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, min_delta=0.1, patience=10)
    # number_of_epoch = 400
    # history = finalModel.fit(x=train_data,
    #                          y=tuple(train_label),
    #                          shuffle=True,
    #                          epochs=number_of_epoch,
    #                          batch_size=32,
    #                          validation_data=(test_data,
    #                                           tuple(test_label)),
    #                          callbacks=[checkpoint],
    #                          verbose=0)
    # finalModel = tf.keras.models.load_model(filepath)
    # scores = finalModel.evaluate(tuple(test_data), tuple(test_label), verbose=0)

    number_of_epoch = 1000
    # opt = tf.keras.optimizers.Adam(learning_rate=shared_hyperparameters['learning_rate'])
    # finalModel.compile(optimizer=opt, loss='binary_crossentropy')

    # checkpoint = ModelCheckpoint(filepath, verbose=2, monitor='val_loss', save_best_only=True, mode='auto')
    task_log_loss_before_applying_gradient = {}
    task_log_loss_after_applying_gradient = {}
    for t in range(0, len(landmine_list)):
        task_log_loss_before_applying_gradient[f'Task_{landmine_list[t]}'] = []
        task_log_loss_after_applying_gradient[f'Task_{landmine_list[t]}_Gradient'] = {}
        for tt in range(0, len(landmine_list)):
            task_log_loss_after_applying_gradient[f'Task_{landmine_list[t]}_Gradient'][f'Task_{landmine_list[tt]}'] = []

    Bin_CrossEntropy = []
    Val_Bin_CrossEntropy = []
    import tqdm
    for epoch in tqdm.tqdm(range(0, number_of_epoch, 1)):

        filepath = f'SavedModels/fifty_Landmine_epoch_{epoch}_fold_{fold}.h5'

        if (epoch > 0 and epoch <= 150) and epoch % 5 == 0:
            task_list = landmine_list
            updated_shared_gradient = {}
            model_from_prev_epoch = f'SavedModels/fifty_Landmine_epoch_{epoch - 1}_fold_{fold}.h5'
            PrevModel = tf.keras.models.load_model(model_from_prev_epoch)

            # print(f'\n\n!!!!!!!!!!!!!\n')
            for t in range(0, len(task_list)):
                with tf.GradientTape() as model_tape:
                    _y_train_pred = PrevModel((train_data), training=True)
                    _y_train_predicted_values = tf.convert_to_tensor(_y_train_pred)
                    log_loss = loss_fn(train_label[t], _y_train_predicted_values[t])

                task_log_loss_before_applying_gradient[f'Task_{task_list[t]}'].append(
                    tf.keras.backend.get_value(log_loss))
                task_gradients = model_tape.gradient(log_loss, PrevModel.trainable_variables)

                '''Store weight and bias for the shared layer'''
                for var, g in zip(PrevModel.trainable_variables, task_gradients):

                    if f'SharedLayer_' in var.name:
                        layer_name = var.name.split('/')[0]
                        if f'Task_{task_list[t]}_{layer_name}' not in updated_shared_gradient.keys():
                            updated_shared_gradient[f'Task_{task_list[t]}_{layer_name}'] = []
                        if f'kernel' in var.name:
                            updated_shared_gradient[f'Task_{task_list[t]}_{layer_name}'].append(g)
                        if f'bias' in var.name:
                            updated_shared_gradient[f'Task_{task_list[t]}_{layer_name}'].append(g)

            '''apply gradient(calculated for each task) to the shared layer'''
            # print(f'\n\n*************\n')
            for t in range(0, len(task_list)):
                model_from_prev_epoch = f'SavedModels/fifty_Landmine_epoch_{epoch - 1}_fold_{fold}.h5'
                PrevModel = tf.keras.models.load_model(model_from_prev_epoch)
                for grad_idx in range(len(prev_gradient)):
                    var = PrevModel.trainable_variables[grad_idx]

                    '''Only update gradients for shared module - based on one task'''
                    new_layer_name = var.name.split('/')
                    if f'SharedLayer_' in var.name:
                        # print(f'var.name: {var.name}\tprev_gradient[grad_idx]: {prev_gradient[grad_idx]}')
                        if f'kernel' in new_layer_name[1]:
                            curr_kernel = updated_shared_gradient[f'Task_{task_list[t]}_{new_layer_name[0]}'][0]
                            prev_gradient[grad_idx] = curr_kernel
                        if f'bias' in new_layer_name[1]:
                            curr_bias = updated_shared_gradient[f'Task_{task_list[t]}_{new_layer_name[0]}'][1]
                            prev_gradient[grad_idx] = curr_bias
                        # print(f'var.name: {var.name}\tprev_gradient[grad_idx]: {prev_gradient[grad_idx]}')

                opt.apply_gradients(zip(prev_gradient, PrevModel.trainable_variables))
                '''Get loss for each task after applying gradient'''
                _y_train_pred = PrevModel((train_data), training=False)
                _y_train_predicted_values = tf.convert_to_tensor(_y_train_pred)
                for tt in range(0, len(task_list)):
                    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
                    log_loss = tf.reduce_sum(
                        bce(train_label[tt], _y_train_predicted_values[tt]))
                    task_log_loss_after_applying_gradient[f'Task_{task_list[t]}_Gradient'][
                        f'Task_{task_list[tt]}'].append(tf.keras.backend.get_value(log_loss))

        if epoch==160:
            before_loss = pd.DataFrame.from_dict(task_log_loss_before_applying_gradient)
            after_loss = pd.DataFrame.from_dict(task_log_loss_after_applying_gradient)
            datasetName = 'Landmine'
            before_loss.to_csv(f'before_loss_{datasetName}_fold_{fold}_{len(landmine_list)}_Tasks.csv')
            after_loss.to_csv(f'after_loss_{datasetName}_fold_{fold}_{len(landmine_list)}_Tasks.csv')


        '''main training loop'''
        batch_size = 128
        # print(int(len(train_data[1]) / batch_size))
        # print(f'train_data = {np.shape(train_data)}\ttrain_label = {np.shape(train_label)}')
        loss_per_batch = []
        for bid in range(int(len(train_data[1]) / batch_size)):
            X = [train_data[t][bid * batch_size: (bid + 1) * batch_size] for t in range(len(landmine_list))]
            y = [train_label[t][bid * batch_size: (bid + 1) * batch_size] for t in range(len(landmine_list))]
            # print(f'X = {np.shape(X)}\ty = {np.shape(y)}')
            loss, finalModel, gradients = train_step(X, y, finalModel, opt, loss_fn, filepath)
            # print(f'e = {epoch}\tbid = {bid}\tloss = {loss}')
            loss_per_batch.append(loss)

        # print(f'epoch = {epoch}\tloss = {loss_per_batch}\t {np.mean(loss_per_batch)}\t{np.sum(loss_per_batch)}')
        log_loss = np.mean(loss_per_batch)
        Bin_CrossEntropy.append(log_loss)

 
        _y_validation_values = finalModel(test_data, training=False)
        _y_validation_values = tf.convert_to_tensor(_y_validation_values)
        validation_log_loss = loss_fn(test_label, _y_validation_values)
   
        Val_Bin_CrossEntropy.append(validation_log_loss)
        # val_acc_metric.reset_states()
        prev_gradient = gradients


        if (epoch % 100 == 0 or epoch == number_of_epoch - 1):
            # print(f"Epoch = {epoch}\tTrain Bin_CrossEntropy: {log_loss}\tValidation Bin_CrossEntropy: {validation_log_loss}")
            scores = finalModel.evaluate(tuple(test_data), tuple(test_label), verbose=0)
            print(f"Epoch = {epoch}\tTest Bin_CrossEntropy: {scores[0]}\t {np.sum(scores[1:])} {np.mean(scores[1:])}")

    print("*" * 500)
    print(
        f"FINAL: Train Bin_CrossEntropy: {log_loss}\tValidation Bin_CrossEntropy: {validation_log_loss}")


    best_epoch = Bin_CrossEntropy.index(min(Bin_CrossEntropy))
    print(f'fold = {fold}, Best Score {min(Bin_CrossEntropy)} at epoch: {best_epoch}')
    filepath = f'SavedModels/fifty_Landmine_epoch_{best_epoch}_fold_{fold}.h5'
    finalModel = tf.keras.models.load_model(filepath)

    scores = finalModel.evaluate(tuple(test_data), tuple(test_label), verbose=0)
    y_pred = finalModel.predict(test_data)
    y_pred = Splitting_Values(y_pred)
    y_test = Splitting_Values(test_label)

    # print('Done')
    for e in range(0, number_of_epoch):
        if e!=best_epoch:
            old_models = f'SavedModels/fifty_Landmine_epoch_{e}_fold_{fold}.h5'
            if os.path.exists(old_models):
                os.remove(os.path.join(old_models))

    
    auc = 0
    try:
        auc = roc_auc_score(y_test, y_pred)
    except ValueError:
        pass

    print(f"fold = {fold}\t AUC = {auc}\tscores = {scores[0]}\tmean scores = {np.mean(scores[1:])}\tsum scores = {np.sum(scores[1:])}")
    return scores[0]


def prep_task_specific_arch(current_task_group):
    TASK_Specific_Arch = {}
    for group_no in current_task_group.keys():
        Number_of_Tasks = current_task_group[group_no]
        initial_task_specific_architecture = {}
        for n in Number_of_Tasks:
            initial_task_specific_architecture.update({n: {'preprocessing_FF_layers': 1,
                                                           'preprocessing_FF_Neurons': [2],
                                                           'postprocessing_FF_layers': 0,
                                                           'postprocessing_FF_Neurons': []
                                                           }})

        TASK_Specific_Arch.update({group_no: initial_task_specific_architecture})
    return TASK_Specific_Arch


def get_ITA():

    task_group = {0: TASKS}
    print(f'task_group = {task_group}')

    TASK_Specific_Arch = prep_task_specific_arch(task_group)
    random_seed = random.randint(0, 100)

    group_no = 0
    task = task_group[group_no]
    tmp = (TASK_Specific_Arch[group_no], initial_shared_architecture, task, group_no, random_seed)
    
    kFold_validation(*tmp)

if __name__ == "__main__":
    datasetName = 'Landmine'
    DataPath = f'../../Dataset/{datasetName.upper()}/'

    ResultPath = '../../Results'

    task_len = {}
    variance_dict = {}
    std_dev_dict = {}
    dist_dict = {}
    Single_res_dict = {}
    loss_dict = {}
    task_info = pd.read_csv(f'{DataPath}Task_Information_{datasetName}.csv')
    task_distance_info = pd.read_csv(f'{DataPath}Task_Distance_{datasetName}.csv')
    single_results = pd.read_csv(f'{ResultPath}STL/STL_{datasetName}_NN.csv')
    pair_results = pd.read_csv(f'{ResultPath}Pairwise/NN/{datasetName}_Results_from_Pairwise_Training_ALL_NN.csv')

    TASKS = [i for i in range(0, 29)]
    # TASKS = [i for i in range(0, 5)]
    print(len(TASKS))
    for Selected_Task in TASKS:
        task_data = task_info[task_info.Task_Name == Selected_Task].reset_index()
        task_len.update({Selected_Task: task_data.Dataset_Size[0]})
        variance_dict.update({Selected_Task: task_data.Variance[0]})
        std_dev_dict.update({Selected_Task: task_data.Std_Dev[0]})
        dist_dict.update({Selected_Task: task_data.Average_Euclidian_Distance_within_Task[0]})
        single_res = single_results[single_results.Task == Selected_Task].reset_index()
        Single_res_dict.update({Selected_Task: single_res.LOSS[0]})

    num_folds = 10


    initial_shared_architecture = {'adaptive_FF_neurons': 6, 'shared_FF_Layers': 2, 'shared_FF_Neurons': [6, 11],
                                   'learning_rate': 0.00020960514116261997}

    get_ITA()




