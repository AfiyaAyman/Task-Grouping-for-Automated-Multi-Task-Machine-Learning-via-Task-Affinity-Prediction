import matplotlib.pyplot as plt
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
    num_folds = 10

    data_param_dict_for_specific_task = {}

    for molecule in molecule_list:

        Sample_Inputs = data_param_dictionary[f'Molecule_{molecule}_FF_Inputs']
        Sample_Label = data_param_dictionary[f'Molecule_{molecule}_Labels']
        
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)

        fold = 0

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
            fold += 1


    ALL_FOLDS = []
    for fold in range(num_folds):

        tmp = (current_task_specific_architecture, current_shared_architecture, molecule_list,
               data_param_dict_for_specific_task,
               Number_of_Features, fold, group_no)

        ALL_FOLDS.append(tmp)

    with ThreadPool(10) as tp:
        all_scores = tp.starmap(final_model, ALL_FOLDS)
    tp.join()

    print(f'all_scores = {all_scores}')
    print(f'final score = {np.mean(all_scores)}')
    
def train_step(X,y, model, optimizer,loss_object,filepath):

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

    return log_loss,model,gradients

def calculate_ITA(task_list, task_log_loss_before_applying_gradient,task_log_loss_after_applying_gradient,opt,
                  loss_fn,prev_gradient,train_data,train_label,epoch,fold):

    updated_shared_gradient = {}
    model_from_prev_epoch = f'SavedModels/fifty_Chemical_epoch_{epoch - 1}_fold_{fold}.h5'
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
        model_from_prev_epoch = f'SavedModels/fifty_Chemical_epoch_{epoch - 1}_fold_{fold}.h5'
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

def final_model(task_hyperparameters, shared_hyperparameters, molecule_list, data_param_dict_for_specific_task,
                Number_of_Features, fold, group_no):
    gpus = tf.config.list_physical_devices('GPU')
    gpu_device = gpus[0]
    core_config = tf.config.experimental.set_visible_devices(gpu_device, 'GPU')
    tf.config.experimental.set_memory_growth(gpu_device, True)
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=core_config))

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
        shared_ff = tf.keras.layers.Dense(shared_hyperparameters['shared_FF_Neurons'][h], activation='relu',name=f'SharedLayer_{h}')
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
    # from tensorflow import keras
    # tf.keras.utils.plot_model(finalModel, f"MTL_Chem_{group_no}.png", show_shapes=True)
    # if fold == 1:
    #     print(finalModel.summary())
    #     tf.keras.utils.plot_model(finalModel, f"MTL_Chem_{group_no}.png", show_shapes=True)
    # for layer in finalModel.layers: print(layer.get_config(), layer.get_weights())
    #

    number_of_epoch = 1000
    opt = tf.keras.optimizers.Adam(learning_rate=shared_hyperparameters['learning_rate'])

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    finalModel.compile(optimizer=opt, loss=loss_fn)

    # train_acc_metric = tf.keras.losses.binary_crossentropy()
    # val_acc_metric = tf.keras.losses.binary_crossentropy()


    # checkpoint = ModelCheckpoint(filepath, verbose=2, monitor='val_loss', save_best_only=True, mode='auto')
    task_log_loss_before_applying_gradient = {}
    task_log_loss_after_applying_gradient = {}
    for t in range(0, len(molecule_list)):
        task_log_loss_before_applying_gradient[f'Task_{molecule_list[t]}'] = []
        task_log_loss_after_applying_gradient[f'Task_{molecule_list[t]}_Gradient'] = {}
        for tt in range(0, len(molecule_list)):
            task_log_loss_after_applying_gradient[f'Task_{molecule_list[t]}_Gradient'][f'Task_{molecule_list[tt]}'] = []

    Bin_CrossEntropy = []
    Val_Bin_CrossEntropy = []
    import tqdm
    for epoch in tqdm.tqdm(range(0, number_of_epoch, 1)):

        filepath = f'SavedModels/fifty_Chemical_epoch_{epoch}_fold_{fold}.h5'

        if (epoch > 0 and epoch <= 200) and epoch % 10 == 0:
            # print('Calculating the ITA')
            # calculate_ITA(molecule_list, task_log_loss_before_applying_gradient, task_log_loss_after_applying_gradient,
            #               opt,loss_fn, prev_gradient, train_data, train_label, epoch, fold)
            task_list = molecule_list
            updated_shared_gradient = {}
            model_from_prev_epoch = f'SavedModels/fifty_Chemical_epoch_{epoch - 1}_fold_{fold}.h5'
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
                model_from_prev_epoch = f'SavedModels/fifty_Chemical_epoch_{epoch - 1}_fold_{fold}.h5'
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


        '''main training loop'''
        batch_size = 264
        # print(int(len(train_data[1]) / batch_size))
        # print(f'train_data = {np.shape(train_data)}\ttrain_label = {np.shape(train_label)}')
        loss_per_batch = []
        for bid in range(int(len(train_data[1]) / batch_size)):

            X = [train_data[t][bid * batch_size: (bid + 1) * batch_size] for t in range(len(molecule_list))]
            y = [train_label[t][bid * batch_size: (bid + 1) * batch_size] for t in range(len(molecule_list))]
            # print(f'X = {np.shape(X)}\ty = {np.shape(y)}')
            loss,finalModel,gradients = train_step(X, y,finalModel,opt,loss_fn,filepath)
            # print(f'e = {epoch}\tbid = {bid}\tloss = {loss}')
            loss_per_batch.append(loss)



        # print(f'epoch = {epoch}\tloss = {loss_per_batch}\t {np.mean(loss_per_batch)}\t{np.sum(loss_per_batch)}')
        log_loss = np.mean(loss_per_batch)
        Bin_CrossEntropy.append(log_loss)

        val_loss_per_batch = []
        for bid in range(int(len(test_data[1]) / batch_size)):
            X_val = [test_data[t][bid * batch_size: (bid + 1) * batch_size] for t in range(len(molecule_list))]
            y_val = [test_label[t][bid * batch_size: (bid + 1) * batch_size] for t in range(len(molecule_list))]

            _y_validation_values = finalModel(X_val, training=False)
            _y_validation_values = tf.convert_to_tensor(_y_validation_values)
            validation_log_loss = loss_fn(y_val, _y_validation_values)
            val_loss_per_batch.append(validation_log_loss)



        validation_log_loss = np.mean(val_loss_per_batch)
        Val_Bin_CrossEntropy.append(validation_log_loss)
        # val_acc_metric.reset_states()
        prev_gradient = gradients
        # Reset training metrics at the end of each epoch
        # train_acc_metric.reset_states()
        # '''Save model after each epoch'''
        # finalModel.save(filepath)

        if (epoch % 100 == 0 or epoch == number_of_epoch - 1):
            # print(f"Epoch = {epoch}\tTrain Bin_CrossEntropy: {log_loss}\tValidation Bin_CrossEntropy: {validation_log_loss}")
            scores = finalModel.evaluate(tuple(test_data), tuple(test_label), verbose=0)
            print(f"Epoch = {epoch}\tTest Bin_CrossEntropy: {scores[0]}\t {np.sum(scores[1:])} {np.mean(scores[1:])}")


    print("*" * 500)
    print(
        f"FINAL: Train Bin_CrossEntropy: {log_loss}\tValidation Bin_CrossEntropy: {validation_log_loss}")

    # plt.plot(Bin_CrossEntropy, label='Train')
    # plt.plot(Val_Bin_CrossEntropy, label='Validation')
    # plt.grid()
    # plt.legend()
    # plt.show()

    before_loss = pd.DataFrame.from_dict(task_log_loss_before_applying_gradient)
    after_loss = pd.DataFrame.from_dict(task_log_loss_after_applying_gradient)

    before_loss.to_csv(f'before_loss_{datasetName}_fold_{fold}_{len(TASKS)}_Tasks.csv')
    after_loss.to_csv(f'after_loss_{datasetName}_fold_{fold}_{len(TASKS)}_Tasks.csv')



    best_epoch = Val_Bin_CrossEntropy.index(min(Val_Bin_CrossEntropy))
    print(f'fold = {fold}, Best Score {min(Val_Bin_CrossEntropy)} at epoch: {best_epoch}')
    filepath = f'SavedModels/fifty_Chemical_epoch_{best_epoch}_fold_{fold}.h5'
    finalModel = tf.keras.models.load_model(filepath)

    scores = finalModel.evaluate(tuple(test_data), tuple(test_label), verbose=0)
    y_pred = finalModel.predict(test_data)
    y_pred = Splitting_Values(y_pred)
    y_test = Splitting_Values(test_label)
    predicted_val = []
    for i in y_pred:
        if i[0] < 0.75:
            predicted_val.append(0)
        else:
            predicted_val.append(1)

    Error = [abs(a_i - b_i) for a_i, b_i in zip(y_test, predicted_val)]
    wrong_pred = Error.count(1)
    errorRate = wrong_pred / len(y_test)


    # if os.path.exists(filepath):
    #     os.remove(os.path.join(filepath))
    trainable_count = finalModel.count_params()
    # print(f"Trainable Count = {trainable_count}")
    print(f"fold = {fold}\t Error Rate = {errorRate}\t scores = {scores[0]}")
    for e in range(0, number_of_epoch-1):
        if e!=best_epoch:
            old_models = f'SavedModels/fifty_Chemical_epoch_{e}_fold_{fold}.h5'
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
                                                           # 'postprocessing_FF_layers': 0,
                                                           # 'postprocessing_FF_Neurons': [0]
                                                           }})

        TASK_Specific_Arch.update({group_no: initial_task_specific_architecture})
    return TASK_Specific_Arch

def get_ITA():

    task_group = {0:TASKS}
    print(f'task_group = {task_group}')

    TASK_Specific_Arch = prep_task_specific_arch(task_group)
    random_seed = random.randint(0, 100)

    group_no = 0
    task = task_group[group_no]

    tmp = (TASK_Specific_Arch[group_no], initial_shared_architecture, task, group_no, random_seed)

    kFold_validation(*tmp)

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
    task_info = pd.read_csv(f'{DataPath}Task_Information_{datasetName}.csv')
    task_distance_info = pd.read_csv(f'{DataPath}Task_Distance_{datasetName}.csv')
    single_results = pd.read_csv(f'{ResultPath}STL/STL_{datasetName}_NN.csv')
    pair_results = pd.read_csv(f'{ResultPath}Pairwise/NN/{datasetName}_Results_from_Pairwise_Training_ALL_NN.csv')

    for Selected_Task in TASKS:
        task_data = task_info[task_info.Molecule == Selected_Task].reset_index()
        task_len.update({Selected_Task: task_data.Dataset_Size[0]})
        variance_dict.update({Selected_Task: task_data.Variance[0]})
        std_dev_dict.update({Selected_Task: task_data.Std_Dev[0]})
        dist_dict.update({Selected_Task: task_data.Average_Hamming_Distance_within_Task[0]})
        single_res = single_results[single_results.Task == Selected_Task].reset_index()
        Single_res_dict.update({Selected_Task: single_res.LOSS[0]})

    initial_shared_architecture = {'adaptive_FF_neurons': 5, 'shared_FF_Layers': 1, 'shared_FF_Neurons': [6],
                                   'learning_rate': 0.00029055748415145487}


    get_ITA()




