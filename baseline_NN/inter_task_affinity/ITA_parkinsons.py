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
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model, backend
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
        # print(df.columns)
        # exit(0)

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


'''MTL for Task Groups Functions'''


def kFold_validation(current_task_specific_architecture, current_shared_architecture, task, random_seed):

    data_param_dictionary, Number_of_Features_FF = readData(task)

    data_param_dict_for_specific_task = {}


    for f in range(num_folds):
        data_param_dict_for_specific_task[f] = {}

    for subj_id in task:
        Sample_Inputs = data_param_dictionary[f'Subject_{subj_id}_FF_Inputs']
        Sample_Label = data_param_dictionary[f'Subject_{subj_id}_Label']
        Sample_Label_target_1 = data_param_dictionary[f'Subject_{subj_id}_Label_1']
        Sample_Label_target_2 = data_param_dictionary[f'Subject_{subj_id}_Label_2']
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

    pool = mp.Pool(number_of_pools)
    all_scores = np.array(pool.starmap(final_model, args))
    pool.close()

    print(f'Time required = {(time.time() - timeStart) / 60} minutes')
    print(f'all_scores = {all_scores}')
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
                Number_of_Features, fold):
    data_param = copy.deepcopy(data_param_dict_for_specific_task)
    # print(f'task_group_list = {task_group_list}')

    filepath = f'SavedModels/fifty_Park_epoch_{fold}.h5'

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

    for subj_id in task_group_list:
        hyperparameters = copy.deepcopy(task_hyperparameters[subj_id])
        last_shared = MTL_model_param[f'sub_{subj_id}_fold_{fold}_last_hidden_layer']
        if hyperparameters['postprocessing_FF_layers'] > 0:

            hidden_ff = Dense(hyperparameters['postprocessing_FF_Neurons'][0], activation=activation_func)(
                last_shared)
            for h in range(1, hyperparameters['postprocessing_FF_layers']):
                hidden_ff = Dense(hyperparameters['postprocessing_FF_Neurons'][h], activation=activation_func)(
                    hidden_ff)
            MTL_model_param[f'sub_{subj_id}_fold_{fold}_ff_postprocessing_model'] = hidden_ff

    # output Neurons
    output_layers = []

    for subj_id in task_group_list:
        shared_model = Model(inputs=MTL_model_param[f'sub_{subj_id}_fold_{fold}_Input_FF'],
                             outputs=MTL_model_param[f'sub_{subj_id}_fold_{fold}_ff_postprocessing_model'])

        outputlayer = Dense(2, activation='linear', name=f'Score_{subj_id}')(shared_model.output)
        output_layers.append(outputlayer)

    finalModel = Model(inputs=input_layers, outputs=output_layers)

    # from keras.utils.layer_utils import count_params
    # trainable_count = count_params(finalModel.trainable_weights)
    # print(f'fold = {fold}\tMTL = {len(MTL_model_param.keys())}\t trainable_count = {trainable_count}')

    opt = tf.keras.optimizers.Adam(learning_rate=shared_hyperparameters['learning_rate'])
    finalModel.compile(optimizer=opt, loss='mse')


    batch_size = 132
    number_of_epoch = 1200

    task_loss_before_applying_gradient = {}
    task_loss_after_applying_gradient = {}
    for t in range(0, len(task_group_list)):
        task_loss_before_applying_gradient[f'Task_{task_group_list[t]}'] = []
        task_loss_after_applying_gradient[f'Task_{task_group_list[t]}_Gradient'] = {}
        for tt in range(0, len(task_group_list)):
            task_loss_after_applying_gradient[f'Task_{task_group_list[t]}_Gradient'][f'Task_{task_group_list[tt]}'] = []

    MSE = []
    Val_MSE = []
    for epoch in tqdm.tqdm(range(0, number_of_epoch, 1)):

        filepath = f'SavedModels/fifty_Park_epoch_{epoch}_fold_{fold}.h5'

        '''Calculate Inter task affinity using task-specific gradient onto the model from prev_epoch'''
        if (epoch > 0 and epoch <= 200) and epoch % 10 == 0:
            updated_shared_gradient = {}
            model_from_prev_epoch = f'SavedModels/fifty_Park_epoch_{epoch - 1}_fold_{fold}.h5'
            PrevModel = tf.keras.models.load_model(model_from_prev_epoch)

            # print(f'\n\n!!!!!!!!!!!!!\n')
            for t in range(0, len(task_group_list)):
                with tf.GradientTape() as model_tape:
                    _y_train_pred = PrevModel((train_data), training=True)
                    _y_train_predicted_values = tf.convert_to_tensor(_y_train_pred)
                    task_mse = tf.math.reduce_mean(tf.math.square(train_label[t] - _y_train_predicted_values[t]))

                task_loss_before_applying_gradient[f'Task_{task_group_list[t]}'].append(
                    tf.keras.backend.get_value(task_mse))
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
                model_from_prev_epoch = f'SavedModels/fifty_Park_epoch_{epoch - 1}_fold_{fold}.h5'
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
                    task_loss_after_applying_gradient[f'Task_{task_group_list[t]}_Gradient'][
                        f'Task_{task_group_list[tt]}'].append(tf.keras.backend.get_value(task_mse))


        if epoch == 210:
            before_loss = pd.DataFrame.from_dict(task_loss_before_applying_gradient)
            after_loss = pd.DataFrame.from_dict(task_loss_after_applying_gradient)
            datasetName = 'Parkinsons'
            before_loss.to_csv(f'before_loss_{datasetName}_fold_{fold}_{len(task_group_list)}_Tasks.csv')
            after_loss.to_csv(f'after_loss_{datasetName}_fold_{fold}_{len(task_group_list)}_Tasks.csv')

        '''main training loop'''
        batch_size = 132
        loss_per_batch = []
        # print(f'train_data.shape: {np.shape(train_data)}, train_label.shape: {np.shape(train_label)}')
        for bid in range(int(len(train_data[1]) / batch_size)):
            X = [train_data[t][bid * batch_size: (bid + 1) * batch_size] for t in range(len(task_group_list))]
            y = [train_label[t][bid * batch_size: (bid + 1) * batch_size] for t in range(len(task_group_list))]
            # print(f'X.shape: {np.shape(X)}, y.shape: {np.shape(y)}')
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
    print(
        f"FINAL: Train MSE: {mse_loss}\tValidation MSE: {validation_mse_loss}\t {validation_mse_loss * len(task_group_list)}")

    # plt.plot(MSE)
    # plt.plot(Val_MSE)
    # plt.show()
    best_epoch = MSE.index(min(MSE))
    print(f'fold = {fold}, Best Score {min(MSE)} at epoch: {best_epoch}')
    filepath = f'SavedModels/fifty_Park_epoch_{best_epoch}_fold_{fold}.h5'
    finalModel = tf.keras.models.load_model(filepath)
    scores = finalModel.evaluate(tuple(test_data), tuple(test_label), verbose=0)
    print(f"fold = {fold}\t scores = {scores[0]}")
    for e in range(0, number_of_epoch):
        if e != best_epoch:
            old_models = f'SavedModels/fifty_Park_epoch_{e}_fold_{fold}.h5'
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
                                                           'postprocessing_FF_layers': 1,
                                                           'postprocessing_FF_Neurons': [5]
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
    tmp = (TASK_Specific_Arch[group_no], initial_shared_architecture, task, random_seed)

    kFold_validation(*tmp)


if __name__ == '__main__':


    num_folds = 10
    initial_shared_architecture = {'adaptive_FF_neurons': 4, 'shared_FF_Layers': 2, 'shared_FF_Neurons': [20,10],
                                   'learning_rate': 0.00779959, 'activation': 'sigmoid'}

    TASKS = [i for i in range(1, 43)]
    datasetName = 'Parkinsons'
    DataPath = f'../../Dataset/{datasetName.upper()}/'
    ResultPath = '../../Results/'


    get_ITA()
