# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import ast
import tqdm
import itertools
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint
from keras import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TGPDataPreparation:

    # def __init__(self, dataset_name,counter,run, modelName):
    #     self.dataset_name = dataset_name
    #     self.DataPath = f'../Dataset/{self.dataset_name.upper()}/'
    #     self.ResultPath = f'Results/Run_{run}/'
    #     self.dataresultPath = 'Results'

    def __init__(self, dataset_name, modelName):
        self.dataset_name = dataset_name
        self.DataPath = f'../Dataset/{self.dataset_name.upper()}/'
        self.ResultPath = f'../Results/'
        self.dataresultPath = '../Results'

        self.task_len = {}
        self.variance_dict = {}
        self.std_dev_dict = {}
        self.dist_dict = {}
        self.single_res_dict = {}
        # self.counter = counter
        self.counter = 0
        self.modelName = modelName
        self.FULL_TRAIN = True

        if self.dataset_name == 'School':
            self.TASKS = [i for i in range(1, 140)]
        if self.dataset_name == 'Landmine':
            self.TASKS = [i for i in range(0, 29)]
        if self.dataset_name == 'Parkinsons':
            self.TASKS = [i for i in range(1, 43)]
        if self.dataset_name == 'Chemical':
            chemical_data = pd.read_csv(f'{self.DataPath}ChemicalData_All.csv', low_memory=False)
            self.TASKS = list(chemical_data['180'].unique())

        if self.modelName == 'SVM' and self.dataset_name == 'Chemical':
            self.task_info = pd.read_csv(f'{self.DataPath}Task_Information_{self.dataset_name}_{self.modelName}.csv')
        else:
            self.task_info = pd.read_csv(f'{self.DataPath}Task_Information_{self.dataset_name}.csv')
        self.task_distance_info = pd.read_csv(f'{self.DataPath}Task_Distance_{self.dataset_name}.csv')
        self.single_results = pd.read_csv(f'{self.dataresultPath}/STL/STL_{self.dataset_name}_{self.modelName}.csv')

        self.pair_results = pd.read_csv(
                f'{self.dataresultPath}/Pairwise/{self.modelName}/{self.dataset_name}_Results_from_Pairwise_Training_ALL_{self.modelName}.csv')


        for selected_task in self.TASKS:
            if self.dataset_name == 'Chemical':
                task_data = self.task_info[self.task_info.Molecule == selected_task].reset_index()
            else:
                task_data = self.task_info[self.task_info.Task_Name == selected_task].reset_index()
            self.task_len.update({selected_task: task_data.Dataset_Size[0]})
            self.variance_dict.update({selected_task: task_data.Variance[0]})
            self.std_dev_dict.update({selected_task: task_data.Std_Dev[0]})
            self.dist_dict.update({selected_task: task_data.Average_Euclidian_Distance_within_Task[0]})
            single_res = self.single_results[self.single_results.Task == selected_task].reset_index()
            self.single_res_dict.update({selected_task: single_res.LOSS[0]})

        # self.predictor_data_prep(task_grouping_results)
        # self.retrain_predictor()

    def predictor_data_prep(self, task_grouping_results):
        single_task_loss = []
        group_info = []
        group_loss = []
        group_dataset_size = []
        group_variance = []
        group_stddev = []
        group_distance = []
        number_of_tasks = []

        pairwise_improvement_average = []
        pairwise_improvement_variance = []
        pairwise_improvement_stddev = []

        # pairwise_ITA_average = []
        # pairwise_Weight_average = []


        change = []



        # count = 0
        for group in tqdm.tqdm(range(len(task_grouping_results))):

            Task_Group = ast.literal_eval(task_grouping_results.Task_group[group])
            Individual_Group_Score = ast.literal_eval(task_grouping_results.Individual_Group_Score[group])
            if self.counter!=0:

                Changed_Groups = ast.literal_eval(task_grouping_results.Changed_Groups[group])

                if Changed_Groups != None:
                    for gr in Changed_Groups:
                        sample_size = 0
                        avg_var = []
                        avg_stddev = []
                        avg_dist = []
                        sum_loss_single_task = 0

                        if len(Task_Group[gr])<=1:
                            continue

                        task_combo = list(itertools.combinations(sorted(Task_Group[gr]), 2))
                        for task_pair in task_combo:
                            task_dist_data = self.task_distance_info[(self.task_distance_info.Task_1 == task_pair[0]) & (
                                    self.task_distance_info.Task_2 == task_pair[1])].reset_index()

                            if len(task_dist_data) == 0:
                                task_dist_data = self.task_distance_info[(self.task_distance_info.Task_1 == task_pair[1]) & (
                                        self.task_distance_info.Task_2 == task_pair[0])].reset_index()

                            if self.dataset_name == 'Chemical':
                                avg_dist.append(task_dist_data.Average_Hamming_Distance[0])
                            else:
                                avg_dist.append(task_dist_data.Average_Euclidean_Distance[0])
                        # group_distance.append(np.mean(avg_dist))

                        paired_improvement = []
                        # paired_ITA = []
                        # paired_weight = []
                        for pair in task_combo:
                            stl_loss = 0
                            stl_loss += self.single_res_dict[pair[0]]
                            stl_loss += self.single_res_dict[pair[1]]
                            pair_specific = self.pair_results[
                                (self.pair_results.Task_1 == pair[0]) & (self.pair_results.Task_2 == pair[1])].reset_index()
                            if len(pair_specific) == 0:
                                pair_specific = self.pair_results[
                                    (self.pair_results.Task_1 == pair[1]) & (self.pair_results.Task_2 == pair[0])].reset_index()
                            paired_improvement.append((stl_loss - pair_specific.Total_Loss[0]) / stl_loss)


                        pairwise_improvement_average.append(np.mean(paired_improvement))
                        pairwise_improvement_variance.append(np.var(paired_improvement))
                        pairwise_improvement_stddev.append(np.std(paired_improvement))





                        # pairwise_ITA_average.append(np.mean(paired_ITA))
                        # pairwise_Weight_average.append(np.mean(paired_weight))

                        for t in Task_Group[gr]:
                            sample_size += self.task_len[t]
                            avg_var.append(self.variance_dict[t])
                            avg_stddev.append(self.std_dev_dict[t])
                            sum_loss_single_task += self.single_res_dict[t]
                        group_info.append(Task_Group[gr])
                        group_loss.append(Individual_Group_Score[gr])
                        number_of_tasks.append(len(Task_Group[gr]))
                        group_dataset_size.append(sample_size / len(Task_Group[gr]))  # individual length

                        group_distance.append(np.mean(avg_dist))
                        group_variance.append(np.mean(avg_var))
                        group_stddev.append(np.mean(avg_stddev))

                        # group_distance.append(np.mean(avg_dist))
                        single_task_loss.append(sum_loss_single_task)

                        change.append((sum_loss_single_task - Individual_Group_Score[gr]) / sum_loss_single_task)
            else:
                for group_no, tasks in Task_Group.items():
                    sample_size = 0
                    avg_var = []
                    avg_stddev = []
                    avg_dist = []
                    sum_loss_single_task = 0
                    if len(tasks)<=1:
                        continue
                    task_combo = list(itertools.combinations(sorted(tasks), 2))
                    for task_pair in task_combo:
                        task_dist_data = self.task_distance_info[(self.task_distance_info.Task_1 == task_pair[0]) & (
                                self.task_distance_info.Task_2 == task_pair[1])].reset_index()
                        if len(task_dist_data) == 0:
                            task_dist_data = self.task_distance_info[(self.task_distance_info.Task_1 == task_pair[1]) & (
                                    self.task_distance_info.Task_2 == task_pair[0])].reset_index()

                        if self.dataset_name == 'Chemical':
                            avg_dist.append(task_dist_data.Average_Hamming_Distance[0])
                        else:
                            avg_dist.append(task_dist_data.Average_Euclidean_Distance[0])


                    paired_improvement = []
                    paired_ITA = []
                    paired_weight = []
                    for pair in task_combo:
                        stl_loss = 0
                        stl_loss += self.single_res_dict[pair[0]]
                        stl_loss += self.single_res_dict[pair[1]]
                        pair_specific = self.pair_results[
                            (self.pair_results.Task_1 == pair[0]) & (self.pair_results.Task_2 == pair[1])].reset_index()
                        if len(pair_specific) == 0:
                            pair_specific = self.pair_results[
                                (self.pair_results.Task_1 == pair[1]) & (self.pair_results.Task_2 == pair[0])].reset_index()
                        paired_improvement.append((stl_loss - pair_specific.Total_Loss[0]) / stl_loss)

                        p = tuple(sorted([pair[0], pair[1]]))
                        # pair_idx = list(self.ITA_data['Pairs']).index(str(p))
                        # paired_ITA.append(list(self.ITA_data.Pairwise_ITA)[pair_idx])

                        # pair_idx = list(self.Weight_Matrix['Pairs']).index(str(p))
                        # paired_weight.append(list(self.Weight_Matrix.Weight)[pair_idx])

                    # pairwise_ITA_average.append(np.mean(paired_ITA))
                    # pairwise_Weight_average.append(np.mean(paired_weight))


                    pairwise_improvement_average.append(np.mean(paired_improvement))
                    pairwise_improvement_variance.append(np.var(paired_improvement))
                    pairwise_improvement_stddev.append(np.std(paired_improvement))
                    for t in tasks:
                        sample_size += self.task_len[t]
                        avg_var.append(self.variance_dict[t])
                        avg_stddev.append(self.std_dev_dict[t])
                        sum_loss_single_task += self.single_res_dict[t]
                    group_info.append(tasks)
                    group_loss.append(Individual_Group_Score[group_no])
                    number_of_tasks.append(len(tasks))
                    group_dataset_size.append(sample_size / len(tasks))  # individual length

                    group_distance.append(np.mean(avg_dist))
                    group_variance.append(np.mean(avg_var))
                    group_stddev.append(np.mean(avg_stddev))

                    change.append((sum_loss_single_task - Individual_Group_Score[group_no]) / sum_loss_single_task)

        predictor_data = pd.DataFrame({
            'group_variance': group_variance,
            'group_stddev': group_stddev,
            'group_distance': group_distance,
            'number_of_tasks': number_of_tasks,
            'group_dataset_size': group_dataset_size,
            'pairwise_improvement_average': pairwise_improvement_average,
            'pairwise_improvement_variance': pairwise_improvement_variance,
            'pairwise_improvement_stddev': pairwise_improvement_stddev,
            # 'pairwise_ITA_average': pairwise_ITA_average,
            # 'pairwise_Weight_average': pairwise_Weight_average,
            'change': change
        })

        # predictor_data.to_csv(f'{self.ResultPath}/Data_for_Predictor_{self.dataset_name}_new_updated.csv', index=False)
        predictor_data.to_csv(
            f'{self.ResultPath}partition_sample/{self.modelName}/group_sample/Data_for_GroupPredictor_{self.dataset_name}_{modelName}.csv',
            index=False)

        # '''


        # import os
        # if self.counter == 0:
        #     predictor_data.to_csv(f'{self.ResultPath}/Data_for_Predictor_{self.dataset_name}_first.csv', index=False)
        #     predictor_data.to_csv(f'{self.ResultPath}/Data_for_Predictor_{self.dataset_name}_updated.csv', index=False)
        #
        # elif self.counter == 'rerun':
        #     old_file = f'{self.ResultPath}/Data_for_Predictor_{self.dataset_name}_first.csv'
        #     if os.path.exists(old_file):
        #         df_1 = pd.read_csv(f'{self.ResultPath}/Data_for_Predictor_{self.dataset_name}_first.csv')
        #         df_2 = pd.read_csv(f'{self.ResultPath}/Data_for_Predictor_{self.dataset_name}_new_updated.csv')
        #         frames = [df_1, df_2]
        #
        #         result = pd.concat(frames)
        #         result.to_csv(f'{self.ResultPath}/Data_for_Predictor_{self.dataset_name}_updated.csv', index=False)
        #
        # else:
        #     old_file = f'{self.ResultPath}/Data_for_Predictor_{self.dataset_name}_updated.csv'
        #     if os.path.exists(old_file):
        #         df_1 = pd.read_csv(f'{self.ResultPath}/Data_for_Predictor_{self.dataset_name}_updated.csv')
        #         df_2 = pd.read_csv(f'{self.ResultPath}/Data_for_Predictor_{self.dataset_name}_new_updated.csv')
        #         frames = [df_1, df_2]
        #
        #         result = pd.concat(frames)
        #         result.to_csv(f'{self.ResultPath}/Data_for_Predictor_{self.dataset_name}_updated.csv', index=False)


    def predictor_network_evaluate(self, x_train, x_test, y_train, y_test, fold):

        if self.dataset_name == 'Chemical':
            # network_architecture = {'FF_Layers': 1, 'FF_Neurons': [10], 'learning_rate': 0.0013850679916489607}
            network_architecture = {'FF_Layers': 2, 'FF_Neurons': [20,10], 'learning_rate': 0.005}
        else:
            network_architecture = {'FF_Layers': 2, 'FF_Neurons': [28, 25], 'learning_rate': 0.0014116832808877841}
            # network_architecture = {'FF_Layers': 1, 'FF_Neurons': [3], 'learning_rate': 0.0012524124635136002}
            # network_architecture = {'FF_Layers': 1, 'FF_Neurons': [12], 'learning_rate': 0.0025}

        number_of_epoch = 100
        filepath = f'{self.ResultPath}/SavedModels/{self.dataset_name}_TG_predictor_{fold}.h5'

        number_of_features = np.shape(x_train)[1]

        Input_FF = tf.keras.layers.Input(shape=(number_of_features,))
        hidden_FF = tf.keras.layers.Dense(network_architecture['FF_Neurons'][0],
                                          activation='sigmoid')(Input_FF)
        for h in range(1, network_architecture['FF_Layers']):
            hidden_FF = tf.keras.layers.Dense(network_architecture['FF_Neurons'][h],
                                                  activation='sigmoid')(hidden_FF)

        output = tf.keras.layers.Dense(1, activation='sigmoid')(hidden_FF)
        # output = tf.keras.layers.Dense(1, activation='linear')(hidden_FF)

        finalModel = Model(inputs=Input_FF, outputs=output)
        opt = tf.keras.optimizers.Adam(learning_rate=network_architecture['learning_rate'])
        finalModel.compile(optimizer=opt, loss='mse')
        # print(finalModel.summary())

        checkpoint = ModelCheckpoint(filepath, verbose=2, monitor='val_loss', save_best_only=True, mode='auto')

        history = finalModel.fit(x=x_train,
                     y=y_train,
                     shuffle=True,
                     epochs=number_of_epoch,
                     batch_size=16,
                     validation_data=(x_test,
                                      y_test),
                     callbacks=checkpoint,
                     verbose=0)

        # import matplotlib.pyplot as plt
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('model loss')
        # plt.show()

        finalModel = tf.keras.models.load_model(filepath)
        train_scores = finalModel.evaluate(x_train, y_train, verbose=0)
        test_scores = finalModel.evaluate(x_test, y_test, verbose=0)

        return train_scores, test_scores, np.var(y_test)

    def predictor_network(self, x_train, y_train):

        if self.dataset_name == 'Chemical':
            network_architecture = {'FF_Layers': 1, 'FF_Neurons': [10], 'learning_rate': 0.0013850679916489607}
        else:
            network_architecture = {'FF_Layers': 2, 'FF_Neurons': [20, 10], 'learning_rate': 0.005}

        number_of_epoch = 100
        filepath = f'{self.ResultPath}/SavedModels/{self.dataset_name}_TG_predictor_Best.h5'

        number_of_features = np.shape(x_train)[1]

        Input_FF = tf.keras.layers.Input(shape=(number_of_features,))
        hidden_FF = tf.keras.layers.Dense(network_architecture['FF_Neurons'][0],
                                          activation='sigmoid')(Input_FF)
        for h in range(1, network_architecture['FF_Layers']):
            hidden_FF = tf.keras.layers.Dense(network_architecture['FF_Neurons'][h],
                                                  activation='sigmoid')(hidden_FF)

        output = tf.keras.layers.Dense(1, activation='sigmoid')(hidden_FF)
        # output = tf.keras.layers.Dense(1, activation='linear')(hidden_FF)

        finalModel = Model(inputs=Input_FF, outputs=output)
        opt = tf.keras.optimizers.Adam(learning_rate=network_architecture['learning_rate'])
        finalModel.compile(optimizer=opt, loss='mse')
        # print(finalModel.summary())

        checkpoint = ModelCheckpoint(filepath, verbose=0, monitor='loss', save_best_only=True, mode='auto')

        history = finalModel.fit(x=x_train,
                     y=y_train,
                     shuffle=True,
                     epochs=number_of_epoch,
                     batch_size=16,
                     callbacks=checkpoint,
                     verbose=0)


    def predict_performance_of_new_group(self,tasks):
        if len(tasks) > 1:
            number_of_tasks = []
            pairwise_improvement_average = []
            pairwise_improvement_variance = []
            pairwise_improvement_stddev = []
            group_variance = []
            group_stddev = []
            group_distance = []
            group_dataset_size = []

            # pairwise_ITA_average = []
            # pairwise_Weight_average = []


            number_of_tasks.append(len(tasks))
            sample_size = 0
            avg_var = []
            avg_stddev = []
            avg_dist = []
            task_combo = list(itertools.combinations(sorted(tasks), 2))
            for task_pair in task_combo:
                task_dist_data = self.task_distance_info[(self.task_distance_info.Task_1 == task_pair[0]) & (
                        self.task_distance_info.Task_2 == task_pair[1])].reset_index()
                if len(task_dist_data) == 0:
                    task_dist_data = self.task_distance_info[(self.task_distance_info.Task_1 == task_pair[1]) & (
                            self.task_distance_info.Task_2 == task_pair[0])].reset_index()

                if self.dataset_name == 'Chemical':
                    avg_dist.append(task_dist_data.Average_Hamming_Distance[0])
                else:
                    avg_dist.append(task_dist_data.Average_Euclidean_Distance[0])

            group_distance.append(np.mean(avg_dist))
            paired_improvement = []
            paired_ITA = []
            paired_weight = []
            for pair in task_combo:
                stl_loss = 0
                stl_loss += self.single_res_dict[pair[0]]
                stl_loss += self.single_res_dict[pair[1]]
                pair_specific = self.pair_results[
                    (self.pair_results.Task_1 == pair[0]) & (self.pair_results.Task_2 == pair[1])].reset_index()
                if len(pair_specific) == 0:
                    pair_specific = self.pair_results[
                        (self.pair_results.Task_1 == pair[1]) & (self.pair_results.Task_2 == pair[0])].reset_index()
                paired_improvement.append((stl_loss - pair_specific.Total_Loss[0]) / stl_loss)

                # p = tuple(sorted([pair[0], pair[1]]))
                # pair_idx = list(self.ITA_data['Pairs']).index(str(p))
                # paired_ITA.append(list(self.ITA_data.Pairwise_ITA)[pair_idx])
                #
                # pair_idx = list(self.Weight_Matrix['Pairs']).index(str(p))
                # paired_weight.append(list(self.Weight_Matrix.Weight)[pair_idx])

            pairwise_improvement_average.append(np.mean(paired_improvement))
            pairwise_improvement_variance.append(np.var(paired_improvement))
            pairwise_improvement_stddev.append(np.std(paired_improvement))
            # pairwise_ITA_average.append(np.mean(paired_ITA))
            # pairwise_Weight_average.append(np.mean(paired_weight))



            single_task_total_loss = 0
            for t in tasks:
                single_task_total_loss += self.single_res_dict[t]
                sample_size += self.task_len[t]
                avg_var.append(self.variance_dict[t])
                avg_stddev.append(self.std_dev_dict[t])

            group_variance.append(np.mean(avg_var))
            group_stddev.append(np.mean(avg_stddev))
            group_dataset_size.append(sample_size / len(self.TASKS))

            new_groups = pd.DataFrame({
                'group_dataset_size': group_dataset_size,
                'group_variance': group_variance,
                'group_stddev': group_stddev,
                'group_distance': group_distance,
                'number_of_tasks': number_of_tasks,
                'pairwise_improvement_average': pairwise_improvement_average,
                'pairwise_improvement_variance': pairwise_improvement_variance,
                'pairwise_improvement_stddev': pairwise_improvement_stddev,
                # 'pairwise_ITA_average': pairwise_ITA_average,
                # 'pairwise_Weight_average': pairwise_Weight_average
            })

            x = np.array(new_groups, dtype='float32')
            filepath = f'{self.ResultPath}/SavedModels/{self.dataset_name}_TG_predictor_Best.h5'
            finalModel = tf.keras.models.load_model(filepath)
            final_score = finalModel.predict(x, verbose=0)

            return final_score[0][0], single_task_total_loss
        else:
            return 0, self.single_res_dict[tasks[0]]

    def linear_Regression(self, X_train, X_test, y_train, y_test, fold):
        from sklearn.linear_model import LinearRegression
        from sklearn import metrics
        regr = LinearRegression(normalize=True)

        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        train_scores = regr.score(X_train, y_train)
        # test_scores = regr.score(X_test, y_test)
        test_scores = metrics.mean_squared_error(y_test, y_pred)
        return train_scores, test_scores, np.var(y_test)

    def retrain_predictor(self):
        predictor_data = pd.read_csv(f'{self.ResultPath}/Data_for_Predictor_{self.dataset_name}_updated.csv')
        print(f'\n\n******* Training Samples = {len(predictor_data)} *******\n\n')

        predictor_data.dropna(inplace=True)
        predictor_data = predictor_data[['group_variance', 'group_stddev', 'group_distance', 'number_of_tasks',
                   'group_dataset_size', 'pairwise_improvement_average',
                   'pairwise_improvement_variance', 'pairwise_improvement_stddev',
                    # 'pairwise_ITA_average', 'pairwise_Weight_average',
                   'change']]

        DataSet = np.array(predictor_data, dtype=float)
        Number_of_Records = np.shape(DataSet)[0]
        number_of_features = np.shape(DataSet)[1]

        Input_Features = predictor_data.columns[:number_of_features - 1]

        Sample_Inputs = np.zeros((Number_of_Records, len(Input_Features)))
        for t in range(Number_of_Records):
            Sample_Inputs[t] = DataSet[t, :len(Input_Features)]
        Sample_Label = np.array(list(predictor_data.change), dtype=float)

        if self.FULL_TRAIN:
            self.predictor_network(Sample_Inputs,Sample_Label)

        else:
            indices = np.arange(len(Sample_Inputs))
            if len(Sample_Inputs) < 100:
                num_folds = 5
            else:
                num_folds = 10
            from sklearn.model_selection import KFold
            import multiprocessing as mp
            kfold = KFold(n_splits=num_folds, shuffle=True)

            fold = 0
            ALL_FOLDS = []

            for train, test in kfold.split(indices):
                x_train = Sample_Inputs[train]
                x_test = Sample_Inputs[test]
                y_train = Sample_Label[train]
                y_test = Sample_Label[test]
                ALL_FOLDS.append((x_train, x_test, y_train, y_test, fold))
                fold += 1

            number_of_pools = len(ALL_FOLDS)
            pool = mp.Pool(number_of_pools)
            all_scores = pool.starmap(self.predictor_network_evaluate, ALL_FOLDS)
            pool.close()
            # print(f'all_scores = {all_scores}')
            train_scores = [x[0] for x in all_scores]
            test_scores = [x[1] for x in all_scores]
            var = [x[2] for x in all_scores]
            print(f'train mse = {np.mean(train_scores)}')
            print(f'test mse = {np.mean(test_scores)}')
            print(f'variance = {np.var(predictor_data.change)}')


            # train, test = train_test_split(indices, shuffle=True, test_size=0.2, random_state=234)
            #
            # x_train = Sample_Inputs[train]
            # x_test = Sample_Inputs[test]
            # y_train = Sample_Label[train]
            # y_test = Sample_Label[test]
            # train_scores, test_scores, var = self.predictor_network_evaluate(x_train, x_test, y_train, y_test, 999)
            # print(f'train mse = {train_scores}')
            # print(f'test mse = {test_scores}')
            # print(f'variance = {np.var(predictor_data.change)}')



if __name__ == '__main__':
    '''Predictor Functions'''
    for modelName in ['XGBoost','SVM']:
        for dataset in ['School', 'Chemical', 'Landmine', 'Parkinsons']:

            DataPath = f"../Dataset/{dataset.upper()}/"
            ResultPath = '../Results/'
            if dataset == 'Chemical':
                ChemicalData = pd.read_csv(f'{DataPath}ChemicalData_All.csv', low_memory=False)
                TASKS = list(ChemicalData['180'].unique())
            if dataset == 'School':
                TASKS = [i for i in range(1, 140)]
            if dataset == 'Landmine':
                TASKS = [i for i in range(0, 29)]
            if dataset == 'Parkinsons':
                TASKS = [i for i in range(1, 43)]

            partition_sample = pd.read_csv(f'../Results/partition_sample/{modelName}/{dataset}_group_sample_{modelName}.csv')

            obj = TGPDataPreparation(dataset_name=dataset,modelName=modelName)
            TGPDataPreparation.predictor_data_prep(obj, partition_sample)
