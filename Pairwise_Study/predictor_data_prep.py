import math
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, cityblock, hamming
from sklearn import preprocessing
import tqdm


def fit_log_curve(ys):
    xs = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    tmp_A = []
    tmp_B = []
    for i in range(len(xs)):
        tmp_A.append([np.log(xs[i]), 1])
        tmp_B.append(ys[i])
    B = np.matrix(tmp_B).T
    A = np.matrix(tmp_A)
    fit = (A.T * A).I * A.T * B
    errors = B - A * fit
    residual = np.linalg.norm(errors)
    return fit


def get_variance_stdDev_for_groups(Tasks_list,dataset):
    DataPath = f"../Dataset/{dataset.upper()}/"
    Target_Group = []
    Target_Group1 = []
    Target_Group2 = []
    avg_var = []
    avg_std = []
    for t in Tasks_list:
        if dataset=='School':
            df = pd.read_csv(f'{DataPath}{t}_School_Data.csv')
            Target_Group = Target_Group + list(df.ExamScore)
        if dataset=='Chemical':
            df = pd.read_csv(f'{DataPath}{t}_Molecule_Data.csv')
            Target_Group = Target_Group + list(df['181'])
        if dataset=='Landmine':
            df = pd.read_csv(f'{DataPath}LandmineData_{t}.csv')
            Target_Group = Target_Group + list(df.Labels)
        if dataset=='Parkinsons':
            df = pd.read_csv(f'{DataPath}parkinsons_subject_{t}.csv')
            Target_Group1 = Target_Group1 + list(df.motor_UPDRS)
            Target_Group2 = Target_Group2 + list(df.total_UPDRS)


    if dataset=='Parkinsons':
        avg_var.append((np.var(Target_Group1) + np.var(Target_Group2)) / len(Tasks_list))
        avg_std.append((np.std(Target_Group1) + np.std(Target_Group2)) / len(Tasks_list))
        return np.mean(avg_var),np.mean(avg_std)
    else:
        return np.var(Target_Group), np.std(Target_Group)


def unified_Distance(Tasks_list,dataset):
    Target_df = []
    for t in Tasks_list:
        if dataset == 'School':
            df = pd.read_csv(f'{DataPath}{t}_School_Data.csv')
        if dataset == 'Chemical':
            df = pd.read_csv(f'{DataPath}{t}_Molecule_Data.csv')
        if dataset == 'Landmine':
            df = pd.read_csv(f'{DataPath}LandmineData_{t}.csv')
        if dataset=='Parkinsons':
            df = pd.read_csv(f'{DataPath}parkinsons_subject_{t}.csv')
        Target_df.append(df)

    Target_DF = pd.concat(Target_df, axis=0, join='outer', ignore_index=True)
    Target_DF = np.array(Target_DF, dtype=float)
    edist = []
    mdist = []
    hdist = []
    for d in range(0, len(Target_DF) - 1):
        edist.append(euclidean(Target_DF[d], Target_DF[d + 1]))
        mdist.append(cityblock(Target_DF[d], Target_DF[d + 1]))
        hdist.append(hamming(Target_DF[d], Target_DF[d + 1]))
    return np.mean(edist), np.mean(mdist), np.mean(hdist)


def unified_Distance_scaled(Tasks_list,dataset):
    # DataPath = f"../Dataset/{dataset.upper()}/"
    Target_df = []
    for t in Tasks_list:
        if dataset == 'School':
            df = pd.read_csv(f'{DataPath}{t}_School_Data.csv')
        if dataset == 'Chemical':
            df = pd.read_csv(f'{DataPath}{t}_Molecule_Data.csv')
        if dataset == 'Landmine':
            df = pd.read_csv(f'{DataPath}LandmineData_{t}.csv')
        if dataset=='Parkinsons':
            df = pd.read_csv(f'{DataPath}parkinsons_subject_{t}.csv')

        Target_df.append(df)

    Target_DF = pd.concat(Target_df, axis=0, join='outer', ignore_index=True)
    # print(len(Target_DF))
    x = Target_DF.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    Target_DF_Scaled = pd.DataFrame(x_scaled)
    Target_DF_Scaled = np.array(Target_DF_Scaled, dtype=float)

    edist = []
    mdist = []
    # print(DataSet[0],DataSet[len(DataSet)-1])
    for d in range(0, len(Target_DF_Scaled) - 1):
        edist.append(euclidean(Target_DF_Scaled[d], Target_DF_Scaled[d + 1]))
        mdist.append(cityblock(Target_DF_Scaled[d], Target_DF_Scaled[d + 1]))
    return np.mean(edist), np.mean(mdist)

def get_task_specific_features(dataset):

    task_info = pd.read_csv(f'{DataPath}Task_Information_{dataset}.csv')
    single_results = pd.read_csv(f'{ResultPath}STL/STL_{dataset}_{modelName}.csv')
    pair_results = pd.read_csv(f'{ResultPath}Pairwise/NN/{dataset}_Results_from_Pairwise_Training_ALL_{modelName}.csv')

    Weight_Matrix = pd.read_csv(f'{ResultPath}/Weight_Matrix/Weight_Affinity_{dataset}.csv', low_memory=False)
    ITA_data = pd.read_csv(f'{ResultPath}/InterTask_Affinity/Pairwise_ITA_{dataset}.csv', low_memory=False)





    task_len = {}
    variance_dict = {}
    std_dev_dict = {}
    euclidean_dist_dict = {}
    manhattan_dist_dict = {}
    hamming_dist_dict = {}
    euclidean_dist_dict_scaled = {}
    manhattan_dist_dict_scaled = {}
    hamming_dist_dict_scaled = {}
    Single_res_dict = {}
    loss_dict = {}

    for Selected_Task in TASKS:

        if dataset == 'Chemical':
            task_data = task_info[task_info.Molecule == Selected_Task].reset_index()
        else:
            task_data = task_info[task_info.Task_Name == Selected_Task].reset_index()

        task_len.update({Selected_Task: task_data.Dataset_Size[0]})
        variance_dict.update({Selected_Task: task_data.Variance[0]})
        std_dev_dict.update({Selected_Task: task_data.Std_Dev[0]})

        if dataset == 'Chemical':
            hamming_dist_dict.update({Selected_Task: task_data.Average_Hamming_Distance_within_Task[0]})
            euclidean_dist_dict.update({Selected_Task: task_data.Average_Euclidian_Distance_within_Task[0]})
            manhattan_dist_dict.update({Selected_Task: task_data.Average_Manhattan_Distance_within_Task[0]})
        else:
            euclidean_dist_dict.update({Selected_Task: task_data.Average_Euclidian_Distance_within_Task[0]})
            manhattan_dist_dict.update({Selected_Task: task_data.Average_Manhattan_Distance_within_Task[0]})

            euclidean_dist_dict_scaled.update(
                {Selected_Task: task_data.Average_Euclidian_Distance_within_Task_after_Scaling[0]})
            manhattan_dist_dict_scaled.update(
                {Selected_Task: task_data.Average_Manhattan_Distance_within_Task_after_Scaling[0]})
        # hamming_dist_dict_scaled.update(
        #     {Selected_Task: task_data.Average_Hamming_Distance_within_Task_after_Scaling[0]})
        single_res = single_results[single_results.Task == Selected_Task].reset_index()
        if dataset == 'School':
            Single_res_dict.update({Selected_Task: single_res.Loss_MSE[0]})
        else:
            Single_res_dict.update({Selected_Task: single_res.LOSS[0]})
        for i in range(10, 100, 10):
            loss_dict.update({(Selected_Task, i): single_res[f'val_loss_{i}'][0]})

    Task_1 = []
    Task_2 = []
    Dataset_Task1 = []
    Dataset_Task2 = []
    Variance_Task1 = []
    Variance_Task2 = []
    StdDev_Task1 = []
    StdDev_Task2 = []
    Loss_Task1 = []
    Loss_Task2 = []
    Distance_Task1 = []
    Distance_Task2 = []
    Fitted_param_a_Task1 = []
    Fitted_param_b_Task1 = []
    Fitted_param_a_Task2 = []
    Fitted_param_b_Task2 = []
    Loss_dict_Task1 = {}
    Loss_dict_Task2 = {}

    for i in range(10, 100, 10):
        Loss_dict_Task1.update({f'lc_task1_{i}': []})
        Loss_dict_Task2.update({f'lc_task2_{i}': []})

    task_combo = []
    count = 0

    for i in range(len(TASKS)):
        for j in range(len(TASKS)):
            if TASKS[i] != TASKS[j]:
                task1 = TASKS[i]
                task2 = TASKS[j]
                Task_1.append(task1)
                Task_2.append(task2)
                if TASKS[i] == 83:
                    count += 1
                task_combo.append([TASKS[i], TASKS[j]])

                Dataset_Task1.append(task_len[task1])
                Dataset_Task2.append(task_len[task2])
                Variance_Task1.append(variance_dict[task1])
                Variance_Task2.append(variance_dict[task2])
                StdDev_Task1.append(std_dev_dict[task1])
                StdDev_Task2.append(std_dev_dict[task2])
                Loss_Task1.append(Single_res_dict[task1])
                Loss_Task2.append(Single_res_dict[task2])

                if dataset != 'Chemical':
                    Distance_Task1.append(euclidean_dist_dict[task1])
                    Distance_Task2.append(euclidean_dist_dict[task2])
                else:
                    Distance_Task1.append(hamming_dist_dict[task1])
                    Distance_Task2.append(hamming_dist_dict[task2])

                for c in range(10, 100, 10):
                    Loss_dict_Task1[f'lc_task1_{c}'].append(loss_dict[(task1, c)])
                    Loss_dict_Task2[f'lc_task2_{c}'].append(loss_dict[(task2, c)])

                ys_task1 = []
                ys_task2 = []
                for c in range(10, 100, 10):
                    Loss_dict_Task1[f'lc_task1_{c}'].append(loss_dict[(task1, c)])
                    ys_task1.append(loss_dict[(task1, c)])
                    Loss_dict_Task2[f'lc_task2_{c}'].append(loss_dict[(task2, c)])
                    ys_task2.append(loss_dict[(task2, c)])

                fit = fit_log_curve(ys_task1)
                Fitted_param_a_Task1.append(fit[0, 0])
                Fitted_param_b_Task1.append(fit[1, 0])

                fit = fit_log_curve(ys_task2)
                Fitted_param_a_Task2.append(fit[0, 0])
                Fitted_param_b_Task2.append(fit[1, 0])

    Loss_dict_Task1_X = {}
    Loss_dict_Task2_X = {}

    for i in range(10, 100, 10):
        Loss_dict_Task1_X.update({f'lc_task1_{i}': []})
        Loss_dict_Task2_X.update({f'lc_task2_{i}': []})
    for i in range(len(TASKS)):
        for j in range(len(TASKS)):
            if TASKS[j] != TASKS[i]:
                task1 = TASKS[i]
                task2 = TASKS[j]
                Task_1.append(task1)
                Task_2.append(task2)
                if TASKS[i] == 83:
                    count += 1
                task_combo.append([TASKS[i], TASKS[j]])

                Dataset_Task1.append(task_len[task1])
                Dataset_Task2.append(task_len[task2])
                Variance_Task1.append(variance_dict[task1])
                Variance_Task2.append(variance_dict[task2])
                StdDev_Task1.append(std_dev_dict[task1])
                StdDev_Task2.append(std_dev_dict[task2])
                Loss_Task1.append(Single_res_dict[task1])
                Loss_Task2.append(Single_res_dict[task2])
                # Distance_Task1.append(euclidean_dist_dict[task1])
                # Distance_Task2.append(euclidean_dist_dict[task2])

                if dataset != 'Chemical':
                    Distance_Task1.append(euclidean_dist_dict[task1])
                    Distance_Task2.append(euclidean_dist_dict[task2])
                else:
                    Distance_Task1.append(hamming_dist_dict[task1])
                    Distance_Task2.append(hamming_dist_dict[task2])

                for c in range(10, 100, 10):
                    Loss_dict_Task1_X[f'lc_task1_{c}'].append(loss_dict[(task1, c)])
                    Loss_dict_Task2_X[f'lc_task2_{c}'].append(loss_dict[(task2, c)])

                ys_task1 = []
                ys_task2 = []
                for c in range(10, 100, 10):
                    Loss_dict_Task1_X[f'lc_task1_{c}'].append(loss_dict[(task1, c)])
                    ys_task1.append(loss_dict[(task1, c)])
                    Loss_dict_Task2_X[f'lc_task2_{c}'].append(loss_dict[(task2, c)])
                    ys_task2.append(loss_dict[(task2, c)])

                # ys_task1 = []
                # ys_task2 = []
                # for c in range(10, 100, 10):
                #     Loss_dict_Task1_X[f'lc_task1_{c}'].append(loss_dict[(task1, c)])
                #     ys_task1.append(loss_dict[(task1, c)])
                #     Loss_dict_Task2_X[f'lc_task2_{c}'].append(loss_dict[(task2, c)])
                #     ys_task2.append(loss_dict[(task2, c)])

                fit = fit_log_curve(ys_task1)
                Fitted_param_a_Task1.append(fit[0, 0])
                Fitted_param_b_Task1.append(fit[1, 0])

                fit = fit_log_curve(ys_task2)
                Fitted_param_a_Task2.append(fit[0, 0])
                Fitted_param_b_Task2.append(fit[1, 0])

    paired_improvement = []
    Weight = []
    InterTaskAffinity = []
    for pair in task_combo:

        stl_loss = 0
        stl_loss += Single_res_dict[pair[0]]
        stl_loss += Single_res_dict[pair[1]]
        pair_specific = pair_results[
            (pair_results.Task_1 == pair[0]) & (pair_results.Task_2 == pair[1])].reset_index()
        if len(pair_specific) == 0:
            pair_specific = pair_results[
                (pair_results.Task_1 == pair[1]) & (pair_results.Task_2 == pair[0])].reset_index()
        paired_improvement.append((stl_loss - pair_specific.Total_Loss[0]) / stl_loss)

        task_pair = tuple(sorted(pair))
        weight_df = Weight_Matrix[Weight_Matrix['Pairs'] == str(task_pair)].reset_index(drop=True)
        ita_df = ITA_data[ITA_data['Pairs'] == str(task_pair)].reset_index(drop=True)
        Weight.append(weight_df.Weight[0])
        InterTaskAffinity.append(ita_df.Pairwise_ITA[0])


    # '''*************'''
    # print('\n\n\n\n*********\n\n\n')
    # print(len(Dataset_Task1))
    # print(len(Dataset_Task2))
    # print(len(Variance_Task1))
    # print(len(Variance_Task2))
    # print(len(StdDev_Task1))
    # print(len(StdDev_Task2))
    # print(len(Loss_Task1))
    # print(len(Loss_Task2))
    # print(len(Distance_Task1))
    # print(len(Distance_Task2))
    # print(len(Loss_dict_Task2))
    # print(len(Loss_dict_Task1))
    # print(len(Loss_dict_Task1_X))
    # print(len(Loss_dict_Task2_X))
    # print(len(Fitted_param_a_Task1), len(Fitted_param_a_Task2))

    df = pd.DataFrame({
        'Task1': Task_1,
        'Task2': Task_2,
        'Dataset_Task1': Dataset_Task1,
        'Dataset_Task2': Dataset_Task2,
        'Variance_Task1': Variance_Task1,
        'Variance_Task2': Variance_Task2,
        'StdDev_Task1': StdDev_Task1,
        'StdDev_Task2': StdDev_Task2,
        'Distance_Task1': Distance_Task1,
        'Distance_Task2': Distance_Task2,
        'Loss_Task1': Loss_Task1,
        'Loss_Task2': Loss_Task2,
        'Fitted_param_a_Task1': Fitted_param_a_Task1,
        'Fitted_param_b_Task1': Fitted_param_b_Task1,
        'Fitted_param_a_Task2': Fitted_param_a_Task2,
        'Fitted_param_b_Task2': Fitted_param_b_Task2,
    })
    df['Change'] = paired_improvement
    df['Weight'] = Weight
    df['InterTaskAffinity'] = InterTaskAffinity

    learning_curve_loss_task1 = pd.DataFrame(Loss_dict_Task1)
    learning_curve_loss_task2 = pd.DataFrame(Loss_dict_Task2)
    learning_curve_loss_task1 = learning_curve_loss_task1.join(learning_curve_loss_task2)

    df = df.join(learning_curve_loss_task1)

    # learning_curve_loss_task1_X = pd.DataFrame(Loss_dict_Task1_X)
    # learning_curve_loss_task2_X = pd.DataFrame(Loss_dict_Task2_X)
    # learning_curve_loss_task1_X = learning_curve_loss_task1_X.join(learning_curve_loss_task2_X)
    #
    # learning_curve_loss_task1 = learning_curve_loss_task1.join(learning_curve_loss_task1_X)

    # df = df.join(learning_curve_loss_task1)
    # df = df.join(learning_curve_loss_task2)

    print(f'dataset: {dataset}, total task-specific features: {len(df.columns)}')

    df.to_csv(f'{ResultPath}/Pairwise/{modelName}/Pairwise_Task_Specific_Features_{dataset}.csv', index=False)


def get_task_relation_features(dataset):
    task_info = pd.read_csv(f'{DataPath}Task_Information_{dataset}.csv')
    task_distance_info = pd.read_csv(f'{DataPath}Task_Distance_{dataset}.csv')

    single_results = pd.read_csv(f'{ResultPath}STL/STL_{dataset}_{modelName}.csv')
    pair_results = pd.read_csv(f'{ResultPath}Pairwise/NN/{dataset}_Results_from_Pairwise_Training_ALL_{modelName}.csv')



    Single_Task_Loss = []
    Group_Dataset_Size = []

    Group_Variance_avg = []
    Group_Variance_Combined_Sum_Normalized = []
    Group_Variance_Combined_Prod_Normalized = []
    Group_Variance_Individual_Sum_Normalized = []
    Group_Variance_Individual_Prod_Normalized = []

    Group_StdDev_avg = []
    Group_StdDev_Combined_Sum_Normalized = []
    Group_StdDev_Combined_Prod_Normalized = []
    Group_StdDev_Individual_Sum_Normalized = []
    Group_StdDev_Individual_Prod_Normalized = []

    Euclidean_Distance_between_Tasks_Diff = []
    Euclidean_Distance_between_Tasks_Prod = []
    Euclidean_Distance_Combined_Sum = []
    Euclidean_Distance_Combined_Prod = []

    Manhattan_Distance_between_Tasks_Diff = []
    Manhattan_Distance_between_Tasks_Prod = []
    Manhattan_Distance_Combined_Sum = []
    Manhattan_Distance_Combined_Prod = []

    Euclidean_Distance_Scaled_between_Tasks_Diff = []
    Euclidean_Distance_Scaled_between_Tasks_Prod = []
    Euclidean_Distance_Scaled_Combined_Sum = []
    Euclidean_Distance_Scaled_Combined_Prod = []

    Manhattan_Distance_Scaled_between_Tasks_Diff = []
    Manhattan_Distance_Scaled_between_Tasks_Prod = []
    Manhattan_Distance_Scaled_Combined_Sum = []
    Manhattan_Distance_Scaled_Combined_Prod = []

    Hamming_Distance_between_Tasks_Diff = []
    Hamming_Distance_between_Tasks_Prod = []
    Hamming_Distance_Combined_Sum = []
    Hamming_Distance_Combined_Prod = []

    Loss_dict_Task1 = {}
    Loss_dict_Task2 = {}
    Loss_curve_diff = {}
    Fitted_param_a_diff = []
    Fitted_param_b_diff = []

    for i in range(10, 100, 10):
        Loss_dict_Task1.update({f'lc_task1_{i}': []})
        Loss_dict_Task2.update({f'lc_task2_{i}': []})
        Loss_curve_diff.update({f'train_curve_{i}': []})

    Change = []
    DatasetSize_diff = []

    task_len = {}
    variance_dict = {}
    std_dev_dict = {}
    euclidean_dist_dict = {}
    manhattan_dist_dict = {}
    hamming_dist_dict = {}
    euclidean_dist_dict_scaled = {}
    manhattan_dist_dict_scaled = {}
    Single_res_dict = {}
    loss_dict = {}

    for Selected_Task in TASKS:

        if dataset == 'Chemical':
            task_data = task_info[task_info.Molecule == Selected_Task].reset_index()
        else:
            task_data = task_info[task_info.Task_Name == Selected_Task].reset_index()

        task_len.update({Selected_Task: task_data.Dataset_Size[0]})
        variance_dict.update({Selected_Task: task_data.Variance[0]})
        std_dev_dict.update({Selected_Task: task_data.Std_Dev[0]})

        if dataset == 'Chemical':
            hamming_dist_dict.update({Selected_Task: task_data.Average_Hamming_Distance_within_Task[0]})
            euclidean_dist_dict.update({Selected_Task: task_data.Average_Euclidian_Distance_within_Task[0]})
            manhattan_dist_dict.update({Selected_Task: task_data.Average_Manhattan_Distance_within_Task[0]})
        else:
            euclidean_dist_dict.update({Selected_Task: task_data.Average_Euclidian_Distance_within_Task[0]})
            manhattan_dist_dict.update({Selected_Task: task_data.Average_Manhattan_Distance_within_Task[0]})
            euclidean_dist_dict_scaled.update(
                {Selected_Task: task_data.Average_Euclidian_Distance_within_Task_after_Scaling[0]})
            manhattan_dist_dict_scaled.update(
                {Selected_Task: task_data.Average_Manhattan_Distance_within_Task_after_Scaling[0]})

        single_res = single_results[single_results.Task == Selected_Task].reset_index()
        if dataset == 'School':
            Single_res_dict.update({Selected_Task: single_res.Loss_MSE[0]})
        else:
            Single_res_dict.update({Selected_Task: single_res.LOSS[0]})
        for i in range(10, 100, 10):
            loss_dict.update({(Selected_Task, i): single_res[f'val_loss_{i}'][0]})

    for group in tqdm.tqdm(range(len(pair_results))):
        tasks = []
        tasks.append(pair_results.Task_1[group])
        tasks.append(pair_results.Task_2[group])

        ys_task1 = []
        ys_task2 = []
        for i in range(10, 100, 10):
            Loss_dict_Task1[f'lc_task1_{i}'].append(loss_dict[(tasks[0], i)])
            ys_task1.append(loss_dict[(tasks[0], i)])
            Loss_dict_Task2[f'lc_task2_{i}'].append(loss_dict[(tasks[1], i)])
            ys_task2.append(loss_dict[(tasks[1], i)])
            d = abs(loss_dict[(tasks[0], i)] - loss_dict[(tasks[1], i)]) / (
                    loss_dict[(tasks[0], i)] + loss_dict[(tasks[1], i)])
            Loss_curve_diff[f'train_curve_{i}'].append(d)

        fit = fit_log_curve(ys_task1)
        a_task1 = fit[0, 0]
        b_task1 = fit[1, 0]
        fit = fit_log_curve(ys_task2)
        a_task2 = fit[0, 0]
        b_task2 = fit[1, 0]

        d = abs(a_task1 - a_task2) / (a_task1 + a_task2)
        Fitted_param_a_diff.append(d)

        d = abs(b_task1 - b_task2) / (b_task1 + b_task2)
        Fitted_param_b_diff.append(d)

        sample_size = 0
        avg_var = []
        avg_stddev = []
        sum_loss_single_task = 0

        task_dist_data = task_distance_info[(task_distance_info.Task_1 == pair_results.Task_1[group]) & (
                task_distance_info.Task_2 == pair_results.Task_2[group])].reset_index()

        for t in tasks:
            sample_size += task_len[t]
            avg_var.append(variance_dict[t])
            avg_stddev.append(std_dev_dict[t])
            sum_loss_single_task += Single_res_dict[t]

        DatasetSize_diff.append((abs(task_len[tasks[0]]-task_len[tasks[1]]))/(task_len[tasks[0]]+task_len[tasks[1]]))

        Euclidean_Distance_between_Tasks_Diff.append(
            abs(euclidean_dist_dict[tasks[0]] - euclidean_dist_dict[tasks[1]]) / (
                    euclidean_dist_dict[tasks[0]] + euclidean_dist_dict[tasks[1]]))
        Euclidean_Distance_between_Tasks_Prod.append(
            (task_dist_data.Average_Euclidean_Distance[0] * task_dist_data.Average_Euclidean_Distance[0]) / (
                    euclidean_dist_dict[tasks[0]] * euclidean_dist_dict[tasks[1]]))
        Euclidean_Distance_Combined_Sum.append(unified_Distance(tasks,dataset)[0] / (
                euclidean_dist_dict[tasks[0]] + euclidean_dist_dict[tasks[1]]))
        Euclidean_Distance_Combined_Prod.append((unified_Distance(tasks,dataset)[0] * unified_Distance(tasks,dataset)[0]) / (
                euclidean_dist_dict[tasks[0]] * euclidean_dist_dict[tasks[1]]))

        Manhattan_Distance_between_Tasks_Diff.append(
            abs(manhattan_dist_dict[tasks[0]] - manhattan_dist_dict[tasks[1]]) / (
                    manhattan_dist_dict[tasks[0]] + manhattan_dist_dict[tasks[1]]))
        Manhattan_Distance_between_Tasks_Prod.append(
            (task_dist_data.Average_Manhattan_Distance[0] * task_dist_data.Average_Manhattan_Distance[0]) / (
                    manhattan_dist_dict[tasks[0]] * manhattan_dist_dict[tasks[1]]))
        Manhattan_Distance_Combined_Sum.append(unified_Distance(tasks,dataset)[1] / (
                manhattan_dist_dict[tasks[0]] + manhattan_dist_dict[tasks[1]]))
        Manhattan_Distance_Combined_Prod.append((unified_Distance(tasks,dataset)[1] * unified_Distance(tasks,dataset)[1]) / (
                manhattan_dist_dict[tasks[0]] * manhattan_dist_dict[tasks[1]]))

        if dataset=='Chemical':
            Hamming_Distance_between_Tasks_Diff.append(
                abs(hamming_dist_dict[tasks[0]] - hamming_dist_dict[tasks[1]]) / (
                        hamming_dist_dict[tasks[0]] + hamming_dist_dict[tasks[1]]))
            Hamming_Distance_between_Tasks_Prod.append(
                (task_dist_data.Average_Hamming_Distance[0] * task_dist_data.Average_Hamming_Distance[0]) / (
                        hamming_dist_dict[tasks[0]] * hamming_dist_dict[tasks[1]]))
            Hamming_Distance_Combined_Sum.append(unified_Distance(tasks,dataset)[2] / (
                    hamming_dist_dict[tasks[0]] + hamming_dist_dict[tasks[1]]))
            Hamming_Distance_Combined_Prod.append((unified_Distance(tasks,dataset)[2] * unified_Distance(tasks,dataset)[2]) / (
                    hamming_dist_dict[tasks[0]] * hamming_dist_dict[tasks[1]]))


        if dataset!='Chemical':
            Euclidean_Distance_Scaled_between_Tasks_Diff.append(
                abs(euclidean_dist_dict_scaled[tasks[0]] - euclidean_dist_dict_scaled[tasks[1]]) / (
                        euclidean_dist_dict_scaled[tasks[0]] + euclidean_dist_dict_scaled[tasks[1]]))
            Euclidean_Distance_Scaled_between_Tasks_Prod.append(
                (task_dist_data.Average_Euclidean_Distance_after_Scaling[0] *
                 task_dist_data.Average_Euclidean_Distance_after_Scaling[0]) / (
                        euclidean_dist_dict_scaled[tasks[0]] * euclidean_dist_dict_scaled[tasks[1]]))
            Euclidean_Distance_Scaled_Combined_Sum.append(unified_Distance_scaled(tasks,dataset)[0] / (
                    euclidean_dist_dict_scaled[tasks[0]] + euclidean_dist_dict_scaled[tasks[1]]))
            Euclidean_Distance_Scaled_Combined_Prod.append(
                (unified_Distance_scaled(tasks,dataset)[0] * unified_Distance_scaled(tasks,dataset)[0]) / (
                        euclidean_dist_dict_scaled[tasks[0]] * euclidean_dist_dict_scaled[tasks[1]]))

            Manhattan_Distance_Scaled_between_Tasks_Diff.append(
                abs(manhattan_dist_dict_scaled[tasks[0]] - manhattan_dist_dict_scaled[tasks[1]]) / (
                        manhattan_dist_dict_scaled[tasks[0]] + manhattan_dist_dict_scaled[tasks[1]]))
            Manhattan_Distance_Scaled_between_Tasks_Prod.append(
                (task_dist_data.Average_Manhattan_Distance_after_Scaling[0] *
                 task_dist_data.Average_Manhattan_Distance_after_Scaling[0]) / (
                        manhattan_dist_dict_scaled[tasks[0]] * manhattan_dist_dict_scaled[tasks[1]]))
            Manhattan_Distance_Scaled_Combined_Sum.append(unified_Distance_scaled(tasks,dataset)[1] / (
                    manhattan_dist_dict_scaled[tasks[0]] + manhattan_dist_dict_scaled[tasks[1]]))
            Manhattan_Distance_Scaled_Combined_Prod.append(
                (unified_Distance_scaled(tasks,dataset)[1] * unified_Distance_scaled(tasks,dataset)[1]) / (
                        manhattan_dist_dict_scaled[tasks[0]] * manhattan_dist_dict_scaled[tasks[1]]))

        Single_Task_Loss.append(sum_loss_single_task)

        Group_Dataset_Size.append(sample_size)  # individual length
        var, std = get_variance_stdDev_for_groups(tasks,dataset)
        Group_Variance_Combined_Sum_Normalized.append(var / (variance_dict[tasks[0]] + variance_dict[tasks[1]]))
        Group_Variance_Combined_Prod_Normalized.append(
            math.pow(var, 2) / (variance_dict[tasks[0]] * variance_dict[tasks[1]]))
        Group_Variance_Individual_Sum_Normalized.append(abs(variance_dict[tasks[0]] - variance_dict[tasks[1]]) / (
                variance_dict[tasks[0]] + variance_dict[tasks[1]]))
        Group_Variance_avg.append(np.mean(avg_var))

        Group_StdDev_avg.append(np.mean(avg_stddev))
        Group_StdDev_Combined_Sum_Normalized.append(std / (std_dev_dict[tasks[0]] + std_dev_dict[tasks[1]]))
        Group_StdDev_Combined_Prod_Normalized.append(
            math.pow(std, 2) / (std_dev_dict[tasks[0]] * std_dev_dict[tasks[1]]))
        Group_StdDev_Individual_Sum_Normalized.append(
            abs(std_dev_dict[tasks[0]] - std_dev_dict[tasks[1]]) / (std_dev_dict[tasks[0]] + std_dev_dict[tasks[1]]))

        Change.append((sum_loss_single_task - pair_results.Total_Loss[group]) / sum_loss_single_task)

    '''*************'''
    # print('\n\n\n\n*********\n\n\n')
    # print(len(Group_Variance_avg))
    # print(len(Group_Variance_Combined_Sum_Normalized))
    # print(len(Group_Variance_Combined_Prod_Normalized))
    # print(len(Group_Variance_Individual_Sum_Normalized))
    # print(len(Group_Variance_Individual_Prod_Normalized))
    #
    # print(len(Group_StdDev_avg))
    # print(len(Group_StdDev_Combined_Sum_Normalized))
    # print(len(Group_StdDev_Combined_Prod_Normalized))
    # print(len(Group_StdDev_Individual_Sum_Normalized))
    # print(len(Group_StdDev_Individual_Prod_Normalized))
    #
    # print(len(Euclidean_Distance_between_Tasks_Diff))
    # print(len(Euclidean_Distance_between_Tasks_Prod))
    # print(len(Euclidean_Distance_Combined_Sum))
    #
    # print(len(Manhattan_Distance_between_Tasks_Diff))
    # print(len(Manhattan_Distance_between_Tasks_Prod))
    # print(len(Manhattan_Distance_Combined_Sum))
    #
    # print(len(Euclidean_Distance_Scaled_between_Tasks_Diff))
    # print(len(Euclidean_Distance_Scaled_between_Tasks_Prod))
    # print(len(Euclidean_Distance_Scaled_Combined_Sum))
    #
    # print(len(Manhattan_Distance_Scaled_between_Tasks_Diff))
    # print(len(Manhattan_Distance_Scaled_between_Tasks_Prod))
    # print(len(Manhattan_Distance_Scaled_Combined_Sum))
    # print('\n\n\n\n*********\n\n\n')
    # print(len(Group_Dataset_Size))
    # print(len(Euclidean_Distance_between_Tasks_Diff))
    # print(len(Loss_dict_Task2))
    # print(len(Loss_dict_Task1))
    if dataset=='Chemical':
        df = pd.DataFrame({
            'DatasetSize_diff': DatasetSize_diff,
            'Group_Variance_Combined_Sum_Normalized': Group_Variance_Combined_Sum_Normalized,
            'Group_Variance_Combined_Prod_Normalized': Group_Variance_Combined_Prod_Normalized,
            'Group_Variance_Individual_Sum_Normalized': Group_Variance_Individual_Sum_Normalized,
            'Group_StdDev_Combined_Sum_Normalized': Group_StdDev_Combined_Sum_Normalized,
            'Group_StdDev_Combined_Prod_Normalized': Group_StdDev_Combined_Prod_Normalized,
            'Group_StdDev_Individual_Sum_Normalized': Group_StdDev_Individual_Sum_Normalized,
            'Group_StdDev_avg': Group_StdDev_avg,
            'Group_Variance_avg': Group_Variance_avg,

            # Distance Features
            'Euclidean_Distance_between_Tasks_Diff': Euclidean_Distance_between_Tasks_Diff,
            'Euclidean_Distance_between_Tasks_Prod': Euclidean_Distance_between_Tasks_Prod,
            'Euclidean_Distance_Combined_Sum': Euclidean_Distance_Combined_Sum,
            'Euclidean_Distance_Combined_Prod': Euclidean_Distance_Combined_Prod,

            'Manhattan_Distance_between_Tasks_Diff': Manhattan_Distance_between_Tasks_Diff,
            'Manhattan_Distance_between_Tasks_Prod': Manhattan_Distance_between_Tasks_Prod,
            'Manhattan_Distance_Combined_Sum': Manhattan_Distance_Combined_Sum,
            'Manhattan_Distance_Combined_Prod': Manhattan_Distance_Combined_Prod,

            'Hamming_Distance_between_Tasks_Diff': Hamming_Distance_between_Tasks_Diff,
            'Hamming_Distance_between_Tasks_Prod': Hamming_Distance_between_Tasks_Prod,
            'Hamming_Distance_Combined_Sum': Hamming_Distance_Combined_Sum,
            'Hamming_Distance_Combined_Prod': Hamming_Distance_Combined_Prod,

            'Fitted_param_a_diff': Fitted_param_a_diff,
            'Fitted_param_b_diff': Fitted_param_b_diff,
            'Change': Change
        })

    else:
        df = pd.DataFrame({
            'DatasetSize_diff': DatasetSize_diff,
            'Group_Variance_Combined_Sum_Normalized': Group_Variance_Combined_Sum_Normalized,
            'Group_Variance_Combined_Prod_Normalized': Group_Variance_Combined_Prod_Normalized,
            'Group_Variance_Individual_Sum_Normalized': Group_Variance_Individual_Sum_Normalized,
            'Group_StdDev_Combined_Sum_Normalized': Group_StdDev_Combined_Sum_Normalized,
            'Group_StdDev_Combined_Prod_Normalized': Group_StdDev_Combined_Prod_Normalized,
            'Group_StdDev_Individual_Sum_Normalized': Group_StdDev_Individual_Sum_Normalized,
            'Group_StdDev_avg': Group_StdDev_avg,
            'Group_Variance_avg': Group_Variance_avg,

            # Distance Features
            'Euclidean_Distance_between_Tasks_Diff': Euclidean_Distance_between_Tasks_Diff,
            'Euclidean_Distance_between_Tasks_Prod': Euclidean_Distance_between_Tasks_Prod,
            'Euclidean_Distance_Combined_Sum': Euclidean_Distance_Combined_Sum,
            'Euclidean_Distance_Combined_Prod': Euclidean_Distance_Combined_Prod,
            'Manhattan_Distance_between_Tasks_Diff': Manhattan_Distance_between_Tasks_Diff,
            'Manhattan_Distance_between_Tasks_Prod': Manhattan_Distance_between_Tasks_Prod,
            'Manhattan_Distance_Combined_Sum': Manhattan_Distance_Combined_Sum,
            'Manhattan_Distance_Combined_Prod': Manhattan_Distance_Combined_Prod,

            'Euclidean_Distance_Scaled_between_Tasks_Diff': Euclidean_Distance_Scaled_between_Tasks_Diff,
            'Euclidean_Distance_Scaled_between_Tasks_Prod': Euclidean_Distance_Scaled_between_Tasks_Prod,
            'Euclidean_Distance_Scaled_Combined_Sum': Euclidean_Distance_Scaled_Combined_Sum,
            'Euclidean_Distance_Scaled_Combined_Prod': Euclidean_Distance_Scaled_Combined_Prod,
            'Manhattan_Distance_Scaled_between_Tasks_Diff': Manhattan_Distance_Scaled_between_Tasks_Diff,
            'Manhattan_Distance_Scaled_between_Tasks_Prod': Manhattan_Distance_Scaled_between_Tasks_Prod,
            'Manhattan_Distance_Scaled_Combined_Sum': Manhattan_Distance_Scaled_Combined_Sum,
            'Manhattan_Distance_Scaled_Combined_Prod': Manhattan_Distance_Scaled_Combined_Prod,

            'Fitted_param_a_diff': Fitted_param_a_diff,
            'Fitted_param_b_diff': Fitted_param_b_diff,
            'Change': Change
        })

    learning_curve_loss_diff = pd.DataFrame(Loss_curve_diff)
    # learning_curve_loss_task1 = pd.DataFrame(Loss_dict_Task1)
    # learning_curve_loss_task2 = pd.DataFrame(Loss_dict_Task2)
    # learning_curve_loss_task1 = learning_curve_loss_task1.join(learning_curve_loss_task2)
    # learning_curve_loss_task1 = learning_curve_loss_task1.join(learning_curve_loss_diff)

    # learning_curve_df = (learning_curve_loss_task1 - learning_curve_loss_task1.min()) / (
    #             learning_curve_loss_task1.max() - learning_curve_loss_task1.min())
    # print(learning_curve_df.describe())
    df = df.join(learning_curve_loss_diff)
    df.to_csv(f'{ResultPath}Pairwise/{modelName}/Pairwise_Task_Relation_Features_{dataset}.csv', index=False)




    print(f'Pairwise_Task_Relation_Features_{dataset}.csv saved with samples {len(df)}')

    Weight_Matrix = pd.read_csv(f'{ResultPath}/Weight_Matrix/Weight_Affinity_{dataset}.csv', low_memory=False)
    Weight = list(Weight_Matrix['Weight'])
    ITA_data = pd.read_csv(f'{ResultPath}/InterTask_Affinity/Pairwise_ITA_{dataset}.csv', low_memory=False)
    InterTaskAffinity = list(ITA_data['Pairwise_ITA'])
    df['Weight'] = Weight
    df['InterTaskAffinity'] = InterTaskAffinity

    print(f'dataset: {dataset}, total task-relation features: {len(df.columns)}')

    df.to_csv(f'{ResultPath}/Pairwise/{modelName}/Pairwise_Task_Relation_Features_{dataset}.csv', index=False)

if __name__ == '__main__':

    modelName = 'NN'
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

        get_task_specific_features(dataset)
        get_task_relation_features(dataset)

