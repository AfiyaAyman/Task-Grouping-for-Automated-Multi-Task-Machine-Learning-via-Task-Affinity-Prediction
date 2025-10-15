import scipy.io
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean,hamming,cityblock
from sklearn import preprocessing

def get_task_information():
    landmineData = scipy.io.loadmat(f'{DataPath}LandmineData.mat')
    mat = {k:v for k, v in landmineData.items() if k[0] != '_'}
    data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})


    Task = []
    Variance = []
    Std_Dev = []
    length = []
    Average_Euclidian_Distance_within_Task = []
    Average_Manhattan_Distance_within_Task = []
    Average_Euclidian_Distance_within_Task_after_Scaling = []
    Average_Manhattan_Distance_within_Task_after_Scaling = []

    for landmine in range(len(data)):
        Task.append(landmine)
        Feature_dict = {}
        feature = list(data.feature[landmine])
        label = list(data.label[landmine])
        Labels = []
        for single_feat in feature:
            for j in range(len(single_feat)):
                if j not in Feature_dict:
                    Feature_dict.update({j:[]})
                Feature_dict[j].append(single_feat[j])

        for lbl in label:
            Labels.append(lbl[0])

        Feature_dict = pd.DataFrame.from_dict(Feature_dict)
        Feature_dict['Labels'] = Labels
        print(f'Task = {landmine}, #Samples = {len(Feature_dict)}, class 1 = {len(Feature_dict[Feature_dict.Labels == 1])}, class 0 = {len(Feature_dict[Feature_dict.Labels == 0])}')
        # Feature_dict.to_csv(f'{DataPath}LandmineData_{landmine}.csv', index=None)


        length.append(len(Feature_dict))
        Variance.append(Feature_dict['Labels'].var())
        Std_Dev.append(Feature_dict['Labels'].std())
        Feature_dict['Labels'] = Labels

        DataSet = np.array(Feature_dict, dtype=float)

        dist = []
        mdist = []
        for i in range(0, len(DataSet) - 1):
            dist.append(euclidean(DataSet[i], DataSet[i + 1]))
            mdist.append(cityblock(DataSet[i], DataSet[i + 1]))
        Average_Euclidian_Distance_within_Task.append(np.mean(dist))
        Average_Manhattan_Distance_within_Task.append(np.mean(mdist))

        '''Feature Scaling'''
        x = Feature_dict.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        DataSet_Scaled = pd.DataFrame(x_scaled)
        DataSet_Scaled = np.array(DataSet_Scaled, dtype=float)
        dist = []
        mdist = []
        for i in range(0, len(DataSet_Scaled) - 1):
            dist.append(euclidean(DataSet_Scaled[i], DataSet_Scaled[i + 1]))
            mdist.append(cityblock(DataSet_Scaled[i], DataSet_Scaled[i + 1]))
        Average_Euclidian_Distance_within_Task_after_Scaling.append(np.mean(dist))
        Average_Manhattan_Distance_within_Task_after_Scaling.append(np.mean(mdist))


    Task_info = pd.DataFrame({"Task_Name":Task,
                              "Dataset_Size":length,
                              "Variance":Variance,
                              "Std_Dev":Std_Dev,
                              "Average_Euclidian_Distance_within_Task":Average_Euclidian_Distance_within_Task,
                              "Average_Manhattan_Distance_within_Task":Average_Manhattan_Distance_within_Task,
                              'Average_Euclidian_Distance_within_Task_after_Scaling':Average_Euclidian_Distance_within_Task_after_Scaling,
                              'Average_Manhattan_Distance_within_Task_after_Scaling':Average_Manhattan_Distance_within_Task_after_Scaling
                              })

    Task_info.to_csv(f"{DataPath}Task_Information_Landmine.csv",index=False)
    return max(length)

def prep_Data_for_mtl():
    for l in landmine:
        df = pd.read_csv(f"{DataPath}LandmineData_{l}.csv")
        print(f'Processing Landmine {l} ----------  {len(df)}')
        samples_to_be_repeated = maxlen - len(df)
        if samples_to_be_repeated > 0:
            df2 = df.sample(n=samples_to_be_repeated, random_state=1234, replace=True)
            df = df.append(df2, ignore_index=True)
        else:
            df = df.sample(n=maxlen)
        csv = (f"{DataPath}DATA/LandmineData_{l}_for_MTL.csv")
        df.to_csv(csv, index=False)


DataPath = '../Dataset/LANDMINE/'
maxlen = get_task_information()
landmine = [i for i in range(0, 29)]
# prep_Data_for_mtl()