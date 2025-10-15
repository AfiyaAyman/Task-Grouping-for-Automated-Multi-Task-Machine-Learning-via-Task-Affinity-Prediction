import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean,hamming,cityblock

def convert_to_csv():
    datafile = f"{DataPath}mhcp.dat"
    file1 = open(datafile, 'r')
    Lines = file1.readlines()


    Feature_dict = {}
    for i in range(182):
        Feature_dict.update({i:[]})


    # Strips the newline character
    for line in Lines:
        line = line.split(' ')
        for elem in range(len(line)):
            Feature_dict[elem].append(int(line[elem]))

    ChemicalData = pd.DataFrame.from_dict(Feature_dict)
    ChemicalData.to_csv(f'../Dataset/CHEMICAL/ChemicalData_All.csv',index=False)

def get_task_information():

    Length = []
    Variance = []
    StdDev = []
    Average_Euclidian_Distance_within_Task = []
    Average_Hamming_Distance_within_Task = []
    Average_Manhattan_Distance_within_Task = []

    count = 0
    for m in Molecule:
        moleculeData = ChemicalData[ChemicalData['180'] == m].reset_index()

        Negative_Sample = moleculeData[moleculeData['181'] == -1]
        Positive_Sample = moleculeData[moleculeData['181'] == 1]

        if len(Positive_Sample)>len(Negative_Sample):
            diff = len(Positive_Sample)-len(Negative_Sample)
            Negative_Sample_length = len(Positive_Sample)-diff
            Positive_Sample = Positive_Sample.sample(n=Negative_Sample_length).reset_index()
        else:
            Negative_Sample_length = len(Positive_Sample)
        Negative_Sample = Negative_Sample.sample(n=Negative_Sample_length).reset_index()


        ALL = [Negative_Sample, Positive_Sample]
        df2 = pd.concat(ALL)
        df2 = df2.sample(frac=1)
        # print(f'molecule = {m}\tLength = {len(moleculeData)}\t Positive {len(Positive_Sample)}\t Negative {len(Negative_Sample)}\tLength = {len(df2)}')
        Length.append(len(df2))

        df = df2[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
                 '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
                 '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45',
                 '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60',
                 '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75',
                 '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90',
                 '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105',
                 '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120',
                 '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135',
                 '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150',
                 '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165',
                 '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '181']]


        df.to_csv(f"../Dataset/CHEMICAL/{m}_Molecule_Data.csv", index=False)

        Variance.append(df['181'].var())
        StdDev.append(df['181'].std())
        dist = []
        hdist = []
        mdist = []

        DataSet = np.array(df, dtype=float)
        for i in range(0, len(DataSet) - 1):
            dist.append(euclidean(DataSet[i], DataSet[i + 1]))
            hdist.append(hamming(DataSet[i], DataSet[i+1]))
            mdist.append(cityblock(DataSet[i],DataSet[i+1]))
        Average_Euclidian_Distance_within_Task.append(np.mean(dist))
        Average_Hamming_Distance_within_Task.append(np.mean(hdist))
        Average_Manhattan_Distance_within_Task.append(np.mean(mdist))

    res = pd.DataFrame({'Molecule':Molecule,
                        "Dataset_Size":Length,
                        'Variance':Variance,
                        'Std_Dev':StdDev,
                        'Average_Euclidian_Distance_within_Task':Average_Euclidian_Distance_within_Task,
                        'Average_Hamming_Distance_within_Task':Average_Hamming_Distance_within_Task,
                        'Average_Manhattan_Distance_within_Task': Average_Manhattan_Distance_within_Task})

    res.to_csv(f"{DataPath}Task_Information_Chemical.csv",index=False)
    return max(Length)

def prep_data_for_mtl():
    for m in Molecule:
        df = pd.read_csv(f"../Dataset/CHEMICAL/{m}_Molecule_Data.csv")
        df = df[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
                 '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
                 '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45',
                 '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60',
                 '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75',
                 '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90',
                 '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105',
                 '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119',
                 '120',
                 '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134',
                 '135',
                 '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149',
                 '150',
                 '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164',
                 '165',
                 '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179',
                 '181']]

        print(f'Processing Molecule {m} ----------  {len(df)}')


        samples_to_be_repeated = maxlen - len(df)
        if samples_to_be_repeated > 0:
            df2 = df.sample(n=samples_to_be_repeated, random_state=1234, replace=True)
            df = df.append(df2, ignore_index=True)
            print(f'Length of new df = {len(df)}')
        else:
            df = df.sample(n=maxlen)
        csv = (f"{mtl_datapath}{m}_Chemical_Data_for_MTL.csv")
        df.to_csv(csv, index=False)


def get_task_information_SVM():

    Length = []
    Variance = []
    StdDev = []
    Average_Euclidian_Distance_within_Task = []
    Average_Hamming_Distance_within_Task = []
    Average_Manhattan_Distance_within_Task = []

    count = 0
    for m in Molecule:
        moleculeData = ChemicalData[ChemicalData['180'] == m].reset_index()

        Negative_Sample = moleculeData[moleculeData['181'] == -1]
        Positive_Sample = moleculeData[moleculeData['181'] == 1]

        if len(Positive_Sample)>len(Negative_Sample):
            if len(Positive_Sample)>450:
                Positive_Sample = Positive_Sample.sample(n=450).reset_index()

            diff = len(Positive_Sample)-len(Negative_Sample)
            Negative_Sample_length = len(Positive_Sample)-diff
            Positive_Sample = Positive_Sample.sample(n=Negative_Sample_length).reset_index()
        else:
            if len(Positive_Sample)>450:
                Positive_Sample = Positive_Sample.sample(n=450).reset_index()
            Negative_Sample_length = len(Positive_Sample)

        Negative_Sample = Negative_Sample.sample(n=Negative_Sample_length).reset_index()


        ALL = [Negative_Sample, Positive_Sample]
        df2 = pd.concat(ALL)
        df2 = df2.sample(frac=1)
        print(f'molecule = {m}\tLength = {len(moleculeData)}\t Positive {len(Positive_Sample)}\t Negative {len(Negative_Sample)}\tLength = {len(df2)}')
        Length.append(len(df2))

        df = df2[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
                 '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
                 '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45',
                 '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60',
                 '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75',
                 '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90',
                 '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105',
                 '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120',
                 '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135',
                 '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150',
                 '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165',
                 '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '181']]


        df.to_csv(f"../Dataset/CHEMICAL/{m}_Molecule_Data_SVM.csv", index=False)

        Variance.append(df['181'].var())
        StdDev.append(df['181'].std())
        dist = []
        hdist = []
        mdist = []

        DataSet = np.array(df, dtype=float)
        for i in range(0, len(DataSet) - 1):
            dist.append(euclidean(DataSet[i], DataSet[i + 1]))
            hdist.append(hamming(DataSet[i], DataSet[i+1]))
            mdist.append(cityblock(DataSet[i],DataSet[i+1]))
        Average_Euclidian_Distance_within_Task.append(np.mean(dist))
        Average_Hamming_Distance_within_Task.append(np.mean(hdist))
        Average_Manhattan_Distance_within_Task.append(np.mean(mdist))

    res = pd.DataFrame({'Molecule':Molecule,
                        "Dataset_Size":Length,
                        'Variance':Variance,
                        'Std_Dev':StdDev,
                        'Average_Euclidian_Distance_within_Task':Average_Euclidian_Distance_within_Task,
                        'Average_Hamming_Distance_within_Task':Average_Hamming_Distance_within_Task,
                        'Average_Manhattan_Distance_within_Task': Average_Manhattan_Distance_within_Task})

    res.to_csv(f"{DataPath}Task_Information_Chemical_SVM.csv",index=False)
    print(Length)
    return max(Length)

DataPath = '../Dataset/CHEMICAL/'
convert_to_csv()
ChemicalData = pd.read_csv(f'../Dataset/CHEMICAL/ChemicalData_All.csv',low_memory=False)
Molecule = ChemicalData['180'].unique()

'''For NN and XGBoost'''
maxlen = get_task_information()
mtl_datapath = f'{DataPath}DATA/'
prep_data_for_mtl()


'''For SVM'''
maxlen = get_task_information_SVM()
# print(f'Maximum Length of the Dataset = {maxlen}')
maxlen = 800
mtl_datapath = f'{DataPath}SVM_DATA/'
prep_data_for_mtl()