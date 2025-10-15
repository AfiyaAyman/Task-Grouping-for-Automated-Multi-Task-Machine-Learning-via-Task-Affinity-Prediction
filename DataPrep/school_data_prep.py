import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean,hamming,cityblock
from sklearn import preprocessing

'''Initial Data Read'''
def convert_to_csv():
    'For the ease of data-processing & personal preference'
    datafile = "../Dataset/SCHOOL/ILEA567.DAT"
    file1 = open(datafile, 'r')
    Lines = file1.readlines()

    Year = []
    School = []
    ExamScore = []
    FSM = []
    VR_Band = []
    Gender = []
    VR_BAND_STudent = []
    Ethnicity = []
    School_Gender = []
    School_Denomination = []

    count = 0
    # Strips the newline character
    for line in Lines:
        count += 1
        Year.append(int(line[0]))
        School.append(int(line[1:4]))
        ExamScore.append(int(line[4:6]))
        FSM.append(int(line[6:8]))
        VR_Band.append(int(line[8:10]))
        Gender.append(int(line[10]))
        VR_BAND_STudent.append(int(line[11]))
        Ethnicity.append(int(line[12:14]))
        School_Gender.append(int(line[14]))
        School_Denomination.append(int(line[15]))
        if count == 15362:
            break

    print(len(Year),len(School),len(ExamScore),len(FSM),len(VR_Band),len(Gender),len(VR_BAND_STudent),len(Ethnicity),len(School_Gender),len(School_Denomination),)

    school_data = pd.DataFrame({'Year':Year,
                       'School':School,
                       'FSM':FSM,
                       'VR_Band':VR_Band,
                       'Gender':Gender,
                       'VR_BAND_Student':VR_BAND_STudent,
                       'Ethnicity':Ethnicity,
                        'School_Gender':School_Gender,
                       'School_Denomination':School_Denomination,
                        'ExamScore': ExamScore
                       })

    school_data.to_csv("../Dataset/SCHOOL/School_Data.csv",index = False)

'''Data Processing and Separate Tasks'''
def data_processing():
    school_data = pd.read_csv("../Dataset/SCHOOL/School_Data.csv")
    school_data = school_data[['School',
                               'Year', 'VR_Band', 'Gender', 'Ethnicity', #Student-dependent
                               'FSM', 'VR_BAND_Student','School_Gender', 'School_Denomination', #School-dependent
                               'ExamScore']]

    one_hot = pd.get_dummies(school_data.Year)
    school_data = school_data.join(one_hot)

    school_data = school_data.rename(
        columns={1: '1985', 2: '1986', 3: '1987'})

    one_hot = pd.get_dummies(school_data.Ethnicity)
    school_data = school_data.join(one_hot)
    school_data = school_data.rename(
        columns={1: 'ESWI', 2: 'African',3: 'Arab', 4: 'Bangladeshi',5: 'Caribbean', 6: 'Greek',7: 'Indian', 8: 'Pakistani',9: 'SE_Asian', 10: 'Turkish',11:'Other'})

    one_hot = pd.get_dummies(school_data.School_Denomination)
    school_data = school_data.join(one_hot)
    school_data = school_data.rename(
        columns={1: 'Maintained', 2: 'Church', 3: 'Roman_Cath'})

    school_data = school_data.drop('Year', 1)
    school_data = school_data.drop('Ethnicity', 1)
    school_data = school_data.drop('School_Denomination', 1)

    school_data['VR_Band'] = (school_data.VR_Band - school_data.VR_Band.mean()) / np.std(school_data.VR_Band)
    school_data['FSM'] = (school_data.FSM - school_data.FSM.mean()) / np.std(school_data.FSM)
    school_data['ExamScore'] = (school_data.ExamScore - school_data.ExamScore.mean()) / np.std(school_data.ExamScore)

    school_data = school_data[['School',
                                '1985','1986','1987',
                               'ESWI', 'African', 'Arab', 'Bangladeshi', 'Caribbean', 'Greek', 'Indian', 'Pakistani', 'SE_Asian', 'Turkish', 'Other',
                                'VR_Band', 'Gender',
                                'FSM', 'VR_BAND_Student', 'School_Gender', 'Maintained', 'Church', 'Roman_Cath',
                                'ExamScore',
                                ]]
    for i in SCHOOLS:
        df = school_data[school_data.School == i].reset_index()
        df.to_csv(f"../Dataset/SCHOOL/{i}_School_Data.csv",index = False)

def get_task_information():

    'Get each tasks data size, variance, std dev in target, distance among samples, etc '
    Task_Name = []
    length = []
    Variance = []
    Std_Dev = []
    Average_Score = []

    Average_Euclidian_Distance_within_Task = []
    Average_Manhattan_Distance_within_Task = []
    Average_Euclidian_Distance_within_Task_after_Scaling = []
    Average_Manhattan_Distance_within_Task_after_Scaling = []

    for sch_id in SCHOOLS:
        csv = (f"{DataPath}/{sch_id}_School_Data.csv")
        df = pd.read_csv(csv, low_memory=False)
        Task_Name.append(sch_id)
        length.append(len(df))
        Variance.append(df.ExamScore.var())
        Std_Dev.append(df.ExamScore.std())
        Average_Score.append(df.ExamScore.mean())

        df = df[['1985', '1986', '1987',
                  'ESWI', 'African', 'Arab', 'Bangladeshi', 'Caribbean', 'Greek', 'Indian', 'Pakistani', 'SE_Asian',
                  'Turkish', 'Other',
                  'VR_Band', 'Gender',
                  'FSM', 'VR_BAND_Student', 'School_Gender', 'Maintained', 'Church', 'Roman_Cath',
                  'ExamScore']]
        DataSet = np.array(df, dtype=float)
        dist = []
        mdist = []
        for i in range(0, len(DataSet) - 1):
            dist.append(euclidean(DataSet[i], DataSet[i + 1]))
            mdist.append(cityblock(DataSet[i], DataSet[i + 1]))
        Average_Euclidian_Distance_within_Task.append(np.mean(dist))
        Average_Manhattan_Distance_within_Task.append(np.mean(mdist))

        '''Perform Scaling'''
        x = df.values  # returns a numpy array
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

    Task_Information = pd.DataFrame({'Task_Name': Task_Name,
                                     'Dataset_Size': length,
                                     'Variance': Variance,
                                     'Std_Dev': Std_Dev,
                                     'Average_Score': Average_Score,
                                     'Average_Euclidian_Distance_within_Task': Average_Euclidian_Distance_within_Task,
                                     'Average_Manhattan_Distance_within_Task': Average_Manhattan_Distance_within_Task,
                                     'Average_Euclidian_Distance_within_Task_after_Scaling': Average_Euclidian_Distance_within_Task_after_Scaling,
                                     'Average_Manhattan_Distance_within_Task_after_Scaling': Average_Manhattan_Distance_within_Task_after_Scaling})
    Task_Information.to_csv(f'{DataPath}Task_Information_School.csv', index=False)
    return max(length)

def prep_data_for_mtl():
    for sch_id in SCHOOLS:
        csv = (f"{DataPath}/{sch_id}_School_Data.csv")
        df = pd.read_csv(csv, low_memory=False)
        samples_to_be_repeated = maxlen-len(df)
        if samples_to_be_repeated>0:
            df2 = df.sample(n=samples_to_be_repeated,random_state=1234,replace=True)
            df = df.append(df2, ignore_index=True)
        else:
            df = df.sample(n=maxlen)
        new_csv = (f"{mtl_datapath}{sch_id}_School_Data_for_MTL.csv")
        df.to_csv(new_csv,index=False)

DataPath = f"../Dataset/SCHOOL/"
convert_to_csv()
SCHOOLS = [i for i in range(1,140)]
data_processing()
maxlen = get_task_information()
mtl_datapath = f'{DataPath}DATA/'
prep_data_for_mtl()
