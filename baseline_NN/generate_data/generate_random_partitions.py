import pandas as pd
from combalg.comblg import set_partition

def compute_bell_number(n):
    s = [[0 for _ in range(n+1)] for _ in range(n+1)]
    for i in range(n+1):
        for j in range(n+1):
            if j > i:
                continue
            elif(i==j):
                s[i][j] = 1
            elif(i==0 or j==0):
                s[i][j]=0
            else:
                s[i][j] = j*s[i-1][j] + s[i-1][j-1]
    ans = 0
    for i in range(0,n+1):
        ans+=s[n][i]
    print(f'Possible Partitions = {ans}')

for dataset in ['Ridership']:
    if dataset == 'Chemical':
        ChemicalData = pd.read_csv(f'Dataset/CHEMICAL/ChemicalData_All.csv', low_memory=False)
        Tasks = list(ChemicalData['180'].unique())
    if dataset == 'School':
        Tasks = [i for i in range(1,140)]
    if dataset == 'Landmine':
        Tasks = [i for i in range(0, 29)]
    if dataset == 'Parkinsons':
        Tasks = [i for i in range(1, 43)]
    if dataset == 'Ridership':
        Tasks = [3,  4,  7, 22, 23, 52, 55, 50, 56, 19, 14, 8, 28, 5, 6, 29, 17, 18, 34, 41, 42,  9]
        print(len(Tasks))

    print(f'Tasks = {len(Tasks)}')
    compute_bell_number(len(Tasks))


    Task_Groups = []
    Number_of_Task_Groups = []
    i = 0
    while len(Task_Groups) <200:
        new_partition = set_partition.random(Tasks)
        number_of_task_groups = len(new_partition)
        new_group = {i: new_partition[i] for i in range(len(new_partition))}
        if new_group not in Task_Groups:
            Task_Groups.append(new_group)
            Number_of_Task_Groups.append(number_of_task_groups)
        else:
            print('Duplicate')

    print(len(Task_Groups),len(Number_of_Task_Groups))
    print(len(Task_Groups), min(Number_of_Task_Groups),max(Number_of_Task_Groups))
    Random_Task_Groups = pd.DataFrame({'Task_Groups': Task_Groups, 'Number_of_Task_Groups': Number_of_Task_Groups})
    Random_Task_Groups.to_csv(f'{dataset}_Random_Task_Groups.csv', index=False)
