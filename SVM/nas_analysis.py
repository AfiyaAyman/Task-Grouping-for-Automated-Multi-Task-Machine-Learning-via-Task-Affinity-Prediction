import ast
import pandas as pd
import matplotlib.pyplot as plt


import math
import random
neurons = 6
for i in range(10000):
    NEURON_CHANGE_PERC = random.randint(-30, 30)
    new = neurons + math.ceil(neurons * NEURON_CHANGE_PERC / 100)
    print(f'perc = {NEURON_CHANGE_PERC}, old = {neurons}, change = {neurons * NEURON_CHANGE_PERC / 100}, new = {new}')
    neurons = new
    if new < 0:
        print(f'break at {i}')
print(f'final = {neurons} after i {i}')
exit(0)
dataset = 'Chemical'
df = pd.read_csv(f'HP_Tuning_SVM_Chemical_200_group_1.csv')

print(df.columns)
# plt.plot(df['Prev_Loss'])
# plt.grid()
# plt.xlabel('Epochs')
# plt.ylabel('Loss for group')
# plt.title(f'SVM {dataset} - accepted mutations')
# plt.show()
# exit(0)


Tweaked_param = ['None']
Current_HP = list(df.Current_HP)
old_hp_dict = ast.literal_eval(Current_HP[0])

for idx in range(1,len(Current_HP)):
    hp_dict = ast.literal_eval(Current_HP[idx])
    for key in hp_dict.keys() and old_hp_dict.keys():
        if hp_dict[key] != old_hp_dict[key]:
            Tweaked_param.append(key)
            # print(hp_dict, old_hp_dict, key)

    if str(df.Switch[idx]) == '0' or str(df.Switch[idx]) == 'yes':
        old_hp_dict = ast.literal_eval(Current_HP[idx])

print(len(Tweaked_param))
df['Tweaked_param'] = Tweaked_param


def significant_hp(df):
    filtered_df = df[df['Switch'] == 'yes']

    # Group by 'Tweaked_param' column and calculate the count
    count_df = filtered_df.groupby('Tweaked_param').size().reset_index(name='Count')

    # Display the count for each tweaked_param
    print(count_df)
    # plt.bar(count_df['Tweaked_param'], count_df['Count'])
    # plt.title(f'Significant HP for {dataset} Dataset - one random group\nlambda1 = task specific regularization\nlambda2 = shared regularization')
    # plt.xticks(rotation=0)
    # plt.show()

def loss_diff_param(df):
    filtered_df = df[df['Switch'] == 'yes']
    print(f'accepted mutations = {len(filtered_df)}')
    # print(filtered_df)
    Iteration_NAS = list(df.Iteration)

    param_loss_diff = {}
    for i in range(len(filtered_df)):
        row = filtered_df.iloc[i]
        indicator = row['Iteration']
        idx = Iteration_NAS.index(indicator)
        param = df.Tweaked_param[idx]
        if param not in param_loss_diff.keys():
            param_loss_diff[param] = []
        param_loss_diff[param].append(df.Prev_Loss[idx - 1] - df.Current_Loss[idx])

    print(len(param_loss_diff.keys()))
    L = 0
    for key in param_loss_diff.keys():
        print(key, len(param_loss_diff[key]))
        L += len(param_loss_diff[key])
    print(L)
    # print(param_loss_diff['Post'])
    #     # print(f'key = {key}, {param_loss_diff[key]}')
    # plt.bar(param_loss_diff['Post'], range(len(param_loss_diff['Post'])), label='Post')
    # plt.grid()
    # # plt.xlabel('Epochs')
    # plt.ylabel('Loss difference')
    # plt.show()
    key = 'lambda1'
    plt.hist(param_loss_diff[f'{key}'], bins=len(param_loss_diff[f'{key}']), label=f'{key}')

    # Add labels and title
    plt.xlabel('Loss Difference')
    plt.ylabel('Frequency')
    plt.title(f'Histogram for loss difference for parameter {key}')

    # Show the plot
    plt.show()


significant_hp(df)
loss_diff_param(df)

