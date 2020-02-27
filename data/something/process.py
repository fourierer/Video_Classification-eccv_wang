import os
import tqdm
import pandas as pd
import subprocess

train = pd.read_csv('annotations/something-something-v1-train.csv', sep=';', header=None)
validation = pd.read_csv('annotations/something-something-v1-validation.csv', sep=';', header=None)
labels = pd.read_csv('annotations/something-something-v1-labels.csv', sep='\n', header=None)
#print(type(validation))  # <class 'pandas.core.frame.DataFrame'>
#print(labels)  # 174*1的DataFrame
#print(labels[0])  # DataFrame的第一列
#print(labels[0].to_dict().items())  # 将第一列的值和行号(类别)组成元组，元组再组成列表

labels = dict((v,k) for k,v in labels[0].to_dict().items())
#print(labels)  # 将第一列的值作为key，将行号(类别)作为value，组成字典

# 将train和validation中第二列的类别描述，换成数字形式(1-174)
train = train.replace({1: labels})
validation = validation.replace({1: labels})


# 将/home/sz/Video_Classification/eccv_wang/video_classification/data/something/raw/20bn-something-something-v1中的数据集分为训练集和测试集两部分，此处的数据是从超链接的数据源中移动得到
for i in tqdm.tqdm(range(len(train))):
    if not os.path.isdir('frames/train/{}'.format(train.loc[i][1])):
        os.makedirs('frames/train/{}'.format(train.loc[i][1]))
    cmd = 'mv raw/20bn-something-something-v1/{} frames/train/{}'.format(train.loc[i][0], train.loc[i][1])
    subprocess.call(cmd, shell=True)


for i in tqdm.tqdm(range(len(validation))):
    if not os.path.isdir('frames/valid/{}'.format(train.loc[i][1])):
        os.makedirs('frames/valid/{}'.format(train.loc[i][1]))
    cmd = 'mv raw/20bn-something-something-v1/{} frames/valid/{}'.format(validation.loc[i][0], validation.loc[i][1])
    subprocess.call(cmd, shell=True)
