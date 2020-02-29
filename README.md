# Video_Classification-eccv_wang
Reproduce the results of Videos as Space-Time Region Graphs's experiments via pytorch.



##### 代码复现：(复现I3D和GCN组合，即文章中table 4中的最后一行，上面其余几行可以注释相应的代码复现)

1.数据集下载

下载something-something数据集压缩文件，网站：https://20bn.com/datasets/something-something/v1，需要注册才能下载。

something数据集是一个中等规模的数据集，是为了识别视频中的动作的，其甚至都不在意具体是什么实施了这个动作，又有谁承载了这个动作。其标注形式就是**Moving something down，Moving something up**这种。这也是数据集名字的由来。从我目前的阅读来看，这个数据集相对小众了一些，目前在V1上的结果少有top1超过50%，在V2上少有超过60%。V1/V2均有174个类别，分别有108499/220847个的视频。处理数据的时候一定要看准数据数量是否准确。发布于2017年。((https://zhuanlan.zhihu.com/p/69064522))

下载25个压缩文件(压缩数据集)以及4个csv文件(标签信息)，

(1)something-something-v1-label.csv中是174个类别的名称，如：

Holding something

Turning something upside down

......

(2)something-something-v1-train.csv和something-something-v1-validation.csv中存储训练集验证集的标签信息，如：

100218;Something falling like a feather or paper

24413;Unfolding something

.......

(3)something-something-v1-test.csv中只有视频类别的序号，没有标签

**数据集可以不用下载到video/classification/data/something/raw当中，只需将解压后的数据集跟该文件夹建立一个超链接即可，因为一般数据集会放在一个统一的文件夹里面(比如服务器中的data文件夹，代码放在home文件夹)，并且后续写其他算法用到这个数据集的时候，就可以省去复制的时间。**



2.数据集的解压与整理

(1)在数据集所在文件夹(something-something，自行设置)中解压数据

```shell
cat 20bn-something-something-v1-?? | tar zx
```

数据集解压之后something-something文件夹中出现20bn-something-something-v1的文件夹，里面共有108499个文件夹，共174个类别。每个类别文件夹中都是已经抽好帧的jpg图片，图片个数根据视频长短决定。

(2)创建超链接

```shell
ln -s .../20bn-something-something-v1 .../video_classification/data/something/raw
ln -s /data/sz/raw/something-somethng/20bn-something-something-v1 /home/sz/Video_Classification/eccv_wang/video_classification/data/something/raw
```

在something-something文件夹下输入指令ln -s .../20bn-something-something-v1 .../video_classification/data/something/raw(下面的实例是我自己的服务器上的路径)，注意在建立软链接时，源文件和目标文件一定要是完整的绝对路径。回车之后在raw文件夹下面生成一个快捷方式文件夹20bn-something-something-v1，这个快捷方式文件夹并不占内存，但是可以访问到原来的20bn-something-something-v1里的所有内容。

(3)整理数据集

video_classification/data/something/process.py

将/home/sz/Video_Classification/eccv_wang/video_classification/data/something/raw/20bn-something-something-v1中的数据集分为训练集和测试集两部分，此处的数据是从超链接的数据源中移动得到。

```python
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


# 将/home/sunzheng/Video_Classification/eccv_wang/video_classification/data/something/raw/20bn-something-something-v1中的数据集分为训练集和测试集两部分，此处的数据是从超链接的数据源中移动得到
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
```

移动之后，源文件就没有了训练和验证集，只有测试集。(可以通过解压再次得到所有数据集)



3.配置环境





4.特征提取