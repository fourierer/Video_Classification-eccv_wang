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

(1)根据detectron2/INSTALL.md安装[detectron2](https://github.com/facebookresearch/detectron2)

这里采用的detectron2是facebook写的一个用于提取bounding box的一个库，由于比较新，所以对服务器上的环境要求比较高，值得注意的有以下几点(不会写轮子只能跟着别人要求走！)：

1)Python >= 3.6，使用anaconda建立虚拟环境时可以解决；

2)Pytorch >= 1.3，搭建pytorch和相应版本的torchvision(通过pytorch官网的指令)。在安装pytorch时，要注意：

i).服务器的驱动版本(nvidia-smi查看)和cuda版本之间的兼容性(不兼容无法使用)；

![cuda-driver_version](/Users/momo/Documents/video/cuda-driver_version.png)

ii).cuda版本与pytorch版本兼容

pytorch官网上可以看到。

3)GCC >= 5.0

如果服务器自带的GCC版本过低就需要升级，但是一般使用服务器的学生没有root权限，下面介绍没有root权限情况下升级GCC版本。

在Linux下，如果有root权限的话，使用sudo apt install 就可以很方便的安装软件，而且同时也会帮你把一些依赖文件也给编译安装好。但是如果不是用的自己的机器，一般情况下是没有root 权限的。所以就需要自己动手下载tar文件，解压安装。在安装中遇到的最大的问题是依赖的问题。
i)首先下载gcc压缩包并解压：

在网址https://ftp.gnu.org/gnu/gcc找到需要下载的版本，这里选择gcc-5.5.0.tar.gz(不要下载太新的版本，可能会不支持，满足要求即可)，上传到服务器(我自己的服务器路径为/home/sunzheng/gcc-5.5.0.tar.gz);

解压：

```shell
tar zxvf gcc-5.5.0.tar.gz
```

解压之后出现文件夹gcc-5.5.0；

进入该文件夹（后续操作都在该解压缩文件夹中进行）；

```shell
cd gcc-5.5.0
```

ii)下载gcc，和gcc依赖的包到文件夹gcc-5.5.0中

```shell
./contrib/download_prerequisites
```

如果运行过程中出现错误，可以依次运行文件中每个命令，来安装或者解压gcc所依赖的包；

iii)编译gcc

在gcc解压缩根目录下(gcc-5.5.0下)新建一个文件夹，然后进入该文件夹配置编译安装：

```shell
mkdir gcc-build
cd gcc-build
../configure --disable-checking --enable-languages=c,c++ --disable-multilib --prefix=/path/to/install --enable-threads=posix
make -j64    # 多线程编译，否则很慢很慢很慢，能多开就多开几个线程
make install
```

`path/to/install`就是要安装GCC的目录，比如我的服务器上就是/home/sunzheng/GCC-5.5.0，一定要是有安装权限的目录，所以第二条指令就是../configure --disable-checking --enable-languages=c,c++ --disable-multilib --prefix=/home/sunzheng/GCC-5.5.0 --enable-threads=posix

iv)为当前用户配置系统环境变量

打开～/.bashrc文件：

```shell
vim ~/.bashrc
```

在末尾加入：

```shell
export PATH=/path/to/install/bin:/path/to/install/lib64:$PATH
export LD_LIBRARY_PATH=/path/to/install/lib/:$LD_LIBRARY_PATH
```

在我的服务器上就是：

```shell
export PATH=/home/sunzheng/GCC-5.5.0/bin:/home/sunzheng/GCC-5.5.0/lib64:$PATH
export LD_LIBRARY_PATH=/home/sunzheng/GCC-5.5.0/lib/:$LD_LIBRARY_PATH
```

一定要确保安装路径在`$LD_LIBRARY_PATH`和`$PATH`之前，这样安装的程序才能取代之前系统默认的程序。同样地，也可以安装别的软件到自己的目录下并采用以上方式指定默认程序。

更新bashrc文件：

```shell
source ~/.bashrc
```

或者重启shell.

v)输入gcc -v检查版本

至此gcc升级完成。

![gcc](/Users/momo/Documents/video/gcc.jpeg)



(2)安装必要的package

```shell
pip install -r requirements.txt
```



**4.数据预处理**

1生成文件索引

```shell
python parse_annotations.py
```

parse_annotations.py:

```python
import os
import glob
import tqdm
import torch
import numpy as np

def parse_annotations(root):

    def parse(directory):

        data = []
        for cls in tqdm.tqdm(os.listdir(directory)):  # 对train或者test文件夹中的类别做循环，cls是类别名称
            cls = os.path.join(directory, cls)  #cls是类别路径
            for frame_dir in os.listdir(cls):  # 对类别中视频文件夹做循环，frame_dir是视频文件夹

                frame_dir = os.path.join(cls, frame_dir)  # frames_dir是视频文件夹路径

                frames = glob.glob('%s/*.jpg'%(frame_dir))  # 返回视频文件夹中所有匹配的文件路径列表
                if len(frames)<32:
                    continue
                frames = sorted(frames)
                frames = [f.replace(root, '') for f in frames]
                data.append({'frames':frames})
        return data

    train_data = parse('%s/frames/train'%root)  # train_data是一个列表，每一个值都是一个字典，这个字典只有一个键，对应的值是训练集中某个视频文件夹中所有帧图片的路径，且该视频文件夹超过32帧
    val_data = parse('%s/frames/valid'%root)  # 同理，test_data存储测试集中大于32帧的视频文件夹中所有帧图片的路径

    annotations = {'train_data':train_data, 'val_data':val_data}
    torch.save(annotations, 'data/something_data.pth')

# if not os.path.exists('data/something_data.pth'):
parse_annotations('data/something')
print ('Annotations created!')
```

该代码作用是将抽好帧的文件夹video_classification/data/frame中的训练集和测试集中的所有帧生成索引，保存在文件video_classification/data/something_data.pth中，可以查看something_data.pth文件中的内容(图中为一部分)：

![something_data_pth](/Users/momo/Documents/video/something_data_pth.png)



2.提取bounding box

提取的box将保存在data/something/bbox中，可以将demo/extract_box.py中73～77行注释取消，只提取部分frame的box，以加快速度：

```shell
cd detectron2
python demo/extract_box.py
```

注意如果这一步如果报错：RuntimeError: CUDA error: no kernel image is available for execution on the device

说明cuda无法使用，用下面两段代码验证：

```python
import torch
print(torch.cuda.is_available())
```

如果输出True，说明CUDA安装正确并且能被Pytorch检测到，并没有说明是否正常使用，要想看Pytorch能不能调用cuda加速，还需要简单地测试：

```python
import torch
a = torch.tensor(5,3)
a = a.cuda()
print(a)
```

此时一般会报同样的错误，原因在于显卡算力和CUDA不匹配。要么更换显卡提高显卡算力，要么降低CUDA的版本。但是由于使用detectron2库对pytorch版本要求很高，所以CUDA版本没办法降低，这里建议使用高算力的显卡～～



3.提取i3d的inference result

提取的feature将保存在data/something/feats中，如果需要去掉non local block，可以将extract_feat.py第90行注释取消：

```shell
python extract_feat.py
```

当未下载i3d pertained weight时，应首先下载权重：

```shell
wget https://dl.fbaipublicfiles.com/video-nonlocal/i3d_baseline_32x2_IN_pretrain_400k.pkl -P pretrained/
wget https://dl.fbaipublicfiles.com/video-nonlocal/i3d_nonlocal_32x2_IN_pretrain_400k.pkl -P pretrained/

python -m utils.convert_weights pretrained/i3d_baseline_32x2_IN_pretrain_400k.pkl pretrained/i3d_r50_kinetics.pth
python -m utils.convert_weights pretrained/i3d_nonlocal_32x2_IN_pretrain_400k.pkl pretrained/i3d_r50_nl_kinetics.pth
```



**5.训练**

不同的GCN模型可从models/gcn_model.py 第100, 101, 102行进行调整：

```python
python main.py

# 从保存的第5 epoch权重开始训练
python main.py --load_state 5
```





非常抱歉，我复现的结果测试集上top-5精确度只有约17%，并没有文章中的45%，不知道哪里出了问题，还希望各位大佬可以多多指教！