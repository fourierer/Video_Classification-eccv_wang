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
