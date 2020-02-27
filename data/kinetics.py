import os
import glob
import torch
import numpy as np
from PIL import Image
import tqdm

import sys
sys.path.append(os.getcwd())

from utils import util


def parse_annotations(root):

    def parse(annotation_csv):
        annotations = open(annotation_csv, 'r').read().strip().split('\n')[1:]
        annotations = [line.split(',') for line in annotations]
        clip_labels, yt_ids, start_times, end_times, _, _ = zip(*annotations)

        labels = map(lambda l: l.strip('"'), clip_labels)
        labels = np.unique(list(labels)).tolist()

        data = []
        for yt_id, start, end, label in tqdm.tqdm(zip(yt_ids, start_times, end_times, clip_labels), total=len(yt_ids)):
            label = label.strip('"')
            frames = glob.glob('%s/%s/%s_%06d_%06d/*.jpg'%(frame_dir, label, yt_id, int(start), int(end)))
            frames = sorted(frames)
            frames = [f.replace(root, '') for f in frames]
            if len(frames)==0:
                #print (yt_id, start, end, '-- Not present')
                continue
            data.append({'frames':frames, 'label':labels.index(label)})
        return data, labels

    frame_dir = '%s/frames/'%root
    train_data, labels = parse('%s/annotations/kinetics-400_train.csv'%root)
    val_data, _ = parse('%s/annotations/kinetics-400_val.csv'%root)

    annotations = {'train_data':train_data, 'val_data':val_data, 'labels':labels}
    torch.save(annotations, 'data/kinetics_data.pth')

#-------------------------------------------------------------------------------------------------------------------#

class Kinetics(torch.utils.data.Dataset):

    def __init__(self, root, split, clip_len):
        super(Kinetics, self).__init__()

        # self.root = os.path.join(root, 'frames')
        self.root = root
        self.bbox = os.path.join(root, 'bbox')
        self.split = split
        self.clip_len = clip_len

        self.sample_frame = [i*8 for i in range(4)]

        if not os.path.exists('data/kinetics_data.pth'):
            parse_annotations(self.root)
            print ('Annotations created!')
        annotations =  torch.load('data/kinetics_data.pth')

        self.labels = annotations['labels']
        self.train_data = annotations['train_data']
        self.val_data = annotations['val_data']
        self.data = self.train_data if self.split=='train' else self.val_data
        print ('%d train clips | %d val clips'%(len(self.train_data), len(self.val_data)))

        self.clip_transform = util.clip_transform(self.split, self.clip_len)
        self.loader = lambda fl: Image.open(os.path.join(self.root, fl)).convert('RGB')

    def sample(self, imgs):
        
        if len(imgs)>self.clip_len:

            # if self.split=='train': # random sample
            #     offset = np.random.randint(0, len(imgs)-self.clip_len)
            #     imgs = imgs[offset:offset+self.clip_len]
            # elif self.split=='val': # center crop
            #     offset = len(imgs)//2 - self.clip_len//2
            #     imgs = imgs[offset:offset+self.clip_len]
            #     assert len(imgs)==self.clip_len, 'frame selection error!'

            offset = len(imgs)//2 - self.clip_len//2
            imgs = imgs[offset:offset+self.clip_len]


        imgs = [img.split('/', 1)[-1] for img in imgs]
        boxes = [img.split('/', 1)[-1][:-4]+'.pth' for img in imgs]
        boxes = [boxes[item] for item in self.sample_frame]

        bbox = [torch.load(os.path.join(self.bbox, box[1:])) for box in boxes]

        imgs = [self.loader(img) for img in imgs]

        return imgs, bbox

    def __getitem__(self, index):

        entry = self.data[index]
        frames, bbox = self.sample(entry['frames'])
        frames = self.clip_transform(frames) # (T, 3, 224, 224)
        frames = frames.permute(1, 0, 2, 3) # (3, T, 224, 224)

        bbox = torch.stack(bbox, 0)

        instance = {'frames':frames, 'label':entry['label'], 'bbox':bbox}

        return instance

    def __len__(self):
        return len(self.data)

#-------------------------------------------------------------------------------------------------------------------#

# Returns 3 Random Crops per frame across 10 uniformly samples clips per video
# Used for evaluation only.
# class KineticsMultiCrop(Kinetics):

#     def __init__(self, root, split, clip_len):
#         super(KineticsMultiCrop, self).__init__(root, split, clip_len)
#         self.clip_transform = util.clip_transform('3crop', self.clip_len)

#     def sample(self, imgs, K=10):

#         # memoize loading images since clips overlap
#         cache = {}
#         def load(img):
#             if img not in cache:
#                 cache[img] = self.loader(img)
#             return cache[img]
        
#         centers = [int(idx) for idx in np.linspace(self.clip_len//2, len(imgs)-self.clip_len//2, K)]

#         clips = []
#         for c in centers:
#             clip = imgs[c-self.clip_len//2:c+self.clip_len//2]
#             clip = [load(img) for img in clip]
#             clips.append(clip)
#         return clips

#     def __getitem__(self, index):

#         entry = self.data[index]
#         clips = self.sample(entry['frames'])

#         frames = []
#         for clip in clips:
#             clip = [self.clip_transform(clip).permute(1, 0, 2, 3) for _ in range(3)] 
#             clip = torch.stack(clip, 0) # (3, 3, 224, 224)
#             frames.append(clip)
#         frames = torch.stack(frames, 0) # (10, 3, 3, 224, 224)

#         instance = {'frames':frames, 'label':entry['label']}

#         return instance


if __name__ == '__main__':
    testset = Kinetics(root='data/kinetics', split='val', clip_len=32)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

    for idx, batch in enumerate(testloader):
        # batch = util.batch_cuda(batch)
        frames, labels, cls_dets = batch['frames'], batch['label'], batch['bbox']

        # print(frames.shape, cls_dets.shape)
        break
        