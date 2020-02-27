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

    def parse(directory):

        data = []
        for cls in tqdm.tqdm(os.listdir(directory)):
            cls = os.path.join(directory, cls)
            for frame_dir in os.listdir(cls):

                frame_dir = os.path.join(cls, frame_dir)

                frames = glob.glob('%s/*.jpg'%(frame_dir))
                frames = sorted(frames)
                frames = [f.replace(root, '') for f in frames]
                if len(frames)==0:
                    continue
                data.append({'frames':frames})
        return data

    train_data = parse('%s/frames/train'%root)
    val_data = parse('%s/frames/valid'%root)

    annotations = {'train_data':train_data, 'val_data':val_data}
    torch.save(annotations, 'data/something_data.pth')

#-------------------------------------------------------------------------------------------------------------------#

class Something(torch.utils.data.Dataset):

    def __init__(self, root, split, clip_len):
        super(Something, self).__init__()

        # self.root = os.path.join(root, 'frames')
        self.root = root
        self.bbox = os.path.join(root, 'bbox')
        self.split = split
        self.clip_len = clip_len

        self.sample_frame = [i*8 for i in range(4)]

        if not os.path.exists('data/something_data.pth'):
            parse_annotations(self.root)
            print ('Annotations created!')
        annotations =  torch.load('data/something_data.pth')

        self.labels = os.listdir('%s/frames/train'%(self.root))
        self.train_data = annotations['train_data']
        self.val_data = annotations['val_data']
        self.data = self.train_data if self.split=='train' else self.val_data
        print ('%d train clips | %d val clips'%(len(self.train_data), len(self.val_data)))

        self.clip_transform = util.clip_transform(self.split, self.clip_len)
        self.loader = lambda fl: Image.open('%s/%s'%(self.root, fl)).convert('RGB')
        self.box_loader = lambda fl: torch.load('%s/%s'%(self.bbox, fl))

    def sample(self, imgs):
        
        offset = len(imgs)//2 - self.clip_len//2
        imgs = imgs[offset:offset+self.clip_len]
        assert len(imgs)==self.clip_len, 'frame selection error!'

        # imgs = [img.split('/', 1)[-1] for img in imgs]
        # print(imgs)
        boxes = [imgs[item] for item in self.sample_frame]

        boxes = [box.split('/', 2)[-1][:-3]+'pth' for box in boxes]

        # bbox = [torch.load(os.path.join(self.bbox, box[1:])) for box in boxes]

        label = imgs[0].split('/', -1)[-3]
        bbox = [self.box_loader(box) for box in boxes]
        imgs = [self.loader(img) for img in imgs]

        return imgs, bbox, label

    def __getitem__(self, index):

        entry = self.data[index]
        frames, bbox, label = self.sample(entry['frames'])
        frames = self.clip_transform(frames) # (T, 3, 224, 224)
        frames = frames.permute(1, 0, 2, 3) # (3, T, 224, 224)

        bbox = torch.stack(bbox, 0)

        instance = {'frames':frames, 'bbox':bbox, 'label': label}

        return instance

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    testset = Kinetics(root='data/kinetics', split='val', clip_len=32)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

    for idx, batch in enumerate(testloader):
        # batch = util.batch_cuda(batch)
        frames, labels, cls_dets = batch['frames'], batch['label'], batch['bbox']

        # print(frames.shape, cls_dets.shape)
        break
        