import sys, os, time
import tqdm
import torch
import torch.nn as nn
sys.path.append(os.getcwd())

from utils import util
from data import something
from models import resnet
from models.iou_cal import build_graph

from detectron2.layers.roi_align import ROIAlign


def mkdir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)


def extract():

    with torch.no_grad():
        for idx, batch in enumerate(tqdm.tqdm(testloader)):

            if cuda:
                batch = util.batch_cuda(batch)
            frames, labels, bboxes = batch['frames'], batch['label'], batch['bbox']  # bboxes is bounding boxes

            if os.path.exists(os.path.join(save_path, labels[0], '{}.pth'.format(idx*batch_size+0))):
                continue

            bboxes = bboxes[:, :, :box_num, :]  # b_z, 4, box_num, 5

            b_s = frames.size(0)

            tmp = [[i]*box_num for i in range(b_s*clips_len//8)]
            tmp = torch.tensor(tmp).reshape(-1)

            # extract base features from i3d model
            feat_base = net(input_frames=frames, include_top=False)  # b_z, 2048, 4, 7, 7
            feat_base = feat_base.permute(0, 2, 1, 3, 4).reshape(-1, 2048, 7, 7)  # b_z*4, 2048, 7, 7
            # print(feat_base.shape)

            # get rid of the batch dim
            # feat_base, labels, bboxes = feat_base[0], labels[0], bboxes[0]
            bboxes = bboxes.reshape(-1, 5)  # b_z*4*box_num, 5
            bboxes[:, 0] = tmp


            # after the i3d model, there will left 4 frames
            # and the order of the bounding boxes has been arranged in dataloader
            feat_box = roi(feat_base, bboxes)  # b_z*n*4, 2048, 7, 7
            feat_box = avg_pool2d(feat_box).squeeze()  # b_z*n*4, 2048
            feat_box = feat_box.reshape(b_s, -1, 2048)  # b_z, N, 2048

            feat_base = feat_base.reshape(b_s, 2048, clips_len//8, 7, 7)
            feat_base = avg_pool3d(feat_base).squeeze() # b_z, 2048

            bboxes = bboxes.reshape(b_s, clips_len//8, box_num, 5)

            for i in range(b_s):
                feat_Gfront, feat_Gback = build_graph(bboxes[i]) # N, N

                save_feat = {'base': feat_base[i], 'box': feat_box[i], 'Gfront': feat_Gfront, 'Gback': feat_Gback}

                save_dir = os.path.join(save_path, labels[i])
                mkdir(save_dir)

                torch.save(save_feat, os.path.join(save_dir, '{}.pth'.format(idx*b_s+i)))


batch_size = 32
box_num = 10
clips_len = 32
dataset = 'something'
cuda = True

path = 'data/{}/feats'.format(dataset)

# make directory for extracted features
mkdir(path)

# set up some layers
roi = ROIAlign((7,7), 7.0/224.0, 0)
avg_pool2d = torch.nn.AdaptiveMaxPool2d((1, 1))
avg_pool3d = nn.AdaptiveAvgPool3d((1, 1, 1))

# set up base network
net = resnet.i3_res50_nl(num_classes=400, pretrained=True)
# net = resnet.i3_res50(num_classes=400, pretrained=True)
if cuda:
    net.cuda()
net = nn.DataParallel(net)
net.eval()

# set up dataloader
testset = something.Something(root='data/{}'.format(dataset), split='val', clip_len=clips_len)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

print('Start extracting validation data.')
save_path = os.path.join(path, 'valid')
mkdir(save_path)
extract()

testset = something.Something(root='data/{}'.format(dataset), split='train', clip_len=clips_len)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

print('Start extracting train data.')
save_path = os.path.join(path, 'train')
mkdir(save_path)
extract()
