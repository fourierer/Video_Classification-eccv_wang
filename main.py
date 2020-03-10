import os
import torch
import numpy as np
import random
import collections
import torchnet as tnt
import torch.nn.functional as F
from data.dataset import Dataset

from models.gcn_model import GCN_model
from utils import util
import opts


def save(epoch):
    print('Saving state, epoch:', epoch)
    state_dict = model.state_dict()
    optim_state = optimizer.state_dict()
    checkpoint = {'model': state_dict, 'optimizer': optim_state}
    if not os.path.isdir('data/%s/save_weight' % opt.dataset):
        os.mkdir('data/%s/save_weight' % opt.dataset)
    torch.save(checkpoint, 'data/%s/save_weight/ckpt_E_%d.pth' % (opt.dataset, epoch))


def train(epoch, loader, is_valid=False):
    state = 'valid' if is_valid else 'train'
    loss_meters = collections.defaultdict(lambda: tnt.meter.AverageValueMeter())

    for idx, input in enumerate(loader):
        X, labels = input
        if opt.cuda:
            X = util.batch_cuda(X)
            labels = labels.cuda()
        pred = model(X)

        loss = F.cross_entropy(pred, labels, reduction='none').mean()
        loss_meters['loss'].add(loss.item())

        if not is_valid:
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

        prec_scores = util.accuracy(pred, labels, topk=topk)
        for k, prec in zip(topk, prec_scores):
            loss_meters['P%s' % k].add(prec.item(), pred.shape[0])

        stats = ' | '.join(['P%s: %.3f' % (k, prec.item()/pred.shape[0]) for k, prec in zip(topk, prec_scores)])
        print('%s epoch %d/%d: step %d/%d. %s | loss: %.3f' % (state, epoch, opt.epochs, idx+1, len(loader), stats, loss.item()))

    stats = ' | '.join(['%s: %.3f' % (k, v.value()[0]) for k, v in loss_meters.items()])
    str_log = '%s epoch %d/%d. %s\n' % (state, epoch, opt.epochs, stats)
    print(str_log)
    with open(log_file, 'a') as f:
        f.write(str_log)


opt = opts.parse_opt()

# set up random seed
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
np.random.seed(opt.seed)  # Numpy module.
random.seed(opt.seed)  # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# set up gpu
if opt.cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

# set up dataloder
trainset = Dataset(root='data/{}/feats/train'.format(opt.dataset))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True)
validset = Dataset(root='data/{}/feats/valid'.format(opt.dataset))
validloader = torch.utils.data.DataLoader(validset, batch_size=opt.batch_size, shuffle=False)

assert len(trainset.labels) == len(validset.labels), 'nclass wrong'

# set up model
model = GCN_model(nfeat=2048, nclass=len(trainset.labels), dropout=0.1)
if opt.cuda:
    model.cuda()
print(model)

# set up optimizer
if opt.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, weight_decay=1e-6, momentum=0.9)
elif opt.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-6)

# load saved state
if opt.load_state != 0:
    checkpoint = torch.load('data/%s/save_weight/ckpt_E_%d.pth' % (opt.dataset, opt.load_state))

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
    print('checkpoint loaded')


topk = [1, 5]
log_file = 'log.txt'

print('\nStart training!\n')
# start training
for epoch in range(opt.load_state+1, opt.epochs+1):
    model.train()
    train(epoch, trainloader, is_valid=False)

    if epoch % 5 == 0:
        with torch.no_grad():
            model.eval()
            train(epoch, validloader, is_valid=True)
        save(epoch)
