import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bs', dest='batch_size',
                        help='batch_size',
                        default=128, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether or not use gpu',
                        default=True, type=bool)
    parser.add_argument('-g', '--gpu', dest='gpu',
                        help='choose to use which gpu',
                        default='0', type=str,
                        choices=['0', '1'])
    parser.add_argument('-e', '--epochs', dest='epochs',
                        help='epoch',
                        default=100, type=int)
    parser.add_argument('--lr',
                        default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('-o', '--optimizer',
                        default='adam', type=str,
                        choices=['adam', 'sgd'],
                        help='sgd | adam')
    parser.add_argument('--dataset',
                        default='something', type=str,
                        help='choose which dataset')
    parser.add_argument('--load_state',
                        type=int, default=0,
                        help='load checkpoint')
    parser.add_argument('--seed',
                        type=int, default=0,
                        help='seed number')

    args = parser.parse_args()

    return args
