import torch
from torch import nn
import torch.nn.functional as F
import math


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.fc.weight.size(1))
        self.fc.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = self.fc(input)
        output = torch.bmm(adj, support)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nfeat)
        self.gc2 = GraphConvolution(nfeat, nfeat)
        self.gc3 = GraphConvolution(nfeat, nfeat)

        # self.ln1 = nn.LayerNorm(nfeat)
        # self.ln2 = nn.LayerNorm(nfeat)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        x = self.gc1(x, adj)

        x = self.dropout(x)
        x = self.relu(self.gc2(x, adj))

        x = self.dropout(x)
        x = self.relu(self.gc3(x, adj))

        return x

    # layernorm + residual
    # def forward(self, x, adj):
    # residual = x
    # x = self.gc1(x, adj)
    # x = self.relu(self.ln1(x) + residual)
    #
    # residual = x
    # x = self.dropout(x)
    # x = self.gc2(x, adj)
    # x = self.relu(self.ln2(x) + residual)
    #
    # residual = x
    # x = self.dropout(x)
    # x = self.dropout(x)
    # x = self.gc3(x, adj) + residual
    #
    # return x


class GCN_model(nn.Module):
    def __init__(self, nfeat, nclass, dropout):
        super(GCN_model, self).__init__()

        self.G_sim = GCN(nfeat, dropout)
        self.G_front = GCN(nfeat, dropout)
        self.G_back = GCN(nfeat, dropout)

        self.trans_1 = nn.Linear(nfeat, nfeat)
        self.trans_2 = nn.Linear(nfeat, nfeat)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(nfeat*2, nclass)

    def forward(self, input):
        x, feat_box, g_front, g_back = input['base'], input['box'], input['Gfront'], input['Gback']

        x1 = self.trans_1(feat_box)
        x2 = self.trans_2(feat_box)

        g_sim = torch.bmm(x1, x2.permute(0, 2, 1))  # bs, N, N
        g_sim = F.softmax(g_sim, dim=2)

        g_sim = self.G_sim(feat_box, g_sim)  # bs, N, 2048
        g_front = self.G_front(feat_box, g_front)
        g_back = self.G_back(feat_box, g_back)

        x_g = g_sim + g_front + g_back
        x_g = x_g.mean(1)  # bs, 2048

        x = torch.cat([x, x_g], dim=1)  # bs, 4096

        x = self.dropout(x)
        pred = self.fc(x)

        return pred

