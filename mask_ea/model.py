from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import math
import torch
import random
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F


class GraphConvolution(Module):
    """
     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj)) # change to leaky relu
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


class NCA_loss(nn.Module):
    def __init__(self, alpha, beta, ep):
        super(NCA_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ep = ep
        self.sim = cosine_sim

    def forward(self, emb, train_links, test_links, device=0):
        emb = F.normalize(emb)
        im = emb[train_links[:, 0]]
        s = emb[train_links[:, 1]]

        if len(test_links) != 0:
            test_links = test_links[random.sample([x for x in np.arange(0, len(test_links))], 4500)]

            im_neg_scores = self.sim(im, emb[test_links[:, 1]])
            s_neg_scores = self.sim(s, emb[test_links[:, 0]])
        bsize = im.size()[0]
        scores = self.sim(im, s)  # + 1
        tmp = torch.eye(bsize).cuda(device)
        s_diag = tmp * scores

        alpha = self.alpha
        alpha_2 = alpha  # / 3.0
        beta = self.beta
        ep = self.ep
        S_ = torch.exp(alpha * (scores - ep))
        S_ = S_ - S_ * tmp  # clear diagnal

        if len(test_links) != 0:
            S_1 = torch.exp(alpha * (im_neg_scores - ep))
            S_2 = torch.exp(alpha * (s_neg_scores - ep))

        loss_diag = - torch.log(1 + F.relu(s_diag.sum(0)))

        loss = torch.sum(
            torch.log(1 + S_.sum(0)) / alpha
            + torch.log(1 + S_.sum(1)) / alpha
            + loss_diag * beta \
            ) / bsize
        if len(test_links) != 0:
            loss_global_neg = (torch.sum(torch.log(1 + S_1.sum(0)) / alpha_2
                                         + torch.log(1 + S_2.sum(0)) / alpha_2)
                               + torch.sum(torch.log(1 + S_1.sum(1)) / alpha_2
                                           + torch.log(1 + S_2.sum(1)) / alpha_2)) / 4500
        if len(test_links) != 0:
            return loss + loss_global_neg
        return loss


class masked_nca_loss(nn.Module):
    def __init__(self, alpha, beta, ep):
        super(masked_nca_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ep = ep
        self.sim = cosine_sim

    def forward(self, emb, train_links, test_links, mask, device=0):
        emb = F.normalize(emb)
        mask = mask.bool().cpu().numpy()
        indices = np.logical_and(mask[train_links[:, 0]], mask[train_links[:, 1]])
        im = emb[train_links[indices, 0]]
        s = emb[train_links[indices, 1]]
        # im = emb[train_links[:, 0]]
        # s = emb[train_links[:, 1]]

        if len(test_links) != 0:
            test_links = test_links[random.sample([x for x in np.arange(0, len(test_links))], 4500)]

            im_neg_scores = self.sim(im, emb[test_links[:, 1]])
            s_neg_scores = self.sim(s, emb[test_links[:, 0]])

        bsize = im.size()[0]
        scores = self.sim(im, s)  # + 1
        tmp = torch.eye(bsize).cuda(device)
        s_diag = tmp * scores

        alpha = self.alpha
        alpha_2 = alpha  # / 3.0
        beta = self.beta
        ep = self.ep
        S_ = torch.exp(alpha * (scores - ep))
        S_ = S_ - S_ * tmp  # clear diagnal

        if len(test_links) != 0:
            S_1 = torch.exp(alpha * (im_neg_scores - ep))
            S_2 = torch.exp(alpha * (s_neg_scores - ep))

        loss_diag = - torch.log(1 + F.relu(s_diag.sum(0)))

        loss = torch.sum(
            torch.log(1 + S_.sum(0)) / alpha
            + torch.log(1 + S_.sum(1)) / alpha
            + loss_diag * beta \
            ) / bsize
        if len(test_links) != 0:
            loss_global_neg = (torch.sum(torch.log(1 + S_1.sum(0)) / alpha_2
                                         + torch.log(1 + S_2.sum(0)) / alpha_2)
                               + torch.sum(torch.log(1 + S_1.sum(1)) / alpha_2
                                           + torch.log(1 + S_2.sum(1)) / alpha_2)) / 4500
        if len(test_links) != 0:
            return loss + loss_global_neg
        return loss


class MASK_MMEA(nn.Module):
    def __init__(self, ent_num, gcn_units: list, drop_out,
                 img_feature_dim, img_emb_dim):
        super(MASK_MMEA, self).__init__()
        input_dim = gcn_units[0]
        self.entity_emb = nn.Embedding(ent_num, input_dim)
        nn.init.normal_(self.entity_emb.weight, std=1.0 / math.sqrt(ent_num))
        self.entity_emb.requires_grad = True
        self.gcn = GCN(gcn_units[0], gcn_units[1], gcn_units[2], dropout=drop_out)
        self.img_linear = nn.Linear(img_feature_dim, img_emb_dim)

    def forward(self, input_idx, adj, img_features, mask):
        struct_emb = self.gcn(self.entity_emb(input_idx), adj)
        img_emb = self.img_linear(img_features)
        mask_mat = mask.reshape(len(img_emb), 1)
        #print(len(mask_mat[torch.where(mask_mat==0)]))
        img_emb = img_emb * mask_mat
        return struct_emb, img_emb


class MMEA(nn.Module):
    def __init__(self, ent_num, gcn_units: list, drop_out,
                 img_feature_dim, img_emb_dim):
        super(MMEA, self).__init__()
        input_dim = gcn_units[0]
        self.entity_emb = nn.Embedding(ent_num, input_dim)
        nn.init.normal_(self.entity_emb.weight, std=1.0 / math.sqrt(ent_num))
        self.entity_emb.requires_grad = True
        self.gcn = GCN(gcn_units[0], gcn_units[1], gcn_units[2], dropout=drop_out)
        self.img_linear = nn.Linear(img_feature_dim, img_emb_dim)

    def forward(self, input_idx, adj, img_features):
        struct_emb = self.gcn(self.entity_emb(input_idx), adj)
        img_emb = self.img_linear(img_features)
        return struct_emb, img_emb

