import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MatrixCrossEntropyMMFYFYFXFX:
    def __init__(self, args):
        self.args = args
        self.m = self.args.batch_size
        self.n = self.args.num_classes

    def __str__(self):
        return "Loss: MatrixCrossEntropy"

    def centering_matrix(self):
        J_m = torch.eye(self.m) - (torch.ones([self.m, 1]) @ torch.ones([1, self.m])) * (1.0 / self.m)
        return J_m.cuda(self.args.gpu).detach()

    def mce(self, P, Q):
        # print("--------")
        P = torch.exp(1. / self.args.mce_tau * P)
        # print(self.args.mce_tau, P)
        P = P / (torch.sum(P) + (0. * torch.eye(1).cuda(self.args.gpu).detach()))
        Q = torch.exp(1. / self.args.mce_tau * Q)
        Q = Q / (torch.sum(Q) + (0. * torch.eye(1).cuda(self.args.gpu).detach()))
        origin_Q = Q
        Q = Q - torch.eye(self.m).cuda(self.args.gpu).detach()
        cur = Q
        res = torch.zeros_like(Q).cuda(self.args.gpu).detach()
        for k in range(1, self.args.mce_order + 1):
            if k % 2 == 1:
                res = res + cur * (1. / float(k))
            else:
                res = res - cur * (1. / float(k))
            cur = cur @ Q
        # print("-----------------")
        # print(P, Q)
        return -(P.detach() @ res).trace() + origin_Q.trace()

    def __call__(self, pseudo_label, fx_ulb_s, use_feature=False):
        # m is batch size, i.e. b in Paper; n is num_classes, i.e. k in paper.
        self.m = pseudo_label.shape[0]
        self.n = fx_ulb_s.shape[1]
        if not use_feature:
            pseudo_label = torch.as_tensor(F.one_hot(pseudo_label, self.n), dtype=torch.float32).detach()
            fx_ulb_s = torch.softmax(fx_ulb_s, dim=-1)
        # print(pseudo_label)
        # print(fx_ulb_s)
        if (pseudo_label.shape[0] < 2):
            return torch.zeros(1).cuda(self.args.gpu)
        C_yy = (1. / self.m * pseudo_label @ pseudo_label.T).detach()
        C_xx= 1. / self.m * fx_ulb_s @ fx_ulb_s.T
        P = C_yy + (self.args.mce_lambd * torch.eye(self.m).cuda(self.args.gpu).detach())
        Q = C_xx + (self.args.mce_mu * torch.eye(self.m).cuda(self.args.gpu).detach())
        # print(P)
        # print(Q)
        return self.mce(P, Q)