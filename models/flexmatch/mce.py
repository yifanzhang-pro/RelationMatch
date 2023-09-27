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

class MatrixCrossEntropyVectorLogFYFYFYFX:
    def __init__(self, args):
        self.args = args
        self.m = self.args.batch_size
        self.n = self.args.num_classes

    def __str__(self):
        return "Loss: MatrixCrossEntropyVectorLogFYFYFYFX"

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
        # print("-----------------")
        # print(P, Q)
        return -(P.detach() @ torch.log(Q)).trace()

    def __call__(self, pseudo_label, fx_ulb_s):
        pseudo_label = torch.as_tensor(F.one_hot(pseudo_label, self.n), dtype=torch.float32).detach()
        fx_ulb_s = torch.softmax(fx_ulb_s, dim=-1)
        # print(pseudo_label)
        # print(fx_ulb_s)
        if (pseudo_label.shape[0] < 2):
            return 0.
        self.m = pseudo_label.shape[0]
        J_m = self.centering_matrix().detach()
        C_yy = (1. / self.m * pseudo_label.T @ J_m @ pseudo_label).detach()
        C_yx= 1. / self.m * pseudo_label.T @ J_m @ fx_ulb_s
        P = C_yy + (self.args.mce_lambd * torch.eye(self.n).cuda(self.args.gpu).detach())
        Q = C_yx + (self.args.mce_mu * torch.eye(self.n).cuda(self.args.gpu).detach())
        # print(P)
        # print(Q)
        return self.mce(P, Q)

class MatrixCrossEntropyVectorLogFYFYFXFX:
    def __init__(self, args):
        self.args = args
        self.m = self.args.batch_size
        self.n = self.args.num_classes

    def __str__(self):
        return "Loss: MatrixCrossEntropyVectorLogFYFYFXFX"

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
        # print("-----------------")
        # print(P, Q)
        return -(P.detach() @ torch.log(Q)).trace()

    def __call__(self, pseudo_label, fx_ulb_s):
        pseudo_label = torch.as_tensor(F.one_hot(pseudo_label, self.n), dtype=torch.float32).detach()
        fx_ulb_s = torch.softmax(fx_ulb_s, dim=-1)
        # print(pseudo_label)
        # print(fx_ulb_s)
        if (pseudo_label.shape[0] < 2):
            return 0.
        self.m = pseudo_label.shape[0]
        J_m = self.centering_matrix().detach()
        C_yy = (1. / self.m * pseudo_label.T @ J_m @ pseudo_label).detach()
        C_xx= 1. / self.m * fx_ulb_s.T @ J_m @ fx_ulb_s
        P = C_yy + (self.args.mce_lambd * torch.eye(self.n).cuda(self.args.gpu).detach())
        Q = C_xx + (self.args.mce_mu * torch.eye(self.n).cuda(self.args.gpu).detach())
        # print(P)
        # print(Q)
        return self.mce(P, Q)


class MatrixCrossEntropyMatrixLogFYFYFXFX:
    def __init__(self, args):
        self.args = args
        self.m = self.args.batch_size
        self.n = self.args.num_classes

    def __str__(self):
        return "Loss: MatrixCrossEntropyMatrixLogFYFYFYFX"

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
        Q = Q - torch.eye(self.n).cuda(self.args.gpu).detach()
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
        return -(P.detach() @ res).trace()

    def __call__(self, pseudo_label, fx_ulb_s):
        pseudo_label = torch.as_tensor(F.one_hot(pseudo_label, self.n), dtype=torch.float32).detach()
        fx_ulb_s = torch.softmax(fx_ulb_s, dim=-1)
        # print(pseudo_label)
        # print(fx_ulb_s)
        if (pseudo_label.shape[0] < 2):
            return 0.
        self.m = pseudo_label.shape[0]
        J_m = self.centering_matrix().detach()
        C_yy = (1. / self.m * pseudo_label.T @ J_m @ pseudo_label).detach()
        C_xx= 1. / self.m * fx_ulb_s.T @ J_m @ fx_ulb_s
        P = C_yy + (self.args.mce_lambd * torch.eye(self.n).cuda(self.args.gpu).detach())
        Q = C_xx + (self.args.mce_mu * torch.eye(self.n).cuda(self.args.gpu).detach())
        # print(P)
        # print(Q)
        return self.mce(P, Q)



class MatrixCrossEntropyVectorLogFYFYFYFX:
    def __init__(self, args):
        self.args = args
        self.m = self.args.batch_size
        self.n = self.args.num_classes

    def __str__(self):
        return "Loss: MatrixCrossEntropyVectorLogFYFYFYFX"

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
        # print("-----------------")
        # print(P, Q)
        return -(P.detach() @ torch.log(Q)).trace()

    def __call__(self, pseudo_label, fx_ulb_s):
        pseudo_label = torch.as_tensor(F.one_hot(pseudo_label, self.n), dtype=torch.float32).detach()
        fx_ulb_s = torch.softmax(fx_ulb_s, dim=-1)
        # print(pseudo_label)
        # print(fx_ulb_s)
        if (pseudo_label.shape[0] < 2):
            return 0.
        self.m = pseudo_label.shape[0]
        J_m = self.centering_matrix().detach()
        C_yy = (1. / self.m * pseudo_label.T @ J_m @ pseudo_label).detach()
        C_yx= 1. / self.m * pseudo_label.T @ J_m @ fx_ulb_s
        P = C_yy + (self.args.mce_lambd * torch.eye(self.n).cuda(self.args.gpu).detach())
        Q = C_yx + (self.args.mce_mu * torch.eye(self.n).cuda(self.args.gpu).detach())
        # print(P)
        # print(Q)
        return self.mce(P, Q)


class MatrixCrossEntropyNoCenterFYFYFXFX:
    def __init__(self, args):
        self.args = args
        self.m = self.args.batch_size
        self.n = self.args.num_classes

    def __str__(self):
        return "Loss: MatrixCrossEntropyNoCenterFYFYFXFX"

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
        Q = Q - torch.eye(self.n).cuda(self.args.gpu).detach()
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

    def __call__(self, pseudo_label, fx_ulb_s):
        pseudo_label = torch.as_tensor(F.one_hot(pseudo_label, self.n), dtype=torch.float32).detach()
        fx_ulb_s = torch.softmax(fx_ulb_s, dim=-1)
        # print(pseudo_label)
        # print(fx_ulb_s)
        if (pseudo_label.shape[0] < 2):
            return 0.
        self.m = pseudo_label.shape[0]
        C_yy = (1. / self.m * pseudo_label.T @ pseudo_label).detach()
        C_xx= 1. / self.m * fx_ulb_s.T @ fx_ulb_s
        P = C_yy + (self.args.mce_lambd * torch.eye(self.n).cuda(self.args.gpu).detach())
        Q = C_xx + (self.args.mce_mu * torch.eye(self.n).cuda(self.args.gpu).detach())
        # print(P)
        # print(Q)
        return self.mce(P, Q)

class MatrixCrossEntropyMMFYFYFXFX:
    def __init__(self, args):
        self.args = args
        self.m = self.args.batch_size
        self.n = self.args.num_classes

    def __str__(self):
        return "Loss: MatrixCrossEntropyMMFYFYFXFX"

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
        self.m = pseudo_label.shape[0]
        self.n = fx_ulb_s.shape[1]
        if not use_feature:
            pseudo_label = torch.as_tensor(F.one_hot(pseudo_label, self.n), dtype=torch.float32).detach()
            fx_ulb_s = torch.softmax(fx_ulb_s, dim=-1)
        # print(pseudo_label)
        # print(fx_ulb_s)
        if (pseudo_label.shape[0] < 2):
            return torch.zeros(1).cuda(self.args.gpu)
        C_yy = (1. / self.n * pseudo_label @ pseudo_label.T).detach()
        C_xx= 1. / self.n * fx_ulb_s @ fx_ulb_s.T
        P = C_yy + (self.args.mce_lambd * torch.eye(self.m).cuda(self.args.gpu).detach())
        Q = C_xx + (self.args.mce_mu * torch.eye(self.m).cuda(self.args.gpu).detach())
        # print(P)
        # print(Q)
        return self.mce(P, Q)


class MatrixCrossEntropyNoCenterFYFYFYFX:
    def __init__(self, args):
        self.args = args
        self.m = self.args.batch_size
        self.n = self.args.num_classes

    def __str__(self):
        return "Loss: MatrixCrossEntropyNoCenterFYFYFYFX"

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
        Q = Q - torch.eye(self.n).cuda(self.args.gpu).detach()
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

    def __call__(self, pseudo_label, fx_ulb_s):
        pseudo_label = torch.as_tensor(F.one_hot(pseudo_label, self.n), dtype=torch.float32).detach()
        fx_ulb_s = torch.softmax(fx_ulb_s, dim=-1)
        # print(pseudo_label)
        # print(fx_ulb_s)
        if (pseudo_label.shape[0] < 2):
            return 0.
        self.m = pseudo_label.shape[0]
        C_yy = (1. / self.m * pseudo_label.T @ pseudo_label).detach()
        C_yx= 1. / self.m * pseudo_label.T @ fx_ulb_s
        P = C_yy + (self.args.mce_lambd * torch.eye(self.n).cuda(self.args.gpu).detach())
        Q = C_yx + (self.args.mce_mu * torch.eye(self.n).cuda(self.args.gpu).detach())
        # print(P)
        # print(Q)
        return self.mce(P, Q)