import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from core.models import NormDistConv, NormDist, NormDistBase
from core.models import BoundTanh, BoundLinear, BoundFinalLinear, BoundConv2d, BoundSequential
from core.models import apply_if_not_none


def check_inf_and_eval(model):
    for m in model.modules():
        if isinstance(m, NormDistBase) and (not math.isinf(m.p) or m.training):
            return False
    return True


class SortMLPModel(nn.Module):
    def __init__(self, depth, width, input_dim, num_classes=10, dropout=0.7,
                 std=1.0, identity_val=None, scalar=False):
        super(SortMLPModel, self).__init__()
        dist_kwargs = {'std': std, 'identity_val': identity_val}
        pixels = input_dim[0] * input_dim[1] * input_dim[2]
        fc_dist = []
        fc_dist.append(NormDist(pixels, width, bias=False, mean_shift=True, **dist_kwargs))
        for i in range(depth - 2):
            fc_dist.append(NormDist(width, width, bias=False, mean_shift=True, dropout=dropout, **dist_kwargs))
        fc_dist.append(NormDist(width, num_classes, bias=True, mean_shift=False, dropout=dropout, **dist_kwargs))
        self.fc_dist = BoundSequential(*fc_dist)
        if isinstance(scalar, float):
            self.scalar = scalar
        else:
            self.scalar = nn.Parameter(torch.ones(1)) if scalar else 1

    def forward(self, x=None, targets=None, eps=0, up=None, down=None):
        if up is not None and down is not None and check_inf_and_eval(self):  # certification
            paras = (x, torch.maximum(x - eps, down), torch.minimum(x + eps, up))
        else:
            paras = (x, None, None)
        paras = apply_if_not_none(paras, lambda z: z.view(z.size(0), -1))
        paras = self.fc_dist(paras)
        x = paras[0]
        if targets is None:
            return -x * self.scalar
        else:
            lower = x - eps if paras[1] is None else paras[1]
            upper = x + eps if paras[2] is None else paras[2]
            x, lower, upper = -x, -upper, -lower
            margin = upper - torch.gather(lower, 1, targets.view(-1, 1))
            margin = margin.scatter(1, targets.view(-1, 1), 0)
            return x * self.scalar, margin / (2 * eps)


class SortHybridModel(nn.Module):
    def __init__(self, depth, width, input_dim, hidden=512, num_classes=10, dropout=0.7,
                 std=1.0, identity_val=None, scalar=True):
        super(SortHybridModel, self).__init__()
        dist_kwargs = {'std': std, 'identity_val': identity_val}
        pixels = input_dim[0] * input_dim[1] * input_dim[2]
        fc_dist = []
        fc_linear = []
        fc_dist.append(NormDist(pixels, width, bias=False, mean_shift=True, **dist_kwargs))
        for i in range(depth - 3):
            fc_dist.append(NormDist(width, width, bias=False, mean_shift=True, dropout=dropout, **dist_kwargs))
        fc_linear.append(BoundLinear(width, hidden, bias=True))
        fc_linear.append(BoundTanh())
        self.fc_dist = BoundSequential(*fc_dist)
        self.fc_linear = BoundSequential(*fc_linear)
        self.fc_final = BoundFinalLinear(hidden, num_classes, bias=True)
        if isinstance(scalar, float):
            self.scalar = scalar
        else:
            self.scalar = nn.Parameter(torch.ones(1)) if scalar else 1

    def forward(self, x=None, targets=None, eps=0, up=None, down=None):
        if up is not None and down is not None and check_inf_and_eval(self):  # certification
            paras = (x, torch.maximum(x - eps, down), torch.minimum(x + eps, up))
        else:
            paras = (x, None, None)
        paras = apply_if_not_none(paras, lambda z: z.view(z.size(0), -1))
        paras = self.fc_dist(paras)
        if targets is not None and (paras[1] is None or paras[2] is None):
            paras = paras[0], paras[0] - eps, paras[0] + eps
        paras = self.fc_linear(paras)
        paras = self.fc_final(*paras, targets=targets)
        if targets is not None:
            paras = (paras[0] * self.scalar, paras[1])
        else:
            paras = paras * self.scalar
        return paras


class SortHybridModel2(nn.Module):
    def __init__(self, depth, width, input_dim, hidden=512, num_classes=10, dropout=0.7, stride=2,
                 std=1.0, identity_val=None, scalar=True):
        super(SortHybridModel2, self).__init__()
        dist_kwargs = {'std': std, 'identity_val': identity_val}
        fc_dist = []
        fc_linear = []
        fc_dist.append(NormDistConv(input_dim[0], width, (input_dim[1] - stride, input_dim[2] - stride), stride=stride,
                                    bias=False, mean_shift=True, **dist_kwargs))
        for i in range(depth - 3):
            fc_dist.append(NormDistConv(width, width, 1, bias=False, mean_shift=True, dropout=dropout, **dist_kwargs))
        fc_linear.append(BoundLinear(width * 4, hidden, bias=True))
        fc_linear.append(BoundTanh())
        self.fc_dist = BoundSequential(*fc_dist)
        self.fc_linear = BoundSequential(*fc_linear)
        self.fc_final = BoundFinalLinear(hidden, num_classes, bias=True)
        if isinstance(scalar, float):
            self.scalar = scalar
        else:
            self.scalar = nn.Parameter(torch.ones(1)) if scalar else 1

    def forward(self, x=None, targets=None, eps=0, up=None, down=None):
        if up is not None and down is not None and check_inf_and_eval(self):  # certification
            paras = (x, torch.maximum(x - eps, down), torch.minimum(x + eps, up))
        else:
            paras = (x, None, None)
        paras = self.fc_dist(paras)
        paras = apply_if_not_none(paras, lambda z: z.view(z.size(0), -1))
        if targets is not None and (paras[1] is None or paras[2] is None):
            paras = paras[0], paras[0] - eps, paras[0] + eps
        paras = self.fc_linear(paras)
        paras = self.fc_final(*paras, targets=targets)
        if targets is not None:
            paras = (paras[0] * self.scalar, paras[1])
        else:
            paras = paras * self.scalar
        return paras


class SortHybridModelAvg(nn.Module):
    def __init__(self, depth, width, input_dim, hidden=512, num_classes=10, dropout=0.7, stride=2,
                 std=1.0, identity_val=None, scalar=True):
        super(SortHybridModelAvg, self).__init__()
        dist_kwargs = {'std': std, 'identity_val': identity_val}
        fc_dist = []
        fc_linear = []
        fc_dist.append(NormDistConv(input_dim[0], width, (input_dim[1] - stride, input_dim[2] - stride), stride=stride,
                                    bias=False, mean_shift=True, **dist_kwargs))
        for i in range(depth - 3):
            fc_dist.append(NormDistConv(width, width, 1, bias=False, mean_shift=True, dropout=dropout, **dist_kwargs))
        fc_linear.append(BoundLinear(width, hidden, bias=True))
        fc_linear.append(BoundTanh())
        self.fc_dist = BoundSequential(*fc_dist)
        self.fc_linear = BoundSequential(*fc_linear)
        self.fc_final = BoundFinalLinear(hidden, num_classes, bias=True)
        if isinstance(scalar, float):
            self.scalar = scalar
        else:
            self.scalar = nn.Parameter(torch.ones(1)) if scalar else 1

    def forward(self, x=None, targets=None, eps=0, up=None, down=None):
        if up is not None and down is not None and check_inf_and_eval(self):  # certification
            paras = (x, torch.maximum(x - eps, down), torch.minimum(x + eps, up))
        else:
            paras = (x, None, None)
        paras = self.fc_dist(paras)
        paras = apply_if_not_none(paras, lambda z: z.view(z.size(0), -1, 4).mean(dim=2))
        if targets is not None and (paras[1] is None or paras[2] is None):
            paras = paras[0], paras[0] - eps, paras[0] + eps
        paras = self.fc_linear(paras)
        paras = self.fc_final(*paras, targets=targets)
        if targets is not None:
            paras = (paras[0] * self.scalar, paras[1])
        else:
            paras = paras * self.scalar
        return paras
