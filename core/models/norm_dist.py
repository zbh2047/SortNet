import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable
from core.norm_dist import norm_dist, bound_inf_dist, norm_dist_dropout, sort_neuron, bound_sort_neuron
from .bound_basic_module import MeanShiftDropout


def apply_if_not_none(paras, func):
    return [None if x is None else func(x) for x in paras]


def sample_mask(batch, groups, channel, q, device):
    count = (torch.rand(channel, device=device) < q).sum().item()
    indices = torch.topk(torch.rand(batch, groups, channel, device=device), dim=-1, k=count, sorted=False)[1].int()
    return torch.sort(indices)[0]


class NormDistBase(nn.Module):
    dropout_list = dict()
    dropout_buffer = None
    tag = 0

    def __init__(self, in_features, out_features, p=float('inf'), groups=1, bias=True, std=1.0, mean_shift=True,
                 dropout=None):
        super(NormDistBase, self).__init__()
        assert (in_features % groups == 0)
        assert (out_features % groups == 0)
        self.weight = nn.Parameter(torch.randn(out_features, in_features // groups) * std)
        self.groups = groups
        self.p = p
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.mean_shift = MeanShiftDropout(out_channels=out_features, affine=False) if mean_shift else None

        NormDistBase.tag += 1
        self.tag = NormDistBase.tag
        self.dropout = dropout
        if dropout is not None:
            NormDistBase.dropout_list[self.tag] = (in_features // groups, out_features // groups, dropout)

    # x, lower and upper should be 3d tensors with shape (B, C, H*W)
    def forward(self, x=None, lower=None, upper=None):
        drop_list = NormDistBase.dropout_list
        has_dropout = len(drop_list) > 0
        running_mean_index = None
        bias = self.bias
        if has_dropout:
            assert x is not None
            requires_grad = torch.is_grad_enabled() and (x.requires_grad or self.weight.requires_grad)
            assert (requires_grad and self.training) or math.isinf(self.p)
            SUB_BATCH = 32
            sub_batch = (x.size(0) - 1) // 32 + 1
            if requires_grad and self.training: # training
                if self.tag in drop_list and self.dropout != 1.0:
                    dim_in, dim_out, prob = drop_list[self.tag]
                    if NormDistBase.dropout_buffer is not None and NormDistBase.dropout_buffer[0] == self.tag:
                        _, w_index_ci = NormDistBase.dropout_buffer
                    else:
                        w_index_ci = sample_mask(sub_batch, self.groups, dim_in, prob, device=x.device)
                        batch_index = torch.div(torch.arange(x.size(0), dtype=torch.long, device=x.device), SUB_BATCH,
                                                rounding_mode='trunc')
                        w_index_ci_unfold = w_index_ci[batch_index]
                        x = x.view(x.size(0), self.groups, -1, x.size(-1))
                        x = torch.gather(x, 2, index=w_index_ci_unfold.unsqueeze(-1).long().expand(-1, -1, -1, x.size(-1)))
                        x = x.view(x.size(0), -1, x.size(-1))
                else:
                    dim_out = self.weight.size(0) // self.groups
                    w_index_ci = None
                if self.tag + 1 in drop_list.keys() and drop_list[self.tag + 1][0] == dim_out:
                    prob_co = drop_list[self.tag + 1][2]
                    if prob_co != 1.0:
                        w_index_co = sample_mask(sub_batch, self.groups, dim_out, prob_co, device=x.device)
                        NormDistBase.dropout_buffer = self.tag + 1, w_index_co
                        if self.bias is not None:
                            batch_index = torch.div(torch.arange(x.size(0), dtype=torch.long, device=x.device),
                                                    SUB_BATCH, rounding_mode='trunc')
                            w_index_co_unfold = w_index_co[batch_index]
                            bias = self.bias.view(1, self.groups, -1).expand(x.size(0), -1, -1)
                            bias = torch.gather(bias, dim=-1, index=w_index_co_unfold.long()).view(x.size(0), -1)
                    else:
                        w_index_co = None
                else:
                    w_index_co = None
                x = norm_dist_dropout(x, self.weight, w_index_ci, w_index_co, self.p, self.groups, tag=self.tag)
                running_mean_index = w_index_co
            else:
                x = sort_neuron(x, self.weight, self.groups, q=self.dropout, tag=self.tag)
                if lower is not None and upper is not None:
                    lower, upper = bound_sort_neuron(lower, upper, self.weight, self.groups, q=self.dropout,
                                                     tag=self.tag)
        else:
            if x is not None:
                x = norm_dist(x, self.weight, self.p, self.groups, tag=self.tag)
            if lower is not None and upper is not None:
                assert math.isinf(self.p)
                lower, upper = bound_inf_dist(lower, upper, self.weight, self.groups, tag=self.tag)
        if self.mean_shift is not None:
            if running_mean_index is not None:
                running_mean_index = running_mean_index.long()
            x, lower, upper = self.mean_shift(x, lower, upper, index=running_mean_index)
        if bias is not None:
            x, lower, upper = apply_if_not_none((x, lower, upper), lambda z: z + bias.unsqueeze(-1))
        return x, lower, upper


class NormDist(NormDistBase):
    def __init__(self, in_features, out_features, groups=1, bias=True, identity_val=None, **kwargs):
        super(NormDist, self).__init__(in_features, out_features, groups=groups, bias=bias, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        if identity_val is not None and in_features <= out_features:
            for i in range(out_features):
                weight = self.weight.data
                weight[i, i % (in_features // groups)] = -identity_val

    def forward(self, x=None, lower=None, upper=None):
        x, lower, upper = apply_if_not_none((x, lower, upper), lambda z: z.unsqueeze(-1))
        x, lower, upper = super(NormDist, self).forward(x, lower, upper)
        x, lower, upper = apply_if_not_none((x, lower, upper), lambda z: z.squeeze(-1))
        return x, lower, upper

    def extra_repr(self):
        s = 'in_features={}, out_features={}, bias={}'
        if self.groups != 1:
            s += ', groups={groups}'
        return s.format(self.in_features, self.out_features, self.bias is not None, groups=self.groups)


class NormDistConv(NormDistBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, identity_val=None, **kwargs):
        if isinstance(kernel_size, Iterable):
            kernel_size = tuple(kernel_size)
        else:
            kernel_size = (kernel_size, kernel_size)
        in_features = in_channels * kernel_size[0] * kernel_size[1]
        assert (in_channels % groups == 0)
        super(NormDistConv, self).__init__(in_features, out_channels, groups=groups, bias=bias, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        if identity_val is not None and in_channels <= out_channels:
            for i in range(out_channels):
                weight = self.weight.data.view(out_channels, -1, *kernel_size)
                weight[i, i % (in_channels // groups), kernel_size[0] // 2, kernel_size[1] // 2] = -identity_val

    def forward(self, x=None, lower=None, upper=None):
        unfold_paras = self.kernel_size, self.dilation, self.padding, self.stride
        h, w = 0, 0
        if x is not None:
            h, w = x.size(2), x.size(3)
            x = F.unfold(x, *unfold_paras)
        if lower is not None and upper is not None:
            h, w = lower.size(2), lower.size(3)
            lower = F.unfold(lower, *unfold_paras)
            upper = F.unfold(upper, *unfold_paras)
        x, lower, upper = super(NormDistConv, self).forward(x, lower, upper)
        h, w = [(s + 2 * self.padding - k) // self.stride + 1 for s, k in zip((h, w), self.kernel_size)]
        x, lower, upper = apply_if_not_none((x, lower, upper), lambda z: z.view(z.size(0), -1, h, w))
        return x, lower, upper

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != 0:
            s += ', padding={padding}'
        if self.dilation != 1:
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

