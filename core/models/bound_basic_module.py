import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_if_not_none(paras, func):
    return [None if x is None else func(x) for x in paras]


class BoundReLU(nn.ReLU):
    def forward(self, x=None, lower=None, upper=None):
        return apply_if_not_none((x, lower, upper), F.relu)


class BoundTanh(nn.Tanh):
    def forward(self, x=None, lower=None, upper=None):
        return apply_if_not_none((x, lower, upper), torch.tanh)


def linear(input, weight, bias, w_scale, b_scale):
    if bias is None:
        return torch.mm(input, weight.T) * w_scale
    return torch.addmm(bias, input, weight.T, alpha=w_scale, beta=b_scale)


class BoundLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, w_scale=1.0, w_lr=1.0, b_lr=1.0):
        super(BoundLinear, self).__init__(in_features, out_features, bias=bias)
        self.weight.data.normal_()
        self.weight.data *= w_scale
        if self.bias is not None:
            self.bias.data.zero_()
        self.w_scale = w_lr / math.sqrt(in_features)
        self.b_scale = b_lr

    def forward(self, x=None, lower=None, upper=None):
        if x is not None:
            x = linear(x, self.weight, self.bias, self.w_scale, self.b_scale)
        if lower is not None and upper is not None:
            x_mul_2 = lower + upper
            r_mul_2 = upper - lower
            z = linear(x_mul_2, self.weight, self.bias, self.w_scale / 2, self.b_scale)
            r_mul_2 = torch.mm(r_mul_2, self.weight.abs().T)
            lower = torch.add(z, r_mul_2, alpha=-self.w_scale / 2)
            upper = torch.add(z, r_mul_2, alpha=self.w_scale / 2)
        return x, lower, upper


class BoundConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 w_scale=1.0, w_lr=1.0, b_lr=1.0):
        super(BoundConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.weight.data.normal_()
        self.weight.data *= w_scale
        if self.bias is not None:
            self.bias.data.zero_()
        self.w_scale = w_lr / math.sqrt(in_channels * kernel_size * kernel_size)
        self.b_scale = b_lr

    def forward(self, x=None, lower=None, upper=None):
        if x is not None:
            x = F.conv2d(x, self.weight * self.w_scale, None if self.bias is None else self.bias * self.b_scale,
                         stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        if lower is not None and upper is not None:
            x_mul_2 = lower + upper
            r_mul_2 = upper - lower
            z = F.conv2d(x_mul_2, self.weight * (self.w_scale / 2),
                         None if self.bias is None else self.bias * self.b_scale,
                         stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
            r_mul_2 = F.conv2d(r_mul_2, self.weight.abs() * (self.w_scale / 2), None,
                               stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
            lower = z - r_mul_2
            upper = z + r_mul_2
        return x, lower, upper


class BoundFinalLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, w_scale=1.0, w_lr=1.0, b_lr=1.0):
        super(BoundFinalLinear, self).__init__(in_features, out_features, bias=bias)
        self.weight.data.normal_()
        self.weight.data *= w_scale
        if self.bias is not None:
            self.bias.data.zero_()
        self.w_scale = w_lr / math.sqrt(in_features)
        self.b_scale = b_lr

    def forward(self, x=None, lower=None, upper=None, targets=None):
        res = None
        if x is not None:
            x = linear(x, self.weight, self.bias, self.w_scale, self.b_scale)
            if lower is None or upper is None or targets is None:
                return x
        if lower is not None and upper is not None and targets is not None:
            x_mul_2 = lower + upper
            r_mul_2 = upper - lower
            z = linear(x_mul_2, self.weight, self.bias, self.w_scale / 2, self.b_scale)
            z = z - torch.gather(z, 1, targets.unsqueeze(1))
            w = self.weight - self.weight.index_select(0, targets).unsqueeze(1)  # B * CO * CI
            r_mul_2 = w.abs().bmm(r_mul_2.unsqueeze(-1)).squeeze(-1)
            res = torch.add(z, r_mul_2, alpha=self.w_scale / 2)
        return x, res


class MeanShiftDropout(nn.Module):
    def __init__(self, out_channels, momentum=0.1, affine=True):
        super(MeanShiftDropout, self).__init__()
        self.out_channels = out_channels
        self.momentum = momentum
        self.affine = affine
        self.register_buffer('running_mean', torch.zeros(out_channels))
        if affine:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        if not hasattr(MeanShiftDropout, 'tag'):
            MeanShiftDropout.tag = 0
        MeanShiftDropout.tag += 1
        self.tag = MeanShiftDropout.tag

    # x, lower and upper should be 3d tensors with shape (B, C, H*W)
    def forward(self, x=None, lower=None, upper=None, index=None):
        SUB_BATCH = 32
        if self.training:
            assert x is not None, 'Currently not supported'
            if index is not None:
                mod = x.size(0) % SUB_BATCH
                groups = index.size(-2)
                cat_x = torch.cat((x, x.new_zeros(SUB_BATCH - mod, x.size(1), x.size(2))), dim=0) if mod != 0 else x
                value = cat_x.view(index.size(0), SUB_BATCH, groups, -1, x.size(-1)).transpose(0, 2).sum(dim=[1, -1])
                index_tmp = index.transpose(0, 1).reshape(groups, -1)
                sum_value = x.new_zeros(self.out_channels).view(groups, -1)
                sum_value = sum_value.scatter_add(1, index_tmp, value.view(groups, -1))
                count = x.new_empty(index.transpose(0, 1).size(), dtype=torch.float).fill_(SUB_BATCH)
                if mod != 0:
                    count[:, -1] = mod
                sum_count = x.new_empty(self.out_channels).view(groups, -1).fill_(1e-10)
                sum_count = sum_count.scatter_add(1, index_tmp, count.view(groups, -1))
                mean = (sum_value / (sum_count * x.size(-1))).view(-1)
            else:
                mean = x.mean(dim=[0, 2])
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
        else:
            mean = self.running_mean
        if self.bias is not None:
            mean = mean - self.bias
        if index is not None:
            batch_index = torch.div(torch.arange(x.size(0), dtype=torch.long, device=x.device), SUB_BATCH,
                                    rounding_mode='trunc')
            index_unfold = index[batch_index]
            mean = mean.view(1, index.size(-2), -1).expand(x.size(0), -1, -1)
            mean = torch.gather(mean, dim=-1, index=index_unfold).view(x.size(0), -1)

        x, lower, upper = apply_if_not_none((x, lower, upper), lambda z: z - mean.unsqueeze(-1))
        return x, lower, upper

    def extra_repr(self):
        return '{out_channels}, affine={affine}'.format(**self.__dict__)


class BoundSequential(nn.Sequential):
    def forward(self, paras):
        for module in self:
            paras = module(*paras)
        return paras
