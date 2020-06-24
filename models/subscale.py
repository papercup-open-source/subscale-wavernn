import torch
import torch.nn as nn
import torch.nn.functional as F


class CondintionNetwork(nn.Module):
    def __init__(self, n_layers, field, condnet_channels, kernelsize, dropout):
        super(CondintionNetwork, self).__init__()
        self.conv_layers = nn.ModuleList([])
        self.LeakyReLU = nn.LeakyReLU()
        self.linproj = nn.Linear(field, condnet_channels)
        self.n_layers = n_layers
        self.dilations = [2 ** (n // 2) for n in range(self.n_layers)]
        self.dropout = dropout
        self.batch_norm_layers = nn.ModuleList([])
        for dilation in self.dilations:
            padding = dilation
            conv_layer = nn.Conv1d(condnet_channels, condnet_channels, kernelsize, stride=1, padding=padding,
                                   dilation=dilation, groups=1, bias=False, padding_mode='zeros')
            self.conv_layers.append(conv_layer)
            batch_norm = nn.BatchNorm1d(condnet_channels)
            # self.batch_norm_layers.append(batch_norm)

    def linear_layer(self, x):
        x = x.permute(0, 2, 1)
        x = self.linproj(x)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x):
        x = self.linear_layer(x)
        for conv_layer, batch_norm_layer in zip(self.conv_layers, self.batch_norm_layers):
            x = F.dropout(x, p=self.dropout)
            x = self.LeakyReLU(conv_layer(x)) + x
            x = batch_norm_layer(x)
        return x


class Subscaler():
    def __init__(self, hparams, debug=False):
        self.batch_factor = hparams.batch_factor
        self.horizon = hparams.horizon
        self.lookback = hparams.lookback

        self.context_width = self.lookback + self.horizon + 1
        self.context_len = self.context_width * self.batch_factor
        self.present = self.context_len - self.batch_factor * self.horizon - 1

    def one2two(self, x):
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1, self.batch_factor)

    def two2one(self, x):
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

    def map_pos(self, pos):
        t = pos // self.batch_factor  # with sr / batch_factor clock rate
        subt = pos % self.batch_factor  # which sub tensor the sample belongs to
        return t, subt

    def inv_map_pos(self, subt, t):
        pos = subt + t * self.batch_factor
        return pos

    def overlay(self, tgt, src, pos):
        """
            tgt := should be size of context
                 i.e. [B, context_width, batch_factor]
            src := tensor from which context is extracted
                 [B, T, batch_factor]
        """
        t, subt = self.map_pos(pos)
        assert(t <= src.shape[1])
        assert(tgt.shape[1] == self.context_width)
        assert(tgt.shape[2] == self.batch_factor)
        assert(tgt.shape[2] == self.batch_factor)

        init = max(0, t - self.lookback)
        end = max(1, t + self.horizon + 1)
        overrun = max(0, end - src.shape[1])
        window = end - init

        edgelim = self.context_width - overrun
        tgt[:, -window: edgelim, 1: subt + 1] = src[:, init: end, :subt].flip(2)
        tgt[:, -window: - self.horizon - 1, 0] = src[:, init: t, subt]
        return tgt

    def extract_context(self, x, pos):
        batch_size = x.shape[0]
        x_2d = self.one2two(x)
        zeros = torch.zeros(batch_size, self.context_len).cuda()
        zeros = self.one2two(zeros)
        context = self.overlay(zeros, x_2d, pos)
        context = self.two2one(context.flip(2))
        return context

    def extract_context_from_train_batch(self, x):
        contexts = torch.zeros_like(x).repeat(self.context_len, 1, 1).permute(1, 2, 0)
        for pos in range(x.shape[1]):
            context = self.extract_context(x, pos)
            contexts[:, pos] = context
        return contexts

    def pad(self, x):
        padding = self.batch_factor - (x.shape[1] % self.batch_factor)
        padding = 0 if padding == self.batch_factor else padding
        x = F.pad(x,  pad=(0, 0, 0, padding))
        return x

    def stack_substensors(self, x):
        # assuming axis 1 is time
        subtensors = [x[:, i::self.batch_factor] for i in range(self.batch_factor)]
        x = torch.cat(subtensors, 0)
        return x

    def flatten_subtensors(self, x):
        # assuming axis 1 is time
        orig_batch_size = int(x.shape[0] / self.batch_factor)
        subtensors = torch.split(x, orig_batch_size)
        subtensors = torch.cat(subtensors, 1)
        base = torch.zeros_like(subtensors)
        width = int(base.shape[1] / self.batch_factor)
        for i in range(self.batch_factor):
            base[:, i::self.batch_factor] = subtensors[:, i * width: (i + 1) * width]
        return base
