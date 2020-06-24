import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from utils.utils import decode_mu_law
from models.layers import (
    UpsampleNetwork
)
from models.subscale import (
    CondintionNetwork,
    Subscaler
)

from hparams import create_hparams

HPARAMS = create_hparams()


class WaveRNN(nn.Module):
    def __init__(self, hparams, debug=False):
        super().__init__()
        self.n_classes = 2 ** hparams.bits
        self.rnn_dims = hparams.rnn_dims
        self.fc_dims = hparams.fc_dims
        self.pad = hparams.pad
        self.upsample_factors = hparams.upsample_factors
        self.feat_dims = hparams.feat_dims
        self.compute_dims = hparams.compute_dims
        self.res_out_dims = hparams.res_out_dims
        self.res_blocks = hparams.res_blocks
        self.hop_length = hparams.hop_length
        self.debug = debug

        self.aux_dims = self.res_out_dims // 4
        self.lut_x = nn.Embedding(
            self.n_classes,
            self.fc_dims,
            max_norm=1.0
        )

        self.subscale = Subscaler(hparams)

        self.conditioning_network = CondintionNetwork(hparams.condnet_n_layers, self.subscale.context_len,
                                                      hparams.condnet_channels, hparams.condnet_kernelsize,
                                                      hparams.condnet_drouput)

        self.upsample = UpsampleNetwork(self.feat_dims, self.upsample_factors, self.compute_dims,
                                        self.res_blocks, self.res_out_dims, self.pad)
        self.fc0 = nn.Linear(self.feat_dims + self.aux_dims + hparams.condnet_channels, self.rnn_dims)
        self.rnn1 = nn.GRU(self.rnn_dims, self.rnn_dims, batch_first=True)
        self.rnn2 = nn.GRU(self.rnn_dims + self.aux_dims, self.rnn_dims, batch_first=True)
        self.fc1 = nn.Linear(self.rnn_dims + self.aux_dims, self.fc_dims)
        self.fc2 = nn.Linear(self.fc_dims + self.aux_dims, self.fc_dims)
        self.fc3 = nn.Linear(self.fc_dims, self.n_classes)

    def int2float(self, x):
        x = 2 * x.float() / (self.n_classes - 1.) - 1.
        return x

    def forward(self, x, mel):
        """Method used during training. Given the chunk of a melspectrogram, and the corresponding
        chunk of a waveform, this decodes autoregressively, i.e. predicts softmax probabilities for
        the output waveform. Note: T_mel * hop_length = T_wav.
            args:
                x ([B, T_wav] torch.LongTensor): chunk of a quantized waveform (corresponds to mel)
                mel ([B, num_bins, T_mel] torch.FloatTensor): chunk of a mel (corresponds to x)
            returns:
               soft ([B, T_wav, 2 ** bits] torch.FloatTensor): predicted softmax probabilities.
        """

        mel, aux = self.upsample(mel)
        mel, aux = self.subscale.pad(mel), self.subscale.pad(aux)
        aux_idx = [self.aux_dims * i for i in range(5)]
        a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
        a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
        a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
        a4 = aux[:, :, aux_idx[3]:aux_idx[4]]

        x = self.int2float(x)
        context = self.subscale.extract_context_from_train_batch(x)
        # B, T, context_len -> B, context_len, T
        context = context.permute(0, 2, 1)
        conditioned = self.conditioning_network(context)
        # B, n_channels, T -> B, T, n_channels
        conditioned = conditioned.permute(0, 2, 1)

        x = torch.cat([conditioned, mel, a1], dim=2)

        x = self.subscale.stack_substensors(x)
        a2 = self.subscale.stack_substensors(a2)
        a3 = self.subscale.stack_substensors(a3)
        a4 = self.subscale.stack_substensors(a4)

        x = self.fc0(x)
        res = x
        self.rnn1.flatten_parameters()
        x, _ = self.rnn1(x)

        x = x + res
        res = x
        x = torch.cat([x, a2], dim=2)
        self.rnn2.flatten_parameters()
        x, _ = self.rnn2(x)

        x = x + res
        x = torch.cat([x, a3], dim=2)
        x = F.relu(self.fc1(x))

        x = torch.cat([x, a4], dim=2)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.subscale.flatten_subtensors(x)

        soft = F.log_softmax(x, dim=-1)
        if self.debug:
            return soft, context, x
        return soft

    def train_mode_generate(self, x, mel):
        if self.debug:
            logprobs, context, x = self.forward(x, mel)
        else:
            logprobs = self.forward(x, mel)
        probs = torch.exp(logprobs)
        output = self.sampler(probs)
        output = self.transform(output)
        if self.debug:
            return output, context, x
        return output

    def make_context_batch(self, x, pos_dict):
        assert(list(pos_dict.keys()) == list(range(min(pos_dict), max(pos_dict) + 1)))
        context_batch = []
        for subt in range(min(pos_dict), max(pos_dict) + 1):
            pos = pos_dict[subt]
            context = self.subscale.extract_context(x, pos)
            context_batch.append(context)
        context_batch = torch.stack(context_batch)
        context_batch = context_batch.permute(0, 2, 1)
        return context_batch

    # pylint: disable=R0913
    def decode(self, context, m_t, a1_t, a2_t, a3_t, a4_t, h1, h2, rnn_cell1, rnn_cell2):
        """Helper function for inference mode when the ground truth waveform is not known.
        """
        conditioned = self.conditioning_network(context)
        # B, n_channels, 1 -> B, n_channels
        conditioned = conditioned.squeeze(2)
        x = torch.cat([conditioned, m_t, a1_t], dim=1)
        x = self.fc0(x)
        h1 = rnn_cell1(x, h1)

        x = x + h1
        inp = torch.cat([x, a2_t], dim=1)
        h2 = rnn_cell2(inp, h2)

        x = x + h2
        x = torch.cat([x, a3_t], dim=1)
        x = F.relu(self.fc1(x))

        x = torch.cat([x, a4_t], dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        posterior = F.softmax(1.0 * x, dim=1)

        if self.debug:
            return (posterior, h1, h2), context, x
        return posterior, h1, h2

    # pylint: disable=R0913
    def transform(self, output):
        output = decode_mu_law(output.float().cpu(), self.n_classes)
        output = output.cpu().numpy()
        return output

    def sampler(self, posterior):
        distrib = torch.distributions.Categorical(posterior)
        x = distrib.sample()
        x = self.int2float(x)
        return x

    def inference(self, mel, use_tqdm=True, gt=None):
        """Given a melspectrogram of arbitrary length, this function generates the corresponding
        predicted waveform. Note that T_wav = T_mel * hop_length
            args:
                mel ([B, num_bins, T_mel] torch.FloatTensor): melspectrogram.
                use_tqdm (bool): flag to use tqdm or not. Useful to reduce logging.
            returns:
                outputs ([B, T_wav] torch.FloatTensor): predicted waveform
        """

        rnn_cell1 = self.get_gru_cell(self.rnn1)
        rnn_cell2 = self.get_gru_cell(self.rnn2)

        with torch.no_grad():

            aux_idx = [self.aux_dims * i for i in range(5)]

            upsampled_mel, aux = self.upsample(mel[:, :, :])
            a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
            a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
            a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
            a4 = aux[:, :, aux_idx[3]:aux_idx[4]]

            output = torch.zeros(upsampled_mel.shape[0], upsampled_mel.shape[1])
            if gt is not None:
                gt = gt[:, :output.shape[1]]
                gt = self.int2float(gt)
            tester_src = torch.zeros_like(output).long()
            tester_tgt = torch.arange(upsampled_mel.shape[1]).repeat(upsampled_mel.shape[0], 1)

            pos_cont = {}
            xs = [0] * output.shape[1]
            for subt in tqdm(range(self.subscale.batch_factor)):
                h1 = mel.new(mel.shape[0], self.rnn_dims).zero_()
                h2 = mel.new(mel.shape[0], self.rnn_dims).zero_()
                for j in tqdm(range(upsampled_mel.shape[1] // self.subscale.batch_factor)):
                    pos = self.subscale.inv_map_pos(subt, j)
                    m_t = upsampled_mel[:, pos, :]
                    a1_t = a1[:, pos, :]
                    a2_t = a2[:, pos, :]
                    a3_t = a3[:, pos, :]
                    a4_t = a4[:, pos, :]

                    context = gt if gt is not None else output
                    context = self.subscale.extract_context(context, pos)
                    context = context.unsqueeze(1)
                    # B, 1, context_len -> B, context_len, 1
                    context = context.permute(0, 2, 1)
                    if self.debug:
                        (posterior, h1, h2), context, x = self.decode(context, m_t, a1_t, a2_t, a3_t, a4_t, h1, h2,
                                                                      rnn_cell1, rnn_cell2)
                        pos_cont[pos] = context
                        xs[pos] = x
                    else:
                        posterior, h1, h2 = self.decode(context, m_t, a1_t, a2_t, a3_t, a4_t, h1, h2, rnn_cell1,
                                                        rnn_cell2)
                    sample = self.sampler(posterior)
                    output[:, pos] = sample
                    tester_src[:, pos] = pos

        assert(torch.eq(tester_src, tester_tgt).all())
        output = self.transform(output)
        if self.debug:
            return output, pos_cont, torch.stack(xs, 1)
        return output

    def subscale_inference(self, mel, use_tqdm=True, gt=None):
        """Given a melspectrogram of arbitrary length, this function generates the corresponding
        predicted waveform. Note that T_wav = T_mel * hop_length
            args:
                mel ([B, num_bins, T_mel] torch.FloatTensor): melspectrogram.
                use_tqdm (bool): flag to use tqdm or not. Useful to reduce logging.
            returns:
                outputs ([B, T_wav] torch.FloatTensor): predicted waveform
        """

        rnn_cell1 = self.get_gru_cell(self.rnn1)
        rnn_cell2 = self.get_gru_cell(self.rnn2)

        with torch.no_grad():
            aux_idx = [self.aux_dims * i for i in range(5)]

            upsampled_mel, aux = self.upsample(mel[:, :, :])
            a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
            a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
            a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
            a4 = aux[:, :, aux_idx[3]:aux_idx[4]]

            output = torch.zeros(upsampled_mel.shape[0], upsampled_mel.shape[1])
            if gt is not None:
                gt = gt[:, :output.shape[1]]
                gt = self.int2float(gt)
            tester_src = torch.zeros_like(output).long()
            tester_tgt = torch.arange(upsampled_mel.shape[1]).repeat(upsampled_mel.shape[0], 1)

            pos_cont = {}
            xs = [0] * output.shape[1]

            hidden_states = {
                subt: [mel.new(mel.shape[0], self.rnn_dims).zero_(), mel.new(mel.shape[0], self.rnn_dims).zero_()]
                for subt in range(self.subscale.batch_factor)
                }
            n_steps_subt_0 = upsampled_mel.shape[1] // self.subscale.batch_factor
            n_steps_remaining = (self.subscale.horizon + 1) * (self.subscale.batch_factor - 1)
            n_steps = n_steps_subt_0 + n_steps_remaining
            for j in tqdm(range(n_steps)):
                m_t, a1_t, a2_t, a3_t, a4_t = [], [], [], [], []
                batch_size_upper_lim = min(j // (self.subscale.horizon + 1) + 1, self.subscale.batch_factor)
                batch_size_lower_lim = max(((j - n_steps_subt_0) // (self.subscale.horizon + 1) + 1), 0)
                # batch_size = batch_size_upper_lim - batch_size_lower_lim
                pos_dict = {}
                for subt in range(batch_size_lower_lim, batch_size_upper_lim):
                    shifted_j = j - (self.subscale.horizon + 1) * subt
                    pos = self.subscale.inv_map_pos(subt, shifted_j)
                    pos_dict[subt] = pos
                    m_t.append(upsampled_mel[:, pos, :])
                    a1_t.append(a1[:, pos, :])
                    a2_t.append(a2[:, pos, :])
                    a3_t.append(a3[:, pos, :])
                    a4_t.append(a4[:, pos, :])

                m_t, a1_t, a2_t, a3_t, a4_t = tuple(map(lambda x: torch.stack(x).squeeze(1), [m_t, a1_t, a2_t,
                                                    a3_t, a4_t]))

                h1 = torch.stack([hidden_states[subt][0]
                                 for subt in range(batch_size_lower_lim, batch_size_upper_lim)]).squeeze(1)
                h2 = torch.stack([hidden_states[subt][1]
                                 for subt in range(batch_size_lower_lim, batch_size_upper_lim)]).squeeze(1)
                context_source = gt if gt is not None else output
                context = self.make_context_batch(context_source, pos_dict)
                if self.debug:
                    (posterior, h1, h2), context, x = self.decode(context, m_t, a1_t, a2_t, a3_t, a4_t, h1, h2,
                                                                  rnn_cell1, rnn_cell2)
                    for i, (subt, pos) in enumerate(pos_dict.items()):
                        pos_cont[pos] = context[i].unsqueeze(0)
                        xs[pos] = x[i].unsqueeze(0)
                else:
                    posterior, h1, h2 = self.decode(context, m_t, a1_t, a2_t, a3_t, a4_t, h1, h2, rnn_cell1, rnn_cell2)

                for i, subt in enumerate(range(batch_size_lower_lim, batch_size_upper_lim)):
                    hidden_states[subt][0] = h1[i].unsqueeze(0)
                    hidden_states[subt][1] = h2[i].unsqueeze(0)
                sample = self.sampler(posterior)
                for i, (subt, pos) in enumerate(pos_dict.items()):
                    output[:, pos] = sample[i]
                    tester_src[:, pos] = pos

        assert(torch.eq(tester_src, tester_tgt).all())
        output = self.transform(output)
        if self.debug:
            return output, pos_cont, torch.stack(xs, 1)
        return output

    @staticmethod
    def get_gru_cell(gru):
        """Given a GRU object, this function returns the corresponding GRU cell with the
        correct weights initialised from the GRU object.
            args:
                gru (nn.GRU): GRU from which to get the cell.
            returns:
                gru_cell (nn.GRUCell): GRU cell where the cell has been initialised from gru.
        """
        gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh.data = gru.weight_hh_l0.data
        gru_cell.weight_ih.data = gru.weight_ih_l0.data
        gru_cell.bias_hh.data = gru.bias_hh_l0.data
        gru_cell.bias_ih.data = gru.bias_ih_l0.data
        return gru_cell

    def load_max_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                logging.warning("This weight is not in model: {}".format(name))
                continue
            try:
                own_state[name].copy_(param)
            except RuntimeError:
                logging.warning(name)
                logging.warning("Model size: {}".format(own_state[name].size()))
                logging.warning("Checkpoint size: {}".format(param.size()))
