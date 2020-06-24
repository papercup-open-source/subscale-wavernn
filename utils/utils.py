import os
import logging
import numpy as np
import torch

from librosa.output import write_wav

from hparams import create_hparams

HPARAMS = create_hparams()


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load a checkpoint into a WaveRNN instance based on a checkpoint
    path. Also loads the corresponding optimizer parameters. Also loads
    the optimizer"""
    assert os.path.isfile(checkpoint_path), 'Path does not exist...'
    logging.info("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_max_state_dict(checkpoint_dict['state_dict'])
    iteration = checkpoint_dict.get('iteration')
    dataset = checkpoint_dict.get('dataset')
    logging.info("Loaded checkpoint '{}' from iteration {} on dataset {}" .format(
        checkpoint_path, iteration, dataset))
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict.get('optimizer'))
        return model, iteration, optimizer
    return model, iteration


def encode_mu_law(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def decode_mu_law(y, mu):
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x


def is_batch_all_silence(m):

    """Takes as input batch of mels m (batch, num_mel_bins, num_frames)
    and returns True if all of the batch elements are entirely silence"""

    return torch.allclose(m, torch.zeros(1))


class ExponentialMovingAverage():
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def update(self, name, x):
        assert name in self.shadow
        update_delta = self.shadow[name] - x
        self.shadow[name] -= (1.0 - self.decay) * update_delta


def init_ema(model, ema_rate):
    ema = ExponentialMovingAverage(ema_rate)
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)
    return ema


def save_wavs(paths, wavs, sample_rate):
    for path, wav in zip(paths, wavs):
        write_wav(
            path,
            np.asfortranarray(wav),
            sr=sample_rate,
        )


def unpad(outputs, lengths):
    """Unpads a padded batch of waveforms to their initial lengths
        args:
            outputs ([B, num_samples_max] array): padded waveforms
            lens (list of ints): initial lengths before padding in features space.
                This needs to be multiplied by hop length to go into sample space.
        returns:
            unpadded ([B,] list of [num_samples] arrays): unpadded waveforms."""
    unpadded = [outputs[i, :length * HPARAMS.hop_length] for i, length in enumerate(lengths)]
    return unpadded


def pad(batch):
    """Pads a batch of features to have the same length along the time dimension
        args:
            batch ([B,] list of [T, num_bins] np.arrays): unpadded features
        returns
            batch ([B, num_bins, T_max] torch.FloatTensor): padded features
    """
    max_len = 0
    for i, m in enumerate(batch):
        if len(m.shape) == 3:
            m = m.squeeze(1)
            batch[i] = m
        if m.shape[0] < 6:
            pad_width = (6 - m.shape[0]) // 2 + 1
            m = np.pad(m, ((pad_width, pad_width), (0, 0)), 'constant')
            batch[i] = m

        max_len = max(max_len, m.shape[0])

    dim = batch[0].shape[1]
    for i, m in enumerate(batch):
        curr_len = m.shape[0]
        pad_len = max_len - curr_len
        padding = np.zeros([pad_len, dim])
        batch[i] = np.concatenate((m, padding)).T

    batch = torch.FloatTensor(np.stack(batch).astype(np.float32))
    return batch


def generate(model, mel_paths, max_mel_len=None):
    """Given a model, some paths to mel spectrograms, this function computes
    the predicted waveforms using the model, and returns them in a list.
        args:
            model (WaveRNN object): instance of WaveRNN model
            mel_paths (list of str): list of paths to [T_i, num_bins] mel spectrograms.
            max_mel_len (int): optionally, truncates the mel-spectrograms at a max length.
        returns:
            outputs (list of arrays): a list containing the predicted waveforms
    """
    # Load mels. Optionally truncate
    if max_mel_len is not None:
        mels = [np.load(f)[:max_mel_len, :] for f in mel_paths]
    else:
        mels = [np.load(f) for f in mel_paths]
    # Compute number of bins and pad batch accordingly.
    batch_lens = [x.shape[0] for x in mels]
    batch = pad(mels).cuda()
    # Generate
    outputs = unpad(model.inference(batch), batch_lens)
    return outputs
