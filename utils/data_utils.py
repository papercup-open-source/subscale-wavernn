import os
import logging

import numpy as np
from librosa import load as librosa_load_wav

import torch

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from utils.utils import encode_mu_law


class FeatDataset(Dataset):
    """
    class defining our dataset of feature inputs and wav sequence labels.
    """
    def __init__(self, names, path, mode, hparams):
        self.path = path
        self.metadata = names
        self.sample_rate = hparams.sample_rate
        self.mode = mode
        self.bits = hparams.bits

    def load_wav(self, path):
        """ Loads the wav into a np array.
        """
        wav, _ = librosa_load_wav(path, sr=self.sample_rate)
        wav = wav.astype(np.float32)
        wav = wav / np.abs(wav).max()
        return wav

    def __getitem__(self, index):
        f = self.metadata[index]
        wav_path = os.path.join(self.path, self.mode, 'wav_24khz', f + '.wav')
        feat_path = os.path.join(self.path, self.mode, 'mel', f + '.npy')
        m = np.load(feat_path)

        x = self.load_wav(wav_path)
        x = encode_mu_law(x, 2 ** self.bits)
        return m.T, x

    def __len__(self):
        return len(self.metadata)


def standardize(batch, hop_length):
    """ Matches lengths of features with lengths
    of audio arrays.
    """
    batch = [x for x in batch if x[0].any()]

    utt_lens = [x[0].shape[-1] for x in batch]
    wav_lens = [x[1].shape[0] for x in batch]

    wav_lens = [min(utt_len * hop_length, wav_len - wav_len // hop_length)
                for utt_len, wav_len in zip(utt_lens, wav_lens)]
    utt_lens = [wav_len // hop_length for wav_len in wav_lens]

    feats = [x[0][:, :utt_len] for x, utt_len in zip(batch, utt_lens)]
    quants = [x[1][: wav_len] for x, wav_len in zip(batch, wav_lens)]

    return feats, quants


def torchize(feats, quants, seq_len):
    """ Converts numpy arrays into torch tensors,
    and converts quantised arrays.
    """
    feats = np.stack(feats).astype(np.float32)
    quants = np.stack(quants).astype(np.int64)

    feats = torch.FloatTensor(feats)
    quants = torch.LongTensor(quants)
    x_input = quants[:, :-1]

    # for subscale, this needs to be the same as x_input
    # y_quants = quants[:, :-1]
    return x_input, feats, x_input


def post_process(all_segments, batch_size):
    """ Chunks up output of collate funcion in test mode,
    in batches of size batch_size.
    """
    x, m, y = all_segments
    x = x.split(batch_size)[:-1]
    m = m.split(batch_size)[:-1]
    y = y.split(batch_size)[:-1]
    return zip(x, m, y)


def get_names(data_path, mode):
    """ Find intersection of mel and wav files and returns those names
        args:
            data_path (str): Path to the data folder
            mode (str): 'train' if training set or 'valid' if valid set
        returns:
            names (list): list of names without suffixes (eg p225_023) to be used for this mode.
    """
    # Compute the intersection
    logging.info(f"Preparing {mode} set:")
    gt_path = os.path.join(data_path, mode, 'wav_24khz')
    gen_path = os.path.join(data_path, mode, 'mel')

    logging.info(f"Sourcing feats from: {gen_path} \n Sourcing gts from: {gt_path}")

    wav_names = [x.replace('.wav', '') for x in os.listdir(gt_path)]
    mel_names = [x.replace('.npy', '') for x in os.listdir(gen_path)]
    names = list(set(wav_names).intersection(mel_names))

    logging.info(f"Found {len(wav_names)} targets, {len(mel_names)} feats, and pick the {len(names)} in common")
    logging.info('----------------------------------------------------------------------')

    return names


def get_loader(data_path, mode, hparams, whole=False):
    """Given root data folder path and mode, defines loader and returns it.
        args:
            data_path (str): Path to root folder of data.
            mode (str): train or valid.
            hparams (Namespace): hyperparameters.
        returns:
            loader (QuantLoader): Loader to iterate
    """
    if whole:
        hparams.batch_size = 1
    names = get_names(data_path, mode)
    dataset = FeatDataset(names, data_path, mode, hparams)
    collate_fn = WaveRNNCollate(
        hparams.hop_length * hparams.feat_sequence_len,
        hparams.hop_length, hparams.pad, mode, whole
    )
    if mode == 'train':
        loader = DataLoader(
            dataset, num_workers=2, shuffle=True, batch_size=hparams.batch_size,
            pin_memory=True, drop_last=False, collate_fn=collate_fn
        )
    elif mode == 'valid':
        loader = DataLoader(
            dataset, num_workers=2, shuffle=False, batch_size=hparams.batch_size,
            pin_memory=True, collate_fn=collate_fn
        )

    return loader


class WaveRNNCollate:
    def __init__(self, seq_len, hop_length, pad, mode, whole=False):
        self.seq_len = seq_len
        self.hop_length = hop_length
        self.pad = pad
        self.whole = whole
        if mode == 'train':
            self.get_segments = self.sample_segments
        elif mode == 'valid':
            self.get_segments = self.get_all_segments
        else:
            raise ValueError('Mode needs to be train or valid...')

    def get_whole_segments(self, feats, quants):
        """ Samples a single segment for each sentence.
        To be used with collate fn in train mode.
        """
        sig_pad = self.pad * self.hop_length

        new_quants = []
        for feat, quant in zip(feats, quants):
            start_index = sig_pad
            end_index = sig_pad + ((feat.shape[1] - 2 * self.pad) * self.hop_length)
            new_quant = quant[start_index: end_index + 1]
            new_quants.append(new_quant)
        return feats, new_quants

    def sample_segments(self, feats, quants):
        """ Samples a single segment for each sentence.
        To be used with collate fn in train mode.
        """
        feat_win = self.seq_len // self.hop_length + 2 * self.pad
        max_offsets = [feat.shape[-1] - feat_win for feat in feats]

        feat_offsets = [np.random.randint(0, offset) for offset in max_offsets]
        sig_offsets = [(offset + self.pad) * self.hop_length for offset in feat_offsets]

        feats = [feat[:, feat_offsets[i]: feat_offsets[i] + feat_win] for i, feat in enumerate(feats)]
        quants = [quant[sig_offsets[i]: sig_offsets[i] + self.seq_len + 1] for i, quant in enumerate(quants)]

        return feats, quants

    def get_all_segments(self, feats, quants):
        """ For each sentence of the batch, divides it in segments,
        and stacks segments from different sentences in one single batch.
        Useful to calculate exact loss at test time.
        WARNING: the batch size of the output batch will be variable,
        dependent on the lenght of the sentences.
        """
        feat_win = self.seq_len // self.hop_length + 2 * self.pad

        all_feats = []
        all_quants = []
        for feat, quant in zip(feats, quants):
            feat_offsets = list(range(0, feat.shape[1] - feat_win, feat_win))
            quant_offsets = [(offset + self.pad) * self.hop_length for offset in feat_offsets]
            feat_chunks = [feat[:, i: i + feat_win] for i in feat_offsets]
            quant_chunks = [quant[i: i + self.seq_len + 1] for i in quant_offsets]
            all_feats.extend(feat_chunks)
            all_quants.extend(quant_chunks)

        return all_feats, all_quants

    def __call__(self, batch):
        """This is the collate function to be used for data loading in this repo.
            args:
                batch (list): list of (np.array melspec, np.array waveform) tuples.
            returns:
                x (torch.LongTensor): shifted GT wav for teacher forcing
                m (torch.FloatTensor): melspectrogram used for conditioning
                y (torch.LongTensor): target waveform for loss computation
        """
        feats, quants = standardize(batch, self.hop_length)

        # ensure that feats are at least feat_win in length, or things go wrong
        feat_win = self.seq_len // self.hop_length + 2 * self.pad
        feats, quants = zip(*[(feat, quant) for (feat, quant) in zip(feats, quants) if feat.shape[-1] >= feat_win])

        if self.whole:
            feats, quants = self.get_whole_segments(feats, quants)
        else:
            feats, quants = self.get_segments(feats, quants)
        x, m, y = torchize(feats, quants, self.seq_len)
        return x, m, y
