import argparse
import os
import torch
import numpy as np

from tqdm import tqdm
from time import time

from models.wavernn import WaveRNN

from utils.utils import generate, load_checkpoint, save_wavs
from utils.data_utils import get_loader
from hparams import create_hparams


def save_wav(out_dir, fname, output, label, sr):
    path = f'{out_dir}/{fname}{label}.wav'
    save_wavs([path], output, sr)


def load_model():
    model = WaveRNN(HPARAMS).cuda()
    model, _ = load_checkpoint(ARGS.checkpoint, model)
    return model


def breakdown_output(out_dir, output, fname, batch_factor, sample_rate):
    for j in range(batch_factor):
        label = '_' + str(j) + 'th_subt'
        sr = sample_rate // batch_factor
        save_wav(out_dir, fname, output[:, j::batch_factor], label, sr)
    return output


def generate_routine(model, whole_segments, out_dir, hparams):
    for i, (x, m, _) in enumerate(whole_segments):
        x, m = x.cuda(), m.cuda()
        model.eval()
        t1 = time()
        output = model.subscale_inference(m)
        print("Subscale inference took: ", time() - t1)
        save_wav(out_dir, i, output, '_subscale_inf_self_context', hparams.sample_rate)
        # breakdown_output
        t2 = time()
        output = model.inference(m)
        print("Standard inference took:", time() - t2)
        save_wav(out_dir, i, output, '_inf_self_context', hparams.sample_rate)
        # breakdown_output(out_dir, output, str(i) + '_inf_self_context', hparams.batch_factor, hparams.sample_rate)
        model.train()
        output = model.train_mode_generate(x, m)
        save_wav(out_dir, i, output, '', hparams.sample_rate)
        # breakdown_output(out_dir, output, i, hparams.batch_factor, hparams.sample_rate)
        output = model.train_mode_generate(x, m * 0.0)
        save_wav(out_dir, i, output, '_zeroed_mel', hparams.sample_rate)
        model.eval()
        output = model.inference(m, gt=x)
        save_wav(out_dir, i, output, '_inf_gt_context', hparams.sample_rate)
        # breakdown_output(out_dir, output, str(i) + '_inf_gt_context', hparams.batch_factor, hparams.sample_rate)


def launch_routine(model):
    data_path = ARGS.data
    if 'valid/mel' in ARGS.data:
        data_path = data_path.replace('valid/mel', '')
    whole_segments = get_loader(data_path, 'valid', HPARAMS, whole=True)
    generate_routine(model, whole_segments, ARGS.out_dir, HPARAMS)


def standard_generate(model):
    model.eval()
    filenames = os.listdir(ARGS.data)
    batched_filenames = [
        filenames[i:i + ARGS.batch_size] for i in range(0, len(filenames), ARGS.batch_size)
    ]
    for batch in tqdm(batched_filenames):
        # Truncate features at ARGS.max_mel_len.
        mel_paths = [os.path.join(ARGS.data, f) for f in batch]
        outputs = generate(model, mel_paths, max_mel_len=ARGS.max_mel_len)
        # Output
        out_paths = [
            os.path.join(ARGS.out_dir, os.path.basename(f.replace('.npy', '.wav'))) for f in mel_paths
        ]
        save_wavs(out_paths, outputs, HPARAMS.sample_rate)
        for output, out_path in zip(outputs, out_paths):
            output = np.expand_dims(output, 0)
            fname = out_path.split('/')[-1].replace('.wav', '')
            breakdown_output(ARGS.out_dir, output, fname, HPARAMS.batch_factor, HPARAMS.sample_rate)
        break


def main():
    model = load_model()
    launch_routine(model)
    standard_generate(model)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Generate wavs from a folder of mels.')
    PARSER.add_argument('--data', default=None, required=True,
                        type=str, help='Data path to the mel directory.')
    PARSER.add_argument('--checkpoint', default=None, required=True,
                        type=str, help='Path to checkpoint to use to generate wavs.')
    PARSER.add_argument('--out_dir', default=None, required=True,
                        type=str, help='Folder to dump wavs')
    PARSER.add_argument('--batch_size', default=400,
                        type=int, help='Batch size')
    PARSER.add_argument('--max_mel_len', default=3000, type=int,
                        help='Maximum number of mel frames after which we truncate')

    ARGS = PARSER.parse_args()
    os.makedirs(ARGS.out_dir, exist_ok=True)
    HPARAMS = create_hparams()

    torch.cuda.set_device(0)
    main()
