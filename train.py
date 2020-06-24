import argparse
import os
import logging

from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter  # pylint: disable=no-name-in-module

from utils.data_utils import (
    get_loader,
    post_process,
)

from utils.utils import (
    init_ema,
    generate,
    save_wavs,
    load_checkpoint,
    is_batch_all_silence
)

# from gen_wavs import generate_routine
from hparams import create_hparams
from models.wavernn import (
    WaveRNN,
)


def save_checkpoint(checkpointdict, iteration, filepath, exp_name):
    """Save the checkpoint dict of a model."""
    torch.save(checkpointdict, filepath)


def clone_as_averaged_model(model, ema):
    averaged_model = WaveRNN(HPARAMS)
    averaged_model.cuda()
    averaged_model.load_state_dict(model.module.state_dict())
    for name, param in averaged_model.named_parameters():
        if name in ema.shadow:
            param.data = ema.shadow[name].clone()
    return averaged_model


def train_step(train_loader, test_loader, whole_loader, model, optimizer, criterion, iteration,
               ema=None):  # pylint: disable=R0913
    running_loss = 0.
    train_enum = tqdm(train_loader)
    running_partials = [0.] * HPARAMS.batch_factor
    num_elements = 0
    for x, m, y in train_enum:
        iteration += 1
        optimizer.zero_grad()

        # If batch is entirely silence, then ignore it
        if is_batch_all_silence(m):
            continue

        x, m, y = x.cuda(), m.cuda(), y.cuda()

        y_hat = model(x, m)
        y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
        y = y.unsqueeze(-1)

        breakdown = []
        partial_losses = []
        weights = [HPARAMS.loss_base_weight ** i for i in range(HPARAMS.batch_factor)]
        weights = weights[::-1]
        weights = [len(weights) * w / sum(weights) for w in weights]
        for i in range(HPARAMS.batch_factor):
            shift = HPARAMS.batch_factor
            partial_loss = criterion(y_hat[:, :, i::shift], y[:, i::shift])
            partial_losses.append(partial_loss)
            partial = partial_loss.item()
            running_partials[i] += partial
            partial = partial / y.numel()
            breakdown.append(partial * HPARAMS.batch_factor)

        weighted_partial_losses = [l * w for l, w in zip(partial_losses, weights)]
        loss = sum(weighted_partial_losses)

        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        curr_elems = y.numel()
        num_elements += curr_elems
        running_loss += current_loss

        # Update EMA.
        if ema is not None:
            for name, param in model.module.named_parameters():
                if name in ema.shadow:
                    ema.update(name, param.data)

        descriptors = (iteration, running_loss / num_elements, current_loss / curr_elems)
        pt1 = f'Iter %d Train (Average Loss %.4f) Loss %.2f |' % descriptors
        pt2 = f' %.2f' * HPARAMS.batch_factor % tuple([p / num_elements * HPARAMS.batch_factor for p in
                                                       running_partials])
        pt3 = f' %.2f' * HPARAMS.batch_factor % tuple(breakdown)
        description = pt1 + pt2 + ' |' + pt3
        train_enum.set_description(
            description,
            descriptors,
        )
        WRITER.add_scalar('train_loss', current_loss / curr_elems, iteration)
        for i, value in enumerate(breakdown):
            WRITER.add_scalar(f'train_loss_{i}', value, iteration)

        if iteration % HPARAMS.iters_per_checkpoint == 0:
            averaged_model = clone_as_averaged_model(model, ema)
            saving_folder = 'checkpoints/{}/iteration_{}'.format(ARGS.expName, iteration)
            # Save and upload to S3.
            os.makedirs(saving_folder, exist_ok=True)
            save_checkpoint(
                {
                    'state_dict': model.module.state_dict(),
                    'iteration': iteration,
                    'dataset': ARGS.data,
                    'optimizer': optimizer.state_dict(),
                }, iteration,
                os.path.join(saving_folder, 'checkpoint.pth'), ARGS.expName
            )
            save_checkpoint(
                {
                    'state_dict': averaged_model.state_dict(),
                    'iteration': iteration,
                    'dataset': ARGS.data,
                    'optimizer': optimizer.state_dict(),
                }, iteration,
                os.path.join(saving_folder, 'ema_model_checkpoint.pth'), ARGS.expName
            )

            with torch.no_grad():
                avg_test_loss = test_step(model, test_loader, criterion, iteration)
                WRITER.add_scalar('test_loss', avg_test_loss, iteration)
                with open(os.path.join('checkpoints', ARGS.expName, 'test_loss.txt'), 'a') as f:
                    f.write(f'Iteration {iteration}: {avg_test_loss}\n')
                averaged_model.eval()
                # Generate now.
                mel_fnames = test_loader.dataset.metadata[:HPARAMS.batch_size]
                # if training on genfeats, we want to listen to eval_mode feats
                mel_paths = [os.path.join(
                    ARGS.data.replace('train_mode', 'eval_mode'), 'valid/mel/', f + '.npy'
                ) for f in mel_fnames]
                outputs = generate(averaged_model, mel_paths, max_mel_len=MAX_MEL_LEN)
                out_dir = f'outputs/{ARGS.expName}/iter_{iteration}/free_inf/'
                os.makedirs(out_dir, exist_ok=True)
                out_paths = [
                    os.path.join(out_dir, os.path.basename(f.replace('.npy', '.wav'))) for f in mel_paths
                ]
                save_wavs(out_paths, outputs, HPARAMS.sample_rate)
                for fname, wave in zip(mel_fnames, outputs):
                    WRITER.add_audio(
                        fname, wave, iteration, sample_rate=HPARAMS.sample_rate
                    )
                # out_dir_routine = 'outputs/{ARGS.expName}/iter_{iteration}/routine/'
                # os.makedirs(out_dir_routine, exist_ok=True)
                # generate_routine(averaged_model, whole_loader, out_dir_routine, HPARAMS)
    return iteration


def test_step(model, test_loader, criterion, iteration, batch_size=64):
    running_test_loss = 0.
    num_elements = 0
    test_enum = tqdm(test_loader)

    for all_segments in test_enum:
        batches = post_process(all_segments, batch_size)

        for x, m, y in batches:
            x, m, y = x.cuda(), m.cuda(), y.cuda()
            y_hat = model(x, m)
            y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
            y = y.unsqueeze(-1)

            test_loss = criterion(y_hat, y)
            current_test_loss = test_loss.item()
            curr_elements = y.numel()
            num_elements += curr_elements
            running_test_loss += current_test_loss

            test_enum.set_description('Iter %d Test (Average Loss %.4f) Loss %.2f' % (
                iteration,
                running_test_loss / num_elements,
                current_test_loss / curr_elements,
            ))
        break

    avg_test_loss = running_test_loss / num_elements

    return avg_test_loss


def train():
    torch.cuda.set_device(0)
    iteration = 0
    model = WaveRNN(HPARAMS)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=HPARAMS.lr)

    if ARGS.checkpoint:
        if os.path.basename(ARGS.checkpoint).startswith('ema_model'):
            ema_checkpoint = ARGS.checkpoint
        else:
            ema_checkpoint = 'ema_model_' + os.path.basename(ARGS.checkpoint)
            ema_checkpoint = os.path.join(os.path.dirname(ARGS.checkpoint), ema_checkpoint)

        # Initialise EMA from the ema checkpoint.
        logging.info('Initialising ema model {}'.format(ema_checkpoint))
        ema_model = WaveRNN(HPARAMS).cuda()
        ema_base_model, _ = load_checkpoint(ema_checkpoint, ema_model)
        ema = init_ema(ema_base_model, HPARAMS.ema_rate)

        # Initialise vanilla model
        logging.info('Loading checkpoint {}'.format(ARGS.checkpoint))
        model, iteration, optimizer = load_checkpoint(ARGS.checkpoint, model, optimizer)

    else:
        # Initialise EMA from scratch.
        ema = init_ema(model, HPARAMS.ema_rate)

    criterion = nn.NLLLoss(reduction='sum').cuda()
    train_loader, test_loader = get_loader(ARGS.data, 'train', HPARAMS), get_loader(ARGS.data, 'valid', HPARAMS)
    whole_loader = get_loader(ARGS.data, 'valid', HPARAMS, whole=True)
    model = nn.DataParallel(model)

    epoch_offset = max(0, int(iteration / len(train_loader)))
    for _ in range(epoch_offset, ARGS.epochs):
        iteration = train_step(
            train_loader, test_loader, whole_loader, model, optimizer,
            criterion, iteration, ema=ema
        )

        averaged_model = clone_as_averaged_model(model, ema)
        save_checkpoint(
            {
                'state_dict': model.module.state_dict(),
                'iteration': iteration,
                'dataset': ARGS.data,
                'optimizer': optimizer.state_dict(),
            }, iteration,
            'checkpoints/{}/lastmodel.pth'.format(ARGS.expName), ARGS.expName,
        )
        save_checkpoint(
            {
                'state_dict': averaged_model.state_dict(),
                'iteration': iteration,
                'dataset': ARGS.data,
                'optimizer': optimizer.state_dict(),
            }, iteration,
            'checkpoints/{}/ema_model_lastmodel.pth'.format(ARGS.expName), ARGS.expName,
        )


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='WaveRNN training script')
    PARSER.add_argument('--checkpoint', default=None,
                        metavar='M', type=str, help='Model path')
    PARSER.add_argument('--data', default=None, required=True,
                        metavar='D', type=str, help='Data path')
    PARSER.add_argument('--expName', default=None, required=True,
                        metavar='N', type=str, help='Experiment name')
    PARSER.add_argument('--epochs', default=30000,
                        metavar='E', type=int, help='Number of total train epochs')
    PARSER.add_argument('--debug', dest='debug', action='store_true',
                        help='Debug mode')

    ARGS = PARSER.parse_args()
    HPARAMS = create_hparams()
    logging.getLogger().setLevel(logging.INFO)
    MAX_MEL_LEN = 500
    if ARGS.debug:
        HPARAMS.iters_per_checkpoint = 5
        MAX_MEL_LEN = 10
    os.makedirs('checkpoints/{}'.format(ARGS.expName), exist_ok=True)
    os.makedirs(os.path.join('tensorboard-runs', ARGS.expName), exist_ok=True)
    WRITER = SummaryWriter(os.path.join('tensorboard-runs', ARGS.expName))

    train()
    WRITER.close()
