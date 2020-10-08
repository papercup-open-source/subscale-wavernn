# Papercup Subscale WaveRNN repository

This code implements the Subscale part of the WaveRNN paper on top of Fatchord's original implementation.

Please refer to this accompanying blogpost for details of our interpretation: [Subscale
WaveRNN](https://papercup.dev/posts/subscale_wavernn/)

Original publication: [Efficient Neural Audio Synthesis](https://arxiv.org/abs/1802.08435)

Initial implementation: [Fatchord's Repo](https://github.com/fatchord/WaveRNN)

Samples after training for 1M iterations on the Sharvard dataset with current hparams: [samples](https://www.dropbox.com/sh/r5ohzs7c1uaqm6s/AAD8_mjcy6l59-b_b-Qp2L1na)

## Quickstart: train a new model

To train the model, use the command below:

`CUDA_VISIBLE_DEVICES={your_gpus} python train.py --data ../data/{your_dataset} --expName {your_experiment}`

THE TRAINING SCRIPT IS MULTI-GPU BY DEFAULT, TO USE SINGLE GPUS PLEASE SET `CUDA_VISIBLE_DEVICES` TO YOUR PREFERRED GPU

__The `{your_dataset}` folder is assumed to have the following folder structure:__

- `{args.data}/train/mel`;
- `{args.data}/train/wav_24khz`;
- `{args.data}/valid/mel`;
- `{args.data}/valid/wav_24khz`

If you are providing the features, the data loader will create a set based on the intersection of filenames present
in the `mel` folder and the `wav_24khz` folder.

You can download a version of the Sharvard dataset with this folder structure
[here](https://www.dropbox.com/s/uqt850z4f3kr529/Sharvard.tar.gz?dl=0)

Extract with:
`tar xvf Sharvard.tar.gz`

## Inspect your trained model with tensorboard 

To use within `WaveRNN`: `tensorboard --logdir tensorboard-runs --port {MY PORT}`.

## To generate from a trained model

Using `gen_wavs.py`:

`CUDA_VISIBLE_DEVICES={} python gen_wavs.py --data {mel_directory} --checkpoint {checkpoint_path} --out_dir {out_folder}`

## Play around with HParams

The most fun hyperparameters to play around with are the three subscale parameters:
- batch_factor
- horizon
- lookback

As well as tweaking the Condition Network. Any issues or cool findings let us know!
