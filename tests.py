import torch
import random

from hparams import create_hparams
from models.wavernn import Subscaler, WaveRNN
from utils.test_utils import run_overlay
from utils.data_utils import get_loader


def test_subscale_vs_standard_inference_partity():
    hparams = create_hparams()
    model = WaveRNN(hparams, debug=True).cuda()
    seq_len = 100
    m = torch.rand(1, hparams.feat_dims, seq_len).cuda()
    x = torch.rand(1, seq_len * hparams.hop_length).cuda()
    _, _, standard_x = model.inference(m, gt=x)
    _, _, subscale_x = model.subscale_inference(m, gt=x)

    assert(abs(standard_x - subscale_x).mean() < 1e-6)


def test_stack_flatten_parity():
    hparams = create_hparams()
    for _ in range(100):
        hparams.batch_factor = random.randint(1, 32)
        hparams.horizon = random.randint(1, 10)
        seq_len = random.randint(1, 10)
        n_channels = random.randint(1, 1000)
        subscale = Subscaler(hparams)

        batch_dim = random.randint(1, 16)
        tensor = torch.rand([batch_dim, seq_len * subscale.context_len, n_channels])

        permuted = subscale.stack_substensors(tensor)
        orig = subscale.flatten_subtensors(permuted)
        assert(torch.eq(tensor, orig).all())


def test_inference_forward_parity():
    hparams = create_hparams()
    model = WaveRNN(hparams, debug=True).cuda()
    model.train()
    data_path = '../data/short_sens/'
    whole_segments = get_loader(data_path, 'valid', hparams, whole=True)
    for i, (x, m, _) in enumerate(whole_segments):
        x, m = x.cuda(), m.cuda()
        forward_output, f_context, f_x = model.train_mode_generate(x, m)
        inference_output, i_cont_dict, i_x = model.inference(m, gt=x)
        assert(abs(i_x - f_x).mean() < 1e-6)
        '''
        f_cont_dict = {}
        for j in range(f_context.shape[1]):
            f_cont_dict[j] = f_context[:, j, :]

        assert(f_cont_dict.keys() == i_cont_dict.keys())

        for k in f_cont_dict.keys():
            assert(torch.eq(f_cont_dict[k], i_cont_dict[k]).all())

        assert((forward_output == inference_output).all())
        break'''


def test_map_pos():
    hparams = create_hparams()
    for _ in range(100):
        hparams.batch_factor = random.randint(1, 32)
        hparams.horizon = random.randint(1, 10)
        subscale = Subscaler(hparams)
        pos = random.randint(0, 100000)
        t, subt = subscale.map_pos(pos)
        assert(pos == subscale.inv_map_pos(subt, t))


def test_permute():
    hparams = create_hparams()
    subscale = Subscaler(hparams)

    indeces = torch.arange(10 * hparams.batch_factor).repeat(10, 1)

    twod = subscale.one2two(indeces)
    oned = subscale.two2one(twod)

    assert(twod.shape[2] == hparams.batch_factor)
    assert(torch.eq(indeces, oned).all())


def test_overlay_general():
    hparams = create_hparams()

    # test 100 random combinations
    for _ in range(100):
        hparams.batch_factor = random.randint(1, 32)
        hparams.horizon = random.randint(1, 10)
        subscale = Subscaler(hparams)

        batch_dim = random.randint(1, 16)

        lensrc = subscale.context_len * random.randint(1, 10)
        indeces = torch.arange(lensrc).repeat(batch_dim, 1)
        pos = random.randint(0, lensrc)

        run_overlay(subscale, indeces, pos)


def test_overlay_last_samples():
    hparams = create_hparams()

    # test 100 random combinations
    for _ in range(100):
        hparams.batch_factor = random.randint(1, 32)
        hparams.horizon = random.randint(1, 10)
        subscale = Subscaler(hparams)
        batch_dim = random.randint(1, 16)

        lensrc = subscale.context_len * random.randint(1, 10)
        indeces = torch.arange(lensrc).repeat(batch_dim, 1)
        pos = random.randint(lensrc - 10, lensrc)

        run_overlay(subscale, indeces, pos)


def test_overlay_first_samples():
    hparams = create_hparams()

    # test 100 random combinations
    for _ in range(100):
        hparams.batch_factor = random.randint(1, 32)
        hparams.horizon = random.randint(1, 10)
        subscale = Subscaler(hparams)
        batch_dim = random.randint(1, 2)

        lensrc = subscale.context_len * random.randint(1, 10)
        indeces = torch.arange(lensrc).repeat(batch_dim, 1)
        pos = random.randint(0, 10)

        run_overlay(subscale, indeces, pos)
