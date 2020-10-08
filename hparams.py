class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.hop_length = int(self.sample_rate * self.frame_length)
        self.segment_length = self.pad * 2 * len(self.upsample_factors)


def create_hparams():
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = Namespace(
        ####################################
        # MISC
        ####################################
        iters_per_checkpoint=10000,

        ####################################
        # MODEL
        ####################################
        rnn_dims=512,
        fc_dims=512,
        bits=8,
        pad=2,
        upsample_factors=(5, 6, 8),
        compute_dims=128,
        res_out_dims=128,
        res_blocks=10,
        # Number of input frames which represent a training example.
        feat_sequence_len=12,

        # SUBSCALE
        batch_factor=16,
        horizon=7,
        lookback=172,
        condnet_n_layers=10,
        condnet_channels=512,
        condnet_kernelsize=3,
        condnet_drouput=0.0,

        loss_base_weight=1.0,

        ####################################
        # OPTIMIZATION
        ####################################
        lr=1e-4,
        ema_rate=0.999,
        batch_size=16,

        ####################################
        # DATA
        ####################################
        # Number of mel bins
        feat_dims=128,
        # Sampling rate of output sound
        sample_rate=24000,
        # Length in second of one input frame
        frame_length=0.01,
    )

    return hparams
