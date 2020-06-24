import torch


def run_overlay(subscale, indeces, pos):
    indeces = subscale.one2two(indeces)
    zeros = torch.zeros(indeces.shape[0], subscale.context_width, subscale.batch_factor)

    out = subscale.overlay(zeros, indeces, pos)
    print(out)
    print(pos)
    flattened = subscale.two2one(out.flip(2))

    # check that masked subtensors are all zeros
    n_future_subtensors = subscale.batch_factor - (pos % subscale.batch_factor) - 1
    for i in range(n_future_subtensors):
        assert((flattened[:, i::subscale.batch_factor] == 0.0).all())

    # check that present and future samples of current subtensor are zeros
    for i in range(subscale.horizon + 1):
        assert((flattened[:, - i * subscale.batch_factor - 1] == 0.0).all())

    # check that available dependencies of non-current subt have exact value expected
    for j in range(n_future_subtensors, subscale.batch_factor - 1):
        for batch_element in flattened[:, j::subscale.batch_factor]:
            for i, x in enumerate(batch_element.flip(0)):
                should_be = pos - (i - subscale.horizon) * subscale.batch_factor
                should_be = should_be - (subscale.batch_factor - j - 1)
                should_be = max(should_be, 0)
                if should_be > indeces.max():
                    should_be = 0
                assert(x == should_be)

    # check that each element on current subt, past is exactly what we expect
    current_subt = flattened[:, subscale.batch_factor - 1::subscale.batch_factor]
    current_subt_past = current_subt[:, :-subscale.horizon - 1]
    for batch_element in current_subt_past:
        for i, x in enumerate(batch_element.flip(0)):
            should_be = pos - (i + 1) * subscale.batch_factor
            should_be = max(should_be, 0)
            assert(x == should_be)
