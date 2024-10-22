# coding=utf-8

import os
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def move_to_cuda(sample):
#
#     def _move_to_cuda(maybe_tensor):
#         if torch.is_tensor(maybe_tensor):
#             return maybe_tensor.cuda()
#         elif isinstance(maybe_tensor, dict):
#             return {
#                 key: _move_to_cuda(value)
#                 for key, value in maybe_tensor.items()
#             }
#         elif isinstance(maybe_tensor, list):
#             return [_move_to_cuda(x) for x in maybe_tensor]
#         else:
#             return maybe_tensor
#
#     return _move_to_cuda(sample)

def move_to_cuda(sample, device):
    for atr in dir(sample):
        value = getattr(sample, atr)
        if torch.is_tensor(value):
            setattr(sample, atr, value.to(device))

    return sample


def load_pretrained_weight(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise IOError('Model file not found: {}'.format(checkpoint_path))

    pretrained_dict = torch.load(checkpoint_path)['model']
    model_dict = model.state_dict()
    inter_dict = dict([(var, pretrained_dict[var]) for var in model_dict if var in pretrained_dict])
    print('load variable from ckpt:', ','.join(inter_dict.keys()))
    model_dict.update(inter_dict)
    model.load_state_dict(model_dict, strict=True)

    return model
