import os
import torch
import torch.nn as nn
import math

import math
from data import Phase11, AudioPhase2


def load_audio_data(finetune=False):
    """If in finetune process, the train dataset and the dev dataset shall not be merged"""
    train_ds=AudioPhase2(audio_path='/data/2015/wav', protocol_path='/data/2015/CM_protocol/cm_train.trn',length=5)
    val_ds=AudioPhase2(audio_path='/data/2015/wav', protocol_path='/data/2015/CM_protocol/cm_develop.ndx', length=5)
    test_ds=AudioPhase2(audio_path='/data/2015/wav', protocol_path='/data/2015/CM_protocol/cm_evaluation.ndx', length=5)
    if not finetune:
        final_ds=torch.utils.data.ConcatDataset([train_ds, val_ds])
        return final_ds, test_ds
    else:
        return train_ds, val_ds


def load_data(finetune=False):
    """If in finetune process, the train dataset and the dev dataset shall not be merged"""
    train_ds=Phase11(genuine_path='/data/Feature/Train/Genuine', spoof_path='/data/Feature/Train/Spoof', protocol_path='/data/ASVspoof2019.LA.cm.train.trn.txt')
    val_ds=Phase11(genuine_path='/data/Feature/Val/Genuine', spoof_path='/data/Feature/Val/Spoof', protocol_path='/data/ASVspoof2019.LA.cm.dev.trl.txt')
    test_ds=Phase11(genuine_path='/data/Feature/Test/Genuine', spoof_path='/data/Feature/Test/Spoof', protocol_path='/data/ASVspoof2019.LA.cm.eval.trl.txt')
    if not finetune:
        final_ds=torch.utils.data.ConcatDataset([train_ds, val_ds])
        return final_ds, test_ds
    else:
        return train_ds, val_ds


def adjust_learning_rate(optimizer, epoch, lr, warmup, epochs=100):
    lr = lr
    if epoch < warmup:
        lr = lr / (warmup - epoch)
    else:
        lr *= 0.5 * (1. + math.cos(math.pi *
                     (epoch - warmup) / (epochs - warmup)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_run_logdir(root_dir):
    import time
    run_id=time.strftime('run_%Y_%m_%d-%H_%M_%S')
    return os.path.join(root_dir, run_id)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

