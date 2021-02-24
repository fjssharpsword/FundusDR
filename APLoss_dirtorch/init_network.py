import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from APLoss_dirtorch.nets.rmac_resnet import resnet101_rmac


def net():
    model = resnet101_rmac()
    pretrained = 'imagenet'
    model.load_pretrained_weights(pretrained)
    return model
