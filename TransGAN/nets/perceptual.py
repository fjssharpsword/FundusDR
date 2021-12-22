import os
import torch
import torchvision
import torch.nn as nn
import numpy as np

#https://github.com/egorzakharov/PerceptualGAN/blob/master/models/perceptual_loss.py
#https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True, matching_loss='l1'):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = nn.ModuleList(blocks)
        self.transform = nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        if matching_loss == 'mse':
            self.criterion = nn.MSELoss()
        elif matching_loss == 'smoothl1':
            self.criterion = nn.SmoothL1Loss()
        elif matching_loss == 'l1':
            self.criterion = nn.L1Loss()

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += self.criterion(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += self.criterion(gram_x, gram_y)
        return loss

if __name__ == "__main__":
    vggptloss = VGGPerceptualLoss(matching_loss='mse').cuda()
    x = torch.rand(2, 3, 256, 256).cuda()
    y = torch.rand(2, 3, 256, 256).cuda()
    loss = vggptloss(x,y)
    print(loss)