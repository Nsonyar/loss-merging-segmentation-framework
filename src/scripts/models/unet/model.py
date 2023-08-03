import torch.nn as nn

from .parts import DoubleConv, Down, Up


class UNet(nn.Module):
    '''
    Architecture based on U-Net: Convolutional Networks for Biomedical Image Segmentation
    Link - https://arxiv.org/abs/1505.04597
    Parameters:
        num_classes (int) - Number of output classes required (default 19 for KITTI dataset)
        num_layers (int) - Number of layers in each side of U-net
        features_start (int) - Number of features in first layer
        bilinear (bool) - Whether to use bilinear interpolation or transposed
            convolutions for upsampling.
    '''

    def __init__(self, num_classes=19, num_layers=5, features_start=64, bilinear=False):
        super().__init__()
        self.num_layers = num_layers

        layers = [DoubleConv(3, features_start)]

        feats = features_start
        for _ in range(num_layers-1):
            layers.append(Down(feats, feats*2))
            feats*=2
        
        for _ in range(num_layers-1):
            layers.append(Up(feats, feats//2))
            #The floor division // operator divides two numbers and rounds the result down to the nearest integer.feats = feats // 2
            feats//=2

        layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        #print('Forward')
        #print(x.shape)
        xi = [self.layers[0](x)]
        
        #print(self.layers[0](x).shape)
        # Down path
        for layer in self.layers[1:self.num_layers]:
            #print('Down')
            #print(layer(xi[-1]).shape)
            xi.append(layer(xi[-1]))
        # Up path
        ''' The slice "self.layers[self.num_layers:-1]" returns a new list containing a subset of the original list "self.layers". The returned list starts with the element at index "self.num_layers" and ends with the second-to-last element of the original list.
        '''
        for i, layer in enumerate(self.layers[self.num_layers:-1]):
            #print('Up')
            #print(layer(xi[-1], xi[-2-i]).shape)
            xi[-1] = layer(xi[-1], xi[-2-i])
        #print(self.layers[-1](xi[-1]).shape)
        return self.layers[-1](xi[-1])