import math
import torch
import torch.nn as nn
import torchvision as vsn

import torch.nn.functional as F
import pretrainedmodels

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        return x * y

class SCSEModule(nn.Module):
    def __init__(self, ch, re=16):
        super().__init__()
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(ch, ch//re, 1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(ch//re, ch, 1),
                                 nn.Sigmoid()
                                )
        self.sSE = nn.Sequential(nn.Conv2d(ch, ch, 1),
                                 nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        #print(x.size())
        return x.contiguous().view(x.size(0), -1)

class Decoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, 
                 kernel_size=3, padding=1, upsample=True):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.se_in = SCSEModule(in_channels)
        self.se_out = SCSEModule(out_channels)
        self.upsample = upsample

    def forward(self, x, y=None):
        if self.upsample:
            x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)

        if y is not None:
            x = torch.cat([x,y], dim=1)

        x = self.se_in(x)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        x = self.se_out(x)

        return x

class ResUNet(nn.Module):
    def __init__(self, use_bool=False):
        super(ResUNet, self).__init__()
        self.resnet_layers = list(pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained='imagenet').children()) 
        #print(self.resnet_layers[:3])
        #self.maxpool = self.resnet_layers[0][3]
        # encoding layers
        #self.conv_in = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        #self.conv_in.weight = self.resnet_layers[0].weight
        #for param_a, param_b in zip(self.conv_in.parameters(), self.resnet_layers[0][0].parameters()):
        #    param_a.data = param_b.data
        #self.bn_in = nn.BatchNorm2d(64)
        #self.bn_in.data = self.resnet_layers[1].data
        #for param_a, param_b in zip(self.bn_in.parameters(), self.resnet_layers[1][0].parameters()):
        #    param_a.data = param_b.data
        self.relu = nn.ReLU(True)
        #self.se_conv1 = SCSEModule(64)
        self.encode_a = nn.Sequential(*self.resnet_layers[0][:3])
        self.encode_b = self.resnet_layers[1]
        self.encode_c = self.resnet_layers[2]
        self.encode_d = self.resnet_layers[3]
        self.encode_e = self.resnet_layers[4]

        self.center = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2))

        self.decoder_a = Decoder(2560, 512, 64, kernel_size=3, padding=1)
        self.decoder_b = Decoder(1088, 256, 64, kernel_size=3, padding=1)
        self.decoder_c = Decoder(576, 128, 64, kernel_size=3, padding=1)
        self.decoder_d = Decoder(320, 96, 64, kernel_size=3, padding=1)
        self.decoder_e = Decoder(64, 32, 64, kernel_size=3, padding=1)

        # hyper column output
        self.se_hcol = SCSEModule(320)
        self.mask_out = self._mask_out(320)
        # deep supervision output
        self.mask_ds = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        
        self.mask_chck = self._mask_check()

        self.dropout = nn.Dropout2d(p=0.5)
        self.use_bool = use_bool
        
    
    def _mask_out(self, in_channels):
        return nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0),
                             nn.ReLU(inplace=True), 
                             nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0))
    
    def _mask_check(self):
        return nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                             Flatten(),
                             nn.Linear(512, 1))
    '''    
    def train(self, mode=True, freeze_bn=False):
        super(ResUNet, self).train(mode)
        if freeze_bn == True and mode == True:
            print('Freezing BatchNorm2d')
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
    '''

    def forward(self, x):
        # encoder layers
        #print('encoders')
        #c1 = self.conv_in(x)
        #c1 = self.bn_in(c1)
        #c1 = self.relu(c1)
        c1 = self.encode_a(x)
        #print(c1.size())
        c2 = self.encode_b(c1)
        #print(c2.size())
        c3 = self.encode_c(c2)
        #print(c3.size())
        c4 = self.encode_d(c3)
        #print(c4.size())
        c5 = self.encode_e(c4)
        #print(c5.size())
        center = self.center(c5)
        #print(center.size())
        #print('decoders')
        d1 = self.decoder_a(center, c5)
        #print(d1.size())
        d2 = self.decoder_b(d1, c4)
        #print(d2.size())
        d3 = self.decoder_c(d2, c3)
        #print(d3.size())
        d4 = self.decoder_d(d3, c2)
        #print(d4.size(), c1.size())
        d5 = self.decoder_e(d4)
        

        mask_d5 = self.mask_ds(d5)
        mask_d4 = self.mask_ds(d4)
        mask_d3 = self.mask_ds(d3)
        mask_d2 = self.mask_ds(d2)
        mask_d1 = self.mask_ds(d1)

        #print(center.size())
        mask_bool = self.mask_chck(self.dropout(center))
       
        #print(d5.size(), d4.size(), d3.size(), d2.size(), d1.size(), mask_bool.size())
        hcol = torch.cat([d5, 
                          F.upsample(d4, scale_factor=2, mode='bilinear',
                                     align_corners=False),
                          F.upsample(d3, scale_factor=4, mode='bilinear', 
                                     align_corners=False),
                          F.upsample(d2, scale_factor=8, mode='bilinear', 
                                     align_corners=False),
                          F.upsample(d1, scale_factor=16, mode='bilinear', 
                                     align_corners=False)], dim=1)
        #print(hcol.size())`
        mask_hcol = self.mask_out(self.dropout(hcol))

        return mask_hcol, mask_d5, mask_d4, mask_d3, mask_d2, mask_d1, mask_bool

if __name__ == '__main__':
    net = ResUNet(use_bool=True)
    x = torch.rand(1, 3, 128, 128)
    output = net(x)
    for out in output:
        print(out.size())

