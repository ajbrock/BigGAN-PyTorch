"""
This code is created for visual + Amygdala + intermediate road with attention block;

if you want to train a model for visual+Amygdala+IntermediateRoad with Attention block, please use this code for training and testing;
if you only want to test the model  best_model_emotion_regression_amygdala_0611_50epoch_lr4_128bs_7AM_ckvideo_middleframe_train_epoch3.pth, please use
Amy_IntermediateRoad_v1.py

"""



from __future__ import print_function, division

import copy

import torch
import torch.nn as nn

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import matplotlib
matplotlib.use('Agg')
from models.eca_module import eca_layer

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',

}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def _vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)

def _vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)

def vgg16_ori(is_freeze = True):
    # Load the pretrained model from pytorch
    model = _vgg16(pretrained=True)

    # Freeze training for all layers except for the final layer
    for param in model.parameters():
        param.requires_grad = not is_freeze

    return model

def vgg16_bn_ori(is_freeze = True):
    # Load the pretrained model from pytorch
    model = _vgg16_bn(pretrained=True)

    # Freeze training for all layers except for the final layer
    for param in model.parameters():
        param.requires_grad = not is_freeze

    return model

class Amy_IntermediateRoad(nn.Module):
    def __init__(self, lowfea_VGGlayer =4, highfea_VGGlayer = 36, is_highroad_only=False, is_gist=False):
        super(Amy_IntermediateRoad, self).__init__()
        """
        #this model is composed of two small networks:  Lateral nucleus (LA) and Central nucleus (CE)
        #LA is composed of 3 small CNNs to integrate low-level features, middle-level features, and high-level features from VGG.
        #CE is composed of several fully connected layers
        Args:

        """
        self.vgg = vgg16_ori()

        self.is_highroad_only = is_highroad_only
        self.highfea_VGGlayer = highfea_VGGlayer
        self.is_gist = is_gist

        if not self.is_highroad_only:

            if not is_gist:
                # settingn which layers features should be extracted
                self.lowfea_VGGlayer = lowfea_VGGlayer

                # seperate the network VGG to different parts
                self.vgg_lowfea_part = self.vgg.features[:self.lowfea_VGGlayer]


                self.attention_eca = eca_layer()


                self.lowroad_maxpool =nn.MaxPool2d(kernel_size=29, stride=14, padding=0, dilation=1, ceil_mode=False)


                self.lowroad_gap1 = nn.AdaptiveAvgPool2d(1)
                self.lowroad_gap2 = nn.AdaptiveAvgPool2d(2)

                self.lowroad_gmp2 = nn.AdaptiveMaxPool2d(2)


                self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=3, padding=0)
                self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=5, padding=0)
                self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=7, padding=0)

                size_input_low_road_FC = 512 + 512
            else:
                size_input_low_road_FC = 960

            self.low_road_output_size = 512
            self.amygdala_low_FC = self.LA_low_road_FC(size_input_low_road_FC)



        self.vgg_highfea_part = self.vgg

        self.vgg_highfea_part.classifier = nn.Sequential(
            *list(self.vgg.children())[2][:(self.highfea_VGGlayer - 30 - 7)])



        self.input_size_CE = 4096+512

        self.amygdala_CE = self.CE(self.input_size_CE)





    def LA_low_road_FC(self, size_input):
        amygdala_low = nn.Sequential(
            nn.Linear(size_input, 1024), nn.ReLU(inplace=True),nn.Dropout(p=0.5),
            nn.Linear(1024,512), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
        )
        return amygdala_low

    def CE(self, input_size_CE):
        amygdala_CE = nn.Sequential(
            nn.Linear(input_size_CE, 1024), nn.ReLU(inplace=True),nn.Dropout(p=0.5),
            nn.Linear(1024, 1024), nn.ReLU(inplace=True),nn.Dropout(p=0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

        return amygdala_CE



    def forward(self, x):
        if self.is_gist:
            input_orinal = x[0]
            input_gist = x[1]
        else:
            input_orinal = x

        y_high = self.vgg_highfea_part(input_orinal)

        if not self.is_highroad_only:

            if not self.is_gist:
                #************************* low road ************************************#
                y_low_raw0 = self.vgg_lowfea_part(x)

                #####SPP##########

                y_low_raw0_mp = self.lowroad_maxpool(y_low_raw0)

                y_low_raw0_gmp2 = self.lowroad_gmp2(y_low_raw0)


                y_low_raw0_pool1 = self.maxpool1(y_low_raw0)
                y_low_raw0_pool1_att = self.lowroad_gap2(y_low_raw0_pool1)

                y_low_raw0_pool2 = self.maxpool2(y_low_raw0)
                y_low_raw0_pool2_att = self.lowroad_gap2(y_low_raw0_pool2)

                y_low_raw0_pool3 = self.maxpool3(y_low_raw0)
                y_low_raw0_pool3_att = self.lowroad_gap2(y_low_raw0_pool3)

                y_low = torch.cat([y_low_raw0_gmp2, y_low_raw0_pool1_att,
                                                 y_low_raw0_pool2_att, y_low_raw0_pool3_att], dim=1)

                #####attention##########

                y_low = self.attention_eca(y_low)

                y_low = self.lowroad_gap1(y_low)
                y_low = y_low.view(y_low.size(0), -1)

                y_low_raw0_mp = y_low_raw0_mp.view(y_low_raw0_mp.size(0), -1)

                y_low = torch.cat([y_low, y_low_raw0_mp], dim=1)
            else:
                if input_gist is not None:
                    y_low = input_gist
                else:
                    print('Gist input is not valid!')
                    return
            #####low road preprocess n##########
            y_low = self.amygdala_low_FC(y_low)
        else:
            y_low = torch.zeros([y_high.shape[0],512]).cuda()
        #************************* High road ************************************#

        # ************************* integrate ************************************#
        flatten_integrated_feature = torch.cat([y_low, y_high], dim=1)

        # ************************* prediction ************************************#
        # CE network
        regression = self.amygdala_CE(flatten_integrated_feature)

        return regression

class FeatureExtractor(nn.Module):
    def __init__(self, model, layer_name, submodel_name):
        super(FeatureExtractor, self).__init__()
        self.layer_name = layer_name
        self.submodel_name = submodel_name


        modules = []

        if submodel_name == 'vgg_ori':
            submodule = model
            modules.append(list(list(submodule.children())[0].children())[0][:self.layer_name+1])
            modules = nn.Sequential(*modules)
        elif submodel_name == 'vgg':
            submodule = copy.deepcopy(model.vgg)
            if layer_name < 30:
                modules = submodule.features[:self.layer_name+1]
            else:
                modules = submodule
                if self.layer_name <= 36:
                    modules.classifier = nn.Sequential(
                        *list(modules.children())[2][:(self.layer_name - 30 - 7)])
        elif submodel_name == 'amygdala_middle':

            self.vgg_lowfea_part =  model.vgg_lowfea_part

            self.amygdala_low = model.amygdala_low

            modules = self.amygdala_low[:self.layer_name + 1]
        elif submodel_name == 'amygdala_LA':

            self.vgg_lowfea_part = model.vgg_lowfea_part
            self.amygdala_low = model.amygdala_low_FC

            #####
            self.lowroad_maxpool = nn.MaxPool2d(kernel_size=29, stride=14, padding=0, dilation=1, ceil_mode=False)

            self.lowroad_gap1 = nn.AdaptiveAvgPool2d(1)
            self.lowroad_gap2 = nn.AdaptiveAvgPool2d(2)

            self.lowroad_gmp2 = nn.AdaptiveMaxPool2d(2)

            self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=3, padding=0)
            self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=5, padding=0)
            self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=7, padding=0)

            self.attention_eca = eca_layer()
            #####

            self.vgg_highfea_part = model.vgg_highfea_part
            self.vgg_highfea_part.classifier = nn.Sequential(
                *list(self.vgg_highfea_part.children())[2][:(-1)])
        elif submodel_name == 'amygdala_CE':

            self.vgg_lowfea_part = model.vgg_lowfea_part
            self.amygdala_low = model.amygdala_low_FC

            #####
            self.lowroad_maxpool = nn.MaxPool2d(kernel_size=29, stride=14, padding=0, dilation=1, ceil_mode=False)

            self.lowroad_gap1 = nn.AdaptiveAvgPool2d(1)
            self.lowroad_gap2 = nn.AdaptiveAvgPool2d(2)

            self.lowroad_gmp2 = nn.AdaptiveMaxPool2d(2)

            self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=3, padding=0)
            self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=5, padding=0)
            self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=7, padding=0)

            self.attention_eca = eca_layer()
            #####


            self.vgg_highfea_part = model.vgg_highfea_part
            self.vgg_highfea_part.classifier = nn.Sequential(
                *list(self.vgg_highfea_part.children())[2][:(-1)])

            self.amygdala_CE = model.amygdala_CE

            modules = self.amygdala_CE[:self.layer_name + 1]


        self.partial_model = modules

    def intermediate_road_attention(self, x):

        # ************************* low road ************************************#
        y_low_raw0 = self.vgg_lowfea_part(x)

        #####SPP##########

        y_low_raw0_mp = self.lowroad_maxpool(y_low_raw0)

        y_low_raw0_gmp2 = self.lowroad_gmp2(y_low_raw0)

        y_low_raw0_pool1 = self.maxpool1(y_low_raw0)
        y_low_raw0_pool1_att = self.lowroad_gap2(y_low_raw0_pool1)

        y_low_raw0_pool2 = self.maxpool2(y_low_raw0)
        y_low_raw0_pool2_att = self.lowroad_gap2(y_low_raw0_pool2)

        y_low_raw0_pool3 = self.maxpool3(y_low_raw0)
        y_low_raw0_pool3_att = self.lowroad_gap2(y_low_raw0_pool3)

        y_low = torch.cat([y_low_raw0_gmp2, y_low_raw0_pool1_att,
                           y_low_raw0_pool2_att, y_low_raw0_pool3_att], dim=1)

        #####attention##########

        y_low = self.attention_eca(y_low)

        y_low = self.lowroad_gap1(y_low)
        y_low = y_low.view(y_low.size(0), -1)

        y_low_raw0_mp = y_low_raw0_mp.view(y_low_raw0_mp.size(0), -1)

        y_low = torch.cat([y_low, y_low_raw0_mp], dim=1)
        return  y_low

    def forward(self, x):
        if self.submodel_name in ['vgg', 'vgg_ori']:

            output = self.partial_model(x)

        elif self.submodel_name == 'amygdala_middle':

            y_low = self.vgg_lowfea_part(x)
            y_low = y_low.view(y_low.size(0), -1)
            output = self.partial_model(y_low)

        elif self.submodel_name == 'amygdala_LA':

            y_low_VGG = self.intermediate_road_attention(x)

            y_low = self.amygdala_low(y_low_VGG)

            y_high = self.vgg_highfea_part(x)

            output = torch.cat([y_low, y_high], dim=1)

        elif self.submodel_name == 'amygdala_CE':

            y_low_VGG = self.intermediate_road_attention(x)

            y_low =  self.amygdala_low(y_low_VGG)

            y_high = self.vgg_highfea_part(x)

            flatten_integrated_feature = torch.cat([y_low, y_high], dim=1)

            output = self.partial_model(flatten_integrated_feature)


        return output










