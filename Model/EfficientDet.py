import tensorboardX
import torch.nn as nn
import torch
from torchvision.ops.boxes import nms as nms_torch
import numpy as np
from EfficientNet import EfficientNet as EffNet
from util import MemoryEfficientSwish, Swish
from util import Conv2dSamePadding, MaxPool2dSamePadding, Anchors
import json
import os

def nms(dets, thresh):
    return nms_torch(dets[:, :4], dets[:, 4], thresh)


class Separable_Conv_Block(nn.Module):
    def __init__(self,in_channels,out_channels=None,norm=True,activation=False,onnx_export=False):
        super(Separable_Conv_Block,self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwiseconv = Conv2dSamePadding(in_channels,in_channels,kernel_size=3,
                                                    stride=1,groups=in_channels,bias=False)
        self.pointwiseconv = Conv2dSamePadding(in_channels,out_channels,kernel_size=1,
                                                    stride=1)

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(out_channels,momentum=0.9,eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self,x):
        x = self.depthwiseconv(x)
        x = self.pointwiseconv(x)
        if self.norm:
            x = self.bn(x)
        if self.activation:
            x = self.swish(x)
        return x


class BiFPN(nn.Module):
    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True):
        super(BiFPN,self).__init__()
        self.eps = epsilon

        self.conv6_up = Separable_Conv_Block(num_channels,onnx_export=onnx_export)
        self.conv5_up = Separable_Conv_Block(num_channels,onnx_export=onnx_export)
        self.conv4_up = Separable_Conv_Block(num_channels,onnx_export=onnx_export)
        self.conv3_up = Separable_Conv_Block(num_channels,onnx_export=onnx_export)

        self.conv4_down = Separable_Conv_Block(num_channels,onnx_export=onnx_export)
        self.conv5_down = Separable_Conv_Block(num_channels,onnx_export=onnx_export)
        self.conv6_down = Separable_Conv_Block(num_channels,onnx_export=onnx_export)
        self.conv7_down = Separable_Conv_Block(num_channels,onnx_export=onnx_export)

        self.p6_upsample = nn.Upsample(scale_factor=2,mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2,mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2,mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2,mode='nearest')

        self.p4_downsample = MaxPool2dSamePadding(3,2)
        self.p5_downsample = MaxPool2dSamePadding(3,2)
        self.p6_downsample = MaxPool2dSamePadding(3,2)
        self.p7_downsample = MaxPool2dSamePadding(3,2)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        self.first_time = first_time

        if self.first_time:
            self.p5_down_channel = nn.Sequential(Conv2dSamePadding(conv_channels[2],num_channels,1),
                                                    nn.BatchNorm2d(num_channels,momentum=0.01,eps=1e-3))
            
            self.p4_down_channel = nn.Sequential(Conv2dSamePadding(conv_channels[1],num_channels,1),
                                                    nn.BatchNorm2d(num_channels,momentum=0.01,eps=1e-3))

            self.p3_down_channel = nn.Sequential(Conv2dSamePadding(conv_channels[0],num_channels,1),
                                                    nn.BatchNorm2d(num_channels,momentum=0.01,eps=1e-3))
            self.p5_to_p6 = nn.Sequential(
                Conv2dSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                MaxPool2dSamePadding(3, 2)
            )
            self.p6_to_p7 = nn.Sequential(
                MaxPool2dSamePadding(3, 2)
            )

            self.p4_down_channel_2 = nn.Sequential(
                Conv2dSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel_2 = nn.Sequential(
                Conv2dSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )


        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()

        self.attention = attention


    def forward(self,x):

        if self.attention:
            p3_out,p4_out,p5_out,p6_out,p7_out = self._fast_forward_attention(x)
        else:
             p3_out,p4_out,p5_out,p6_out,p7_out = self._forward(x)

        return p3_out,p4_out,p5_out,p6_out,p7_out

    def _fast_forward_attention(self,x):
        if self.first_time:
            p3,p4,p5 = x

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)
        else:
            p3_in,p4_in,p5_in,p6_in,p7_in = x


        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1/(torch.sum(p6_w1,dim=0)+self.eps)
        p6_up = self.conv6_up(self.swish(weight[0]*p6_in+weight[1]*self.p6_upsample(p7_in)))

        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1/(torch.sum(p5_w1,dim=0)+self.eps)
        p5_up = self.conv5_up(self.swish(weight[0]*p5_in+weight[1]*self.p5_upsample(p6_up)))

        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1/(torch.sum(p4_w1,dim=0)+self.eps)
        p4_up = self.conv4_up(self.swish(weight[0]*p4_in+weight[1]*self.p4_upsample(p5_up)))

        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1/(torch.sum(p3_w1,dim=0)+self.eps)
        p3_out = self.conv3_up(self.swish(weight[0]*p3_in+weight[1]*self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2/(torch.sum(p4_w2,dim=0)+self.eps)
        p4_out = self.conv4_down(self.swish(weight[0]*p4_in+weight[1]*p4_up+self.p4_downsample(p3_out)))

        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2/(torch.sum(p5_w2,dim=0)+self.eps)
        p5_out = self.conv5_down(self.swish(weight[0]*p5_in+weight[1]*p5_up+self.p5_downsample(p4_out)))

        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2/(torch.sum(p6_w2,dim=0)+self.eps)
        p6_out = self.conv6_down(self.swish(weight[0]*p6_in+weight[1]*p6_up+self.p6_downsample(p5_out)))

        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.eps)
        p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward(self,x):
        if self.first_time:
            p3,p4,p5 = x
            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p3_down_channel(p4)
            p5_in = self.p3_down_channel(p5)
        else:
            p3_in,p4_in,p5_in,p6_in,p7_in = x


        p6_up = self.conv6_up(self.swish(p6_in+self.p6_upsample(p7_in)))
        p5_up = self.conv5_up(self.swish(p5_in+self.p5_upsample(p6_up)))
        p4_up = self.conv4_up(self.swish(p4_in+self.p4_upsample(p5_up)))
        p3_out = self.conv3_up(self.swish(p3_in+self.p3_upsample(p4_up)))
        
        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        p4_out = self.conv4_down(self.swish(p4_in+p4_up+self.p4_downsample(p3_out)))
        p5_out = self.conv5_down(self.swish(p5_in+p5_up+self.p5_downsample(p4_out)))
        p6_out = self.conv6_down(self.swish(p6_in+p6_up+self.p6_downsample(p5_out)))
        p7_out = self.conv7_down(self.swish(p7_in+self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out

class UpsampleBlock(nn.Module):
    """
    A block to upscale the feature map with transpose convolutions.
    Each pass, doubles the map edge size (quadropouls resolution).
    This is used to build the hour glass decoder from different resolutions. 
    """
    def __init__(self,in_channels,scale_factor):
        super(UpsampleBlock,self).__init__()
        self.sf = scale_factor
        self.upscale = torch.nn.ConvTranspose2d(in_channels,in_channels,kernel_size=2,stride=2)
        self.conv = Separable_Conv_Block(in_channels, in_channels, norm=False, activation=False)
    
    def forward(self,input):
        for i in range(self.sf):
            input = self.upscale(input)
            input = self.conv(input)
        return input

class PoseMap(nn.Module):
    def __init__(self,in_channels,num_layers,onnx_export=False):
        super(PoseMap,self).__init__()
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.conv_list = nn.ModuleList(
            [Separable_Conv_Block(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(3)])
        self.up_list = nn.ModuleList([UpsampleBlock(1,scale_factor=i) for i in [2,3,4]])
        self.header = Separable_Conv_Block(in_channels, 1, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()
        self.final = Separable_Conv_Block(3,1,norm=False,activation=False)
        self.swish2 = MemoryEfficientSwish() if not onnx_export else Swish()
        self.down_sample = torch.nn.Conv2d(1,1,1,2)
    
    def forward(self,inputs):
        feats = []
        device = inputs[0].device
        prev_map = torch.zeros((inputs[0].shape[0],1,inputs[0].shape[2],inputs[0].shape[3])).to(device)
        for feat, bn_list, up in zip(inputs[:3], self.bn_list,self.up_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            
            feat = self.header(feat)
            scale = prev_map.shape[-1]//feat.shape[-1]
            for j in range(scale-1):
                prev_map = self.down_sample(prev_map)
            feat = torch.add(feat,prev_map)
            prev_map = feat
            feat = up(feat)
            feats.append(feat)
 
        pmap = torch.cat(feats,dim=1)
        pmap = self.final(pmap)
        pmap = self.swish2(pmap)
        return pmap

class PoseMap2(nn.Module):
    "UNET-type regressor"
    def __init__(self,in_channels,onnx_export = False):
        super(PoseMap2,self).__init__()
                
        self.conv_11 = Separable_Conv_Block(in_channels[0],activation=True)
        self.conv_12 = Separable_Conv_Block(in_channels[0],activation=True)

        self.conv_21 = Separable_Conv_Block(in_channels[1],activation=True)
        self.conv_22 = Separable_Conv_Block(in_channels[1],activation=True)

        self.conv_31 = Separable_Conv_Block(in_channels[2],activation=True)
        self.conv_32 = Separable_Conv_Block(in_channels[2],activation=True)

        self.conv_41 = Separable_Conv_Block(in_channels[3],activation=True)
        self.conv_42 = Separable_Conv_Block(in_channels[3],activation=True)

        self.up = torch.nn.Upsample(scale_factor=2,mode='nearest')
        self.conv_3u2 = Separable_Conv_Block(in_channels[3],40,activation=True)
        self.conv_2u2 = Separable_Conv_Block(40,40,activation=True)
        self.conv_2u1 = Separable_Conv_Block(in_channels[2]+40,32,activation=True)
        self.conv_1u1 = Separable_Conv_Block(32,32,activation=True)
        self.conv_1u0 = Separable_Conv_Block(in_channels[1]+32,16,activation=True)
        self.conv_0u0 = Separable_Conv_Block(16,16,activation=True)
        self.conv_0 = Separable_Conv_Block(3+16,1,activation=True)
        self.final = Separable_Conv_Block(1,1,norm=False,activation=False)
        self.sigm = torch.nn.ReLU()
        
    def forward(self,inputs):
        l0,l1,l2,l3 = inputs
        l0 = self.conv_11(l0)
        l0 = self.conv_12(l0)
        l1 = self.conv_21(l1)
        l1 = self.conv_22(l1)
        l2 = self.conv_31(l2)
        l2 = self.conv_32(l2)
        l3 = self.conv_41(l3)
        l3 = self.conv_42(l3)
        l3 = self.conv_3u2(l3)
        l3 = self.conv_2u2(l3)

        out = torch.cat((self.up(l3),l2),dim = 1)
        out = self.conv_2u1(out)
        out = self.conv_1u1(out)
        out = torch.cat((self.up(out),l1),dim=1)
        out = self.conv_1u0(out)
        out = self.conv_0u0(out)
        out = torch.cat((self.up(out),l0),dim=1)
        out = self.conv_0(out)
        out = self.final(out)
        out = self.sigm(out)
        return out
        

class PoseMap3(nn.Module):
    "UNET-type regressor"
    def __init__(self,onnx_export = False):
        super(PoseMap3,self).__init__()
        
        self.preconv1 = torch.nn.ModuleList([Separable_Conv_Block(64,activation=True) for i in range(5)])
        self.preconv2 = torch.nn.ModuleList([Separable_Conv_Block(64,activation=True) for i in range(5)])

        self.up5_4 = torch.nn.Upsample(scale_factor = 2, mode = "nearest")
        self.conv5pre = Separable_Conv_Block(64,activation=True)
        self.conv5post = Separable_Conv_Block(128,64,activation=True)

        self.up4_3 = torch.nn.Upsample(scale_factor = 2, mode = "nearest")
        self.conv4pre = Separable_Conv_Block(64,activation=True)
        self.conv4post = Separable_Conv_Block(128,64,activation=True)
        
        self.up3_2 = torch.nn.Upsample(scale_factor = 2, mode = "nearest")
        self.conv3pre = Separable_Conv_Block(64,activation=True)
        self.conv3post = Separable_Conv_Block(128,64,activation=True)

        self.up2_1 = torch.nn.Upsample(scale_factor = 2, mode = "nearest")
        self.conv2pre = Separable_Conv_Block(64,activation=True)
        self.conv2post = Separable_Conv_Block(128,64,activation=True)

        self.up1_0 = torch.nn.Upsample(scale_factor = 2, mode = "nearest")
        self.conv1pre = Separable_Conv_Block(64,activation=True)
        self.conv1post = Separable_Conv_Block(64,64,activation=True)

        self.preconv = torch.nn.ModuleList([Separable_Conv_Block(i,i//2,activation=True) for i in [64,32]])
        self.postconv = torch.nn.ModuleList([Separable_Conv_Block(i,i,activation=True) for i in [32,16]])
        self.up = torch.nn.ModuleList([torch.nn.Upsample(scale_factor=2,mode="nearest") for i in range(2)])
        self.last = Separable_Conv_Block(16,1,activation=True,norm=False)


    def forward(self,inputs):
        
        inpts = [*inputs]
        for i in range(len(inpts)):
            inpts[i] = self.preconv1[i](inpts[i])
            inpts[i] = self.preconv2[i](inpts[i])
        p1,p2,p3,p4,p5 = inpts
        p5 = self.up5_4(p5)
        p5 = self.conv5pre(p5)
        p4 = torch.cat((p4,p5),dim=1)
        p4 = self.conv5post(p4)
        p4 = self.up4_3(p4)
        p4 = self.conv4pre(p4)
        p3 = torch.cat((p3,p4),dim=1)
        p3 = self.conv4post(p3)
        p3 = self.up3_2(p3)
        p3 = self.conv3pre(p3)
        p2 = torch.cat((p3,p2),dim=1)
        p2 = self.conv3post(p2)
        p2 = self.up2_1(p2)
        p2 = self.conv2pre(p2)
        p1 =  torch.cat((p2,p1),dim=1)
        p1 = self.conv2post(p1)
        p1 = self.up1_0(p1)
        p1 = self.conv1pre(p1)
        
        out = self.conv1post(p1)
        for c1,c2,up in zip(self.preconv,self.postconv,self.up):
            out = c1(out)
            out = up(out)
            out = c2(out)
        
        out = self.last(out)
        return out


        

        return out


        

class Regressor(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_channels, num_anchors, num_layers,reg_points=4, onnx_export=False):
        super(Regressor, self).__init__()
        self.num_layers = num_layers
        self.reg_points = reg_points # number of regression points
        self.conv_list = nn.ModuleList(
            [Separable_Conv_Block(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(5)])
        self.header = Separable_Conv_Block(in_channels, num_anchors * self.reg_points, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], -1, self.reg_points)

            feats.append(feat)

        feats = torch.cat(feats, dim=1)

        return feats

class Classifier(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_channels, num_anchors, num_classes, num_layers, onnx_export=False):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.conv_list = nn.ModuleList(
            [Separable_Conv_Block(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(5)])
        self.header = Separable_Conv_Block(in_channels, num_anchors * num_classes, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], feat.shape[1], feat.shape[2], self.num_anchors,
                                          self.num_classes)
            feat = feat.contiguous().view(feat.shape[0], -1, self.num_classes)

            feats.append(feat)

        feats = torch.cat(feats, dim=1)
        feats = feats.sigmoid()

        return feats

class EfficientNet(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, compound_coef, load_weights=False):
        super(EfficientNet, self).__init__()
        model = EffNet.from_pretrained(f'efficientnet-b{compound_coef}', load_weights)
        del model.conv_head
        del model.bn1
        del model.avg_pooling
        del model.dropout
        del model.fc
        self.model = model

    def forward(self, x):
        x = self.model.conv_stem(x)
        x = self.model.bn0(x)
        x = self.model.swish(x)
        feature_maps = []

        # TODO: temporarily storing extra tensor last_x and del it later might not be a good idea,
        #  try recording stride changing when creating efficientnet,
        #  and then apply it here.
        last_x = None
        for idx, block in enumerate(self.model.blocks):
            drop_connect_rate = self.model.global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model.blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            if block.depthwise_conv.stride == [2, 2]:
                feature_maps.append(last_x)
            elif idx == len(self.model.blocks) - 1:
                feature_maps.append(x)
            last_x = x
        del last_x
        return feature_maps[:]


class EfficientDetBackbone(nn.Module):
    def __init__(self, num_classes=80, compound_coef=0, load_weights=False, **kwargs):
        super(EfficientDetBackbone, self).__init__()
        self.compound_coef = compound_coef

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5.]
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
        }

        num_anchors = len(self.aspect_ratios) * self.num_scales

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True if compound_coef < 6 else False)
              for _ in range(self.fpn_cell_repeats[compound_coef])])

        self.num_classes = num_classes
        self.regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                   num_layers=self.box_class_repeats[self.compound_coef])
        self.classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                     num_classes=num_classes,
                                     num_layers=self.box_class_repeats[self.compound_coef])

        self.anchors = Anchors(anchor_scale=self.anchor_scale[compound_coef], **kwargs)

        self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef], load_weights)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        max_size = inputs.shape[-1]

        _, p3, p4, p5 = self.backbone_net(inputs)

        features = (p3, p4, p5)
        features = self.bifpn(features)

        regression = self.regressor(features)
        classification = self.classifier(features)
        anchors = self.anchors(inputs, inputs.dtype)

        return features, regression, classification, anchors

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')



class EfficientDetMultiBackbone(nn.Module):
    """
    Efficientdet with Multihead output
    """
    def __init__(self,
                root_dir, 
                compound_coef=0,
                heads=['person','face','pose','face_landmarks','age','gender','emotion','skin'],
                load_weights=False, 
                **kwargs):
        super(EfficientDetMultiBackbone, self).__init__()
        self.compound_coef = compound_coef
        self.heads = heads
        self.root_dir = root_dir
        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5.]
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
        }

        num_anchors = len(self.aspect_ratios) * self.num_scales

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True if compound_coef < 6 else False)
              for _ in range(self.fpn_cell_repeats[compound_coef])])

        if len(self.heads) == 0:
            print('Detecting persons only')
        with open(os.path.join(self.root_dir,'config.json'),'r') as f:
            config = json.load(f)
        self.gender_calsses  = len(config['gender'])
        self.race_classes    = len(config['race'])
        self.emotion_classes = len(config['emotions'])
        self.skin_classes    = len(config['skin'])
        
        
        self.gender_classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                     num_classes=self.gender_calsses,
                                     num_layers=self.box_class_repeats[self.compound_coef])
        if 'race' in self.heads:
            self.race_classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                     num_classes=self.race_classes,
                                     num_layers=self.box_class_repeats[self.compound_coef])
        if 'emotions' in self.heads:
            self.emotion_classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                     num_classes=self.emotion_classes,
                                     num_layers=self.box_class_repeats[self.compound_coef])
        if 'skin' in self.heads:
            self.skin_classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                     num_classes=self.skin_classes,
                                     num_layers=self.box_class_repeats[self.compound_coef])
        

        self.pbox_regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                   num_layers=self.box_class_repeats[self.compound_coef])
        if 'face' in self.heads:
            self.fbox_regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                   num_layers=self.box_class_repeats[self.compound_coef])
        if 'pose' in self.heads:
            self.pose_regressor = PoseMap3()
        if 'face_landmarks' in self.heads:
            #self.fl_regressor = PoseMap2(in_channels=[3,16,24,40])
            self.fl_regressor = PoseMap3()
        if 'age' in self.heads:
            self.age_regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,reg_points=1,
                                   num_layers=self.box_class_repeats[self.compound_coef])
        

        self.anchors = Anchors(anchor_scale=self.anchor_scale[compound_coef], **kwargs)

        self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef], load_weights)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        out = {'person':None,'anchors':None}
        max_size = inputs.shape[-1]
        p0 = inputs
        p1, p2, p3, p4, p5 = self.backbone_net(inputs)

        features = (p3, p4, p5)
        features = self.bifpn(features)
        out['features'] = features        
        person_bbox = self.pbox_regressor(features)
        out['person'] = person_bbox
        if 'face' in self.heads:
            face_bbox = self.fbox_regressor(features)
            out['face'] = face_bbox
        if 'pose' in self.heads:
            pose = self.pose_regressor(features)
            out['pose'] = pose
        if 'age' in self.heads:
            age = self.age_regressor(features)
            out['age'] = age
        if "face_landmarks" in self.heads:
            #face_landmarks = self.fl_regressor((p0,p1,p2,p3))
            face_landmarks = self.fl_regressor(features)
            out['face_landmarks'] = face_landmarks
        gender  = self.gender_classifier(features)
        out['gender'] = gender
        if 'race' in self.heads:
            race = self.race_classifier(features)
            out['race'] = race
        if 'skin' in self.heads:
            skin = self.skin_classifier(features)
            out['skin'] = skin
        if 'emotions' in self.heads:
            emotion = self.emotion_classifier(features)
            out['emotions'] = emotion

        anchors = self.anchors(inputs, inputs.dtype)
        out['anchors'] = anchors
        return out

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')


if __name__ == '__main__':
    """
    p3 = torch.randn((3,10,64,64))
    p4 = torch.randn((3,10,32,32))
    p5 = torch.randn((3,10,16,16))
    summary = tensorboardX.SummaryWriter(logdir='logs',filename_suffix=f'POSEMAP',comment='try1')
    pm = PoseMap(10,3)
    out = pm([p3,p4,p5])
    summary.add_graph(pm,input_to_model=[[p3,p4,p5]],verbose=True)
    summary.close()
    print(out.shape)
    """
    p1 = torch.randn((3,64,64,64))
    p2 = torch.randn((3,64,32,32))
    p3 = torch.randn((3,64,16,16))
    p4 = torch.randn((3,64,8,8))
    p5 = torch.randn((3,64,4,4))
    #p5 = torch.randn((3,320,16,16))
    pm = PoseMap3()
    out = pm([p1,p2,p3,p4,p5])
    print(out.shape)
    #summary = tensorboardX.SummaryWriter(logdir='logs',filename_suffix=f'POSEMAP2',comment='try1')
    #summary.add_graph(pm,input_to_model=[[p0,p1,p2,p3]],verbose=True)
    #summary.close()
    