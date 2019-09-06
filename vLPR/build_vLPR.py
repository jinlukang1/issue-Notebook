import torch
from torch import nn
import sys
sys.path.insert(0, '/ghome/jinlk/jinlukang/vLPR/vLPR_END')
from model.build_contextpath import build_contextpath

provNum, alphaNum, adNum = 31, 24, 34
class ConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))


class Spatial_path(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=3, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock3 = ConvBlock(in_channels=128, out_channels=256)

    def forward(self, input):
        x = self.convblock1(input)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x


class AttentionRefinementModule(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        # x = self.sigmoid(x)
        x = self.sigmoid(self.bn(x)) # now is x = self.sigmoid(x)
        # channels of input and x should be same
        x = torch.mul(input, x)
        return x


class FeatureFusionModule(torch.nn.Module):

    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = self.avgpool(feature)
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        # x = self.sigmoid(self.conv2(x)) # x = self.sigmoid(self.relu(self.conv2(x)))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x


class BiSeNet(torch.nn.Module):

    def __init__(self, num_classes, num_char, context_path):
        super().__init__()
        # build spatial path
        self.saptial_path = Spatial_path()

        # build context path
        self.context_path = build_contextpath(name=context_path)

        # build attention refinement module
        if context_path == 'resnet101':
            self.attention_refinement_module1 = AttentionRefinementModule(1024, 1024)
            self.attention_refinement_module2 = AttentionRefinementModule(2048, 2048)
            self.feature_fusion_module1 = FeatureFusionModule(num_classes, 3328)
            self.feature_fusion_module2 = FeatureFusionModule(num_char, 3328)
        else:
            self.attention_refinement_module1 = AttentionRefinementModule(256, 256)
            self.attention_refinement_module2 = AttentionRefinementModule(512, 512)
            self.feature_fusion_module1 = FeatureFusionModule(num_classes, 1024)
            self.feature_fusion_module2 = FeatureFusionModule(num_char, 1024)
        # build final convolution
        self.conv_seg = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)
        self.conv_pos = nn.Conv2d(in_channels=num_char, out_channels=num_char, kernel_size=1)

        self.convblock = nn.Sequential(
            ConvBlock(in_channels=num_classes-1, out_channels=1000, kernel_size=5, stride=4, padding=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            )
        # self.convblock1 = nn.Sequential(
        #     ConvBlock(in_channels=num_classes-1, out_channels=256, kernel_size=5, stride=4, padding=2),
        #     ConvBlock(in_channels=256, out_channels=1000, kernel_size=5, stride=4, padding=2),
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     )
        # self.convblock2 = nn.Sequential(
        #     ConvBlock(in_channels=num_classes-42, out_channels=1000, kernel_size=5, stride=4, padding=2),
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     )
        # self.convblock3 = nn.Sequential(
        #     ConvBlock(in_channels=num_classes-32, out_channels=1000, kernel_size=5, stride=4, padding=2),
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     )
      
        self.classifier1 = nn.Linear(1000, provNum)
        self.classifier2 = nn.Linear(1000, alphaNum)
        self.classifier3 = nn.Linear(1000, adNum)
        self.classifier4 = nn.Linear(1000, adNum)
        self.classifier5 = nn.Linear(1000, adNum)
        self.classifier6 = nn.Linear(1000, adNum)
        self.classifier7 = nn.Linear(1000, adNum)

    def forward(self, input, label=None):
        # output of spatial path
        sx = self.saptial_path(input)
        # output of context path
        cx1, cx2, tail = self.context_path(input)
        cx1 = self.attention_refinement_module1(cx1)
        cx2 = self.attention_refinement_module2(cx2)
        cx2 = torch.mul(cx2, tail)
        # upsampling
        cx1 = torch.nn.functional.upsample(cx1, size=(sx.shape[2], sx.shape[3]), mode='bilinear', align_corners=True)
        cx2 = torch.nn.functional.upsample(cx2, size=(sx.shape[2], sx.shape[3]), mode='bilinear', align_corners=True)
        cx = torch.cat((cx1, cx2), dim=1)

        # output of feature fusion module
        segmap = self.feature_fusion_module1(sx, cx)
        posmap = self.feature_fusion_module2(sx, cx)

        segmap = torch.nn.functional.upsample(segmap, size=(input.shape[2], input.shape[3]), mode='bilinear', align_corners=True)
        segmap = self.conv_seg(segmap)
        posmap = torch.nn.functional.upsample(posmap, size=(input.shape[2], input.shape[3]), mode='bilinear', align_corners=True)
        posmap = self.conv_pos(posmap)
        posmap_s = nn.functional.softmax(posmap, dim=1)
        segmap_s = nn.functional.softmax(segmap, dim=1)

        out = []
        # seg_pos = segmap_s.narrow(1, 35, 31).mul(posmap_s.narrow(1, 1, 1))
        # out.append(seg_pos)
        # seg_pos = segmap_s.narrow(1, 11, 24).mul(posmap_s.narrow(1, 2, 1))
        # out.append(seg_pos)
        # for i in range(posmap_s.size()[1] - 3):
        #     seg_pos = segmap_s.narrow(1, 1, 34).mul(posmap_s.narrow(1, i+3, 1))#mod
            # out.append(seg_pos)
        for i in range(posmap_s.size()[1] - 1):
            seg_pos = segmap_s.narrow(1, 1, 65).mul(posmap_s.narrow(1, i+1, 1))#mod
            out.append(seg_pos)

        cls_feat1 = self.convblock(out[0])
        cls_feat2 = self.convblock(out[1])
        cls_feat3 = self.convblock(out[2])
        cls_feat4 = self.convblock(out[3])
        cls_feat5 = self.convblock(out[4])
        cls_feat6 = self.convblock(out[5])
        cls_feat7 = self.convblock(out[6])

        y0 = self.classifier1(cls_feat1.view(cls_feat1.size()[0], -1))
        y1 = self.classifier2(cls_feat2.view(cls_feat1.size()[0], -1))
        y2 = self.classifier3(cls_feat3.view(cls_feat1.size()[0], -1))
        y3 = self.classifier4(cls_feat4.view(cls_feat1.size()[0], -1))
        y4 = self.classifier5(cls_feat5.view(cls_feat1.size()[0], -1))
        y5 = self.classifier6(cls_feat6.view(cls_feat1.size()[0], -1))
        y6 = self.classifier7(cls_feat7.view(cls_feat1.size()[0], -1))
        # print(out.shape)

        return segmap, posmap, [y0, y1, y2, y3, y4, y5, y6]
            
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'

    model = BiSeNet(num_classes=36, num_char=8, context_path='resnet101')
    # print(model)
    model = nn.DataParallel(model)

    model = model.cuda()
    for name, key in model.named_parameters():
        print(name)
    x = torch.rand(2, 3, 50, 160)
    record = model.parameters()
    y = model(x)
    print(y.shape)
