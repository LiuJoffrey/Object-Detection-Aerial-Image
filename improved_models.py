import torch.nn as nn
import math
import torch
import torch.utils.model_zoo as model_zoo
from torchvision import datasets, transforms, models
import torch.nn.functional as F

__all__ = ['vgg16_bn']
model_urls = {
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}


class VGG(nn.Module):

    def __init__(self, features, output_size=1274, image_size=448):
        super(VGG, self).__init__()
        self.features = features
        self.image_size = image_size

        self.yolo = nn.Sequential(
            #TODO
            #nn.Linear(in_features=7*7*512, out_features=4096),
            nn.Linear(in_features=7*7*512, out_features=4096),
            #nn.BatchNorm1d(4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=4096, out_features=7 * 7 * (2 * 5 + 16))

        )
        
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        print(x.shape)
        exit()
        x = x.view(x.size(0), -1) # (batch, 25088)
        x = self.yolo(x)
        x = torch.sigmoid(x) 
        x = x.view(-1,7,7,26)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    s = 1
    first_flag=True
    for v in cfg:
        s=1
        if (v==64 and first_flag):
            s=2
            first_flag=False
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=s, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def conv_bn_relu(in_channels,out_channels,kernel_size=3,stride=2,padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding,stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )


cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}




def Yolov1_vgg16bn(pretrained=False, **kwargs):
    """
    VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    yolo = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        vgg_state_dict = model_zoo.load_url(model_urls['vgg16_bn'])
        yolo_state_dict = yolo.state_dict()
        for k in vgg_state_dict.keys():
            if k in yolo_state_dict.keys() and k.startswith('features'):
                yolo_state_dict[k] = vgg_state_dict[k]
    yolo.load_state_dict(yolo_state_dict)
    return yolo






class Resnet(nn.Module):

    def __init__(self, features, output_size=1274, image_size=448):
        super(Resnet, self).__init__()
        self.features = features
        self.image_size = image_size
        
        self.yolo = nn.Sequential(
            #TODO
            #nn.Linear(in_features=7*7*512, out_features=4096),
            nn.Linear(in_features=14*14*512, out_features=4096),
            #nn.BatchNorm1d(4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=4096, out_features=14 * 14 * (2 * 5 + 16))

        )

        self.detnet = self._make_detnet_layer(2048)

        self.yolo2 = nn.Sequential(
            #TODO
            #nn.Linear(in_features=7*7*512, out_features=4096),
            nn.Conv2d(256, 26, kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(26),
            
            #nn.Linear(in_features=4096, out_features=14 * 14 * (2 * 5 + 16))

        )
        
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.detnet(x)
        
        #x = x.view(x.size(0), -1) # (batch, 25088)
        x = self.yolo2(x)
        
        x = torch.sigmoid(x) 
        x = x.permute(0, 2,3,1)
        
        #x = x.view(-1,7,7,26)
        return x

    def _make_detnet_layer(self,in_channels):
        layers = []
        layers.append(detnet_bottleneck(in_planes=in_channels, planes=256, block_type='B'))
        layers.append(detnet_bottleneck(in_planes=256, planes=256, block_type='A'))
        layers.append(detnet_bottleneck(in_planes=256, planes=256, block_type='A'))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



def Yolov1_Resnet(pretrained=False, **kwargs):
    resnet50 = models.resnet50(pretrained=pretrained)
    newmodel = torch.nn.Sequential(*(list(resnet50.children())[:-2]))

    yolo = Resnet(newmodel, **kwargs)
    return yolo
    #return resnet152
    #exit()
    #newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))




class detnet_bottleneck(nn.Module):
    # no expansion
    # dilation = 2
    # type B use 1x1 conv
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_type='A'):
        super(detnet_bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=2, bias=False,dilation=2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes or block_type=='B':
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out




def test():
    import torch
    #model = Yolov1_vgg16bn(pretrained=True)
    model = Yolov1_Resnet(pretrained=True)
    img = torch.rand(1,3,448,448)
    output = model(img)
    print(output.size())

if __name__ == '__main__':
    test()
