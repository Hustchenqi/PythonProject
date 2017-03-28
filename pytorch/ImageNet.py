import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(inplanes, outplanes, stride = 1):
    "1x1 convolution without padding"
    return nn.Conv2d(inplanes, outplanes, kernel_size = 1, stride = stride)

# Basic Block
class BasicBlock(nn.Module):

    def __init__(self, planes):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        #self.relu
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual
        return out

class BasicBlock_V1(nn.Module):
    """docstring for BasicBlock_V1"""
    def __init__(self, inoutplanes, midplanes):
        super(BasicBlock_V1, self).__init__()
        self.bn1 = nn.BatchNorm2d(inoutplanes)
        self.relu = nn.ReLU(inplace = True)
        self.conv1 = conv3x3(inoutplanes, midplanes)
        self.bn2 = nn.BatchNorm2d(midplanes)
        #self.relu
        self.conv2 = conv3x3(midplanes, inoutplanes)

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual
        return out

# VOC_Model_V2 first 3 downsample blocks
class DownSampleBlock1(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1):
        super(DownSampleBlock1, self).__init__()
        self.conv1 = conv3x3(inplanes, outplanes, stride = stride)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(outplanes, outplanes)

        self.residual = conv1x1(inplanes, outplanes, stride = stride)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)

        res = self.residual(x)
        out += res
        return out

class DownSampleBlock2(nn.Module):

    def __init__(self, inplanes, outplanes, stride = 1):
        super(DownSampleBlock2, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(inplanes, outplanes)

        self.residual = conv1x1(inplanes, outplanes, stride = stride)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)

        res = self.residual(x)
        out += res
        return out

class Bottleneck(nn.Module):

    def __init__(self, inplanes, midplanes, outplanes):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, midplanes)
        self.bn1 = nn.BatchNorm2d(midplanes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(midplanes, inplanes)
        self.bn2 = nn.BatchNorm2d(inplanes)
        #self.relu
        self.conv3 = conv1x1(inplanes, outplanes)
        self.conv_res = conv1x1(inplanes, outplanes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        residual = self.conv_res(x)

        out += residual
        return out

class ResNet(nn.Module):
    def __init__(self, blocks, layers, class_number = 1000):
        super(ResNet, self).__init__()

        self.top = nn.Sequential(
            conv3x3(3, 64),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
        )
        #### BasicBlock layer1 = 1+2
        self.layer1 = nn.Sequential(
            DownSampleBlock1(64,128,1),
            self._make_layer(blocks[0], 128, layers[0]),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
        )

        #### BasicBlock layer2 = 1+2
        self.layer2 = nn.Sequential(
            DownSampleBlock1(128,256,1),
            self._make_layer(blocks[0], 256, layers[1]),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
        )

        #### BasicBlock layer3 = 1+5
        self.layer3 = nn.Sequential(
            DownSampleBlock1(256,512,1),
            self._make_layer(blocks[0], 512, layers[2]),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
        )

        #### BasicBlockV1 layer4 = 1+2
        self.layer4 = nn.Sequential(
            DownSampleBlock2(512,1024,1),
            self._make_layer_v1(blocks[1], 1024, 512, layers[3]),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace = True),
        )

        #### Bottleneck layer5 + layer6
        self.Bottleneck_layers = nn.Sequential(
            Bottleneck(1024, 512, 2048),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace = True),
            Bottleneck(2048, 1024, 4096),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace = True),
        )
        #### classifier
        self.classifier = nn.AvgPool2d(7)
        self.fc = nn.Linear(4096, class_number)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    #### for Basic
    def _make_layer(self, block, planes, blocks):
        layers = []
        for i in range(blocks):
            layers.append(block(planes))
        return nn.Sequential(*layers)
    #### for BasicV1
    def _make_layer_v1(self, block, inoutplanes, midplanes, blocks):
        layers = []
        for i in range(blocks):
            layers.append(block(inoutplanes, midplanes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.top(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.Bottleneck_layers(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x

def Net(pretrained = False, **kwargs):
    blocks = [BasicBlock, BasicBlock_V1]
    layers = [2,2,5,2]
    model = ResNet(blocks, layers, **kwargs)
    if pretrained:
        model.load_state_dict('model_best.pth')
    return model
