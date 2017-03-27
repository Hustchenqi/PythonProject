import torch
import torch.nn as nn
import torchvision
import torch.utils.model_zoo as model_zoo

model_urls = {
    'alexnet': 'https://s3.amazonaws.com/pytorch/models/alexnet-owt-4df8aa71.pth',
    'vgg16': 'https://s3.amazonaws.com/pytorch/models/vgg16-82412952.pth',
    'vgg19': 'https://s3.amazonaws.com/pytorch/models/vgg19-341d7465.pth',
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth'
}

def alexnet(pretrained = True, num_classes = 10):
    model = torchvision.models.alexnet()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        weights = model.state_dict()
        fc = nn.Linear(4096, num_classes)
        weights['classifier.6.weight'] = fc.weight
        weights['classifier.6.bias'] = fc.bias

        model.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        model.load_state_dict(weights)

    return model

def vgg16(pretrained=True, num_classes = 10):

    model = torchvision.models.vgg16()

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
        fc = nn.Linear(4096, num_classes)
        weights = model.state_dict()
        weights['classifier.6.weight'] = fc.weight
        weights['classifier.6.bias'] = fc.bias

        model.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )

        model.load_state_dict(weights)

    return model

def vgg19(pretrained=True, num_classes = 10):

    model = torchvision.models.vgg19()

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
        fc = nn.Linear(4096, num_classes)
        weights = model.state_dict()
        weights['classifier.6.weight'] = fc.weight
        weights['classifier.6.bias'] = fc.bias

        model.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )

        model.load_state_dict(weights)

    return model

def resnet18(pretrained = True, num_classes = 10):

    model = torchvision.models.resnet18()

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.fc = nn.Linear(512, num_classes)

    return model

def resnet34(pretrained = True, num_classes = 10):

    model = torchvision.models.resnet34()

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    model.fc = nn.Linear(512, num_classes)

    return model

def resnet50(pretrained = True, num_classes = 10):

    model = torchvision.models.resnet50()

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    model.fc = nn.Linear(2048, num_classes)

    return model

def resnet101(pretrained = True, num_classes = 10):

    model = torchvision.models.resnet101()

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    model.fc = nn.Linear(2048, num_classes)

    return model

def resnet152(pretrained = True, num_classes = 10):

    model = torchvision.models.resnet152()

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    model.fc = nn.Linear(2048, num_classes)

    return model
