import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F


def load_model_for_inference(model_path):
    ld = torch.load(model_path)
    state_dict, num_bottleneck, img_height, img_width, model_name, engine_type = \
        ld['state_dict'], ld['num_bottleneck'], ld['img_height'], ld['img_width'], ld['model_name'], ld['engine_type']

    if engine_type == 'pytorch':
        model = select_model(model_name, num_bottleneck=num_bottleneck)
        model.load_state_dict(state_dict)

        # Remove the final fc layer and classifier layer
        model.classifier.classifier = nn.Sequential()
        model = model.cuda().half()

    elif engine_type == 'tensorrt':
        from torch2trt import TRTModule

        print('Loading deep trt feature extractor...')
        model = TRTModule()
        model.load_state_dict(state_dict)

    model = model.eval()
    return model, num_bottleneck, img_height, img_width, model_name


def select_model(model_name, class_num=751, droprate=0.5, circle=False, num_bottleneck=512):
    tested_models = ('ResNet50', 'ResNet18', 'SqueezeNet', 'MobileNet', 'Deep')

    assert model_name in tested_models, f'model_name must be one of the following: {tested_models}, found {model_name}'
    if model_name == 'ResNet50':
        model = res_net50(class_num=class_num, droprate=droprate, circle=circle, num_bottleneck=num_bottleneck)
    elif model_name == 'ResNet18':
        model = res_net18(class_num=class_num, droprate=droprate, circle=circle, num_bottleneck=num_bottleneck)
    elif model_name == 'SqueezeNet':
        model = squeeze_net(class_num=class_num, droprate=droprate, circle=circle, num_bottleneck=num_bottleneck)
    elif model_name == 'MobileNet':
        model = mob_net(class_num=class_num, droprate=droprate, circle=circle, num_bottleneck=num_bottleneck)
    else:
        model = deep_net(class_num=class_num, droprate=droprate, circle=circle, num_bottleneck=num_bottleneck)
    return model


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, is_downsample=False):
        super(BasicBlock, self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(c_out)
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
                nn.BatchNorm2d(c_out)
            )
            self.is_downsample = True

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y), True)


def make_layers(c_in, c_out, repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [BasicBlock(c_in, c_out, is_downsample=is_downsample), ]
        else:
            blocks += [BasicBlock(c_out, c_out), ]
    return nn.Sequential(*blocks)


class Deep(nn.Module):
    def __init__(self):
        super(Deep, self).__init__()
        # 3 128 64
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        # 32 64 32
        self.layer1 = make_layers(64, 64, 2, False)
        # 32 64 32
        self.layer2 = make_layers(64, 128, 2, True)
        # 64 32 16
        self.layer3 = make_layers(128, 256, 2, True)
        # 128 16 8
        self.layer4 = make_layers(256, 512, 2, True)
        # 256 8 4

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True,
                 return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck, bias=False)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x, f]
        else:
            x = self.classifier(x)
            return x


# Define the ResNet50-based Model
class res_net50(nn.Module):

    def __init__(self, class_num, droprate=0.5, circle=False, num_bottleneck=512):
        super(res_net50, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.circle = circle
        self.classifier = ClassBlock(2048, class_num, droprate, return_f=circle, num_bottleneck=num_bottleneck)
        del model_ft.fc


    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


# Define the ResNet50-based Model
class res_net18(nn.Module):

    def __init__(self, class_num, droprate=0.5, circle=False, num_bottleneck=512):
        super(res_net18, self).__init__()
        model_ft = models.resnet18(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.circle = circle
        self.classifier = ClassBlock(512, class_num, droprate, return_f=circle, num_bottleneck=num_bottleneck)
        del model_ft.fc

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


# Define the MobilenetV2 based Model
class mob_net(nn.Module):

    def __init__(self, class_num=751, droprate=0.5, circle=False, num_bottleneck=512):
        super(mob_net, self).__init__()
        model_ft = models.mobilenet_v2(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.circle = circle
        # removing last bottleneck and classifier and change them
        del model_ft.classifier, model_ft.features[18]
        self.classifier = ClassBlock(320, class_num, droprate, return_f=circle, num_bottleneck=num_bottleneck)

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


# Define the squeeze_net-based Model
class squeeze_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, circle=False, num_bottleneck=512):
        super(squeeze_net, self).__init__()
        model_ft = models.squeezenet1_1(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.circle = circle
        self.classifier = ClassBlock(512, class_num, droprate, return_f=circle, num_bottleneck=num_bottleneck)
        del model_ft.classifier

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


class deep_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, circle=False, num_bottleneck=512):
        super(deep_net, self).__init__()
        model_ft = Deep()
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.circle = circle
        self.classifier = ClassBlock(512, class_num, droprate, return_f=circle, num_bottleneck=num_bottleneck)

    def forward(self, x):
        x = self.model(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x
'''
# debug model structure
# Run this code with:
python models.py
'''
if __name__ == '__main__':
    # Here I left a simple forward function.
    # Test the model, before you train it.
    net = select_model('Deep', num_bottleneck=128)
    print(net)
    print('Removing: \n', net.classifier.classifier)
    # remove last fc from classifier part
    net.classifier.classifier = nn.Sequential()
    input = Variable(torch.FloatTensor(8, 3, 128, 64))
    output = net(input)
    print('net output size:')
    print(output.shape)

