import argparse
import os
import yaml

from torch2trt import torch2trt
import torch
import torch.nn as nn

# from feature_extractor import select_model
from models import res_net50, mob_net, squeeze_net, res_net18

# todo: duplicata
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--model_path', type=str, help='trained pytorch weight.pth')
    parser.add_argument('--width', default=64, type=int, help='person crop width')
    parser.add_argument('--height', default=128, type=int, help='person crop height')
    parser.add_argument('--max_batchsize', default=64, type=int, help='max batchsize for trt inference')
    opt = parser.parse_args()
    print(opt)

    model_path, width, height, max_batchsize = opt.model_path, opt.width, opt.height, opt.max_batchsize

    assert model_path.endswith('.pth') and os.path.exists(model_path), 'argument --model_path is not a valid argument'

    # load the training config
    config_path = '/'.join(model_path.split('/')[:-1]) + '/opts.yaml'
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    num_bottleneck, model_name = config['num_bottleneck'], config['name']

    # load network and weights, then remove classifier to perform feature extraction
    model = select_model(model_name, num_bottleneck=num_bottleneck)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.classifier.classifier = nn.Sequential()

    # eval to cuda to fp16
    model = model.eval().cuda().half()

    # create example data
    x = torch.ones((1, 3, height, width)).cuda().half()

    # convert to TensorRT feeding sample data as input
    print('Generating tensorrt engine...')
    model_trt = torch2trt(model, [x], fp16_mode=True, max_batch_size=max_batchsize)

    new_filename = model_path.replace('.pth', '_trt.pth')
    torch.save(model_trt.state_dict(), new_filename)
