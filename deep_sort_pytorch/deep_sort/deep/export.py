import argparse
import os

from torch2trt import torch2trt
import torch
import torch.nn as nn

from models import select_model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--model_path', type=str, help='trained pytorch weight.pth')
    parser.add_argument('--max_batchsize', default=64, type=int, help='max batchsize for trt inference')
    opt = parser.parse_args()
    print(opt)

    model_path, max_batchsize = opt.model_path, opt.max_batchsize

    assert model_path.endswith('.pth') and os.path.exists(model_path), 'argument --model_path is not a valid argument'

    # load the training config
    loaded_dict = torch.load(model_path, map_location='cpu')
    state_dict = loaded_dict['state_dict']
    num_bottleneck = loaded_dict['num_bottleneck']
    img_height = loaded_dict['img_height']
    img_width = loaded_dict['img_width']
    model_name = loaded_dict['model_name']

    # load network and weights, then remove classifier to perform feature extraction
    model = select_model(model_name, num_bottleneck=num_bottleneck)
    model.load_state_dict(state_dict)
    model.classifier.classifier = nn.Sequential()

    # eval to cuda to fp16
    model = model.eval().cuda().half()

    # create example data
    x = torch.ones((1, 3, img_height, img_width)).cuda().half()

    # convert to TensorRT feeding sample data as input
    print('Generating tensorrt engine...')
    model_trt = torch2trt(model, [x], fp16_mode=True, max_batch_size=max_batchsize)

    new_filename = model_path.replace('.pth', '_trt.pth')

    to_save_dict = {
        'state_dict': model_trt.state_dict(), # todo: check se è portabile su cpu
        'num_bottleneck': num_bottleneck,
        'img_height': img_height,
        'img_width': img_width,
        'model_name': model_name,
        'max_batchsize': max_batchsize,
        'engine_type': 'tensorrt',
    }

    torch.save(to_save_dict, new_filename)
    print('model saved at: ', new_filename)
