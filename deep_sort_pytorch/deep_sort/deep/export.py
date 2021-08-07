import argparse
import os

from torch2trt import torch2trt
import torch
import torch.nn as nn

from models import load_model_for_inference


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--model_path', type=str, help='trained pytorch weight.pth')
    parser.add_argument('--max_batchsize', default=64, type=int, help='max batchsize for trt inference')
    opt = parser.parse_args()
    print(opt)

    model_path, max_batchsize = opt.model_path, opt.max_batchsize

    # load model and training config
    model, num_bottleneck, img_height, img_width, model_name = load_model_for_inference(model_path=model_path)

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
