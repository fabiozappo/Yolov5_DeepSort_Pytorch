import os
import yaml

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging # todo: a che serve?

from torch2trt import torch2trt

from .models import res_net50, mob_net, squeeze_net, res_net18

tested_models = ('ResNet50', 'ResNet18', 'SqueezeNet', 'MobileNet', 'Deep')

# todo: duplicata
def select_model(model_name, class_num=751, droprate=0.5, circle=False, num_bottleneck=512):

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

class Extractor(object):
    def __init__(self, model_path, use_cuda=True, use_trt=True, use_fp16=True, max_batchsize=64):

        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.use_fp16 = use_fp16
        self.person_width = 64
        self.person_heigth = 128

        # load the training config
        config_path = '/'.join(model_path.split('/')[:-1]) + '/opts.yaml'
        with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)
        num_bottleneck, model_name = config['num_bottleneck'], config['name']

        # load network and weights, then remove classifier to perform feature extraction
        net = select_model(model_name, num_bottleneck=num_bottleneck)
        net.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
        net.classifier.classifier = nn.Sequential()

        logger = logging.getLogger("root.tracker") # todo: a che serve?
        logger.info("Loading weights from {}... Done!".format(model_path)) # todo: a che serve?

        # Inference mode
        net = net.eval().to(self.device)

        if use_fp16:
            net = net.half()

        # todo: fa crashare il programma su immagini senza detection
        if use_trt:
            x = torch.FloatTensor(1, 3, self.person_heigth, self.person_width).to(self.device)
            x = x.half() if use_fp16 else x
            print('Generating tensorrt engine...')
            net = torch2trt(net, [x], fp16_mode=use_fp16, max_batch_size=max_batchsize)

        self.net = net
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        # todo: si può far meglio...
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, (self.person_width, self.person_heigth))).unsqueeze(0) for im in im_crops], dim=0).float()
        im_batch = im_batch.half() if self.use_fp16 else im_batch.float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
            features = features.div(features.norm(p=2, dim=1, keepdim=True))  # normalization added
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)
