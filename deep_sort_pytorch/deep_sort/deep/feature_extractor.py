import os
import yaml

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging # todo: a che serve?

from torch2trt import torch2trt

from .models import select_model


class Extractor(object):
    def __init__(self, model_path):

        self.device = "cuda"

        # TUTTA ROBA DUPLICATA
        #########################################
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
        #########################################

        logger = logging.getLogger("root.tracker") # todo: a che serve?
        logger.info("Loading weights from {}... Done!".format(model_path)) # todo: a che serve?

        self.person_width = img_width
        self.person_heigth = img_height
        self.net = model
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

        im_batch = torch.cat([self.norm(_resize(im, (self.person_width, self.person_heigth))).unsqueeze(0) for im in im_crops], dim=0)
        im_batch = im_batch.half()
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
