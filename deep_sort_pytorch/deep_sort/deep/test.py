# -*- coding: utf-8 -*-

from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import os
import scipy.io
import yaml
from models import load_model_for_inference
from tqdm import tqdm


def extract_feature(model, dataloaders, ft_dim, scales=(1, 1.1)):
    features = torch.FloatTensor()

    for data in tqdm(dataloaders):
        img, label = data
        n, c, h, w = img.size()

        ff = torch.FloatTensor(n, ft_dim).zero_().to(device).half()
        img = img.to(device).half()

        if opt.augment:
            for i in range(2):
                if i == 1:
                    img = img.flip(3)  # flips (2-ud, 3-lr)

                for scale in scales:
                    if scale != 1:
                        img = nn.functional.interpolate(img, scale_factor=scale, mode='bicubic', align_corners=False)
                    outputs = model(img)
                    ff += outputs
        else:
            outputs = model(img)
            ff += outputs

        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff.data.cpu()), 0)
    return features


def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        # filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--model_path', type=str, help='trained pytorch or trt weight.pth')
    parser.add_argument('--data_dir', default='../Market/pytorch', type=str, help='./test_data')
    parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
    parser.add_argument('--multi', action='store_true', help='use multiple query')
    parser.add_argument('--augment', action='store_true',
                        help='use horizontal flips and different scales in inference.')
    parser.add_argument('--trt', action='store_true', help='use trt instead of pytorch inference.')
    opt = parser.parse_args()
    print(opt)

    model_path, data_dir, batchsize = opt.model_path, opt.data_dir, opt.batchsize

    device = 'cuda' # if torch.cuda.is_available() else 'cpu'

    # load model and training config
    model, num_bottleneck, img_height, img_width, model_name = load_model_for_inference(model_path=model_path)

    # data transformers
    data_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),  # default interpolation is bilinear
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if opt.multi:
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in
                          ['gallery', 'query', 'multi-query']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,
                                                      shuffle=False, num_workers=8) for x in
                       ['gallery', 'query', 'multi-query']}
    else:
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in
                          ['gallery', 'query']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,
                                                      shuffle=False, num_workers=8) for x in ['gallery', 'query']}
    class_names = image_datasets['query'].classes

    gallery_path = image_datasets['gallery'].imgs
    query_path = image_datasets['query'].imgs

    gallery_cam, gallery_label = get_id(gallery_path)
    query_cam, query_label = get_id(query_path)

    if opt.multi:
        mquery_path = image_datasets['multi-query'].imgs
        mquery_cam, mquery_label = get_id(mquery_path)

    print('-------test-----------')

    # Extract feature
    with torch.no_grad():
        gallery_feature = extract_feature(model, dataloaders['gallery'], ft_dim=num_bottleneck)
        query_feature = extract_feature(model, dataloaders['query'], ft_dim=num_bottleneck)
        if opt.multi:
            mquery_feature = extract_feature(model, dataloaders['multi-query'])

    # Save to Matlab for check
    result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_cam': gallery_cam,
              'query_f': query_feature.numpy(), 'query_label': query_label, 'query_cam': query_cam}
    scipy.io.savemat('pytorch_result.mat', result)

    print(model_name)
    result = './model/%s/result.txt' % model_name
    os.system('python evaluate_gpu.py | tee -a %s' % result)

    if opt.multi:
        result = {'mquery_f': mquery_feature.numpy(), 'mquery_label': mquery_label, 'mquery_cam': mquery_cam}
        scipy.io.savemat('multi_query.mat', result)
