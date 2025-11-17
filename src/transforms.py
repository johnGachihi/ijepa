# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from typing import Tuple

from PIL import ImageFilter

import torch
import torchvision.transforms as transforms
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF

_GLOBAL_SEED = 0
logger = getLogger()


def make_transforms(
    crop_size=224,
    crop_scale=(0.3, 1.0),
    color_jitter=1.0,
    horizontal_flip=False,
    color_distortion=False,
    gaussian_blur=False,
    normalization=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225))
):
    logger.info('making imagenet data transforms')

    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    transform_list = []
    transform_list += [transforms.RandomResizedCrop(crop_size, scale=crop_scale)]
    if horizontal_flip:
        transform_list += [transforms.RandomHorizontalFlip()]
    if color_distortion:
        transform_list += [get_color_distortion(s=color_jitter)]
    if gaussian_blur:
        transform_list += [GaussianBlur(p=0.5)]
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(normalization[0], normalization[1])]

    transform = transforms.Compose(transform_list)
    return transform


def paired_random_crop_resize(
    hr_img, lr_img,
    size: Tuple[int, int], hr_img_size=None,
    scale=(0.3, 1.0), ratio=(3 / 4, 4 / 3)
):
  """
  Apply same random crop to both hr and lr images, then resize to target size
  """
  if hr_img_size is None:
    hr_img_size = size

  def lr_crop_params(i, j, h, w):
    hr_h, hr_w = hr_img.shape[-2:]
    lr_h, lr_w = lr_img.shape[-2:]

    scaled_i = int(i * lr_h / hr_h)
    scaled_j = int(j * lr_w / hr_w)
    scaled_h = int(h * lr_h / hr_h)
    scaled_w = int(w * lr_w / hr_w)

    return scaled_i, scaled_j, scaled_h, scaled_w

  i, j, h, w = T.RandomResizedCrop.get_params(hr_img, scale=scale, ratio=ratio)
  lr_i, lr_j, lr_h, lr_w = lr_crop_params(i, j, h, w)

  hr_img = TF.crop(hr_img, i, j, h, w)
  lr_img = TF.crop(lr_img, lr_i, lr_j, lr_h, lr_w)

  hr_img = TF.resize(hr_img, hr_img_size)
  lr_img = TF.resize(lr_img, size, interpolation=T.InterpolationMode.NEAREST)

  return hr_img, lr_img


def paired_resize(hr_img, lr_img, size: Tuple[int, int], hr_img_size=None):
  """
  Resize both hr and lr images to the target size
  """
  if hr_img_size is None:
    hr_img_size = size

  hr_img = TF.resize(hr_img, hr_img_size)
  lr_img = TF.resize(lr_img, size, interpolation=T.InterpolationMode.NEAREST)

  return hr_img, lr_img


class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
