import json
from pathlib import Path
from typing import Optional

import geobench
import numpy as np
import torch.multiprocessing
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
from src.utils.tensors import to2tuple
from src.transforms import random_crop_resize_img_and_mask, resize_img_and_mask

torch.multiprocessing.set_sharing_strategy("file_system")

DEFAULT_SEED = 42

import torch

from logging import getLogger

logger = getLogger()


def make_m_cashew_plant_dataset(
    batch_size,
    partition,
    norm_operation='standardize',
    band_names=["02 - Blue", "03 - Green", "04 - Red", "08 - NIR"],
    img_size=64,
    drop_last=True,
    pin_mem=True,
    num_workers=10,
    persistent_workers=True,
    world_size=1,
    rank=0,
):
  train_dataset = GeobenchDataset(
    split='train',
    partition=partition,  # eg: "1.00x_train", "0.50x_train", "0.20x_train",... (refers to json partion files)
    dataset_name="m-cashew-plantation",
    norm_operation=norm_operation,
    benchmark_name="segmentation_v0.9.1",
    band_names=band_names,
    img_size=img_size
  )
  logger.info('Train dataset created')

  train_dist_sampler = torch.utils.data.distributed.DistributedSampler(
    dataset=train_dataset,
    num_replicas=world_size,
    rank=rank)

  train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    sampler=train_dist_sampler,
    batch_size=batch_size,
    drop_last=drop_last,
    pin_memory=pin_mem,
    num_workers=num_workers,
    persistent_workers=persistent_workers)
  logger.info('Train dataloader created')

  # Val
  val_dataset = GeobenchDataset(
    split='valid',
    partition=partition,  # eg: "1.00x_train", "0.50x_train", "0.20x_train",... (refers to json partion files)
    dataset_name="m-cashew-plantation",
    norm_operation=norm_operation,
    benchmark_name="segmentation_v0.9.1",
    band_names=band_names,
    img_size=img_size
  )
  logger.info('Validation dataset created')

  # TODO: Not quite sure what this is for?
  val_dist_sampler = torch.utils.data.distributed.DistributedSampler(
    dataset=val_dataset,
    num_replicas=world_size,
    rank=rank)

  val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    pin_memory=pin_mem,
    num_workers=num_workers,
    persistent_workers=persistent_workers)
  logger.info('Validation dataloader created')

  # Test
  test_dataset = GeobenchDataset(
    split='test',
    partition=partition,  # eg: "1.00x_train", "0.50x_train", "0.20x_train",... (refers to json partion files)
    dataset_name="m-cashew-plantation",
    norm_operation=norm_operation,
    benchmark_name="segmentation_v0.9.1",
    band_names=band_names,
    img_size=img_size
  )
  logger.info('Test dataset created')

  test_dist_sampler = torch.utils.data.distributed.DistributedSampler(
    dataset=test_dataset,
    num_replicas=world_size,
    rank=rank)

  test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    pin_memory=pin_mem,
    num_workers=num_workers,
    persistent_workers=persistent_workers)
  logger.info('Test dataloader created')

  return (
    train_dataloader,
    train_dist_sampler,
    val_dataloader,
    val_dist_sampler,
    test_dataloader,
    test_dist_sampler
  )


def make_m_sa_crop_type_dataset(
    batch_size,
    partition,
    norm_operation='standardize',
    band_names=["02 - Blue", "03 - Green", "04 - Red", "08 - NIR"],
    img_size=64,
    drop_last=True,
    pin_mem=True,
    num_workers=10,
    persistent_workers=True,
    world_size=1,
    rank=0,
):
  train_dataset = GeobenchDataset(
    split='train',
    partition=partition,  # eg: "1.00x_train", "0.50x_train", "0.20x_train",... (refers to json partion files)
    dataset_name="m-SA-crop-type",
    norm_operation=norm_operation,
    benchmark_name="segmentation_v0.9.1",
    band_names=band_names,
    img_size=img_size
  )
  logger.info(f'Train dataset created. Num samples {len(train_dataset)}')

  train_dist_sampler = torch.utils.data.distributed.DistributedSampler(
    dataset=train_dataset,
    num_replicas=world_size,
    rank=rank)

  train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    sampler=train_dist_sampler,
    batch_size=batch_size,
    drop_last=drop_last,
    pin_memory=pin_mem,
    num_workers=num_workers,
    persistent_workers=persistent_workers)
  logger.info(f'Train dataloader created. Num batches: {len(train_dataloader)}')

  # Val
  val_dataset = GeobenchDataset(
    split='valid',
    partition=partition,  # eg: "1.00x_train", "0.50x_train", "0.20x_train",... (refers to json partion files)
    dataset_name="m-SA-crop-type",
    norm_operation=norm_operation,
    benchmark_name="segmentation_v0.9.1",
    band_names=band_names,
    img_size=img_size
  )
  logger.info('Validation dataset created')

  # TODO: Not quite sure what this is for?
  val_dist_sampler = torch.utils.data.distributed.DistributedSampler(
    dataset=val_dataset,
    num_replicas=world_size,
    rank=rank)

  val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    pin_memory=pin_mem,
    num_workers=num_workers,
    persistent_workers=persistent_workers)
  logger.info('Validation dataloader created')

  # Test
  test_dataset = GeobenchDataset(
    split='test',
    partition=partition,  # eg: "1.00x_train", "0.50x_train", "0.20x_train",... (refers to json partion files)
    dataset_name="m-SA-crop-type",
    norm_operation=norm_operation,
    benchmark_name="segmentation_v0.9.1",
    band_names=band_names,
    img_size=img_size
  )
  logger.info('Test dataset created')

  test_dist_sampler = torch.utils.data.distributed.DistributedSampler(
    dataset=test_dataset,
    num_replicas=world_size,
    rank=rank)

  test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    pin_memory=pin_mem,
    num_workers=num_workers,
    persistent_workers=persistent_workers)
  logger.info('Test dataloader created')

  return (
    train_dataloader,
    train_dist_sampler,
    val_dataloader,
    val_dist_sampler,
    test_dataloader,
    test_dist_sampler
  )


class GeobenchDataset(Dataset):
  """
  Class implementation inspired by: https://github.com/vishalned/MMEarth-train/tree/main
  """

  def __init__(
      self,
      split: str,
      partition,  # eg: "1.00x_train", "0.50x_train", "0.20x_train",... (refers to json partion files)
      dataset_name: str,
      augmentation=None,
      norm_operation='norm_yes_clip',
      benchmark_name: str = "segmentation_v0.9.1",
      band_names: list[str] = ["02 - Blue", "03 - Green", "04 - Red", "08 - NIR"],  # sentinel-2 bands
      img_size=64
  ):

    assert split in ["train", "valid", "test"]

    self.split = split  # train, valid, test
    self.norm_operation = norm_operation
    self.augmentation = augmentation
    self.partition = partition
    self.band_names = band_names
    self.img_size = to2tuple(img_size)

    assert dataset_name in ["m-SA-crop-type", "m-cashew-plantation"]
    # for cashew plant and SA crop type
    # images are 256x256, no tiling

    for task in geobench.task_iterator(benchmark_name=benchmark_name):
      if task.dataset_name == dataset_name:
        break

    self.dataset = task.get_dataset(split=self.split, partition_name=self.partition, band_names=band_names)
    print(
      f"In dataset length for split {split} and partition {partition}: length = {len(self.dataset)}"
    )

    original_band_names = [
      self.dataset[0].bands[i].band_info.name for i in range(len(self.dataset[0].bands))
    ]

    self.band_indices = [original_band_names.index(band_name) for band_name in self.band_names]

    # Cache normalization stats to avoid recomputing on every sample
    self.norm_stats = self.dataset.normalization_stats()

  def __getitem__(self, idx):
    sample = self.dataset[idx]
    label = sample.label

    # Load bands directly without intermediate list
    x = np.stack([sample.bands[band_idx].data for band_idx in self.band_indices], axis=2)  # (h, w, C)
    assert x.shape[-1] == len(self.band_names), f"Datasets must have {len(self.band_names)} channels, not {x.shape[-1]}"
    x = torch.tensor(normalize_bands(x, norm_type=self.norm_operation, norm_stats=self.norm_stats))

    # check if label is an object or a number
    if not (isinstance(label, int) or isinstance(label, list)):
      label = label.data
      # label is a memoryview object, convert it to a list, and then to a numpy array
      label = np.array(list(label))

    target = torch.tensor(label, dtype=torch.long)

    if self.augmentation is not None:
      x, target = self.augmentation.apply(x, target)

    # Channel last
    x = x.permute(2, 0, 1)
    target = target.unsqueeze(0)

    if self.split == 'train':
      x, target = random_crop_resize_img_and_mask(img=x, mask=target, size=self.img_size, scale=(0.3, 1.0),
                                                  ratio=(3 / 4, 4 / 3))
    else:
      x, target = resize_img_and_mask(img=x, mask=target, size=self.img_size)

    x = x.float()

    return x, target

  def __len__(self):
    return len(self.dataset)


def normalize_bands(image, norm_type, norm_stats):
  if norm_type == "satlas":
    image = image / 8160
    image = np.clip(image, 0, 1)
    return image

  original_dtype = image.dtype
  means, stds = norm_stats[0], norm_stats[1]

  means = np.array(means)
  stds = np.array(stds)

  if norm_type == "standardize":
    image = (image - means) / stds
  else:
    min_value = means - stds
    max_value = means + stds
    image = (image - min_value) / (max_value - min_value)

    if norm_type == "norm_yes_clip":
      image = np.clip(image, 0, 1)
    elif norm_type == "norm_yes_clip_int":
      # same as clipping between 0 and 1 but rounds to the nearest 1/255
      image = image * 255  # scale
      image = np.clip(image, 0, 255).astype(np.uint8)  # convert to 8-bit integers
      image = image.astype(original_dtype) / 255  # back to original_dtype between 0 and 1
    elif norm_type == "norm_no_clip":
      pass
    else:
      raise ValueError(
        f"norm type must norm_yes_clip, norm_yes_clip_int, norm_no_clip, or standardize, not {norm_type}"
      )
  return image
