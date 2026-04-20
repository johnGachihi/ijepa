import json
import os
from logging import getLogger

import h5py
import numpy as np
import torch
import torch.multiprocessing
from torch.utils.data import Dataset

from src.transforms import random_crop_resize_img_and_mask, resize_img_and_mask
from src.utils.tensors import to2tuple

logger = getLogger()


def make_pastis_dataset(
    data_root,
    batch_size,
    img_size=224,
    tiles_per_img=4,
    drop_last=True,
    pin_mem=True,
    num_workers=10,
    world_size=1,
    rank=0,
):
  # Train
  train_dataset = PASTISDataset(
    data_root=data_root,
    split="train",
    img_size=img_size,
    tiles_per_img=tiles_per_img)
  logger.info(f'Train dataset created. Num samples: {len(train_dataset)}')

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
    persistent_workers=True)
  logger.info(f'Train dataloader created. No. batches: {len(train_dataloader)}')

  # Val
  val_dataset = PASTISDataset(
    data_root=data_root,
    split="val",
    img_size=img_size,
    tiles_per_img=tiles_per_img)
  logger.info(f'Validation dataset created. Num samples: {len(val_dataset)}')

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
    persistent_workers=True)
  logger.info(f'Validation dataloader created. No. batches: {len(val_dataloader)}')

  # Test
  test_dataset = PASTISDataset(
    data_root=data_root,
    split="test",
    img_size=img_size,
    tiles_per_img=tiles_per_img)
  logger.info(f'Test dataset created. Num samples: {len(test_dataset)}')

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
    persistent_workers=True)
  logger.info(f'Test dataloader created. No. batches: {len(test_dataloader)}')

  return (
    train_dataloader,
    train_dist_sampler,
    val_dataloader,
    val_dist_sampler,
    test_dataloader,
    test_dist_sampler
  )


class PASTISDataset(Dataset):
  """
  PASTIS semantic segmentation dataset.
  Folds {1,2,3} = train, {4} = val, {5} = test.
  Uses the full S2 time series (T=13) and BGR+NIR bands ([0,1,2,6]).
  Each 128x128 image is split into `tiles_per_img` non-overlapping tiles.
  """

  SPLIT_FOLDS = {'train': [1, 2, 3], 'val': [4], 'test': [5]}
  FILTER_BANDS = [0, 1, 2, 6]  # BGR + NIR

  def __init__(
      self,
      data_root: str,
      split: str = 'train',
      img_size: int = 224,
      normalize: bool = True,
      tiles_per_img: int = 4,
      h5_file: str = 'pastis.h5',
      norm_stats_file: str = 'PASTIS/NORM_S2_patch.json'
  ):
    self.h5_path = os.path.join(data_root, h5_file)
    self.split = split
    self.img_size = to2tuple(img_size)
    self.normalize = normalize
    self.tiles_per_img = tiles_per_img

    assert split in self.SPLIT_FOLDS, \
      f"Invalid split: {split}. Must be one of {list(self.SPLIT_FOLDS)}"
    assert tiles_per_img in (1, 4), "tiles_per_img must be 1 or 4"

    self.norm_config = json.load(open(os.path.join(data_root, norm_stats_file), 'r'))

    with h5py.File(self.h5_path, 'r') as f:
      folds = f['fold'][:]
      self.img_indices = np.where(np.isin(folds, self.SPLIT_FOLDS[split]))[0]

    self._h5_file = None

  def __len__(self):
    return len(self.img_indices) * self.tiles_per_img

  def _get_h5_file(self):
    if self._h5_file is None:
      self._h5_file = h5py.File(self.h5_path, 'r', swmr=True)
    return self._h5_file

  def __del__(self):
    if self._h5_file is not None:
      self._h5_file.close()

  def _normalize(self, image, fold):
    stats = self.norm_config[f'Fold_{int(fold)}']
    mean = torch.tensor(stats['mean']).reshape(-1, 1, 1)
    std = torch.tensor(stats['std']).reshape(-1, 1, 1)
    return (image - mean) / std

  def _resize_img_and_mask(self, img, mask):
    if self.split == 'train':
      return random_crop_resize_img_and_mask(img, mask, self.img_size, scale=(0.3, 1.0), ratio=(3 / 4, 4 / 3))
    return resize_img_and_mask(img, mask, self.img_size)

  def __getitem__(self, idx):
    h5 = self._get_h5_file()

    img_idx = self.img_indices[idx // self.tiles_per_img]
    image = torch.from_numpy(h5['sentinel2_ts'][img_idx]).float()     # (T=13, 10, 128, 128)
    mask = torch.from_numpy(h5['label'][img_idx]).long().squeeze(0)   # (128, 128)
    fold = h5['fold'][img_idx]

    # Remap labels: 0 (background) and 19 (void) -> -1 (ignored); shift 1..18 -> 0..17
    mask = torch.where((mask == 0) | (mask == 19), torch.tensor(-1), mask - 1)

    if self.normalize:
      image = self._normalize(image, fold)  # broadcasts over T

    if self.tiles_per_img == 4:
      subtile_idx = idx % self.tiles_per_img
      tile_size = image.shape[-1] // 2
      r, c = subtile_idx // 2, subtile_idx % 2
      image = image[:, :, r * tile_size:(r + 1) * tile_size, c * tile_size:(c + 1) * tile_size]
      mask = mask[r * tile_size:(r + 1) * tile_size, c * tile_size:(c + 1) * tile_size]

    image = image[:, self.FILTER_BANDS]  # (T, 4, h, w)
    mask = mask.unsqueeze(0)             # (1, h, w)

    image, mask = self._resize_img_and_mask(image, mask)  # broadcasts over T

    return image, mask.long()
