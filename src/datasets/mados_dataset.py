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

def make_mados_dataset(
    data_root,
    batch_size,
    img_size=80,
    drop_last=True,
    pin_mem=True,
    num_workers=10,
    world_size=1,
    rank=0,
):
  # Train
  train_dataset = MADOSDataset(
    data_root=data_root, 
    split="train",
    img_size=img_size)
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
  val_dataset = MADOSDataset(
    data_root=data_root,
    split="val",
    img_size=img_size)
  logger.info(f'Validation dataset created. Num samples: {len(val_dataset)}')

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
    persistent_workers=True)
  logger.info(f'Validation dataloader created. No. batches: {len(val_dataloader)}')

  # Test
  test_dataset = MADOSDataset(
    data_root=data_root,
    split="test",
    img_size=img_size)
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


class MADOSDataset(Dataset):
    def __init__(self, data_root, split: str, img_size=80, augmentation=None):
        self.h5_path = os.path.join(data_root, 'mados.h5')
        self.split = split
        norm_stats_path = os.path.join(data_root, 'NORM_CONFIG.json')

        self.img_size = to2tuple(img_size)
        self.augmentation = augmentation
        self.norm_config = json.load(open(norm_stats_path, "r"))

        self.h5_file = None  # Will be opened lazily

        assert split in ["train", "val", "test"], f"Invalid split: {split}. Must be one of ['train', 'val', 'test']"

        with h5py.File(self.h5_path, "r") as f:
            all_splits = f["split"][:]  
            self.indices = np.where(np.isin(all_splits, [split.encode('utf-8')]))[0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
 
        image = torch.from_numpy(self.h5_file["images"][self.indices[idx]]) # (10, 80, 80)
        label = torch.from_numpy(self.h5_file["label"][self.indices[idx]]).long()  # (80, 80)

        # -- fill and ignore nan pixels
        nan_mask = torch.isnan(image).any(dim=0)
        image = torch.where(torch.isnan(image), 0.0, image)
        label = torch.where(nan_mask, -1, label)

        # -- Set 0 (no-data labels) to -1 (ignored index)
        label = torch.where(label == 0, -1, label)
        # -- Shift labels 1-15 to 0-14 (only for non-ignored labels)
        label = torch.where(label > 0, label - 1, label)

        # -- Add channel dim to label
        label = label.unsqueeze(0)  # (1, 80, 80)

        image = normalize_bands(image, self.norm_config)

        # -- Resize image and mask
        if self.split == "train":
          image, label = random_crop_resize_img_and_mask(img=image, mask=label, size=self.img_size)
        else:
          image, label = resize_img_and_mask(img=image, mask=label, size=self.img_size)

        # -- Filter BGR+NIR for now
        image = image[[1, 2, 3, 6]]

        return image, label



def normalize_bands(image, norm_cfg):
    means, stds = norm_cfg["mean"], norm_cfg["std"]

    means = torch.tensor(means).reshape(-1, 1, 1)
    stds = torch.tensor(stds).reshape(-1, 1, 1)
    image = (image - means) / stds

    return image
