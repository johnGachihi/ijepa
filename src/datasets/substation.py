import json
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset

from src.transforms import random_crop_resize_img_and_mask, resize_img_and_mask
from src.utils.tensors import to2tuple

from logging import getLogger
logger = getLogger()


def make_substation_dataset(
    data_root,
    batch_size,
    img_size=224,
    drop_last=True,
    pin_mem=True,
    num_workers=10,
    world_size=1,
    rank=0,
):
  # Train
  train_ds = SubstationDataset(
    data_path=data_root,
    split="train",
    img_size=img_size)
  logger.info(f'Train dataset created. Num samples: {len(train_ds)}')

  train_dist_sampler = torch.utils.data.distributed.DistributedSampler(
    dataset=train_ds,
    num_replicas=world_size,
    rank=rank)

  train_dl = torch.utils.data.DataLoader(
    train_ds,
    sampler=train_dist_sampler,
    batch_size=batch_size,
    drop_last=drop_last,
    pin_memory=pin_mem,
    num_workers=num_workers,
    persistent_workers=True)
  logger.info(f'Train dataloader created. No. batches: {len(train_dl)}')

  # Val
  val_ds = SubstationDataset(
    data_path=data_root,
    split="val",
    img_size=img_size)
  logger.info(f'Validation dataset created. Num samples: {len(val_ds)}')

  val_dist_sampler = torch.utils.data.distributed.DistributedSampler(
    dataset=val_ds,
    num_replicas=world_size,
    rank=rank)

  val_dl = torch.utils.data.DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    pin_memory=pin_mem,
    num_workers=num_workers,
    persistent_workers=True)
  logger.info(f'Validation dataloader created. No. batches: {len(val_dl)}')

  # Test
  test_ds = SubstationDataset(
    data_path=data_root,
    split="test",
    img_size=img_size)
  logger.info(f'Test dataset created. Num samples: {len(test_ds)}')

  test_dist_sampler = torch.utils.data.distributed.DistributedSampler(
    dataset=test_ds,
    num_replicas=world_size,
    rank=rank)

  test_dl = torch.utils.data.DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    pin_memory=pin_mem,
    num_workers=num_workers,
    persistent_workers=True)
  logger.info(f'Test dataloader created. No. batches: {len(test_dl)}')

  return (
    train_dl,
    train_dist_sampler,
    val_dl,
    val_dist_sampler,
    test_dl,
    test_dist_sampler
  )


class SubstationDataset(Dataset):
  BANDS = [1, 2, 3, 7]
  MEANS = torch.tensor([1431, 1233, 1209, 1192, 1448, 2238, 2609,
                        2537, 2828, 884, 20, 2226, 1537])[BANDS]
  STDS = torch.tensor([157, 254, 290, 420, 363, 457, 575,
                       606, 630, 156, 3, 554, 523])[BANDS]

  def __init__(
      self,
      data_path: str,
      split: str = "train",
      img_size: int = 224,
      normalize: bool = True,
      h5_file: str = "substation.h5",
      splits_file: str = "splits.json"
  ):
    data_path = Path(data_path)
    self.h5_path = data_path / h5_file
    self.split = split
    self.img_size = to2tuple(img_size)
    self.normalize = normalize

    assert split in ["train", "val", "test"], \
      f"Invalid split: {split}. Must be one of ['train', 'val', 'test']"

    splits_file = data_path / splits_file
    with open(splits_file, 'r') as f:
      self.indices = json.load(f)[split]

    logger.info(f"Substation {split} set: {len(self.indices)} samples")

    self._h5_file = None

  def __len__(self):
    return len(self.indices)

  def _normalize_bands(self, image):
    """Normalize image bands using precomputed mean and std."""
    return (image - self.MEANS.view(-1, 1, 1)) / self.STDS.view(-1, 1, 1)

  def _resize_img_and_mask(self, img, mask):
    if self.split == "train":
      return random_crop_resize_img_and_mask(img, mask, self.img_size)
    else:
      return resize_img_and_mask(img, mask, self.img_size)

  def _get_h5_file(self):
    if self._h5_file is None:
      self._h5_file = h5py.File(self.h5_path, "r", swmr=True)
    return self._h5_file

  def __del__(self):
    if self._h5_file is not None:
      self._h5_file.close()

  def __getitem__(self, idx):
    h5 = self._get_h5_file()

    sample_idx = self.indices[idx]
    image = torch.from_numpy(h5['image'][sample_idx]).float()  # (C, H, W)
    mask = torch.from_numpy(h5['mask'][sample_idx]).long()  # (H, W)

    # ?? from https://github.com/Lindsay-Lab/substation-seg/blob/main/dataloader.py
    mask[mask != 3] = 0
    mask[mask == 3] = 1

    # Handle NaN pixels
    nan_mask = torch.isnan(image).any(dim=0)
    image = torch.where(torch.isnan(image), 0.0, image)
    mask = torch.where(nan_mask, -1, mask)

    mask = mask.unsqueeze(0)  # (1, H, W)

    if self.normalize:
      image = self._normalize_bands(image)

    image, mask = self._resize_img_and_mask(image, mask)

    return image, mask.long()
