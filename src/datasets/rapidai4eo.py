import json

import numpy as np
import torch

import h5py

from src.transforms import paired_random_crop_resize, paired_resize
from src.utils.tensors import to2tuple

from logging import getLogger

logger = getLogger()


def make_rapidai4eo_dataloader(
    data_root: str,
    splits_file_path: str,
    split: str = 'train',
    img_size=(224, 224),
    hr_img_size=None,
    use_hr_image: bool = True,
    load_both_images: bool = False,
    batch_size: int = 64,
    collator=None,
    drop_last: bool = True,
    pin_mem: bool = True,
    num_workers: int = 16,
    world_size: int = 1,
    rank: int = 0
):
  dataset = RapidAI4EO(
    hdf5_file=data_root,
    splits_file=splits_file_path,
    split=split,
    img_size=img_size,
    hr_img_size=hr_img_size,
    use_hr_image=use_hr_image,
    load_both_images=load_both_images,
    random_crop_resize=True if split == 'train' else False)
  if load_both_images:
    logger.info('RapidAI4EO dataset created (loading BOTH HR (Planet) and LR (Sentinel2) images)')
  else:
    logger.info(f'RapidAI4EO dataset created (using {"HR (Planet)" if use_hr_image else "LR (Sentinel2)"} images)')

  dist_sampler = torch.utils.data.distributed.DistributedSampler(
    dataset=dataset,
    num_replicas=world_size,
    rank=rank)

  dataloader = torch.utils.data.DataLoader(
    dataset,
    collate_fn=collator,
    sampler=dist_sampler,
    batch_size=batch_size,
    drop_last=drop_last,
    pin_memory=pin_mem,
    num_workers=num_workers,
    persistent_workers=True)
  logger.info('RapidAI4EO dataloader created')

  return dataset, dataloader, dist_sampler


class RapidAI4EO(torch.utils.data.Dataset):
  """
  RapidAI4EO dataset for self-supervised pretraining.
  Contains paired high-resolution (Planet, 200x200) and low-resolution (Sentinel-2, 60x60) satellite images.

  HDF5 structure:
      - sentinel2: (N, 12, 60, 60) - 12 bands at 10m resolution
      - planet: (N, 4, 200, 200) - 4 bands (BGRN) at 3m resolution
      - sample_ids: sample identifiers
      - dates: acquisition dates
  """
  # Normalization statistics (computed from dataset)
  # Planet: 4 bands (BGRN)
  PLANET_MEANS = [528.3, 744.1, 849.1, 2692.8]
  PLANET_STDS = [336.2, 418.1, 607.8, 886.1]

  # Sentinel-2: first 4 bands (B, G, R, NIR)
  SENTINEL2_MEANS = [557.5, 828.4, 900.5, 2652.0]
  SENTINEL2_STDS = [396.1, 477.6, 665.9, 946.3]

  def __init__(
      self,
      hdf5_file,
      splits_file,
      split="train",
      img_size=(224, 224),
      hr_img_size=None,
      use_hr_image=True,
      load_both_images=False,
      random_crop_resize=True,
      normalise=True
  ):
    """
    Args:
        hdf5_file: Path to HDF5 file containing RapidAI4EO data
        splits_file: Path to JSON file containing train/val/test splits
        split: Which split to use ('train', 'val', or 'test')
        img_size: Size to resize LR (Sentinel2) images to
        hr_img_size: Size for high-resolution images (Planet)
        use_hr_image: If True, use HR (Planet) images for training; if False, use LR (Sentinel2)
        load_both_images: If True, return both HR and LR images; if False, return only selected image
        random_crop_resize: Whether to apply random crop and resize
        normalise: Whether to normalize images
    """
    self.hdf5_file = hdf5_file
    self.img_size = to2tuple(img_size)
    self.hr_img_size = to2tuple(hr_img_size) if hr_img_size is not None else None
    self.use_hr_image = use_hr_image
    self.load_both_images = load_both_images
    self.random_crop_resize = random_crop_resize
    self.normalise = normalise

    with open(splits_file, "r") as f:
      self.indices = json.load(f)[split]

  def __len__(self):
    return len(self.indices)

  def __getitem__(self, idx):
    with h5py.File(self.hdf5_file, "r") as data_full:
      # Planet has 4 bands, Sentinel-2 has 12 bands (take first 4: B, G, R, NIR)
      planet_img = torch.from_numpy(data_full["planet"][self.indices[idx]].astype(np.float32))
      sentinel2_img = torch.from_numpy(data_full["sentinel2"][self.indices[idx]][:4].astype(np.float32))

    if self.img_size is not None:
      if self.random_crop_resize:
        planet_img, sentinel2_img = paired_random_crop_resize(
          hr_img=planet_img, lr_img=sentinel2_img,
          size=self.img_size, hr_img_size=self.hr_img_size)
      else:
        planet_img, sentinel2_img = paired_resize(
          hr_img=planet_img, lr_img=sentinel2_img,
          size=self.img_size, hr_img_size=self.hr_img_size)

    # convert images to float32
    planet_img = planet_img.float()
    sentinel2_img = sentinel2_img.float()

    # normalize images
    if self.normalise:
      planet_img = (planet_img - torch.tensor(self.PLANET_MEANS).view(-1, 1, 1)) / torch.tensor(self.PLANET_STDS).view(-1, 1, 1)  # todo: revert after experiment
      sentinel2_img = (sentinel2_img - torch.tensor(self.SENTINEL2_MEANS).view(-1, 1, 1)) / torch.tensor(self.SENTINEL2_STDS).view(-1, 1, 1)

    # Return selected image(s) based on load_both_images parameter
    # Return as tuple (img, target) for compatibility with mask collator
    # target is not used in self-supervised learning, so we return 0
    if self.load_both_images:
      # Return both HR and LR images as (hr_img, lr_img, target)
      return planet_img, sentinel2_img, 0
    elif self.use_hr_image:
      return planet_img, 0
    else:
      return sentinel2_img, 0
