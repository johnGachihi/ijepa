import json

import torch

import h5py

from src.transforms import paired_random_crop_resize, paired_resize
from src.utils.tensors import to2tuple

from logging import getLogger

logger = getLogger()


def make_sen2venus_dataloader(
    data_root: str,
    splits_file_path: str,
    split: str = 'train',
    img_size=(256, 256),
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
  dataset = Sen2Venus(
    hdf5_file=data_root,
    splits_file=splits_file_path,
    split=split,
    img_size=img_size,
    hr_img_size=hr_img_size,
    use_hr_image=use_hr_image,
    load_both_images=load_both_images,
    random_crop_resize=True if split == 'train' else False)
  if load_both_images:
    logger.info('Sen2Venus dataset created (loading BOTH HR (Venus) and LR (Sentinel2) images)')
  else:
    logger.info(f'Sen2Venus dataset created (using {"HR (Venus)" if use_hr_image else "LR (Sentinel2)"} images)')

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
    persistent_workers=False)
  logger.info('Sen2Venus dataloader created')

  return dataset, dataloader, dist_sampler


class Sen2Venus(torch.utils.data.Dataset):
  VENUS_MEANS = [444.3010, 716.1393, 813.6448, 2605.5037]
  VENUS_STDS = [279.9104, 385.4034, 648.5869, 797.1441]

  SENTINEL2_MEANS = [443.8853, 715.6812, 813.2707, 2603.5852]
  SENTINEL2_STDS = [283.9673, 389.3354, 651.2451, 811.8329]

  def __init__(
      self,
      hdf5_file,
      splits_file,
      split="train",
      img_size=(256, 256),
      hr_img_size=None,
      use_hr_image=True,
      load_both_images=False,
      random_crop_resize=True,
      normalise=True
  ):
    """
    Args:
        hdf5_file: Path to HDF5 file containing Sen2Venus data
        splits_file: Path to JSON file containing train/val/test splits
        split: Which split to use ('train', 'val', or 'test')
        img_size: Size to resize images to
        hr_img_size: Size for high-resolution images (Venus)
        use_hr_image: If True, use HR (Venus) images for training; if False, use LR (Sentinel2)
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
      venus_img = torch.from_numpy(data_full["venus"][self.indices[idx]])
      sentinel2_img = torch.from_numpy(data_full["sentinel2"][self.indices[idx]])

    if self.img_size is not None:
      if self.random_crop_resize:
        venus_img, sentinel2_img = paired_random_crop_resize(
          hr_img=venus_img, lr_img=sentinel2_img,
          size=self.img_size, hr_img_size=self.hr_img_size)
      else:
        venus_img, sentinel2_img = paired_resize(
          hr_img=venus_img, lr_img=sentinel2_img,
          size=self.img_size, hr_img_size=self.hr_img_size)

    # convert images to float32
    venus_img = venus_img.float()
    sentinel2_img = sentinel2_img.float()

    # normalize images
    if self.normalise:
      venus_img = (venus_img - torch.tensor(self.VENUS_MEANS).view(-1, 1, 1)) / torch.tensor(self.VENUS_STDS).view(-1, 1, 1)
      sentinel2_img = (sentinel2_img - torch.tensor(self.SENTINEL2_MEANS).view(-1, 1, 1)) / torch.tensor(self.SENTINEL2_STDS).view(-1, 1, 1)

    # Return selected image(s) based on load_both_images parameter
    # Return as tuple (img, target) for compatibility with mask collator
    # target is not used in self-supervised learning, so we return 0
    if self.load_both_images:
      # Return both HR and LR images as (hr_img, lr_img, target)
      return venus_img, sentinel2_img, 0
    elif self.use_hr_image:
      return venus_img, 0
    else:
      return sentinel2_img, 0
