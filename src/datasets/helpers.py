from src.datasets.mados_dataset import make_mados_dataset


def make_eval_dataloaders(
    dataset_name,
    data_root,
    img_size=112,
    batch_size=64,
    drop_last=False,
    pin_mem=True,
    num_workers=10,
    world_size=1,
    rank=0,
):
    """
    Create train/val/test dataloaders for evaluation.

    Args:
        dataset_name: Name of the dataset ('mados', etc.)
        data_root: Root path to the dataset
        img_size: Size to resize images to
        batch_size: Batch size
        drop_last: Whether to drop the last incomplete batch
        pin_mem: Whether to use pinned memory
        num_workers: Number of data loading workers
        world_size: Number of distributed processes
        rank: Rank of current process

    Returns:
        train_loader, train_sampler, val_loader, val_sampler, test_loader, test_sampler
    """
    if dataset_name == "mados":
        return make_mados_dataset(
            data_root=data_root,
            batch_size=batch_size,
            img_size=img_size,
            drop_last=drop_last,
            pin_mem=pin_mem,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
