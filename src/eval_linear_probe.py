# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import logging
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.functional.classification import multiclass_jaccard_index

from src.datasets.helpers import make_eval_dataloaders
from src.utils.distributed import init_distributed, AllReduce

from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)

import src.models.vision_transformer as vit
import src.models.linear_head as linear_head

# --
log_timings = True
log_freq = 10
checkpoint_freq = 10
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    model_name = args['meta']['model_name']
    patch_size = args['meta']['patch_size']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    pretrained_checkpoint = args['meta']['pretrained_checkpoint']
    use_bfloat16 = args['meta']['use_bfloat16']
    seg_n_cls = args['meta']['seg_n_cls']
    use_batchnorm = args['meta'].get('use_batchnorm', False)
    eval_type = args['meta'].get('eval_type', 'linear_probing')  # 'linear_probing' or 'finetuning'

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    dataset_name = args['data']['dataset_name']
    data_root = args['data']['data_root']
    crop_size = args['data']['crop_size']
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    drop_last = args['data'].get('drop_last', False)

    # -- OPTIMIZATION
    ipe_scale = args['optimization']['ipe_scale']
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']

    dump = os.path.join(folder, 'params-linear-probe.yaml')
    os.makedirs(folder, exist_ok=True)
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log and checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    val_log_file = os.path.join(folder, f'{tag}_val_r{rank}.csv')
    test_log_file = os.path.join(folder, f'{tag}_test_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    best_path = os.path.join(folder, f'{tag}-best.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mIoU'),
                           ('%d', 'time (ms)'))

    # -- make validation csv_logger
    val_csv_logger = CSVLogger(val_log_file,
                               ('%d', 'epoch'),
                               ('%.5f', 'val_loss'),
                               ('%.5f', 'val_mIoU'))

    # -- make test csv_logger
    test_csv_logger = CSVLogger(test_log_file,
                                ('%.5f', 'test_loss'),
                                ('%.5f', 'test_mIoU'))

    # -- init data loader
    (
        train_loader,
        train_sampler,
        val_loader,
        val_sampler,
        test_loader,
        test_sampler
    ) = make_eval_dataloaders(
        dataset_name=dataset_name,
        data_root=data_root,
        img_size=crop_size,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_mem=pin_mem,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank
    )
    ipe = len(train_loader)

    # Get number of input channels from dataset
    in_chans = None
    for img, _ in train_loader:
        in_chans = img.shape[1]
        break

    # -- init encoder
    encoder = vit.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size,
        in_chans=in_chans)
    encoder.to(device)

    # -- init segmentation head
    seg_head = linear_head.linear_segmentation_head(
        encoder=encoder,
        n_cls=seg_n_cls,
        use_batchnorm=use_batchnorm)
    seg_head.to(device)

    logger.info(f'Initialized encoder: {model_name}')
    logger.info(f'Initialized linear segmentation head with {seg_n_cls} classes')

    # Wrap models with DistributedDataParallel before loading checkpoints
    encoder = DistributedDataParallel(encoder, static_graph=True)
    seg_head = DistributedDataParallel(seg_head, static_graph=True)

    # Load pretrained encoder (checkpoint has module.* keys from DDP)
    if pretrained_checkpoint is not None and os.path.exists(pretrained_checkpoint):
        try:
            checkpoint = torch.load(pretrained_checkpoint, map_location=torch.device('cpu'))
            if 'encoder' in checkpoint:
                pretrained_dict = checkpoint['encoder']
                # Remove pos_embed to allow different image sizes
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'pos_embed' not in k}
                msg = encoder.load_state_dict(pretrained_dict, strict=False)
                logger.info(f'Loaded pretrained encoder with msg: {msg}')
            logger.info(f'Loaded pretrained checkpoint from {pretrained_checkpoint}')
        except Exception as e:
            logger.error(f'Error loading pretrained checkpoint: {e}')
            raise e

    # -- init optimizer and scheduler
    if eval_type == 'linear_probing':
        from src.helper import init_opt_linear_probe
        optimizer, scaler, scheduler, wd_scheduler = init_opt_linear_probe(
            seg_head=seg_head,
            wd=wd,
            final_wd=final_wd,
            start_lr=start_lr,
            ref_lr=lr,
            final_lr=final_lr,
            iterations_per_epoch=ipe,
            warmup=warmup,
            num_epochs=num_epochs,
            ipe_scale=ipe_scale,
            use_bfloat16=use_bfloat16)
        logger.info('Using linear probing mode (only seg_head trainable)')
    elif eval_type == 'finetuning':
        from src.helper import init_opt_finetune
        optimizer, scaler, scheduler, wd_scheduler = init_opt_finetune(
            encoder=encoder,
            seg_head=seg_head,
            wd=wd,
            final_wd=final_wd,
            start_lr=start_lr,
            ref_lr=lr,
            final_lr=final_lr,
            iterations_per_epoch=ipe,
            warmup=warmup,
            num_epochs=num_epochs,
            ipe_scale=ipe_scale,
            use_bfloat16=use_bfloat16)
        logger.info('Using finetuning mode (both encoder and seg_head trainable)')
    else:
        raise ValueError(f"Unknown eval_type: {eval_type}. Must be 'linear_probing' or 'finetuning'")

    # Freeze encoder for linear probing
    if eval_type == 'linear_probing':
        for p in encoder.parameters():
            p.requires_grad = False
        logger.info('Encoder frozen for linear probing')

    start_epoch = 0
    best_val_miou = 0.0

    # -- load training checkpoint if resuming
    if load_model and os.path.exists(load_path):
        try:
            checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
            seg_head.load_state_dict(checkpoint['seg_head'])
            # Load encoder if finetuning
            if eval_type == 'finetuning' and 'encoder' in checkpoint:
                encoder.load_state_dict(checkpoint['encoder'])
                logger.info('Loaded encoder state from checkpoint')
            optimizer.load_state_dict(checkpoint['opt'])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            start_epoch = checkpoint.get('epoch', 0)
            best_val_miou = checkpoint.get('best_val_miou', 0.0)
            logger.info(f'Resumed from epoch {start_epoch}')
        except Exception as e:
            logger.info(f'Could not load checkpoint: {e}')

    def save_checkpoint(epoch, is_best=False):
        save_dict = {
            'seg_head': seg_head.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'best_val_miou': best_val_miou,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr,
            'eval_type': eval_type
        }
        # Save encoder state dict for finetuning
        if eval_type == 'finetuning':
            save_dict['encoder'] = encoder.state_dict()

        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))
            if is_best:
                torch.save(save_dict, best_path)
                logger.info(f'Saved best checkpoint with mIoU: {best_val_miou:.4f}')

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        train_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        miou_meter = AverageMeter()
        time_meter = AverageMeter()

        if eval_type == 'linear_probing':
            encoder.eval()
        else:  # finetuning
            encoder.train()
        seg_head.train()

        for itr, (images, masks) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Diagnostic: Check for batches with all masked pixels
            if epoch == 24 and itr == 182:  # epoch+1=25 in logs
                valid_pixels = (masks != -1).sum()
                total_pixels = masks.numel()
                logger.info(f'DIAGNOSTIC [ep={epoch+1}, itr={itr}]: '
                           f'valid_pixels={valid_pixels}/{total_pixels}, '
                           f'batch_size={images.shape[0]}, '
                           f'masks_unique={masks.unique().cpu().tolist()}')

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                def forward():
                    if eval_type == 'linear_probing':
                        with torch.no_grad():
                            # Get encoder features (frozen)
                            h = encoder(images)
                    else:  # finetuning
                        # Get encoder features (trainable)
                        h = encoder(images)

                    # Get segmentation predictions
                    o_size = (masks.shape[2], masks.shape[3])
                    s = seg_head(h, im_size=o_size)
                    return s

                def loss_fn(s):
                    # Segmentation loss
                    if dataset_name == 'substation':
                        seg_loss = F.cross_entropy(
                            s.permute(0, 2, 3, 1).reshape(-1, s.shape[1]),
                            masks.permute(0, 2, 3, 1).reshape(-1),
                            ignore_index=-1,
                            weight=torch.tensor([1.0, 3.0], device=s.device))
                    else:
                        seg_loss = F.cross_entropy(
                            s.permute(0, 2, 3, 1).reshape(-1, s.shape[1]),
                            masks.permute(0, 2, 3, 1).reshape(-1),
                            ignore_index=-1)

                    seg_loss = AllReduce.apply(seg_loss)
                    return seg_loss

                def metric_fn(s):
                    miou = multiclass_jaccard_index(
                        s.permute(0, 2, 3, 1).reshape(-1, s.shape[1]),
                        masks.permute(0, 2, 3, 1).reshape(-1),
                        num_classes=seg_n_cls,
                        ignore_index=-1)
                    return miou

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    s = forward()
                    loss = loss_fn(s)
                    miou = metric_fn(s)

                # Step 2. Backward & step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                if eval_type == 'finetuning':
                    # Log gradients for both encoder and seg_head
                    import itertools
                    grad_stats = grad_logger(itertools.chain(
                        encoder.named_parameters(),
                        seg_head.named_parameters()))
                else:
                    # Log gradients only for seg_head
                    grad_stats = grad_logger(seg_head.named_parameters())
                optimizer.zero_grad()

                return (float(loss), float(miou), _new_lr, _new_wd, grad_stats)

            (loss, miou, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            miou_meter.update(miou)
            time_meter.update(etime)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, miou, etime)

                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f mIoU: %.3f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   miou_meter.avg,
                                   _new_wd,
                                   _new_lr,
                                   torch.cuda.max_memory_allocated() / 1024. ** 2,
                                   time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))

            log_stats()
            assert not np.isnan(loss), 'loss is nan'

        # -- Epoch logging
        logger.info('avg. loss %.3f, avg. mIoU %.3f' % (loss_meter.avg, miou_meter.avg))

        # -- VALIDATION
        encoder.eval()
        seg_head.eval()

        val_loss_meter = AverageMeter()
        val_miou_meter = AverageMeter()

        with torch.no_grad():
            for itr, (images, masks) in enumerate(val_loader):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    # Get encoder features
                    h = encoder(images)

                    # Get segmentation predictions
                    o_size = (masks.shape[2], masks.shape[3])
                    s = seg_head(h, im_size=o_size)

                    # Loss
                    if dataset_name == 'substation':
                        loss = F.cross_entropy(
                            s.permute(0, 2, 3, 1).reshape(-1, s.shape[1]),
                            masks.permute(0, 2, 3, 1).reshape(-1),
                            ignore_index=-1,
                            weight=torch.tensor([1.0, 3.0], device=s.device))
                    else:
                        loss = F.cross_entropy(
                            s.permute(0, 2, 3, 1).reshape(-1, s.shape[1]),
                            masks.permute(0, 2, 3, 1).reshape(-1),
                            ignore_index=-1)

                    # mIoU
                    miou = multiclass_jaccard_index(
                        s.permute(0, 2, 3, 1).reshape(-1, s.shape[1]),
                        masks.permute(0, 2, 3, 1).reshape(-1),
                        num_classes=seg_n_cls,
                        ignore_index=-1)

                val_loss_meter.update(float(loss))
                val_miou_meter.update(float(miou))

        logger.info(f'Validation - Loss: {val_loss_meter.avg:.4f}, mIoU: {val_miou_meter.avg:.4f}')

        # Log validation results to CSV
        val_csv_logger.log(epoch + 1, val_loss_meter.avg, val_miou_meter.avg)

        # Save checkpoint
        is_best = val_miou_meter.avg > best_val_miou
        if is_best:
            best_val_miou = val_miou_meter.avg
        save_checkpoint(epoch + 1, is_best=is_best)

    # -- TEST LOOP
    logger.info('Starting test evaluation...')
    encoder.eval()
    seg_head.eval()

    # Load best checkpoint for testing
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        seg_head.load_state_dict(checkpoint['seg_head'])
        # Load encoder if finetuning
        if eval_type == 'finetuning' and 'encoder' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder'])
            logger.info('Loaded encoder state from best checkpoint')
        logger.info(f'Loaded best checkpoint for testing (mIoU: {checkpoint["best_val_miou"]:.4f})')

    test_loss_meter = AverageMeter()
    test_miou_meter = AverageMeter()

    with torch.no_grad():
        for itr, (images, masks) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                # Get encoder features
                h = encoder(images)

                # Get segmentation predictions
                o_size = (masks.shape[2], masks.shape[3])
                s = seg_head(h, im_size=o_size)

                # Loss
                loss = F.cross_entropy(
                    s.permute(0, 2, 3, 1).reshape(-1, s.shape[1]),
                    masks.permute(0, 2, 3, 1).reshape(-1),
                    ignore_index=-1)

                # mIoU
                miou = multiclass_jaccard_index(
                    s.permute(0, 2, 3, 1).reshape(-1, s.shape[1]),
                    masks.permute(0, 2, 3, 1).reshape(-1),
                    num_classes=seg_n_cls,
                    ignore_index=-1)

            test_loss_meter.update(float(loss))
            test_miou_meter.update(float(miou))

            if itr % 50 == 0:
                logger.info(f'Test [{itr}/{len(test_loader)}] '
                           f'Loss: {test_loss_meter.avg:.4f} '
                           f'mIoU: {test_miou_meter.avg:.4f}')

    logger.info('=' * 80)
    logger.info('FINAL TEST RESULTS:')
    logger.info(f'Test Loss: {test_loss_meter.avg:.4f}')
    logger.info(f'Test mIoU: {test_miou_meter.avg:.4f}')
    logger.info('=' * 80)

    # Log test results to CSV
    test_csv_logger.log(test_loss_meter.avg, test_miou_meter.avg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Linear probe evaluation from YAML config")
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config file")
    parser.add_argument("--resume-preempt", action="store_true",
                        help="If set, treat as resumed preemptible run")
    args_ns = parser.parse_args()

    with open(args_ns.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
