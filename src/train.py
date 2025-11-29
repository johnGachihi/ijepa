# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import logging
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks, scale_masks
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch
from src.datasets.imagenet1k import make_imagenet1k
from src.datasets.sen2venus import make_sen2venus_dataloader

from src.helper import (
    load_checkpoint,
    init_model,
    init_opt, init_hr_gram_teacher)
from src.transforms import make_transforms

# --
log_timings = True
log_freq = 10
checkpoint_freq = 10  # 50
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
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    use_hr_gram_loss = args['meta'].get('use_hr_gram_loss', False)
    hr_gram_downsample_method = args['meta'].get('hr_gram_downsample_method', 'bilinear')  # bilinear, bicubic, learned
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    # --
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    # -- dataset specific
    dataset_name = args['data'].get('dataset_name', 'imagenet1k')
    if dataset_name == 'imagenet1k':
        root_path = args['data']['root_path']
        image_folder = args['data']['image_folder']
        in_chans = 3  # RGB images
    elif dataset_name == 'sen2venus':
        dataset_root = args['data']['dataset_root']
        splits_file_path = args['data']['splits_file_path']
        use_hr_image = args['data'].get('use_hr_image', False)
        hr_crop_size = args['data'].get('hr_crop_size', None)
        load_both_images = use_hr_gram_loss  # Load both images when using hr_gram_loss
        if not load_both_images:
            assert hr_crop_size is None, "hr_crop_size should only be set when load_both_images is true"
        in_chans = 4  # RGB + NIR images
    # --

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
    patch_size = args['mask']['patch_size']  # patch-size for model training
    num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
    min_keep = args['mask']['min_keep']  # min number of patches in context block
    enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
    num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
    pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks
    aspect_ratio = args['mask']['aspect_ratio']  # aspect ratio of target blocks
    # --

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
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

    dump = os.path.join(folder, 'params-ijepa.yaml')
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

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    if use_hr_gram_loss:
        csv_logger = CSVLogger(log_file,
                               ('%d', 'epoch'),
                               ('%d', 'itr'),
                               ('%.5f', 'loss'),
                               ('%.5f', 'ijepa loss'),
                               ('%.5f', 'gram loss'),
                               ('%.5f', 'mask-A'),
                               ('%.5f', 'mask-B'),
                               ('%d', 'time (ms)'))
    else:
        csv_logger = CSVLogger(log_file,
                               ('%d', 'epoch'),
                               ('%d', 'itr'),
                               ('%.5f', 'loss'),
                               ('%.5f', 'mask-A'),
                               ('%.5f', 'mask-B'),
                               ('%d', 'time (ms)'))

    # -- init model
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
        in_chans=in_chans)
    target_encoder = copy.deepcopy(encoder)
    hr_gram_teacher = None
    if use_hr_gram_loss:
        hr_gram_teacher = init_hr_gram_teacher(
            device=device,
            model_name=model_name,
            patch_size=patch_size,
            crop_size=hr_crop_size,  # Note: hr_crop_size not crop_size
            in_chans=in_chans
        )

    # -- make data transforms
    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep)

    # -- init data-loaders/samplers
    if dataset_name == 'imagenet1k':
        transform = make_transforms(
            crop_size=crop_size,
            crop_scale=crop_scale,
            gaussian_blur=use_gaussian_blur,
            horizontal_flip=use_horizontal_flip,
            color_distortion=use_color_distortion,
            color_jitter=color_jitter)

        _, unsupervised_loader, unsupervised_sampler = make_imagenet1k(
                transform=transform,
                batch_size=batch_size,
                collator=mask_collator,
                pin_mem=pin_mem,
                training=True,
                num_workers=num_workers,
                world_size=world_size,
                rank=rank,
                root_path=root_path,
                image_folder=image_folder,
                copy_data=copy_data,
                drop_last=True)

    elif dataset_name == 'sen2venus':
        _, unsupervised_loader, unsupervised_sampler = make_sen2venus_dataloader(
                data_root=dataset_root,
                splits_file_path=splits_file_path,
                split='train',
                img_size=crop_size,
                hr_img_size=hr_crop_size,
                use_hr_image=use_hr_image,
                load_both_images=load_both_images,
                batch_size=batch_size,
                collator=mask_collator,
                drop_last=True,
                pin_mem=pin_mem,
                num_workers=num_workers,
                world_size=world_size,
                rank=rank)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    ipe = len(unsupervised_loader)

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
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
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False
    if use_hr_gram_loss:
        hr_gram_teacher = DistributedDataParallel(hr_gram_teacher)
        for p in hr_gram_teacher.parameters():
            p.requires_grad = False
        # Initialize hr_gram_teacher with encoder parameters (skip pos_embed)
        encoder_state = encoder.module.state_dict()
        encoder_state = {k: v for k, v in encoder_state.items() if 'pos_embed' not in k}
        hr_gram_teacher.module.load_state_dict(encoder_state, strict=False)


    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            hr_gram_teacher=hr_gram_teacher,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'hr_gram_teacher': None if hr_gram_teacher is None else hr_gram_teacher.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        if use_hr_gram_loss:
            ijepa_loss_meter = AverageMeter()
            gram_loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):

            def load_imgs():
                # -- unsupervised imgs
                if use_hr_gram_loss:
                    imgs = udata[1].to(device, non_blocking=True)
                    hr_imgs = udata[0].to(device, non_blocking=True)
                else:
                    imgs = udata[0].to(device, non_blocking=True)
                    hr_imgs = None

                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                return (imgs, hr_imgs, masks_1, masks_2)
            imgs, hr_imgs, masks_enc, masks_pred = load_imgs()
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --

                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(imgs)
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                        B = len(h)
                        # -- create targets (masked regions of h)
                        h = apply_masks(h, masks_pred)
                        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                        return h

                def forward_hr_gram_teacher():
                    with torch.no_grad():
                        masks_enc_hr = scale_masks(masks_enc, crop_size, hr_crop_size, patch_size)
                        k = hr_gram_teacher(hr_imgs, masks_enc_hr)
                        return k

                def forward_context():
                    ctx_emb = encoder(imgs, masks_enc)
                    z = predictor(ctx_emb, masks_enc, masks_pred)
                    return z, ctx_emb

                def gram_loss_fn(z, k):
                    # Downscale k (hr gram teacher output) to match z size
                    if imgs.shape[-1] != hr_imgs.shape[-1]:
                        # k has shape [B, N*(scale_factor ** 2), D]; downscale to [B, N, D]
                        B, num_patches, D = k.shape
                        scale_factor = hr_crop_size // crop_size  # HR to LR scale factor
                        N = num_patches // (scale_factor ** 2)

                        # Reshape to (B, N, scale_factor, scale_factor, D)
                        k_reshaped = k.reshape(B, N, scale_factor, scale_factor, D)

                        # Reshape to (B, N, D, scale_factor, scale_factor) for interpolation
                        k_reshaped = k_reshaped.permute(0, 1, 4, 2, 3)

                        # Reshape to (B*N, D, scale_factor, scale_factor) for interpolation
                        k_reshaped = k_reshaped.reshape(B * N, D, scale_factor, scale_factor)

                        # Apply bilinear interpolation to downscale scale_factor x scale_factor → 1x1
                        if hr_gram_downsample_method == 'bilinear' or hr_gram_downsample_method == 'bicubic':
                            k_interp = torch.nn.functional.interpolate(
                                k_reshaped, size=(1, 1), mode=hr_gram_downsample_method, align_corners=False)
                        else:
                            raise NotImplemented()

                        # Reshape back to (B, N, D)
                        k = k_interp.reshape(B, N, D)

                    output_feats = z.float()
                    target_feats = k.float()

                    if True:  # self.apply_norm:
                        target_feats = F.normalize(target_feats, dim=-1)

                    # Compute similarities
                    target_sim = torch.matmul(target_feats, target_feats.transpose(-1, -2))

                    # Patch correlation
                    if True:  # self.apply_norm:
                        output_feats = F.normalize(z, dim=-1)

                    # Compute similarities
                    student_sim = torch.matmul(output_feats, output_feats.transpose(-1, -2))

                    # TODO: consider testing these hparams
                    # if False:  # self.remove_neg:
                    #     target_sim[target_sim < 0] = 0.0
                    #     student_sim[student_sim < 0] = 0.0
                    # elif False:  # self.remove_only_teacher_neg:
                    #     # Remove only the negative sim values of the teacher
                    #     target_sim[target_sim < 0] = 0.0
                    #     student_sim[(student_sim < 0) & (target_sim < 0)] = 0.0

                    return F.mse_loss(student_sim, target_sim)

                def loss_fn(z, h, ctx_emb, k):
                    if use_hr_gram_loss:
                        ijepa_loss = F.smooth_l1_loss(z, h)
                        gram_loss = gram_loss_fn(ctx_emb, k)
                        loss = ijepa_loss + gram_loss
                        loss = AllReduce.apply(loss)
                        return loss, ijepa_loss, gram_loss
                    else:
                        loss = F.smooth_l1_loss(z, h)
                        return loss, 0.0, 0.0

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    h = forward_target()
                    if use_hr_gram_loss:
                        k = forward_hr_gram_teacher()
                    else:
                        k = None
                    z, ctx_emb = forward_context()
                    loss, ijepa_loss, gram_loss = loss_fn(z, h, ctx_emb, k)

                #  Step 2. Backward & step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder and hr gram teacher
                with torch.no_grad():
                    m = next(momentum_scheduler)

                    # target encoder
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                    # hr gram teacher - only update layers, skip pos_embed
                    if use_hr_gram_loss:
                        encoder_params = dict(encoder.module.named_parameters())
                        hr_gram_params = dict(hr_gram_teacher.module.named_parameters())

                        for name, param_q in encoder_params.items():
                            if name in hr_gram_params and 'pos_embed' not in name:
                                param_k = hr_gram_params[name]
                                param_k.data.mul_(m).add_((1. - m) * param_q.detach().data)


                return (float(loss), float(ijepa_loss), float(gram_loss), _new_lr, _new_wd, grad_stats)
            (loss, ijepa_loss, gram_loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            if use_hr_gram_loss:
                ijepa_loss_meter.update(ijepa_loss)
                gram_loss_meter.update(gram_loss)
            time_meter.update(etime)

            # -- Logging
            def log_stats():
                if use_hr_gram_loss:
                    csv_logger.log(epoch + 1, itr, loss, ijepa_loss, gram_loss, maskA_meter.val, maskB_meter.val, etime)
                else:
                    csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)

                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    if use_hr_gram_loss:
                        logger.info('[%d, %5d] loss: %.3f '
                                    'ijepa loss: %.3f '
                                    'gram loss: %.3f '
                                    'masks: %.1f %.1f '
                                    '[wd: %.2e] [lr: %.2e] '
                                    '[mem: %.2e] '
                                    '(%.1f ms)'
                                    % (epoch + 1, itr,
                                       loss_meter.avg,
                                       ijepa_loss_meter.avg,
                                       gram_loss_meter.avg,
                                       maskA_meter.avg,
                                       maskB_meter.avg,
                                       _new_wd,
                                       _new_lr,
                                       torch.cuda.max_memory_allocated() / 1024.**2,
                                       time_meter.avg))
                    else:
                        logger.info('[%d, %5d] loss: %.3f '
                                    'masks: %.1f %.1f '
                                    '[wd: %.2e] [lr: %.2e] '
                                    '[mem: %.2e] '
                                    '(%.1f ms)'
                                    % (epoch + 1, itr,
                                       loss_meter.avg,
                                       maskA_meter.avg,
                                       maskB_meter.avg,
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

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch+1)


if __name__ == "__main__":
    main()
