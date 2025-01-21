#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import itertools
import os
import random
from tqdm import tqdm
from common import get_autoencoder, get_pdn_small, get_pdn_medium, \
    ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader
from sklearn.metrics import roc_auc_score
from logger_setup import *
from MOMDataset import MOMDataset
import torch.nn.functional as F

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='mvtec_ad',
                        choices=['mvtec_ad', 'mvtec_loco'])
    parser.add_argument('-s', '--subdataset', default='bottle',
                        help='One of 15 sub-datasets of Mvtec AD or 5' +
                             'sub-datasets of Mvtec LOCO')
    parser.add_argument('-o', '--output_dir', default='output/1')
    parser.add_argument('-m', '--model_size', default='small',
                        choices=['small', 'medium'])
    parser.add_argument('-w', '--weights', default='models/teacher_small.pth')
    parser.add_argument('-i', '--imagenet_train_path',
                        default='none',
                        help='Set to "none" to disable ImageNet' +
                             'pretraining penalty. Or see README.md to' +
                             'download ImageNet and set to ImageNet path')
    parser.add_argument('-a', '--mvtec_ad_path',
                        default='./mvtec_anomaly_detection',
                        help='Downloaded Mvtec AD dataset')
    parser.add_argument('-b', '--mvtec_loco_path',
                        default='./mvtec_loco_anomaly_detection',
                        help='Downloaded Mvtec LOCO dataset')
    parser.add_argument('-t', '--train_steps', type=int, default=70000)
    parser.add_argument('-l', '--log_path',
                        default='./logs',
                        help='Path of the logs')
    # Add coefficients for loss terms
    parser.add_argument('--coeff_hard', type=float, default=1.0,
                        help='Coefficient for hard loss')
    parser.add_argument('--coeff_penalty', type=float, default=1.0,
                        help='Coefficient for penalty loss')
    parser.add_argument('--coeff_ae', type=float, default=1.0,
                        help='Coefficient for autoencoder loss')
    parser.add_argument('--coeff_stae', type=float, default=1.0,
                        help='Coefficient for student-autoencoder loss')

    return parser.parse_args()

# constants
seed = 42
on_gpu = torch.cuda.is_available()
out_channels = 384
image_size = 256

# data loading
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_ae = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.2),
    transforms.ColorJitter(contrast=0.2),
    transforms.ColorJitter(saturation=0.2)
])

def train_transform(image):
    return default_transform(image), default_transform(transform_ae(image))

def main():
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = get_argparse()

    # Set up logging
    logger = setup_logger(config.log_path)
    
    logger.info(f"The outputs are being saved in {config.log_path}")
    logger.info(f"Data set path: {config.mvtec_ad_path}")
    logger.info(f"Sub Dataset path: {config.subdataset}")
    logger.info(f"number of total iterations: {config.train_steps}")
    logger.info(f"output path: {config.output_dir}")


    if config.dataset == 'mvtec_ad':
        dataset_path = config.mvtec_ad_path
    elif config.dataset == 'mvtec_loco':
        dataset_path = config.mvtec_loco_path
    else:
        raise Exception('Unknown config.dataset')

    pretrain_penalty = True
    if config.imagenet_train_path == 'none':
        pretrain_penalty = False

    # create output dir
    train_output_dir = os.path.join(config.output_dir, 'trainings',
                                    config.dataset, config.subdataset)
    test_output_dir = os.path.join(config.output_dir, 'anomaly_maps',
                                   config.dataset, config.subdataset, 'test')
    os.makedirs(train_output_dir)
    os.makedirs(test_output_dir)

    # load data
    train_transform_final = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    full_train_set = MOMDataset(
        root_dir=os.path.join(dataset_path, config.subdataset, 'train'),
        perturbed=True,            # enable synthetic anomaly patches
        transform=train_transform_final,
        skip_background=False      # or True, if you want to skip background patches
    )



    test_set = ImageFolderWithPath(
        os.path.join(dataset_path, config.subdataset, 'test'))
    if config.dataset == 'mvtec_ad':
        # mvtec dataset paper recommend 10% validation set
        train_size = int(0.9 * len(full_train_set))
        validation_size = len(full_train_set) - train_size
        rng = torch.Generator().manual_seed(seed)
        train_set, validation_set = torch.utils.data.random_split(full_train_set,
                                                           [train_size,
                                                            validation_size],
                                                           rng)
        logger.info(f"training size: {train_size},  validation size: {validation_size}")                                                   
    elif config.dataset == 'mvtec_loco':
        train_set = full_train_set
        validation_set = ImageFolderWithoutTarget(
            os.path.join(dataset_path, config.subdataset, 'validation'),
            transform=transforms.Lambda(train_transform))
    else:
        raise Exception('Unknown config.dataset')


    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              num_workers=4, pin_memory=True)
    train_loader_infinite = InfiniteDataloader(train_loader)
    validation_loader = DataLoader(validation_set, batch_size=1)

    if pretrain_penalty:

        logger.info(f"ImageNet Penalty is available at {config.imagenet_train_path}")
        # load pretraining data for penalty
        penalty_transform = transforms.Compose([
            transforms.Resize((2 * image_size, 2 * image_size)),
            transforms.RandomGrayscale(0.3),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                                  0.225])
        ])
        penalty_set = ImageFolderWithoutTarget(config.imagenet_train_path,
                                               transform=penalty_transform)
        penalty_loader = DataLoader(penalty_set, batch_size=1, shuffle=True,
                                    num_workers=4, pin_memory=True)
        penalty_loader_infinite = InfiniteDataloader(penalty_loader)
    else:
        penalty_loader_infinite = itertools.repeat(None)
        logger.info("ImageNet Penalty is not available!")

    logger.info(f"training is based on {config.model_size} model size")
    # create models
    if config.model_size == 'small':
        teacher = get_pdn_small(out_channels)
        student = get_pdn_small(2 * out_channels)
    elif config.model_size == 'medium':
        teacher = get_pdn_medium(out_channels)
        student = get_pdn_medium(2 * out_channels)
    else:
        raise Exception()
    state_dict = torch.load(config.weights, map_location='cpu')
    teacher.load_state_dict(state_dict)
    autoencoder = get_autoencoder(out_channels)

    # teacher frozen
    teacher.eval()
    student.train()
    autoencoder.train()

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()
        logger.info("training is running on GPU!")
    else:
        logger.info("training is running on CPU!!!")

    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)
    

    optimizer = torch.optim.Adam(itertools.chain(student.parameters(),
                                                 autoencoder.parameters()),
                                 lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(0.95 * config.train_steps), gamma=0.1)
    tqdm_obj = tqdm(range(config.train_steps))
    

    logger.info(f"coefficients of losses: hard:{config.coeff_hard}, penalty:{config.coeff_penalty}, stae:{config.coeff_stae}, ae:{config.coeff_ae}")
    for iteration, (image_st, image_ae, mask_st, name_st), image_penalty in zip(
        tqdm_obj,
        train_loader_infinite,
        penalty_loader_infinite
    ):

        if on_gpu:
            image_st = image_st.cuda()
            image_ae = image_ae.cuda()
            mask_st = mask_st.cuda()   # shape: [B, 1, H, W]
            if image_penalty is not None:
                image_penalty = image_penalty.cuda()
        with torch.no_grad():
            teacher_output_st = teacher(image_st)
            teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std
        student_output_st = student(image_st)[:, :out_channels]
        distance_st = (teacher_output_st - student_output_st) ** 2
        d_hard = torch.quantile(distance_st, q=0.999)
        loss_hard = torch.mean(distance_st[distance_st >= d_hard])

        if image_penalty is not None:
            student_output_penalty = student(image_penalty)[:, :out_channels]
            loss_penalty = torch.mean(student_output_penalty ** 2)
            loss_st = config.coeff_hard * loss_hard + config.coeff_penalty * loss_penalty
        else:
            loss_st = config.coeff_hard * loss_hard

        # -------------------------------------------------------
        # 2) MOM LOSS 
        # -------------------------------------------------------
        # Downsample mask to match teacher-student resolution
        # 1) Resize the mask to match feature resolution
        mask_resized = F.interpolate(
            mask_st, size=(teacher_output_st.shape[2], teacher_output_st.shape[3]), mode='nearest'
        )
        fee = F.normalize(teacher_feats, p=2, dim=1)
        faa = F.normalize(student_feats, p=2, dim=1)

        # 2) Compute the CDO loss
        loss_cdo = cdo_loss_function(
            teacher_feats=fee,      # [B, out_channels, H, W]
            student_feats=faa,      # [B, out_channels, H, W]
            mask=mask_resized,                    # [B, 1, H, W]
            oom=True,                             # If you want OOM weighting
            gamma=1.0,                            # Or parse from config
        )

        loss_st_cdo = loss_st + loss_cdo



        ae_output = autoencoder(image_ae)
        with torch.no_grad():
            teacher_output_ae = teacher(image_ae)
            teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std
        student_output_ae = student(image_ae)[:, out_channels:]
        distance_ae = (teacher_output_ae - ae_output) ** 2
        distance_stae = (ae_output - student_output_ae) ** 2
        loss_ae = torch.mean(distance_ae)
        loss_stae = torch.mean(distance_stae)
        loss_total = (loss_st_cdo +
                      config.coeff_ae * loss_ae +
                      config.coeff_stae * loss_stae)

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        if iteration % 10 == 0:
            tqdm_obj.set_description(
                "Current loss: {:.4f}  ".format(loss_total.item()))

        if iteration % 1000 == 0:
            torch.save(teacher, os.path.join(train_output_dir,
                                            'teacher_tmp.pth'))
            torch.save(student, os.path.join(train_output_dir,
                                            'student_tmp.pth'))
            torch.save(autoencoder, os.path.join(train_output_dir,
                                                'autoencoder_tmp.pth'))
            logger.info(f"Checkpoint saved at iteration {iteration}")
            if pretrain_penalty:
                logger.info(
                    f"l_total:{loss_total.item():.4f}, "
                    f"l_hard: {config.coeff_hard * loss_hard.item():.4f}, "
                    f"l_ood: {config.coeff_penalty * loss_penalty.item():.4f}, "
                    f"l_ae: {config.coeff_ae * loss_ae.item():.4f}, "
                    f"l_stae: {config.coeff_stae * loss_stae.item():.4f},"
                    f"loss_cdo: {loss_cdo.item():.4f}")
            else:
                logger.info(
                    f"l_total:{loss_total.item():.4f}, "
                    f"l_hard: {config.coeff_hard * loss_hard.item():.4f}, "
                    f"l_ae: {config.coeff_ae * loss_ae.item():.4f}, "
                    f"l_stae: {config.coeff_stae * loss_stae.item():.4f}, "
                    f"loss_cdo: {loss_cdo.item():.4f}")
                                  

        if iteration % 10000 == 0 and iteration > 0:
            # run intermediate evaluation
            logger.info("Running intermediate evaluation...")
            teacher.eval()
            student.eval()
            autoencoder.eval()

            q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
                validation_loader=validation_loader, teacher=teacher,
                student=student, autoencoder=autoencoder,
                teacher_mean=teacher_mean, teacher_std=teacher_std,
                desc='Intermediate map normalization')
            auc = test(
                test_set=test_set, teacher=teacher, student=student,
                autoencoder=autoencoder, teacher_mean=teacher_mean,
                teacher_std=teacher_std, q_st_start=q_st_start,
                q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end,
                test_output_dir=None, desc='Intermediate inference')
            logger.info(f"Intermediate AUC at iteration {iteration}: {auc:.4f}")

            # teacher frozen
            teacher.eval()
            student.train()
            autoencoder.train()

    teacher.eval()
    student.eval()
    autoencoder.eval()

    torch.save(teacher, os.path.join(train_output_dir, 'teacher_final.pth'))
    torch.save(student, os.path.join(train_output_dir, 'student_final.pth'))
    torch.save(autoencoder, os.path.join(train_output_dir,
                                         'autoencoder_final.pth'))
    logger.info("Final models saved.")

    q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
        validation_loader=validation_loader, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, desc='Final map normalization')
    auc = test(
        test_set=test_set, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
        q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        test_output_dir=test_output_dir, desc='Final inference')
    logger.info(f"Final AUC: {auc:.4f}")


def test(test_set, teacher, student, autoencoder, teacher_mean, teacher_std,
         q_st_start, q_st_end, q_ae_start, q_ae_end, test_output_dir=None,
         desc='Running inference'):
    y_true = []
    y_score = []
    for image, target, path in tqdm(test_set, desc=desc):
        orig_width = image.width
        orig_height = image.height
        image = default_transform(image)
        image = image[None]
        if on_gpu:
            image = image.cuda()
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
            q_ae_start=q_ae_start, q_ae_end=q_ae_end)
        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_height, orig_width), mode='bilinear')
        map_combined = map_combined[0, 0].cpu().numpy()

        defect_class = os.path.basename(os.path.dirname(path))
        if test_output_dir is not None:
            img_nm = os.path.split(path)[1].split('.')[0]
            if not os.path.exists(os.path.join(test_output_dir, defect_class)):
                os.makedirs(os.path.join(test_output_dir, defect_class))
            file = os.path.join(test_output_dir, defect_class, img_nm + '.tiff')
            tifffile.imwrite(file, map_combined)

        y_true_image = 0 if defect_class == 'good' else 1
        y_score_image = np.max(map_combined)
        y_true.append(y_true_image)
        y_score.append(y_score_image)
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    return auc * 100

@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    map_st = torch.mean((teacher_output - student_output[:, :out_channels])**2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output -
                         student_output[:, out_channels:])**2,
                        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae

@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder,
                      teacher_mean, teacher_std, desc='Map normalization'):
    maps_st = []
    maps_ae = []
    # Now each batch from validation_loader is (image_st, mask_st, label, name)
    # We only need `image_st`. The others can be discarded via _ placeholders.
    for image_st, _, _, _ in tqdm(validation_loader, desc=desc):
        if on_gpu:
            image_st = image_st.cuda()

        # Use your existing `predict` function
        map_combined, map_st, map_ae = predict(
            image=image_st,
            teacher=teacher,
            student=student,
            autoencoder=autoencoder,
            teacher_mean=teacher_mean,
            teacher_std=teacher_std
        )
        maps_st.append(map_st)
        maps_ae.append(map_ae)

    # Concatenate all maps along batch dimension
    maps_st = torch.cat(maps_st, dim=0)
    maps_ae = torch.cat(maps_ae, dim=0)

    # Compute quantiles
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end   = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end   = torch.quantile(maps_ae, q=0.995)

    return q_st_start, q_st_end, q_ae_start, q_ae_end


@torch.no_grad()
def teacher_normalization(teacher, train_loader):
    mean_outputs = []

    # Now each batch from train_loader is (image_st, mask_st, label, name)
    for image_st, _, _, _ in tqdm(train_loader, desc='Computing mean of features'):
        if on_gpu:
            image_st = image_st.cuda()
        # Forward pass through teacher
        teacher_output = teacher(image_st)
        # Spatial average (batch, channels, H, W) -> channels
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)

    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    # Repeat for computing variance
    for image_st, _, _, _ in tqdm(train_loader, desc='Computing std of features'):
        if on_gpu:
            image_st = image_st.cuda()
        teacher_output = teacher(image_st)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)

    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std


def cdo_loss_function(
    teacher_feats:   torch.Tensor,  # [B, C, H, W]
    student_feats:   torch.Tensor,  # [B, C, H, W]
    mask:            torch.Tensor,  # [B, 1, H, W], 0=normal, 1=synthetic anomaly
    oom:             bool = True,   # If you want OOM weighting
    gamma:           float = 4.0,
) -> torch.Tensor:
    """
    Compute single-scale CDO loss:
      L = ( sum_{normal}(w_n * d_p) - sum_{abnormal}(w_s * d_p ) ) / ( sum(w_n) + sum(w_s) )

    Where:
      - d_p = ||teacher_feats - student_feats||^2 at each pixel
      - w_n = ( d_p / mu_p )^gamma  for normal pixels
      - w_s = ( mu_p / d_p )^gamma  for abnormal pixels
      - mu_p = mean(d_p) over all pixels
    """
    B, C, H, W = teacher_feats.shape

    # 1) Flatten everything for easy indexing
    #    shape => [B*H*W, C]
    fe = teacher_feats.permute(0, 2, 3, 1).reshape(-1, C)
    fa = student_feats.permute(0, 2, 3, 1).reshape(-1, C)
    mask_flat = mask.permute(0, 2, 3, 1).reshape(-1)  # 0 or 1

    # 2) Split normal vs. synthetic anomaly
    normal_idx   = (mask_flat < 0.5)
    abnormal_idx = (mask_flat >= 0.5)

    fe_n = fe[normal_idx]  # normal teacher feats
    fa_n = fa[normal_idx]  # normal student feats
    fe_s = fe[abnormal_idx]
    fa_s = fa[abnormal_idx]

    # 3) Squared differences
    #    shape => [num_pixels_in_that_group]
    d_p_n = torch.sum((fe_n - fa_n) ** 2, dim=1)
    d_p_s = torch.sum((fe_s - fa_s) ** 2, dim=1)

    # If no normal or no anomaly pixels, return 0
    if d_p_n.numel() == 0 or d_p_s.numel() == 0:
        return torch.tensor(0.0, device=teacher_feats.device)

    # 4) mu_p = mean(d_p) for ALL pixels combined
    d_p_all = torch.sum((fe - fa) ** 2, dim=1)
    mu_p = d_p_all.mean()  # scalar

    # 5) Weights w_n, w_s for OOM
    if oom:
        w_n = (d_p_n / mu_p) ** gamma   # normal
        w_s = (mu_p / d_p_s) ** gamma  # abnormal
    else:
        # If no OOM weighting, all weights = 1
        w_n = torch.ones_like(d_p_n)
        w_s = torch.ones_like(d_p_s)

    sum_n     = torch.sum(d_p_n * w_n)
    sum_s     = torch.sum(d_p_s * w_s)
    weight_n  = torch.sum(w_n)
    weight_s  = torch.sum(w_s)

    denom = (weight_n + weight_s)
    if denom < 1e-12:
        return torch.tensor(0.0, device=teacher_feats.device)

    # Same formula as CDO: we multiply by B if you want to match CDO code exactly:
    # "loss += ((loss_n - loss_s)/(weight_n + weight_s) * B)"
    # but you can omit "* B" if your batch size is always 1. We'll keep it for fidelity:
    return (sum_n - sum_s) / denom * float(B)


if __name__ == '__main__':
    main()
