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
    return default_transform(image)

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
    full_train_set = ImageFolderWithoutTarget(
        os.path.join(dataset_path, config.subdataset, 'train'),
        transform=transforms.Lambda(train_transform))
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
    #autoencoder = get_autoencoder(out_channels)

    # teacher frozen
    teacher.eval()
    student.train()
    #autoencoder.train()

    if on_gpu:
        teacher.cuda()
        student.cuda()
        #autoencoder.cuda()
        logger.info("training is running on GPU!")
    else:
        logger.info("training is running on CPU!!!")

    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(0.95 * config.train_steps), gamma=0.1)
    tqdm_obj = tqdm(range(config.train_steps))
    

    logger.info(f"coefficients of losses: hard:{config.coeff_hard}, penalty:{config.coeff_penalty}, stae:{config.coeff_stae}, ae:{config.coeff_ae}")
    #training loop
    for iteration, image_st, image_penalty in zip(tqdm_obj, train_loader_infinite, penalty_loader_infinite):

        if on_gpu:
            image_st = image_st.cuda()
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

        #ae_output = autoencoder(image_ae)
        #with torch.no_grad():
            #teacher_output_ae = teacher(image_ae)
            #teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std
        #student_output_ae = student(image_ae)[:, out_channels:]
        #distance_ae = (teacher_output_ae - ae_output) ** 2
        #distance_stae = (ae_output - student_output_ae) ** 2
        #loss_ae = torch.mean(distance_ae)
        #loss_stae = torch.mean(distance_stae)
        #loss_total =  loss_st

        optimizer.zero_grad()
        loss_st.backward()
        optimizer.step()
        scheduler.step()

        if iteration % 10 == 0:
            tqdm_obj.set_description(
                "Current loss: {:.4f}  ".format(loss_st.item()))

        if iteration % 1000 == 0:
            torch.save(teacher, os.path.join(train_output_dir,
                                            'teacher_tmp.pth'))
            torch.save(student, os.path.join(train_output_dir,
                                            'student_tmp.pth'))
            logger.info(f"Checkpoint saved at iteration {iteration}")
            if pretrain_penalty:
                logger.info(
                    f"l_total:{loss_st.item():.4f}, "
                    f"l_hard: {config.coeff_hard * loss_hard.item():.4f}, "
                    f"l_ood: {config.coeff_penalty * loss_penalty.item():.4f},")
            else:
                logger.info(
                    f"l_total:{loss_st.item():.4f}, "
                    f"l_hard: {config.coeff_hard * loss_hard.item():.4f}")
                                  

        if iteration % 10000 == 0 and iteration > 0:
            # run intermediate evaluation
            logger.info("Running intermediate evaluation...")
            teacher.eval()
            student.eval()
            #autoencoder.eval()

            q_st_start, q_st_end = map_normalization(
                validation_loader=validation_loader, teacher=teacher,
                student=student,
                teacher_mean=teacher_mean, teacher_std=teacher_std,
                desc='Intermediate map normalization')
            auc = test(
                test_set=test_set, teacher=teacher, student=student,
                teacher_mean=teacher_mean,
                teacher_std=teacher_std, q_st_start=q_st_start,
                q_st_end=q_st_end,
                test_output_dir=None, desc='Intermediate inference')
            logger.info(f"Intermediate AUC at iteration {iteration}: {auc:.4f}")

            # teacher frozen
            teacher.eval()
            student.train()
            #autoencoder.train()

    teacher.eval()
    student.eval()
    #autoencoder.eval()

    torch.save(teacher, os.path.join(train_output_dir, 'teacher_final.pth'))
    torch.save(student, os.path.join(train_output_dir, 'student_final.pth'))
    logger.info("Final models saved.")

    q_st_start, q_st_end = map_normalization(
        validation_loader=validation_loader, teacher=teacher, student=student,
        teacher_mean=teacher_mean,
        teacher_std=teacher_std, desc='Final map normalization')
    auc = test(
        test_set=test_set, teacher=teacher, student=student,
        teacher_mean=teacher_mean,
        teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
        test_output_dir=test_output_dir, desc='Final inference')
    logger.info(f"Final AUC: {auc:.4f}")


def test(test_set, teacher, student, teacher_mean, teacher_std,
         q_st_start, q_st_end, test_output_dir=None, desc='Running inference'):
    y_true = []
    y_score = []
    for image, target, path in tqdm(test_set, desc=desc):
        orig_width, orig_height = image.width, image.height
        image = default_transform(image).unsqueeze(0)
        if on_gpu:
            image = image.cuda()

        map_st = predict(image, teacher, student, teacher_mean, teacher_std,
                         q_st_start, q_st_end)
        map_st = torch.nn.functional.interpolate(map_st, (orig_height, orig_width), mode='bilinear')
        map_st = map_st[0, 0].cpu().numpy()

        defect_class = os.path.basename(os.path.dirname(path))
        if test_output_dir:
            img_nm = os.path.splitext(os.path.basename(path))[0]
            defect_path = os.path.join(test_output_dir, defect_class)
            os.makedirs(defect_path, exist_ok=True)
            tifffile.imwrite(os.path.join(defect_path, f"{img_nm}.tiff"), map_st)

        y_true.append(0 if defect_class == 'good' else 1)
        y_score.append(np.max(map_st))

    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    return auc * 100


@torch.no_grad()
def predict(image, teacher, student, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None):
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)[:, :out_channels]

    # Calculate the anomaly map using the teacher-student difference
    map_st = torch.mean((teacher_output - student_output) ** 2, dim=1, keepdim=True)

    # Normalize the map if quantiles are provided
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)

    return map_st



@torch.no_grad()
@torch.no_grad()
def map_normalization(validation_loader, teacher, student, teacher_mean, teacher_std, desc='Map normalization'):
    maps_st = []
    for image in tqdm(validation_loader, desc=desc):
        if on_gpu:
            image = image.cuda()
        map_st = predict(image, teacher, student, teacher_mean, teacher_std)
        maps_st.append(map_st)
    maps_st = torch.cat(maps_st)
    q_st_start = torch.quantile(maps_st, 0.9)
    q_st_end = torch.quantile(maps_st, 0.995)
    return q_st_start, q_st_end



@torch.no_grad()
def teacher_normalization(teacher, train_loader):

    mean_outputs = []
    for train_image in tqdm(train_loader, desc='Computing mean of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for train_image in tqdm(train_loader, desc='Computing std of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std

if __name__ == '__main__':
    main()