#!/bin/bash

 /home/zf/Thesis/AnomalyDetection/venv/bin/python3.12 efficientad.py \
    --dataset mvtec_ad \
    --subdataset plastic_nut \
    --output_dir output/plastic_nut_mini_100_imagenet_full \
    --model_size medium \
    --weights models/teacher_medium.pth \
    --imagenet_train_path /home/zf/Thesis/Datasets/ImageNet \
    --mvtec_ad_path  /home/zf/Thesis/Datasets/Real-IAD-mini-100 \
    --train_steps 70000 
