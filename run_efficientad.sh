#!/bin/bash

 /home/zf/Thesis/AnomalyDetection/venv/bin/python3.12 efficientad_CDO.py \
    --dataset mvtec_ad \
    --subdataset plastic_nut \
    --output_dir output/plastic_nut_CDO_Gamma1 \
    --model_size medium \
    --weights models/teacher_medium.pth \
    --imagenet_train_path none \
    --mvtec_ad_path  /home/zf/Thesis/Datasets/Real-IAD \
    --train_steps 70000 
