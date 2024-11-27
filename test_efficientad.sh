#!/bin/bash

/home/zf/Thesis/AnomalyDetection/venv/bin/python3.12  mvtec_ad_evaluation/evaluate_experiment.py \
    --dataset_base_dir '/home/zf/Thesis/Datasets/Real-IAD-mini' \
    --anomaly_maps_dir './output/plastic_nut_mini_imagenet_full_penalty0.5/anomaly_maps/mvtec_ad/' \
    --output_dir './output/plastic_nut_mini_imagenet_full_penalty0.5/metrics/mvtec_ad/' \
    --evaluated_objects plastic_nut