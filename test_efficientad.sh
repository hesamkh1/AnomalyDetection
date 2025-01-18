#!/bin/bash

/home/zf/Thesis/AnomalyDetection/venv/bin/python3.12  mvtec_ad_evaluation/evaluate_experiment.py \
    --dataset_base_dir '/home/zf/Thesis/Datasets/Real-IAD' \
    --anomaly_maps_dir './output/plastic_nut_imagenet_full_NoAutoencoder/anomaly_maps/mvtec_ad/' \
    --output_dir './output/plastic_nut_imagenet_full_NoAutoencoder/metrics/mvtec_ad/' \
    --evaluated_objects plastic_nut