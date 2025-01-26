"""Compute evaluation metrics for a single experiment."""

__author__ = "Paul Bergmann, David Sattlegger"
__copyright__ = "2021, MVTec Software GmbH"

import json
from os import listdir, makedirs, path

import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import generic_util as util
from pro_curve_util import compute_pro
from roc_curve_util import compute_classification_roc
from threshold_computation import compute_threshold, compute_Fscore
import matplotlib.cm as cm  # e.g., "jet", "viridis", etc.


def parse_user_arguments():
    """Parse user arguments for the evaluation of a method on the MVTec AD
    dataset.

    Returns:
        Parsed user arguments.
    """
    parser = argparse.ArgumentParser(description="""Parse user arguments.""")

    parser.add_argument('--anomaly_maps_dir',
                        required=True,
                        help="""Path to the directory that contains the anomaly
                                maps of the evaluated method.""")

    parser.add_argument('--dataset_base_dir',
                        required=True,
                        help="""Path to the directory that contains the dataset
                                images of the MVTec AD dataset.""")

    parser.add_argument('--output_dir',
                        help="""Path to the directory to store evaluation
                                results. If no output directory is specified,
                                the results are not written to drive.""")

    parser.add_argument('--pro_integration_limit',
                        type=float,
                        default=0.3,
                        help="""Integration limit to compute the area under
                                the PRO curve. Must lie within the interval
                                of (0.0, 1.0].""")

    parser.add_argument('--evaluated_objects',
                        nargs='+',
                        help="""List of objects to be evaluated. By default,
                                all dataset objects will be evaluated.""",
                        choices=util.OBJECT_NAMES,
                        default=util.OBJECT_NAMES)
    parser.add_argument('--beta',
                        type=float,
                        default=1,
                        help="""Coefficient to compute the F(beta)score""")

    args = parser.parse_args()

    # Check that the PRO integration limit is within the valid range.
    assert 0.0 < args.pro_integration_limit <= 1.0

    return args


def parse_dataset_files(object_name, dataset_base_dir, anomaly_maps_dir):
    # Same as before:
    test_dir = path.join(dataset_base_dir, object_name, 'test')
    gt_base_dir = path.join(dataset_base_dir, object_name, 'ground_truth')
    anomaly_maps_base_dir = path.join(anomaly_maps_dir, object_name, 'test')

    gt_filenames = []
    prediction_filenames = []
    test_image_filenames = []

    for subdir in listdir(str(test_dir)):
        if not subdir.replace('_', '').isalpha():
            continue

        # Example: all the *.png test images in this subdir
        test_images = [path.splitext(file)[0]
                       for file in listdir(path.join(test_dir, subdir))
                       if path.splitext(file)[1] == '.png']

        # If subdir != 'good', we have ground-truth masks:
        if subdir != 'good':
            gt_filenames.extend(
                [path.join(gt_base_dir, subdir, file + '_mask.png')
                 for file in test_images])
        else:
            gt_filenames.extend([None]*len(test_images))

        # Corresponding anomaly-map filenames
        prediction_filenames.extend(
            [path.join(anomaly_maps_base_dir, subdir, file)
             for file in test_images])

        # Also store the **original test image** path
        test_image_filenames.extend(
            [path.join(test_dir, subdir, file + '.png')
             for file in test_images]
        )

    print(f"Parsed {len(gt_filenames)} ground truth image files for {object_name}.")

    return gt_filenames, prediction_filenames, test_image_filenames



def calculate_au_pro_au_roc(gt_filenames,
                            prediction_filenames,
                            integration_limit):
    """Compute the area under the PRO curve for a set of ground truth images
    and corresponding anomaly images.

    In addition, the function computes the area under the ROC curve for image
    level classification.

    Args:
        gt_filenames: List of filenames that contain the ground truth images
          for a single dataset object.
        prediction_filenames: List of filenames that contain the corresponding
          anomaly images for each ground truth image.
        integration_limit: Integration limit to use when computing the area
          under the PRO curve.

    Returns:
        au_pro: Area under the PRO curve computed up to the given integration
          limit.
        au_roc: Area under the ROC curve.
        pro_curve: PRO curve values for localization (fpr,pro).
        roc_curve: ROC curve values for image level classifiction (fpr,tpr).
    """
    # Read all ground truth and anomaly images.
    ground_truth = []
    predictions = []

    print("Read ground truth files and corresponding predictions...")
    for (gt_name, pred_name) in tqdm(zip(gt_filenames, prediction_filenames),
                                     total=len(gt_filenames)):
        prediction = util.read_tiff(pred_name)
        predictions.append(prediction)

        if gt_name is not None:
            ground_truth.append(np.asarray(Image.open(gt_name)))
        else:
            ground_truth.append(np.zeros(prediction.shape))

    # Compute the PRO curve.
    pro_curve = compute_pro(
        anomaly_maps=predictions,
        ground_truth_maps=ground_truth)

    # Compute the area under the PRO curve.
    au_pro = util.trapezoid(
        pro_curve[0], pro_curve[1], x_max=integration_limit)
    au_pro /= integration_limit
    print(f"AU-PRO (FPR limit: {integration_limit}): {au_pro}")

    # Derive binary labels for each input image:
    # (0 = anomaly free, 1 = anomalous).
    binary_labels = [int(np.any(x > 0)) for x in ground_truth]
    del ground_truth

    # Compute the classification ROC curve.
    roc_curve = compute_classification_roc(
        anomaly_maps=predictions,
        scoring_function=np.max,
        ground_truth_labels=binary_labels)

    # Compute the area under the classification ROC curve.
    au_roc = util.trapezoid(roc_curve[0], roc_curve[1])
    print(f"Image-level classification AU-ROC: {au_roc}")

    # Return the evaluation metrics.
    return au_pro, au_roc, pro_curve, roc_curve


def calculate_threshold(gt_filenames,
                            prediction_filenames):
                    
    """ Compute the best threshold and the best possible accuracy
    (obtained by that threshold setting) """

    # Read all ground truth and anomaly images.
    ground_truth = []
    predictions = []

    print("Read ground truth files and corresponding predictions...")
    for (gt_name, pred_name) in tqdm(zip(gt_filenames, prediction_filenames),
                                     total=len(gt_filenames)):
        prediction = util.read_tiff(pred_name)
        predictions.append(prediction)

        if gt_name is not None:
            ground_truth.append(np.asarray(Image.open(gt_name)))
        else:
            ground_truth.append(np.zeros(prediction.shape))

    # Derive binary labels for each input image:
    # (0 = anomaly free, 1 = anomalous).
    binary_labels = [int(np.any(x > 0)) for x in ground_truth]
    del ground_truth

    # Compute the threshold and accuracy
    threshold, best_acc = compute_threshold(
        anomaly_maps=predictions,
        scoring_function=np.max,
        ground_truth_labels=binary_labels)

    print(f"Optimum Threshold: {threshold}, Best Accuracy: {best_acc}")

    # Return the threshold and the best accuracy associated
    return threshold, best_acc


def calculate_Fscore(beta, gt_filenames,
                            prediction_filenames):
                    
    """ Compute the best Fscore and the best possible threshold
    (obtained by that threshold setting) """

    # Read all ground truth and anomaly images.
    ground_truth = []
    predictions = []

    print("Read ground truth files and corresponding predictions...")
    for (gt_name, pred_name) in tqdm(zip(gt_filenames, prediction_filenames),
                                     total=len(gt_filenames)):
        prediction = util.read_tiff(pred_name)
        predictions.append(prediction)

        if gt_name is not None:
            ground_truth.append(np.asarray(Image.open(gt_name)))
        else:
            ground_truth.append(np.zeros(prediction.shape))

    # Derive binary labels for each input image:
    # (0 = anomaly free, 1 = anomalous).
    binary_labels = [int(np.any(x > 0)) for x in ground_truth]
    del ground_truth

    # Compute the Fscore
    threshold, best_Fscore = compute_Fscore(
        beta = beta,
        anomaly_maps=predictions,
        scoring_function=np.max,
        ground_truth_labels=binary_labels)

    print(f"Optimum Threshold for Fscore: {threshold}, Best F{beta}score: {best_Fscore}")

    # Return the threshold and the best accuracy associated
    return threshold, best_Fscore


def main():
    """Calculate the performance metrics for a single experiment on the
    MVTec AD dataset.
    """
    # Parse user arguments.
    args = parse_user_arguments()

    # Store evaluation results in this dictionary.
    evaluation_dict = dict()

    # Keep track of the mean performance measures.
    au_pros = []
    au_rocs = []

    # Extract beta
    beta = args.beta

    # Metrics
    metrics_dir = path.join(args.output_dir, 'metrics'+'.json')

    # Evaluate each dataset object separately.
    for obj in args.evaluated_objects:
        print(f"=== Evaluate {obj} ===")
        evaluation_dict[obj] = dict()

        # Parse the filenames of all ground truth and corresponding anomaly
        # images for this object.
        gt_filenames, prediction_filenames, test_image_filenames = \
            parse_dataset_files(
                object_name=obj,
                dataset_base_dir=args.dataset_base_dir,
                anomaly_maps_dir=args.anomaly_maps_dir
            )


        # Calculate the PRO and ROC curves.
        au_pro, au_roc, pro_curve, roc_curve = \
            calculate_au_pro_au_roc(
                gt_filenames,
                prediction_filenames,
                args.pro_integration_limit)

        # Calculate threshold and best accuracy.
        threshold, best_acc = \
            calculate_threshold(
              gt_filenames,
              prediction_filenames)

        # Calculate threshold and best Fscore.
        thresholdF, best_Fscore = \
            calculate_Fscore(
              beta,
              gt_filenames,
              prediction_filenames)
        
        save_misclassified_images(
        obj_name=obj,
        test_image_filenames=test_image_filenames,
        gt_filenames=gt_filenames,
        prediction_filenames=prediction_filenames,
        threshold=threshold,
        output_dir=args.output_dir)


        evaluation_dict[obj]['au_pro'] = au_pro
        evaluation_dict[obj]['classification_au_roc'] = au_roc

        evaluation_dict[obj]['classification_roc_curve_fpr'] = roc_curve[0]
        evaluation_dict[obj]['classification_roc_curve_tpr'] = roc_curve[1]

        evaluation_dict[obj]['optimal_threshold'] = threshold
        evaluation_dict[obj]['best_accuracy'] = best_acc

        evaluation_dict[obj][f'threshold_F{beta}score'] = thresholdF
        evaluation_dict[obj][f'best_F{beta}score'] = best_Fscore

        # Keep track of the mean performance measures.
        au_pros.append(au_pro)
        au_rocs.append(au_roc)

        print('\n')

    # Compute the mean of the performance measures.
    evaluation_dict['mean_au_pro'] = np.mean(au_pros).item()
    evaluation_dict['mean_classification_au_roc'] = np.mean(au_rocs).item()

    # If required, write evaluation metrics to drive.
    if args.output_dir is not None:
        makedirs(args.output_dir, exist_ok=True)

        clean_dict(evaluation_dict)

        with open(path.join(args.output_dir, 'metrics.json'), 'w') as file:
            json.dump(evaluation_dict, file, indent=4)

        print(f"Wrote metrics to {args.output_dir}")


def clean_dict(d):
    for key, value in d.items():
        if isinstance(value, np.float32):
            d[key] = float(value)
        elif isinstance(value, np.integer):
            d[key] = int(value)
        elif isinstance(value, dict):
            clean_dict(value)
        elif isinstance(value, np.ndarray):
            d[key] = value.tolist()

def save_misclassified_images(obj_name,
                              test_image_filenames,
                              gt_filenames,
                              prediction_filenames,
                              threshold,
                              output_dir):
    """
    Saves test images + anomaly maps that are misclassified at image level.
    A misclassification is:
      - ground-truth = normal, predicted anomaly
      - ground-truth = anomaly, predicted normal
    """
    from PIL import Image
    import numpy as np
    import os

    wrong_dir = path.join(output_dir, "wrong_predictions", obj_name)
    makedirs(wrong_dir, exist_ok=True)

    # Prepare ground-truth *binary labels* for each image:
    #   0 => normal (subdir 'good'), 1 => anomalous.
    # Because 'gt_filenames[i] = None' if subdir == 'good'.
    binary_labels = []
    for gt_file in gt_filenames:
        if gt_file is not None:
            binary_labels.append(1)
        else:
            binary_labels.append(0)

    # Loop over all images
    for i in range(len(test_image_filenames)):
        test_image_path = test_image_filenames[i]
        pred_map_path   = prediction_filenames[i]
        gt_label        = binary_labels[i]

        # Read the predicted anomaly map
        anomaly_map = util.read_tiff(pred_map_path)   # shape: (H,W), float
        score       = np.max(anomaly_map)
        pred_label  = 1 if score > threshold else 0

        # If misclassified, save to disk
        if pred_label != gt_label:
            base = path.splitext(path.basename(test_image_path))[0]

            # 1) Save original test image
            test_img = Image.open(test_image_path).convert('RGB')
            out_img_path = path.join(wrong_dir, f"{base}_orig.png")
            test_img.save(out_img_path)

            # 2) Save anomaly map as grayscale
            #    (Optionally normalize to [0..255] for better viewing)
            max_val = anomaly_map.max()
            if max_val > 1e-12:
                anomaly_map_255 = (anomaly_map / max_val * 255).astype(np.uint8)
            else:
                anomaly_map_255 = anomaly_map.astype(np.uint8)
            
            out_map_path = path.join(wrong_dir, f"{base}_map.png")
            save_colored_heatmap(anomaly_map_255, out_map_path)
            
            # (Optional) You could also overlay the map on the original image
            # but for simplicity, we just save them separately.

            print(f"Saved misclassified sample: {out_img_path}, {out_map_path}")



def save_colored_heatmap(anomaly_map, save_path):
    """
    Saves the given anomaly_map as a color-coded heatmap using e.g. the 'jet' colormap.
    """
    # 1) Normalize anomaly map to [0,1] for colormap
    amin, amax = anomaly_map.min(), anomaly_map.max()
    if amax > amin:
        anomaly_map_norm = (anomaly_map - amin) / (amax - amin)
    else:
        anomaly_map_norm = anomaly_map  # all zeros or a single value

    # 2) Apply a matplotlib colormap (e.g. 'jet')
    colormap = cm.get_cmap('jet')
    colored_map = colormap(anomaly_map_norm)  # shape: (H,W,4) RGBA

    # 3) Convert [0..1] float RGBA → uint8 [0..255], drop alpha if desired
    colored_map = np.uint8(255 * colored_map[..., :3])  # shape: (H,W,3)

    # 4) Create PIL Image and save
    heatmap_pil = Image.fromarray(colored_map)
    heatmap_pil.save(save_path)
    # Optionally return it if you want to do something else
    # return heatmap_pil


if __name__ == "__main__":
    main()
