"""
Utility functions that compute a ROC curve and integrate its area from a set
of anomaly maps and corresponding ground truth classification labels.
"""
import numpy as np


def compute_threshold(
        anomaly_maps,
        scoring_function,
        ground_truth_labels):
    """Compute the ROC curve for anomaly classification on the image level.

    Args:
        anomaly_maps: List of anomaly maps (2D numpy arrays) that contain
          a real-valued anomaly score at each pixel.
        scoring_function: Function that turns anomaly maps into a single
          real valued anomaly score.

        ground_truth_labels: List of integers that indicate the ground truth
          class for each input image. 0 corresponds to an anomaly-free sample
          while a value != 0 indicates an anomalous sample.

    Returns:
        fprs: List of false positive rates.
        tprs: List of correspoding true positive rates.
    """
    assert len(anomaly_maps) == len(ground_truth_labels)

    # Compute the anomaly score for each anomaly map.
    anomaly_scores = map(scoring_function, anomaly_maps)
    num_scores = len(anomaly_maps)

    # Sort samples by anomaly score. Keep track of ground truth label.
    sorted_samples = \
        sorted(zip(anomaly_scores, ground_truth_labels), key=lambda x: x[0])

    # Compute the number of OK and NOK samples from the ground truth.
    ground_truth_labels_np = np.array(ground_truth_labels)
    num_nok = ground_truth_labels_np[ground_truth_labels_np != 0].size
    num_ok = ground_truth_labels_np[ground_truth_labels_np == 0].size

    # Keep track of the current number of false and true positive predictions.
    num_fp = num_ok
    num_tp = num_nok

    # Compute new true and false positive rates when successively increasing
    # the threshold.
    next_score = None
    acc = 0
    best_acc = 0
    threshold = sorted_samples[0][0]

    for i, (current_score, label) in enumerate(sorted_samples):

        if label == 0:
            num_fp -= 1
        else:
            num_tp -= 1

        if i < num_scores - 1:
            next_score = sorted_samples[i + 1][0]
        else:
            next_score = None  # end of list

        if (next_score != current_score) or (next_score is None):
            acc = (num_tp + num_ok - num_fp)/(num_ok + num_nok)
            if acc >= best_acc:
              best_acc = acc
              threshold = (next_score + current_score)/2

    return threshold, best_acc


def compute_Fscore(
        beta,
        anomaly_maps,
        scoring_function,
        ground_truth_labels):
        
    assert len(anomaly_maps) == len(ground_truth_labels)

    # Compute the anomaly score for each anomaly map.
    anomaly_scores = map(scoring_function, anomaly_maps)
    num_scores = len(anomaly_maps)

    # Sort samples by anomaly score. Keep track of ground truth label.
    sorted_samples = \
        sorted(zip(anomaly_scores, ground_truth_labels), key=lambda x: x[0])

    # Compute the number of OK and NOK samples from the ground truth.
    ground_truth_labels_np = np.array(ground_truth_labels)
    num_nok = ground_truth_labels_np[ground_truth_labels_np != 0].size
    num_ok = ground_truth_labels_np[ground_truth_labels_np == 0].size

    # Keep track of the current number of false and true positive predictions.
    num_fp = num_ok
    num_tp = num_nok

    # Compute new true and false positive rates when successively increasing
    # the threshold.
    next_score = None
    Fscore = 0
    best_Fscore = 0
    threshold = sorted_samples[0][0]

    for i, (current_score, label) in enumerate(sorted_samples):

        if label == 0:
            num_fp -= 1
        else:
            num_tp -= 1

        if i < num_scores - 1:
            next_score = sorted_samples[i + 1][0]
        else:
            next_score = None  # end of list

        if (next_score != current_score) or (next_score is None):
            Fscore = (1+beta**2)*num_tp/((1+beta**2)*num_tp+(beta**2)*(num_nok-num_tp)+num_fp)
            if Fscore >= best_Fscore:
              best_Fscore = Fscore
              threshold = (next_score + current_score)/2

    return threshold, best_Fscore


def main():
    """
    Compute the area under the ROC curve for a toy dataset and an algorithm
    that randomly assigns anomaly scores to each image pixel.
    """

    from generic_util import trapezoid, generate_toy_dataset

    # Generate a toy dataset.
    anomaly_maps, _ = generate_toy_dataset(
        num_images=10000, image_width=30, image_height=30, gt_size=0)

    # Assign a random classification label to each image.
    np.random.seed(42)
    labels = np.random.randint(2, size=len(anomaly_maps))

    # Compute the threshold
    threshold, best_acc = compute_threshold(anomaly_maps=anomaly_maps,
                                                    scoring_function=np.max,
                                                    ground_truth_labels=labels)
    print(f"Threshold: {threshold} Best Accuracy: {best_acc}")


if __name__ == "__main__":
    main()