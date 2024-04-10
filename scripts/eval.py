import numpy as np


def calculate_mAP(predictions, ground_truth):
    """
    Calculate the mean Average Precision (mAP) from predictions and ground truth.

    Parameters:
        predictions (dict): Predictions dictionary containing 'boxes', 'labels', and 'scores'.
        ground_truth (dict): Ground truth dictionary containing 'boxes' and 'labels'.

    Returns:
        float: Mean Average Precision (mAP) value.
    """
    # Assuming predictions and ground truth contain bounding boxes in format (x1, y1, x2, y2)
    # You may need to adjust this based on your actual data format
    # Also assuming labels are integers

    # Convert predictions to format required for evaluation
    pred_boxes = predictions[0]["boxes"]
    pred_labels = predictions[0]["labels"]
    pred_scores = predictions[0]["scores"]

    gt_boxes = ground_truth["boxes"]
    gt_labels = ground_truth["labels"]

    # Calculate AP for each class
    # Here, we'll calculate AP for a single class (you may need to adjust this for multi-class)
    # We assume that the class of interest is label 1, you can adjust this as per your data
    class_of_interest = 1

    # Select predictions and ground truth for the class of interest
    pred_mask = pred_labels == class_of_interest
    gt_mask = gt_labels == class_of_interest

    pred_boxes_class = pred_boxes[pred_mask]
    pred_scores_class = pred_scores[pred_mask]
    gt_boxes_class = gt_boxes[gt_mask]

    # Sort predictions by score
    sorted_indices = np.argsort(pred_scores_class)[::-1]
    pred_boxes_class = pred_boxes_class[sorted_indices]

    # Initialize variables for computing precision and recall
    true_positives = np.zeros(len(pred_boxes_class))
    false_positives = np.zeros(len(pred_boxes_class))
    num_gt_boxes = len(gt_boxes_class)

    # Compute true positives and false positives
    for i, pred_box in enumerate(pred_boxes_class):
        iou_max = 0
        for j, gt_box in enumerate(gt_boxes_class):
            iou = calculate_iou(pred_box, gt_box)
            if iou > iou_max:
                iou_max = iou
                best_gt_index = j

        if iou_max >= 0.5:
            true_positives[i] = 1
            gt_boxes_class = np.delete(gt_boxes_class, best_gt_index, axis=0)
        else:
            false_positives[i] = 1

    # Compute precision and recall
    cum_true_positives = np.cumsum(true_positives)
    cum_false_positives = np.cumsum(false_positives)
    recall = cum_true_positives / num_gt_boxes
    precision = cum_true_positives / (cum_true_positives + cum_false_positives)

    # Compute Average Precision (AP)
    ap = 0
    for i in range(len(precision) - 1):
        ap += (recall[i + 1] - recall[i]) * precision[i + 1]

    return ap


# Helper function to calculate IoU (Intersection over Union)
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = intersection / float(box1_area + box2_area - intersection)

    return iou


# Example usage:
# mAP = calculate_mAP(predictions, ground_truth)
