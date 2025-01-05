import mmengine
import matplotlib.pyplot as plt
from mmdet.apis import init_detector, inference_detector
import numpy as np
import cv2
import os
import matplotlib.patches as patches

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import defaultdict


def get_annotation(ann_path):
    annotation = mmengine.load(ann_path)
    # 建立 name2index 索引
    name2annid = {}
    for image in annotation['images']:
        ann_id = []
        for i, ann in enumerate(annotation['annotations']):
            if ann['image_id'] == image['id']:
                ann_id.append(i)
        name2annid[image['file_name']] = ann_id
    return annotation, name2annid


def get_gt_box(annotation, name2index, file_name):
    bboxes_coco = [annotation['annotations'][i]['bbox']
                   for i in name2index[file_name]]  # [[x, y, w, h], [x2, y2, w2, h2], ...]
    gt_cat_ids = [annotation['annotations'][i]['category_id']
                  for i in name2index[file_name]]
    bboxes_x1y1x2y2 = [[bboxes_xywh[0], bboxes_xywh[1], bboxes_xywh[0]+bboxes_xywh[2], bboxes_xywh[1] +
                        bboxes_xywh[3]] for bboxes_xywh in bboxes_coco]   # convert to [x1, y1, x2, y2] format
    gt_labels = [gt_cat_id-1 for gt_cat_id in gt_cat_ids]
    return bboxes_x1y1x2y2, gt_labels


def inter_class_nms(bboxes, scores, labels, nms_thr=0.5):
    """
    Perform Non-Maximum Suppression (NMS) for all bounding boxes without class separation.

    Args:
        bboxes (list): List of bounding boxes, each box is [x1, y1, x2, y2].
        scores (list): List of confidence scores for each bounding box.
        labels (list): List of category labels for each bounding box.
        nms_thr (float): IoU threshold for NMS.

    Returns:
        list, list, list: Filtered bboxes, scores, and labels.
    """
    # If no bounding boxes, return empty lists
    if not bboxes:
        return [], [], []

    bboxes = np.array(bboxes)
    scores = np.array(scores)
    labels = np.array(labels)

    # Sort by scores in descending order
    order = np.argsort(scores)[::-1]
    bboxes = bboxes[order]
    scores = scores[order]
    labels = labels[order]

    keep_bboxes = []
    keep_scores = []
    keep_labels = []

    while len(bboxes) > 0:
        # Always keep the box with the highest score
        keep_bboxes.append(bboxes[0])
        keep_scores.append(scores[0])
        keep_labels.append(labels[0])

        if len(bboxes) == 1:
            break  # Only one box left, stop

        # Compute IoU with the remaining boxes
        iou = compute_iou(bboxes[0], bboxes[1:])

        # Keep boxes with IoU <= threshold
        keep_indices = np.where(iou <= nms_thr)[0]
        bboxes = bboxes[keep_indices + 1]  # Shift by 1 to skip the current box
        scores = scores[keep_indices + 1]
        labels = labels[keep_indices + 1]

    return keep_bboxes, keep_scores, keep_labels


def detect(model, img, inter_class_nms_thr=0.4):
    # check whether img is string
    if isinstance(img, str):
        img = cv2.imread(img)
    pred = inference_detector(model, img)
    pred_instances = pred.pred_instances
    scores = pred_instances.scores.tolist()
    bboxes = pred_instances.bboxes.cpu().tolist()
    labels = pred_instances.labels.tolist()
    bboxes, scores, labels = inter_class_nms(
        bboxes, scores, labels, nms_thr=inter_class_nms_thr)
    return bboxes, scores, labels


def compute_iou(box, boxes):
    """
    Compute IoU between a box and a set of boxes.

    Args:
        box (array): Single box [x1, y1, x2, y2].
        boxes (array): Set of boxes [[x1, y1, x2, y2], ...].

    Returns:
        array: IoU values for each box in `boxes`.
    """
    if boxes.shape[0] == 0:  # No remaining boxes
        return np.array([])

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union_area = box_area + boxes_area - inter_area
    # Add small epsilon to avoid division by zero
    iou = inter_area / (union_area + 1e-6)

    return iou


def evaluate_model(model, annotation_file, img_dir, inter_class_nms_thr=0.8, return_result=False, eval_iou=0):
    """
    Evaluate the model using COCO-style evaluation and compute AP@50 for each class.

    Parameters:
        model: The detection model for inference.
        annotation_file: Path to the COCO annotation file.
        img_dir: Directory containing the evaluation images.

    Returns:
        None (prints AP@50 for each class).
    """
    # Load the COCO annotations
    coco_gt = COCO(annotation_file)

    # Prepare results in COCO format
    results = []
    for img_id in coco_gt.getImgIds():
        # Load image info
        img_info = coco_gt.loadImgs([img_id])[0]
        img_path = os.path.join(img_dir, img_info['file_name'])

        # Perform inference
        img = cv2.imread(img_path)
        pred = inference_detector(model, img)
        pred_instances = pred.pred_instances

        # Extract predictions
        bboxes = pred_instances.bboxes.cpu().numpy().tolist()
        scores = pred_instances.scores.cpu().numpy().tolist()
        labels = pred_instances.labels.cpu().numpy().tolist()
        bboxes, scores, labels = inter_class_nms(
            bboxes, scores, labels, nms_thr=inter_class_nms_thr)
        # Convert predictions to COCO format
        for bbox, score, label in zip(bboxes, scores, labels):
            x1, y1, x2, y2 = bbox
            width, height = x2 - x1, y2 - y1
            results.append({
                # Ensure image_id is an int
                "image_id": int(img_id),
                # Ensure category_id is an int
                "category_id": int(label + 1),
                # Convert to float
                "bbox": [float(x1), float(y1), float(width), float(height)],
                "score": float(score)                # Convert score to float
            })
    # print(results)
    # Load predictions into COCO format
    if return_result:
        return coco_gt, results
    coco_dt = coco_gt.loadRes(results)

    # Initialize COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    # coco_eval.params.iouThrs = [iou_threshold]  # Set IoU threshold to 0.50
    coco_eval.evaluate()
    coco_eval.accumulate()
    if eval_iou == 0.25:
        coco_eval.params.iouThrs = np.array([eval_iou])
    coco_eval.summarize()

    # Check annotation file
    from collections import Counter
    category_counts = Counter([ann['category_id']
                              for ann in coco_gt.dataset['annotations']])
    print("\nAnnotations per category:", category_counts)

    # Check predictions
    predicted_categories = [res['category_id'] for res in results]
    print("Predictions per category:", Counter(predicted_categories))

    # Print and collect AP50 by category
    category_ap50 = {}
    print("\nPer-category AP50 values:")
    for idx, cat_id in enumerate(coco_gt.getCatIds()):
        category_name = coco_gt.loadCats(cat_id)[0]["name"]
        # Get per-category AP50 from coco_eval
        # AP50 (IoU=0.5, area=all, maxDets=100)
        ap50 = coco_eval.eval['precision'][0, :, idx, 0, 2].mean()
        if not np.isnan(ap50):  # Handle cases where AP might be NaN
            category_ap50[category_name] = ap50
            print(f"Category: {category_name}, AP50: {ap50:.4f}")
        else:
            category_ap50[category_name] = 0.0
            print(f"Category: {category_name}, AP50: 0.0000")
    print(f'mAP50: {sum(category_ap50.values())/len(category_ap50)}')


def compute_confusion_matrix(coco_gt, coco_results, iou_thr=0.5, num_classes=3):
    """
    Compute extended confusion matrix for object detection, including FP and FN.

    Args:
        coco_gt (COCO): Ground truth COCO annotations.
        coco_results (list): Predictions in COCO format.
        iou_thr (float): IoU threshold for a valid detection.
        num_classes (int): Number of target classes (excluding background).

    Returns:
        np.ndarray: Extended confusion matrix of shape (num_classes+1, num_classes+1).
    """
    # Initialize confusion matrix
    confusion_mat = np.zeros((num_classes + 1, num_classes + 1), dtype=int)

    # Load ground truth annotations
    gt_boxes = defaultdict(list)
    gt_labels = defaultdict(list)
    for img_id in coco_gt.getImgIds():
        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        anns = coco_gt.loadAnns(ann_ids)
        for ann in anns:
            gt_boxes[img_id].append(ann['bbox'])
            gt_labels[img_id].append(ann['category_id'])

    # Prepare predictions by image
    pred_boxes = defaultdict(list)
    pred_scores = defaultdict(list)
    pred_labels = defaultdict(list)
    for res in coco_results:
        img_id = res['image_id']
        pred_boxes[img_id].append(res['bbox'])
        pred_scores[img_id].append(res['score'])
        pred_labels[img_id].append(res['category_id'])

    # Match predictions to ground truth
    for img_id in coco_gt.getImgIds():
        gt_b = np.array(gt_boxes[img_id])
        gt_l = np.array(gt_labels[img_id])
        pred_b = np.array(pred_boxes[img_id])
        pred_s = np.array(pred_scores[img_id])
        pred_l = np.array(pred_labels[img_id])

        # Convert bbox format from [x1, y1, w, h] to [x1, y1, x2, y2]
        if len(gt_b) > 0:
            gt_b[:, 2] += gt_b[:, 0]
            gt_b[:, 3] += gt_b[:, 1]
        if len(pred_b) > 0:
            pred_b[:, 2] += pred_b[:, 0]
            pred_b[:, 3] += pred_b[:, 1]

        matched_gt = set()
        matched_pred = set()

        # IoU matching
        for i, gt_box in enumerate(gt_b):
            ious = compute_iou(gt_box, pred_b)
            valid_preds = np.where(ious > iou_thr)[0]

            if len(valid_preds) > 0:
                # Match to the prediction with the highest score
                best_pred = valid_preds[np.argmax(pred_s[valid_preds])]
                if best_pred not in matched_pred:
                    matched_gt.add(i)
                    matched_pred.add(best_pred)
                    # Convert 1-based category ID to 0-based
                    gt_cls = gt_l[i] - 1
                    pred_cls = pred_l[best_pred] - 1
                    confusion_mat[gt_cls, pred_cls] += 1

        # Count FN for unmatched ground truth
        for i in range(len(gt_b)):
            if i not in matched_gt:
                gt_cls = gt_l[i] - 1
                # FN: Background column
                confusion_mat[gt_cls, num_classes] += 1

        # Count FP for unmatched predictions
        for i in range(len(pred_b)):
            if i not in matched_pred:
                pred_cls = pred_l[i] - 1
                confusion_mat[num_classes, pred_cls] += 1  # FP: Background row

    return confusion_mat


def calculate_precision_recall(confusion_mat, class_names, metrix_class='certain whale'):
    """
    Calculate precision and recall for each class and their averages.

    Args:, class_names (list): List of class names.
    metrix_class (str): The class name to return. certain whale or avg
    """

    num_classes = len(class_names)

    # Initialize precision and recall lists
    precisions = []
    recalls = []
    f1_scores = []

    print("Category-wise Precision and Recall:\n")
    for i in range(num_classes):
        # True Positives (TP) for class i
        tp = confusion_mat[i, i]

        # False Positives (FP) for class i: Sum of column i excluding TP
        fp = np.sum(confusion_mat[:, i]) - tp

        # False Negatives (FN) for class i: Sum of row i excluding TP
        fn = np.sum(confusion_mat[i, :]) - tp

        # Precision: TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Recall: TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
        f1 = 2 * (precision * recall) / (precision +
                                         recall) if (precision + recall) > 0 else 0
        # Append to lists
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        if metrix_class == 'certain whale':
            return precisions[0], recalls[0], f1_scores[0]
        # Print precision and recall for this category
        print(
            f"Class '{class_names[i]}': Precision = {precision:.4f}, Recall = {recall:.4f}", f"F1 Score = {f1:.4f}")

    # Calculate averages
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)

    print("\nOverall Performance:")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    if metrix_class == "avg":
        return avg_precision, avg_recall, avg_f1


def calculate_f1_for_all(confusion_mat, class_names, metrix_class='certain whale'):
    """
    Calculate F1 scores for each class and return them.

    Args:
        confusion_mat (np.ndarray): Confusion matrix.
        class_names (list): List of class names.
        metrix_class (str): The class name to return. certain whale or avg
    """

    num_classes = len(class_names)

    # Initialize F1 score list
    f1_scores = []

    print("Category-wise F1 Scores:\n")
    for i in range(num_classes):
        # True Positives (TP) for class i
        tp = confusion_mat[i, i]

        # False Positives (FP) for class i: Sum of column i excluding TP
        fp = np.sum(confusion_mat[:, i]) - tp

        # False Negatives (FN) for class i: Sum of row i excluding TP
        fn = np.sum(confusion_mat[i, :]) - tp

        # Precision: TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Recall: TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
        f1 = 2 * (precision * recall) / (precision +
                                         recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

        # Print F1 score for this category
        print(f"Class '{class_names[i]}': F1 Score = {f1:.4f}")

    # If metrix_class is 'certain whale', return the F1 score for that class
    if metrix_class == 'certain whale':
        return f1_scores[0]

    # If metrix_class is 'avg', calculate and return the average F1 score
    if metrix_class == "avg":
        avg_f1 = np.mean(f1_scores)
        print(f"\nOverall Average F1 Score: {avg_f1:.4f}")
        return avg_f1

    return f1_scores


def combine_whale_categories(confusion_matrix):
    # Combine the first two rows and the first two columns to represent 'whale'
    whale_row = confusion_matrix[0] + confusion_matrix[1]
    harp_seal_row = confusion_matrix[2]
    background_row = confusion_matrix[3]

    # Combine the first two columns
    whale_col = whale_row[0:2].sum()
    harp_seal_col = whale_row[2]
    background_col = whale_row[3]

    # Create the new confusion matrix
    new_confusion_matrix = np.array([
        [whale_col, harp_seal_col, background_col],
        [harp_seal_row[0:2].sum(), harp_seal_row[2], harp_seal_row[3]],
        [background_row[0:2].sum(), background_row[2], background_row[3]]
    ])

    return new_confusion_matrix


# Define colors for each label
COLORS = {
    0: 'red',
    1: '#0080FF',
    2: 'green'
}
class_names = ["certain whale", "uncertain whale", "harp seal"]


def plot_bbox_on_ax(ax, img_rgb, bboxes, labels, title, confidences=None):
    ax.imshow(img_rgb)
    ax.set_title(title)
    if not bboxes:
        print('No box detected')
    else:
        # print(bboxes)
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            label = labels[i]
            # Default to black if label not found
            edgecolor = COLORS.get(label, 'black')
            rect = patches.Rectangle(
                (x1, y1), w, h, linewidth=1, edgecolor=edgecolor, facecolor='none', alpha=0.6)
            ax.add_patch(rect)

            # Draw confidence score if provided
            if confidences is not None and i < len(confidences):
                confidence_text = f'{confidences[i]:.2f}'
                ax.text(rect.xy[0] - 1.3, rect.xy[1] - 1.5, confidence_text,
                        va='center', ha='center', fontsize=5, color=edgecolor,
                        bbox=dict(facecolor='w', lw=0, pad=0.5, alpha=0.5))

    # Create a legend
    handles = [patches.Patch(
        color=color, label=f'{class_names[label]}') for label, color in COLORS.items()]
    ax.legend(handles=handles, loc='upper right', fontsize=12)
    ax.axis('off')  # Hide axes


def visulize_predictions(img, bboxes, gt_bboxes, labels, gt_labels, confidences=None, dpi=100, save_path=None):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, axs = plt.subplots(1, 2, dpi=dpi, figsize=(10, 5))
    plot_bbox_on_ax(axs[0], img_rgb, gt_bboxes,
                    gt_labels, title="Ground Truth")
    plot_bbox_on_ax(axs[1], img_rgb, bboxes, labels,
                    title="Prediction", confidences=confidences)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved as {save_path}")
    plt.show()


def visualize_comparison(img, gt_bboxes_sam, gt_labels_sam, bboxes_sam, labels_sam, scores_sam,
                         gt_bboxes_box, gt_labels_box, bboxes_box, labels_box, scores_box,
                         save_dir=None, filename=None, show_plot=True, dpi=100):
    """
    Visualize and compare SAM and box-based annotations and predictions.

    Parameters:
        img: Original image (numpy array).
        gt_bboxes_sam: Ground truth bounding boxes for SAM annotation.
        gt_labels_sam: Ground truth labels for SAM annotation.
        bboxes_sam: Predicted bounding boxes by SAM-based model.
        labels_sam: Predicted labels by SAM-based model.
        scores_sam: Confidence scores by SAM-based model.
        gt_bboxes_box: Ground truth bounding boxes for box-based annotation.
        gt_labels_box: Ground truth labels for box-based annotation.
        bboxes_box: Predicted bounding boxes by box-based model.
        labels_box: Predicted labels by box-based model.
        scores_box: Confidence scores by box-based model.
        save_dir: Directory to save the visualized image. If None, won't save.
        filename: Name of the image file being processed.
        dpi: DPI for the output image.
    """
    # Convert image to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create a 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(12, 12), dpi=dpi)

    # Plot SAM GT
    plot_bbox_on_ax(axs[0, 0], img_rgb, gt_bboxes_sam, gt_labels_sam,
                    title="SAM-based Annotations", confidences=None)

    # Plot SAM predictions
    plot_bbox_on_ax(axs[0, 1], img_rgb, bboxes_sam, labels_sam,
                    title="YOLO-SAM Predictions", confidences=scores_sam)

    # Plot Box GT
    plot_bbox_on_ax(axs[1, 0], img_rgb, gt_bboxes_box, gt_labels_box,
                    title="Buffer Box-based Annotations", confidences=None)

    # Plot Box predictions
    plot_bbox_on_ax(axs[1, 1], img_rgb, bboxes_box, labels_box,
                    title="YOLO-Buffer Predictions", confidences=scores_box)

    # Adjust layout and save if needed
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{filename}_comparison.png")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Comparison figure saved at {save_path}")

    # Show the figure
    if show_plot:
        plt.show()