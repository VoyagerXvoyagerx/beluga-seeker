import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches


CATEGORIES = ['certain whale', 'certain whale', 'uncertain whale']

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image, alpha=0.5)
    
def show_points(coords, labels, ax, marker_size=1, categories=None):
    """
    Visualize points on a matplotlib axis with different markers and colors based on their labels.

    Parameters:
        coords (numpy.ndarray): A 2D array of point coordinates, where each row represents a point [x, y].
        labels (numpy.ndarray or list): Labels for the points. For binary labels (e.g., positive or negative),
                                         use 1 for positive and 0 for negative. For categorical labels,
                                         provide descriptive strings (e.g., 'certain whale', 'uncertain whale').
        ax (matplotlib.axes.Axes): The matplotlib axis where points will be plotted.
        marker_size (int): The size of the scatter plot markers. Default is 1.
        categories (bool or None): If `None`, assumes binary labels (1 for positive, 0 for negative).
                                   If `True`, assumes categorical labels (e.g., specific categories of points).
    """
    if not categories:  # Binary labels case
        pos_points = coords[labels == 1]  # Positive points
        neg_points = coords[labels == 0]  # Negative points
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size, label='Positive')
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=marker_size, label='Negative')
    else:  # Categorical labels case
        certain_points = coords[[i for i, label in enumerate(categories) if label == 'certain whale']]  # Certain whale points
        uncertain_points = coords[[i for i, label in enumerate(categories) if label == 'uncertain whale']]  # Uncertain whale points
        seal_points = coords[[i for i, label in enumerate(categories) if label == 'harp seal']]  # Harp seal points
        ax.scatter(certain_points[:, 0], certain_points[:, 1], color='red', marker='.', s=marker_size, label='Certain Whale')
        ax.scatter(uncertain_points[:, 0], uncertain_points[:, 1], color='blue', marker='.', s=marker_size, label='Uncertain Whale')
        ax.scatter(seal_points[:, 0], seal_points[:, 1], color='green', marker='.', s=marker_size, label='Harp Seal')
    
        # Add legend to the plot
        ax.legend(loc='best', frameon=True)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=1))

def plot_whale_mask(image, input_points, input_labels, input_box, masks, scores, figsize=(10,10), dpi=100):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(image)
        # show_points(input_points, input_labels, plt.gca())
        # show_box(input_box, plt.gca())
        show_mask(mask, plt.gca())
        plt.contour(mask, colors='b', linewidths=0.6)
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

def visualize_all_masks(image, masks, points, categories=None, figsize=(10, 10)):
    """
    Visualize all masks on the given image with random colors.

    Parameters:
    - image (np.array): The original image as a NumPy array (H x W x 3).
    - masks (list of np.array): A list of binary masks to visualize.
    - figsize (tuple): The size of the visualization figure.
    
    Returns:
    - None: Displays the visualization.
    """
    plt.figure(figsize=figsize)
    plt.imshow(image)
    ax = plt.gca()

    for mask in masks:
        show_mask(mask, ax, random_color=True)  # Use random colors to show each mask
        plt.contour(mask, colors='b', linewidths=0.6)

    # Visualize points
    coords = np.array(points)
    labels = np.ones(len(points))  # Assume all points are foreground
    show_points(coords, labels, ax, marker_size=30, categories=categories)
    
    plt.axis("off")
    plt.title(f"Visualization of {len(masks)} Masks", fontsize=16)
    plt.show()  

def plot_whale_img(image, input_points, input_labels, input_box, masks, scores, figsize=(12,12), dpi=120):
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(image)
    # show_points(input_points, input_labels, plt.gca())
    plt.axis('off')
    plt.show() 

def read_cropped_image(cropped_image_path, filename, save_as_png=False, save_dir=None):
    image = cv2.imread(cropped_image_path, cv2.IMREAD_UNCHANGED)
    image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = image_normalized.astype(np.uint8)
    if len(image.shape) == 2:  # grayscale
        image = cv2.merge([image, image, image])
    elif len(image.shape) == 3 and image.shape[2] == 3:  # BGR2RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pass  # 不做任何处理
    if save_as_png:
        filename = filename[:-4] + '.png'
        cv2.imwrite(os.path.join(save_dir, filename), image)
    return image

def calculate_bounding_box(mask):
    # Get coordinates of non-zero values
    coords = np.column_stack(np.where(mask > 0))
    assert coords.size != 0
    y, x, h, w = cv2.boundingRect(coords)
    return [x, y, w, h]

def visualize_ann_bboxes(image, masks, points, categories=None, figsize=(10, 10)):
    """
    Visualize masks, points, and bounding boxes on the given image.

    Parameters:
    - image (np.array): The original image as a NumPy array (H x W x 3).
    - masks (list of np.array): A list of binary masks to visualize.
    - points (list of list): A list of [x, y] coordinates for each mask's annotation point.
    - figsize (tuple): The size of the visualization figure.
    
    Returns:
    - None: Displays the visualization.
    """
    plt.figure(figsize=figsize)
    plt.imshow(image)
    ax = plt.gca()

    coords = np.array(points)
    labels = np.ones(len(points))  # Assume all points are foreground
    show_points(coords, labels, ax, marker_size=10, categories=categories)

    # Visualize bounding boxes
    for mask in masks:
        bbox = calculate_bounding_box(mask)
        # print(f'bbox: {bbox}')
        if bbox:
            x, y, w, h = bbox
            color = [random.random() for _ in range(3)]  # Generate random color
            rect = plt.Rectangle((x, y), w, h,
                                  edgecolor=color, facecolor='none', linewidth=1, linestyle='-')
            ax.add_patch(rect)

    plt.axis("off")
    plt.title(f"Visualization of Masks, Points, and Bounding Boxes", fontsize=16)
    plt.show()

def visualize_baseline_bboxes(image, refined_masks, all_points, buffer_size=15, categories=None, figsize=(10, 10)):
    """
    Visualize bounding boxes generated by refined_masks and baseline (points + buffer size).

    Parameters:
    - image (np.array): The original image as a NumPy array (H x W x 3).
    - refined_masks (list of np.array): A list of binary masks used to compute bounding boxes.
    - all_points (list of list): A list of [x, y] coordinates for points.
    - buffer_size (int): Half the length of the baseline bounding box's width and height.
    - figsize (tuple): The size of the visualization figure.

    Returns:
    - None: Displays the visualization.
    """
    # Start visualization
    plt.figure(figsize=figsize)
    plt.imshow(image)
    ax = plt.gca()

    # Visualize baseline bounding boxes (points + buffer size)
    for point in all_points:
        x, y = point
        x_min, y_min = x - buffer_size, y - buffer_size
        x_max, y_max = x + buffer_size, y + buffer_size
        color = [random.random() for _ in range(3)]  # Generate random color
        rect = patches.Rectangle((x_min, y_min), 2 * buffer_size, 2 * buffer_size,
                                  edgecolor=color, facecolor='none', linewidth=1, linestyle=':'
                                  )
        ax.add_patch(rect)

    # Visualize points
    coords = np.array(all_points)
    labels = np.ones(len(all_points))  # Assume all points are foreground
    show_points(coords, labels, ax, marker_size=10, categories=categories)

    plt.axis("off")
    plt.title("Comparison of Refined and Baseline Bounding Boxes", fontsize=16)
    plt.show()

def visualize_points(image, all_points, figsize=(10, 10)):
    """
    Visualize bounding boxes generated by refined_masks and baseline (points + buffer size).

    Parameters:
    - image (np.array): The original image as a NumPy array (H x W x 3).
    - refined_masks (list of np.array): A list of binary masks used to compute bounding boxes.
    - all_points (list of list): A list of [x, y] coordinates for points.
    - buffer_size (int): Half the length of the baseline bounding box's width and height.
    - figsize (tuple): The size of the visualization figure.

    Returns:
    - None: Displays the visualization.
    """
    # Start visualization
    plt.figure(figsize=figsize)
    plt.imshow(image)
    ax = plt.gca()

    # Visualize points
    coords = np.array(all_points)
    labels = np.ones(len(all_points))  # Assume all points are foreground
    show_points(coords, labels, ax, marker_size=30)

    plt.axis("off")
    plt.title("Comparison of Refined and Baseline Bounding Boxes", fontsize=16)
    plt.show()

def visualize_baseline_bboxes_individually(image, refined_masks, all_points, buffer_size=12, figsize=(10, 10), output_dir=None):
    """
    Visualize each bounding box and corresponding point in separate images.

    Parameters:
    - image (np.array): The original image as a NumPy array (H x W x 3).
    - refined_masks (list of np.array): A list of binary masks.
    - all_points (list of list): A list of [x, y] coordinates for each mask's annotation point.
    - buffer_size (int): Half the length of the baseline bounding box's width and height.
    - output_dir (str): Directory to save generated images. If None, images are just displayed.

    Returns:
    - None: Displays or saves individual visualizations for each bounding box and point.
    """

    
    for i, (mask, point) in enumerate(zip(refined_masks, all_points)):
        # Create a new figure for each bounding box
        plt.figure(figsize=figsize)
        plt.imshow(image)
        ax = plt.gca()
        
        # Visualize the baseline bounding box
        x, y = point
        x_min, y_min = x - buffer_size, y - buffer_size
        x_max, y_max = x + buffer_size, y + buffer_size
        color = [random.random() for _ in range(3)]  # Generate random color
        rect = patches.Rectangle((x_min, y_min), 2 * buffer_size, 2 * buffer_size,
                                  edgecolor=color, facecolor='none', linewidth=2, linestyle=':')
        ax.add_patch(rect)
        
        # Visualize the point
        ax.scatter(x, y, color="red", s=20, label="Point")
        
        # # Visualize refined mask bounding box
        # bbox = calculate_bounding_box(mask)
        # if bbox:
        #     x, y, w, h = bbox
        #     color = [random.random() for _ in range(3)]
        #     rect_refined = patches.Rectangle((x, y), w, h,
        #                                      edgecolor=color, facecolor='none', linewidth=2, linestyle='--')
        #     ax.add_patch(rect_refined)

        # Set title
        plt.title(f"Visualization {i+1}", fontsize=16)
        plt.axis("off")

        # Save or display the figure
        if output_dir:
            plt.savefig(f"{output_dir}/visualization_{i+1}.png", bbox_inches="tight")
        else:
            plt.show()

        plt.close()