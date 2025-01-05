import numpy as np
import cv2


def optimize_index(image, idx, window_size=5):
    """
    Optimizes the annotation point by finding the highest intensity pixel within
    a defined window around the original point using NumPy for efficient computation.

    Args:
        image (np.array): The image array in which to search for the highest intensity pixel.
        x (int): The x-coordinate of the original annotation.
        y (int): The y-coordinate of the original annotation.
        window_size (int): The size of the window around the point to consider for optimization, should be odd.

    Returns:
        (int, int): The optimized x, y coordinates.
    """
    x, y = idx[0], idx[1]
    half_window = window_size // 2

    # Ensure the window is within the bounds of the image
    start_x = max(0, x - half_window)
    end_x = min(image.shape[1], x + half_window + 1)
    start_y = max(0, y - half_window)
    end_y = min(image.shape[0], y + half_window + 1)

    # Extract the window region
    window_region = image[start_y:end_y, start_x:end_x]

    # Find the index of the maximum intensity pixel in the window
    local_max_idx = np.unravel_index(
        np.argmax(window_region), window_region.shape)

    # Convert local index to global index
    opt_y = local_max_idx[0] + start_y
    opt_x = local_max_idx[1] + start_x

    return [opt_x, opt_y]


def count_connected_components(mask):
    """
    Count the number of connected components in a binary mask.

    Parameters:
    - mask (np.array): A 2D numpy array with boolean values, where True represents the foreground.

    Returns:
    - int: The number of connected components in the mask.
    """
    # Convert boolean mask to uint8 (binary 0 and 1), as required by cv2.connectedComponents
    mask_uint8 = mask.astype(np.uint8)
    # Find connected components
    num_labels, _ = cv2.connectedComponents(mask_uint8)
    # Return the number of labels, subtracting one if you want to ignore the background component
    return num_labels - 1  # Subtract 1 to exclude the background label if needed


def post_process_mask(mask, input_box):
    """
    Post-process the mask to ensure it fits within the input box and is continuous.

    Parameters:
    mask (np.array): The binary mask to be processed with shape (1, width, height).
    input_box (list): The bounding box [x1, y1, x2, y2] specifying the region of interest.

    Returns:
    np.array: The post-processed binary mask with shape (1, width, height).
    """
    # Ensure mask has the shape (1, width, height)
    if len(mask.shape) != 3 or mask.shape[0] != 1:
        raise ValueError(
            "The input mask must have the shape (1, width, height)")

    mask = mask[0]  # Remove the first dimension to simplify processing
    x1, y1, x2, y2 = map(int, input_box)  # Convert to integers

    # Step 1: Apply bounding box constraint
    mask_cropped = mask

    # Step 2: Connected components analysis to select the largest component
    num_labels, labels_im = cv2.connectedComponents(
        mask_cropped.astype(np.uint8))

    # Find the largest connected component
    if num_labels > 1:
        max_label = 1 + np.argmax([np.sum(labels_im == i)
                                  for i in range(1, num_labels)])
        largest_mask = (labels_im == max_label)
    else:
        largest_mask = mask_cropped

    # Add the first dimension back to the mask
    largest_mask = largest_mask[np.newaxis, ...]

    return largest_mask


def refine_overlapping_masks(masks, points):
    """
    Refine overlapping masks by assigning overlapping pixels to the nearest annotated point.

    Parameters:
    - masks (list of np.array): List of binary masks for each object.
    - points (list of list): List of [x, y] coordinates for each object's annotation point.

    Returns:
    - list of np.array: Refined masks with no overlaps.
    """
    # Create a single combined mask with unique labels for each mask
    combined_mask = np.zeros_like(masks[0], dtype=np.int32)
    for i, mask in enumerate(masks):
        combined_mask[mask > 0] = i + 1  # Assign a unique label to each mask

    # Identify overlapping regions
    overlap_mask = np.sum(np.stack(masks, axis=0), axis=0) > 1

    if np.any(overlap_mask):
        # For overlapping regions, assign pixels to the nearest annotation point
        height, width = masks[0].shape
        y_indices, x_indices = np.meshgrid(
            np.arange(height), np.arange(width), indexing='ij')
        distance_maps = []

        for i, point in enumerate(points):
            # Compute distance map for each point
            dist_map = np.sqrt(
                (x_indices - point[0])**2 + (y_indices - point[1])**2)
            distance_maps.append(dist_map)

        # Stack distance maps and find the nearest point for each pixel
        distance_maps = np.stack(distance_maps, axis=0)
        nearest_mask_indices = np.argmin(distance_maps, axis=0)

        # Update assignments in overlapping regions
        for i in range(len(masks)):
            mask = (nearest_mask_indices == i) & overlap_mask
            combined_mask[mask] = i + 1

    # Generate refined masks
    refined_masks = [(combined_mask == (i + 1)).astype(np.uint8)
                     for i in range(len(masks))]
    return refined_masks


def filter_largest_connected_component(mask):
    """
    Retain only the largest connected component in a binary mask.

    Parameters:
    - mask (np.array): A binary mask (2D array) where foreground pixels are 1 and background pixels are 0.

    Returns:
    - np.array: A binary mask with only the largest connected component retained.
    """
    # Ensure the mask is binary
    mask = mask.astype(np.uint8)

    # Find connected components
    num_labels, labels_im = cv2.connectedComponents(mask)

    if num_labels <= 1:
        # Return the original mask if no connected components are found
        return mask

    # Measure the area of each connected component
    areas = [np.sum(labels_im == i) for i in range(1, num_labels)]

    # Find the label of the largest connected component
    largest_label = 1 + np.argmax(areas)

    # Create a binary mask for the largest connected component
    largest_component_mask = (labels_im == largest_label).astype(np.uint8)

    return largest_component_mask
