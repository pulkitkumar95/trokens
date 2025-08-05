import torch
import numpy as np
from PIL import Image
from collections import defaultdict

colors = [
        # Original 8 colors
        (255, 205, 0),   # Vibrant yellow
        (0, 200, 124),   # Bright jade
        (0, 92, 180),    # Crisp blue
        (226, 26, 91),   # Magenta-red
        (150, 111, 51),  # Caramel brown
        (255, 110, 36),  # Bright orange
        (124, 0, 160),   # Royal purple
        (128, 128, 128), # Medium gray

        # 24 additional colors
        (255, 0, 127),   # Hot pink
        (0, 180, 216),   # Azure blue
        (144, 238, 144), # Light green
        (255, 69, 0),    # Red-orange
        (147, 112, 219), # Medium purple
        (0, 163, 108),   # Sea green
        (255, 174, 66),  # Light orange
        (106, 90, 205),  # Slate blue
        (250, 128, 114), # Salmon
        (72, 209, 204),  # Turquoise
        (255, 218, 185), # Peach
        (153, 50, 204),  # Dark orchid
        (0, 139, 139),   # Dark cyan
        (255, 99, 71),   # Tomato
        (186, 85, 211),  # Medium orchid
        (60, 179, 113),  # Medium sea green
        (221, 160, 221), # Plum
        (100, 149, 237), # Cornflower blue
        (219, 112, 147), # Pale violet red
        (176, 196, 222), # Light steel blue
        (255, 127, 80),  # Coral
        (102, 205, 170), # Medium aquamarine
        (238, 130, 238), # Violet
        (64, 224, 208),  # Turquoise blue
    ]

def convert_points_for_tracking(points_list, labels_list, frames_id_dict=None,
                                component_labels_list=None,
                                use_connected_components=False,
                                device=None):
    """Convert points to query points for tracking.

    Args:
        points_list (list): points list
        labels_list (list): points lael list
        frames_id_dict (dict, optional): frames id dict. Defaults to None.
        component_labels_list (list, optional): component labels list. Defaults to None.
        use_connected_components (bool, optional): use connected components. Defaults to False.
        device (torch.device, optional): device. Defaults to None.

    Returns:
        queries_points_all_frames (torch.Tensor): queries points all frames
        query_labels_all_frames (np.ndarray): query labels all frames
    """

    n_frames = len(points_list)
    assert (n_frames == len(labels_list)), "points_list and labels_list must have the same length"
    #inference_state = predictor.init_state(video_path=video_dir)
    cluster_ids_all_frames = []
    cluster_id = 0

    queries_points_all_frames = []
    query_labels_all_frames = []
    query_component_labels_all_frames = []
    for fid in range(n_frames):
        points = points_list[fid]
        labels = labels_list[fid]
        if use_connected_components:
            component_labels = component_labels_list[fid]
            query_component_labels_all_frames.extend(component_labels)
        else:
            component_labels = None
        #predictor.reset_state(inference_state)


        n_points = len(points)

        if n_points==0:
            continue

        assert (points.shape==(n_points,2)), "points must be of shape (n_points,2)"
        assert (labels.shape==(n_points,)), "labels must be of shape (n_points,)"

        points = np.array(points)


        points = points.reshape(-1,2)
        queries_points = torch.tensor(points, device=device).unsqueeze(0) # B M 2
        # make queries points of shape B,M,3, by padding queries_points[:,:,0]=0, and queries_points[:,:,1:]=original queries_points
        fid = frames_id_dict[fid]

        queries_points = torch.cat([torch.ones_like(queries_points[:,:,:1]).float()*fid, queries_points], dim=2) # B M 3
        queries_points_all_frames.append(queries_points)
        query_labels_all_frames.extend(labels)

    queries_points_all_frames = torch.cat(queries_points_all_frames, dim=1) # B M 3
    query_labels_all_frames = np.array(query_labels_all_frames)
    query_component_labels_all_frames = np.array(query_component_labels_all_frames)
    if use_connected_components:
        unique_labels = np.unique(query_labels_all_frames)
        new_labels = np.zeros_like(query_labels_all_frames)
        unique_labels.sort()
        current_max_label = 0
        for label in unique_labels:
            mask = (query_labels_all_frames == label)
            component_labels = query_component_labels_all_frames[mask]
            new_labels[mask] = current_max_label + component_labels
            current_max_label += np.max(component_labels) + 1
        query_labels_all_frames = new_labels

    return queries_points_all_frames.float(), query_labels_all_frames

def get_cluster_peak_frames(cluster_labels, num_clusters=None):
    """
    Find the frame where each cluster has its maximum presence.

    Args:
        cluster_labels: numpy array of shape [T, H, W] containing cluster labels
        num_clusters: optional, maximum number of clusters to consider

    Returns:
        peak_frames: dictionary mapping cluster_id to frame_id.
    """
    T, H, W = cluster_labels.shape

    # If num_clusters not provided, determine from data
    if num_clusters is None:
        num_clusters = np.max(cluster_labels) + 1

    peak_frames = {}

    # For each cluster, count its presence in each frame
    for cluster_id in range(num_clusters):
        # Count number of pixels/points belonging to this cluster in each frame
        cluster_counts = np.sum(cluster_labels == cluster_id, axis=(1,2))

        # Find frame with maximum count
        if np.any(cluster_counts > 0):  # Only include clusters that appear
            peak_frame = np.argmax(cluster_counts)
            peak_frames[cluster_id] = {
                'frame_id': peak_frame,
                'count': cluster_counts[peak_frame]
            }
    peak_frames_dict = defaultdict(list)
    for cluster_id, peak_frame_dict in peak_frames.items():
        peak_frames_dict[peak_frame_dict['frame_id']].append(cluster_id)


    return peak_frames_dict

def find_connected_components(mask, connectivity=4):
    """Find connected components in a binary mask.

    Args:
        mask: Binary mask of shape (H, W)
        connectivity: 4 or 8 for connectivity type

    Returns:
        components: List of arrays containing (y,x) coordinates for each component
    """
    from scipy.ndimage import label

    # Define connectivity structure
    if connectivity == 4:
        structure = np.array([[0,1,0],
                            [1,1,1],
                            [0,1,0]])
    else:  # connectivity == 8
        structure = np.ones((3,3))

    # Label connected components
    labeled_array, num_features = label(mask, structure=structure)

    # Get coordinates for each component
    components = []
    for i in range(1, num_features + 1):
        y_indices, x_indices = np.where(labeled_array == i)
        component_coords = np.stack([y_indices, x_indices], axis=1)
        components.append(component_coords)

    return components

def sample_points_from_mask(mask, n_points, method='random'):
    """
    Sample N points from a binary mask using different strategies.

    Args:
        mask: Binary mask of shape (H, W)
        n_points: Number of points to sample
        method: Sampling method ('random', 'grid', 'distance', 'contour', 'action', 'balanced')

    Returns:
        points: Array of shape (N, 2) containing (x, y) coordinates
    """
    if method not in ['random', 'grid', 'distance', 'contour', 'action', 'balanced']:
        raise ValueError(f"Unknown sampling method: {method}")

    import cv2

    # Fallback to random sampling if mask is empty
    y_indices, x_indices = np.where(mask)

    if len(y_indices) == 0:
        return np.zeros((0, 2))

    if method == 'balanced':
        # Combined boundary and interior sampling (60% boundary, 40% interior)
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            # Fallback to random sampling if no contours found
            indices = np.random.choice(len(y_indices), size=n_points, replace=len(y_indices) < n_points)
            return np.stack([x_indices[indices], y_indices[indices]], axis=1)

        # Combine all contours
        all_contour_points = np.vstack([cont.squeeze() for cont in contours])

        # Sample boundary points (60% of total points)
        n_boundary_points = int(n_points * 0.6)
        boundary_indices = np.random.choice(
            len(all_contour_points),
            size=n_boundary_points,
            replace=len(all_contour_points) < n_boundary_points
        )
        boundary_points = all_contour_points[boundary_indices]

        # Sample interior points (40% of total points) using distance transform
        n_interior_points = n_points - n_boundary_points
        dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
        dist_transform = dist_transform / np.max(dist_transform)

        probs = dist_transform.flatten()
        probs[~mask.flatten()] = 0
        probs = probs / probs.sum()

        flat_indices = np.random.choice(
            len(probs),
            size=n_interior_points,
            p=probs,
            replace=False
        )
        y_coords = flat_indices // mask.shape[1]
        x_coords = flat_indices % mask.shape[1]
        interior_points = np.stack([x_coords, y_coords], axis=1)

        # Combine boundary and interior points
        points = np.vstack([boundary_points, interior_points])


    elif method == 'random':
        # Simple random sampling
        if len(y_indices) < n_points:
            indices = np.random.choice(len(y_indices), size=n_points, replace=True)
        else:
            indices = np.random.choice(len(y_indices), size=n_points, replace=False)
        points = np.stack([x_indices[indices], y_indices[indices]], axis=1)

    elif method == 'grid':
        # Grid-based sampling that tries to cover the mask uniformly
        from sklearn.cluster import KMeans

        # Get number of available coordinates
        n_available = len(x_indices)

        if n_available < n_points:
            # If we need more points than available, use repetition
            coords = np.stack([x_indices, y_indices], axis=1)
            points = []

            # First add all available points
            points.extend(coords)

            # Then randomly sample remaining points with replacement
            remaining = n_points - n_available
            random_indices = np.random.choice(n_available, size=remaining, replace=True)
            points.extend(coords[random_indices])

            points = np.array(points)

        else:
            # Original grid-based sampling logic
            coords = np.stack([x_indices, y_indices], axis=1)
            kmeans = KMeans(n_clusters=n_points, n_init=1)
            kmeans.fit(coords)

            # For each center, find the closest actual mask point
            centers = kmeans.cluster_centers_
            points = []
            for center in centers:
                distances = np.sqrt(np.sum((coords - center) ** 2, axis=1))
                closest_idx = np.argmin(distances)
                points.append(coords[closest_idx])
            points = np.array(points)

    elif method == 'distance':
        # Distance transform based sampling (focus on skeleton/medial axis)
        from scipy.ndimage import distance_transform_edt

        # Compute distance transform
        dist_transform = distance_transform_edt(mask)

        # Normalize distances
        dist_transform = dist_transform / np.max(dist_transform)

        # Use distances as probabilities for sampling
        probs = dist_transform.flatten()
        probs[~mask.flatten()] = 0  # Zero probability for non-mask pixels
        probs = probs / probs.sum()

        # Sample points based on distance transform
        flat_indices = np.random.choice(
            len(probs),
            size=n_points,
            p=probs,
            replace=False
        )
        y_coords = flat_indices // mask.shape[1]
        x_coords = flat_indices % mask.shape[1]
        points = np.stack([x_coords, y_coords], axis=1)

    elif method == 'contour':
        # Contour-based sampling (focus on boundaries and important features)
        import cv2

        # Find contours
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            return sample_points_from_mask(mask, n_points, method='random')

        # Combine all contours
        all_contour_points = np.vstack([cont.squeeze() for cont in contours])

        # Sample points along contours
        n_contour_points = n_points // 2
        if len(all_contour_points) < n_contour_points:
            contour_indices = np.random.choice(
                len(all_contour_points),
                size=n_contour_points,
                replace=True
            )
        else:
            contour_indices = np.random.choice(
                len(all_contour_points),
                size=n_contour_points,
                replace=False
            )
        contour_points = all_contour_points[contour_indices]

        # Sample remaining points from inside the mask
        inner_points = sample_points_from_mask(
            mask,
            n_points - n_contour_points,
            method='distance'
        )

        # Combine contour and inner points
        points = np.vstack([contour_points, inner_points])

    else:
        raise ValueError(f"Unknown sampling method: {method}")

    return points



# vis utils

def create_overlay_mask(image, labels):
    """Create a colored overlay mask based on clustering labels"""
    # Define 8 distinct and darker colors (R,G,B)
    global colors


    # Convert labels to RGB mask
    h, w = labels.shape
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(len(colors)):
        mask[labels == i] = colors[i]

    # Resize mask to match image size
    mask = Image.fromarray(mask).resize(image.size, Image.Resampling.NEAREST)

    # Blend with original image
    return Image.blend(image, mask, 0.5)


def save_video(frames, save_path):
    frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=100, loop=0)