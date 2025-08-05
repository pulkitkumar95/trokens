import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from einops import rearrange
import torch.nn.functional as F
import os
import socket
import cv2
import random
from matplotlib import cm
from einops import rearrange
import colorsys
from collections import defaultdict
import pickle
import argparse
import pandas as pd
import sys
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hod import run_hod_obj_id_sampling_per_class
from clustering import get_temporal_bipartite_clusters, TorchKMeansVectorizedCluster
#set seeds
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

import warnings

os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'


class feature_extract(nn.Module):
    def __init__(self):
        super().__init__()
        # Add SigLIP initialization
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "mps" if torch.backends.mps.is_available()
                                   else "cpu")
        self.dinov2.to(self.device)
        self.dinov2.eval()


    @torch.no_grad()
    def forward(self, frames, model_type='dino'):
        """
        Args:
            frames: numpy array of shape (bs, num_frames, height, width, channel) in uint8 format
            model_type: one of 'dino', 'clip', or 'siglip'
        """
        batch_size, num_frames, _, _, _ = frames.shape

        if model_type == 'dino':
            # DINO preprocessing
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])

            # Process all frames
            processed_frames = []
            for i in range(batch_size):
                for j in range(num_frames):
                    frame_pil = Image.fromarray(frames[i, j])
                    frame_tensor = transform(frame_pil)
                    processed_frames.append(frame_tensor)

            # Stack and reshape
            x = torch.stack(processed_frames)
            x = x.to(self.device)

            # Get features
            feat = self.dinov2.forward_features(x)['x_norm_patchtokens']

        else:
            raise ValueError(f"Invalid model type: {model_type}")

        # Reshape features
        feat = rearrange(feat, '(b t) p d -> b t p d', b=batch_size, t=num_frames)
        patch_size = int(feat.shape[2] ** 0.5)
        feat = rearrange(feat, 'b t (p q) d -> b t p q d', p=patch_size)

        return feat

    def cluster_features(self, feat, method='dbscan', n_clusters=8, prev_centers=None, global_clustering=False, use_torch=False):
        """
        Args:
            feat: Features to cluster
            method: Clustering method ('dbscan' or 'kmeans')
            n_clusters: Number of clusters for kmeans
            prev_centers: Previous cluster centers for temporal consistency
            global_clustering: If True, cluster all frames together
        """
        if global_clustering:
            # For global clustering, reshape to (n_frames * n_patches, n_features)
            n_frames, n_patches_h, n_patches_w, feat_dim = feat.shape

            if use_torch:
                feat_2d = feat.reshape(-1, feat_dim)
            else:
                feat_2d = feat.reshape(-1, feat_dim).cpu().numpy()

        else:
            # Original per-frame clustering
            n_patches = feat.shape[0] * feat.shape[1]

            if use_torch:
                feat_2d = feat.reshape(n_patches, -1)
            else:
                feat_2d = feat.reshape(n_patches, -1).cpu().numpy()

        if method == 'dbscan':
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=0.5, min_samples=5).fit(feat_2d)
            labels = clustering.labels_
            centers = None
        elif method == 'kmeans':
            if use_torch:
                #print('verbose...usetorchkmeans')
                device = feat_2d.device
                clustering = TorchKMeansVectorizedCluster(n_clusters=n_clusters)
                labels, centers = clustering(feat_2d, prev_centers=torch.tensor(prev_centers).to(device) if prev_centers is not None else None)
                labels = labels.cpu().numpy()
                centers = centers.cpu().numpy()

            else:
                from sklearn.cluster import KMeans
                if prev_centers is not None:
                    clustering = KMeans(n_clusters=n_clusters, random_state=42, init=prev_centers, n_init=1)
                else:
                    clustering = KMeans(n_clusters=n_clusters, random_state=42)

                clustering.fit(feat_2d)
                labels = clustering.labels_
                centers = clustering.cluster_centers_
        else:
            raise ValueError("Method must be either 'dbscan' or 'kmeans'")

        if global_clustering:
            # Reshape labels back to (n_frames, patch_h, patch_w)
            labels = labels.reshape(n_frames, n_patches_h, n_patches_w)
        else:
            # Reshape labels back to patch grid
            labels = labels.reshape(feat.shape[0], feat.shape[1])

        return labels, centers



def create_overlay_mask(image, labels, colors):
    """Create a colored overlay mask based on clustering labels"""
    # Define 8 distinct and darker colors (R,G,B)


    # Convert labels to RGB mask
    h, w = labels.shape
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(len(colors)):
        mask[labels == i] = colors[i]

    # Resize mask to match image size
    mask = Image.fromarray(mask).resize(image.size, Image.Resampling.NEAREST)

    # Blend with original image
    return Image.blend(image, mask, 0.5)


def cluster_coordinates(coords: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster coordinates based on their spatial positions using K-means and find points
    closest to cluster centers.

    Args:
        coords: Array of shape (N, 2) containing (x, y) coordinates
        n_clusters: Number of clusters to create

    Returns:
        tuple of:
            labels: Array of shape (N,) containing cluster assignments
            center_points: Array of shape (n_clusters, 2) containing coordinates of points
                         closest to each cluster center
    """
    from sklearn.cluster import KMeans

    # Ensure coords is the right shape
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be of shape (N, 2), got {coords.shape}")

    # Handle case where N < n_clusters
    if len(coords) < n_clusters:
        # Assign each point to its own cluster
        labels = np.arange(len(coords))
        # Use the actual points as centers for available clusters
        center_points = coords
        return center_points

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(coords)
    centroids = kmeans.cluster_centers_

    # Find points closest to each centroid
    center_points = np.zeros((n_clusters, 2))
    for i in range(n_clusters):
        # Get points in this cluster
        cluster_points = coords[labels == i]
        # Calculate distances to centroid
        distances = np.sqrt(np.sum((cluster_points - centroids[i]) ** 2, axis=1))
        # Find the closest point
        closest_idx = np.argmin(distances)
        center_points[i] = cluster_points[closest_idx]

    return center_points



###
def cluster_coordinates_per_component(components, num_points_total, cluster_coordinates_fn):
    """Cluster coordinates within each connected component.

    Args:
        components: List of arrays containing (y,x) coordinates for each component
        num_points_total: Total number of points to sample across all components
        cluster_coordinates_fn: Function to cluster coordinates within a component

    Returns:
        clustered_points: Array of shape (N, 2) containing final sampled points
    """
    # Calculate points per component proportional to component size
    total_pixels = sum(len(comp) for comp in components)
    points_per_component = []

    # for comp in components:
    #     # Allocate points proportionally with minimum of 1 point per component
    #     comp_points = max(1, int(round(len(comp) / total_pixels * num_points_total)))
    #     points_per_component.append(comp_points)

    # # Adjust to match total points exactly
    # while sum(points_per_component) > num_points_total:
    #     idx = np.argmax(points_per_component)
    #     points_per_component[idx] -= 1
    # while sum(points_per_component) < num_points_total:
    #     idx = np.argmax([len(comp) for comp in components])
    #     points_per_component[idx] += 1
    for comp in components:
        #print('debug....num points per component', num_points_total)
        points_per_component.append(num_points_total)

    # Cluster within each component using provided cluster_coordinates function
    all_clustered_points = []
    point_component_labels =[]
    for component_idx, (component, n_points) in enumerate(zip(components, points_per_component)):
        if len(component) <= n_points:
            clustered_points = component
        else:
            clustered_points = cluster_coordinates_fn(component, n_points)
        all_clustered_points.append(clustered_points)
        point_component_labels.append(np.ones(len(clustered_points)) * component_idx)
    return np.vstack(all_clustered_points), np.concatenate(point_component_labels).astype(int)


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

def get_representative_points(labels, grid_size=16, image_size=None, clustered_points=False, num_points=-1, clusters_to_consider=None, use_connected_components = True):
    """Get one representative point per cluster (centroid or random sample)
    Args:
        labels: tensor of shape [H, W] containing cluster labels (16x16 grid)
        grid_size: size of the feature grid (16 for DINO ViT-B/14)
        image_size: tuple of (height, width) for the target image. If None, defaults to (224, 224)
    Returns:
        points: numpy array of shape (N, 2) containing x,y coordinates
        labels: numpy array of shape (N,) containing corresponding labels
    """
    if image_size is None:
        image_size = (224, 224)
    img_height, img_width = image_size

    # Convert labels to tensor if it's numpy array
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    points_list = []
    labels_list = []
    max_current_class_label = 0
    unique_labels = torch.unique(labels).cpu().numpy()
    component_labels_list = []
    for label in clusters_to_consider:
        mask = (labels == label)

        mask = torch.nn.functional.interpolate(
            mask[None, None, :, :].float(),  # Add batch and channel dims
            size=(img_height, img_width),
            mode='nearest'
        )[0, 0]  # Remove batch and channel dims

        y_indices, x_indices = torch.where(mask)
        if clustered_points:
            if use_connected_components:
                components = find_connected_components(mask.cpu().numpy(), connectivity=4)
                if not components:
                    warnings.warn(f"No connected components found for label {label}")
                    continue
                cluster_points, component_labels = cluster_coordinates_per_component(components, num_points, cluster_coordinates_fn=cluster_coordinates)
                # scaled_y = cluster_points[:,0:1] * (img_height/grid_size)
                # scaled_x = cluster_points[:,1:2] * (img_width/grid_size)

                # # Add padding to ensure points fall within mask boundaries
                # scaled_x = (cluster_points[:,1:2] + 0.5) * (img_width/grid_size)  # Add 0.5 to center within grid cell
                # scaled_y = (cluster_points[:,0:1] + 0.5) * (img_height/grid_size)

                scaled_x = cluster_points[:,1:2]
                scaled_y = cluster_points[:,0:1]

                # Optional: Clamp values to ensure they stay within image boundaries
                scaled_x = np.clip(scaled_x, 0, img_width - 1)
                scaled_y = np.clip(scaled_y, 0, img_height - 1)

                points_list.extend(np.concatenate([scaled_x, scaled_y], axis=1).tolist())
                labels_list.extend([label] * len(scaled_x))
                component_labels_list.extend(component_labels.tolist())
            else:
                stacked_indices = torch.stack([y_indices, x_indices], dim=1).cpu().numpy()
                cluster_points = cluster_coordinates(stacked_indices, num_points)
                # scaled_y = cluster_points[:,0:1] * (img_height/grid_size)
                # scaled_x = cluster_points[:,1:2] * (img_width/grid_size)

                scaled_x = cluster_points[:,1:2]
                scaled_y = cluster_points[:,0:1]

                # Optional: Clamp values to ensure they stay within image boundaries
                scaled_x = np.clip(scaled_x, 0, img_width - 1)
                scaled_y = np.clip(scaled_y, 0, img_height - 1)

                points_list.extend(np.concatenate([scaled_x, scaled_y], axis=1).tolist())
                labels_list.extend([label] * len(scaled_x))

        else:
            if len(y_indices) > 0:
                # Calculate centroid
                centroid_y = y_indices.float().mean().item()
                centroid_x = x_indices.float().mean().item()

                # Check if centroid falls on the mask
                centroid_y_int = int(round(centroid_y))
                centroid_x_int = int(round(centroid_x))

                if mask[centroid_y_int, centroid_x_int]:
                    # Use centroid if it falls on the mask
                    point_y, point_x = centroid_y, centroid_x
                else:
                    # Random sample from the mask
                    random_idx = torch.randint(0, len(y_indices), (1,))
                    point_y = y_indices[random_idx].float().item()
                    point_x = x_indices[random_idx].float().item()

                # Scale coordinates from grid space to image space
                # scaled_x = point_x * (img_width/grid_size)
                # scaled_y = point_y * (img_height/grid_size)

                scaled_x = point_x
                scaled_y = point_y

                points_list.append([scaled_x, scaled_y])
                labels_list.append(label)

    # Convert to numpy arrays
    points = np.array(points_list)  # Shape: (N, 2)
    point_labels = np.array(labels_list)  # Shape: (N,)
    component_labels = np.array(component_labels_list)

    return points, point_labels, component_labels


def get_cluster_peak_frames(cluster_labels, num_clusters=None):
    """
    Find the frame where each cluster has its maximum presence.

    Args:
        cluster_labels: numpy array of shape [T, H, W] containing cluster labels
        num_clusters: optional, maximum number of clusters to consider

    Returns:
        peak_frames: dictionary mapping cluster_id to frame_id where that cluster has maximum presence
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

def get_points_and_labels_from_global_cluster_main(video_frames, video_uniq_id,
                                                    save_root, extractor,
                                                    merge_ratio=25, num_iters=11,
                                                    n_clusters=8, vis = True,
                                                    use_connected_components=False,
                                                    num_points_per_entity=8,
                                                    clustering_method='bipartite'):
    '''
    output query points and labels (all positive i.e. 1) for each frame

    '''


    dino_features = extractor(video_frames, model_type='dino').squeeze() # (T, 16, 16, 768)

     # Apply clustering for all frames for all three models

    # Global clustering for DINO torch
    if clustering_method == 'bipartite':
        global_labels_dino_torch = get_temporal_bipartite_clusters(dino_features, merge_ratio=merge_ratio, num_iters=num_iters)
    elif clustering_method == 'kmeans':
        global_labels_dino_torch, _ = extractor.cluster_features(
            dino_features, # only use first frame for clustering
            method='kmeans',
            n_clusters=n_clusters,
            global_clustering=True,
            use_torch=True
        )

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

    points_list = []
    point_positive_labels_list = []
    clusters_considerd = set()
    if args.dataset == 'paper_vis':
        peak_frames = {0 : np.unique(global_labels_dino_torch[0]).tolist()}
    else:
        peak_frames = get_cluster_peak_frames(global_labels_dino_torch)
    point_labels_list = []
    component_labels_list = []

    for fid in range(n_frames):
        if fid not in peak_frames:
            continue
        clusters_to_consider = peak_frames[fid]
        points, cluster_labels, component_labels= get_representative_points(torch.tensor(global_labels_dino_torch[fid]).to(dino_features.device), grid_size=16, image_size=(h,w), clustered_points=True, num_points=num_points_per_entity, clusters_to_consider=clusters_to_consider,use_connected_components=use_connected_components)
        clusters_considerd.update(set(cluster_labels))

        point_positive_labels = np.ones(len(points), dtype=np.int32)
        points_list.append(points)
        point_positive_labels_list.append(point_positive_labels)
        point_labels_list.append(cluster_labels)
        if use_connected_components:
            component_labels_list.append(component_labels)

    ### visualize clustering
    # Plot original and both DINO overlays (all at 224x224)

        if vis:

            original_frame = Image.fromarray(video_frames[0,fid])

            overlay_dino_global_torch = create_overlay_mask(original_frame, global_labels_dino_torch[fid], colors)

            # Create a figure with one row and two columns
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

            # Left subplot - Original frame
            ax1.imshow(video_frames[0,fid])
            ax1.set_title(f'Frame {fid} ({h}x{w})', fontsize=14)
            ax1.axis('off')

            # Right subplot - DINO Cluster with points
            ax2.imshow(overlay_dino_global_torch)
            ax2.set_title('DINO Global Cluster', fontsize=14)
            ax2.axis('off')


            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(os.path.join(save_root, f"debug_cluster_pts_{video_uniq_id}_frame{fid}_connected{use_connected_components}.png"),
                    bbox_inches='tight', dpi=150)
            print(os.path.join(save_root, f"debug_cluster_pts_{video_uniq_id}_frame{fid}_connected{use_connected_components}.png"))
            plt.close()

    return points_list, point_positive_labels_list, point_labels_list, component_labels_list

def apply_mask_nms(masks, obj_ids, prompts, iou_threshold=0.5):
    """Apply non-maximum suppression to masks based on their area and IoU.

    Args:
        masks: List of binary masks
        obj_ids: List of object IDs corresponding to masks
        prompts: Dictionary mapping object IDs to their prompts
        iou_threshold: IoU threshold for suppressing overlapping masks

    Returns:
        filtered_masks: List of masks after NMS
        filtered_obj_ids: List of object IDs after NMS
        filtered_prompts: Dictionary of prompts after NMS
    """

    # Calculate areas
    masks = [mask.astype(bool) for mask in masks]
    areas = [mask.sum() for mask in masks]
    keep_indices = []

    # Sort masks by area (largest first)
    sorted_indices = sorted(range(len(areas)), key=lambda k: areas[k], reverse=True)

    for i, idx in enumerate(sorted_indices):
        # Keep first mask
        if i == 0:
            keep_indices.append(idx)
            continue

        # Check IoU with all kept masks
        current_mask = masks[idx]
        should_keep = True

        for keep_idx in keep_indices:
            kept_mask = masks[keep_idx]
            intersection = (current_mask & kept_mask).sum()
            union = (current_mask | kept_mask).sum()
            iou = intersection / union if union > 0 else 0
            overlap_ratio = intersection / current_mask.sum() if current_mask.sum() > 0 else 0

            if iou > iou_threshold or overlap_ratio > 0.90:
                should_keep = False
                break

        if should_keep:
            keep_indices.append(idx)

    # Filter masks and update related data
    filtered_masks = [masks[i] for i in keep_indices]
    if obj_ids is not None:
        filtered_obj_ids = [obj_ids[i] for i in keep_indices]
    else:
        filtered_obj_ids = None
    if prompts is not None:
        filtered_prompts = {}
        for new_idx, old_idx in enumerate(keep_indices):
            filtered_prompts[new_idx] = prompts[old_idx]
    else:
        filtered_prompts = None


    return filtered_masks, filtered_obj_ids, filtered_prompts


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


def convert_points_for_tracking(points_list, labels_list, frames_id_dict=None, component_labels_list=None, use_connected_components=False):
    '''
    video_frames: (1, T, H, W, 3) tensor
    points_list: list of points
    labels_list: list of labels
    video_dir: path to video directory
    video_uniq_id: unique id of video
    save_root: path to save root
    vis: if True, visualize the results
    '''

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

def construct_sam_model_image_seg(model_size='tiny'):
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    if model_size == 'large':
        sam2_checkpoint = "libs/sam2/checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        print("loading large sam model from ", sam2_checkpoint)
    elif model_size == 'tiny':
        sam2_checkpoint = "libs/sam2/checkpoints/sam2.1_hiera_tiny.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        print("loading tiny sam model from ", sam2_checkpoint)



    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

    mask_generator = SAM2AutomaticMaskGenerator(sam2,points_per_side=10)

    return mask_generator

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)



def load_video_decord(vid_path, return_tensor=False, use_float=False, num_frames=8, sample_all_frames=False, fps=None):
    '''
    load video from file with regular interval sampling using Decord.
    Args:
        vid_path: path to video file
        return_tensor: if True, return torch tensor, otherwise numpy array
        device: device to load tensor to
        use_float: if True, convert frames to float32, otherwise keep uint8
        num_frames: number of frames to sample (default=8)
        sample_all_frames: if True, return all frames without subsampling
        fps: if set, load video at this frame rate
    Returns:
        frames: (B, T, C, H, W) numpy array or tensor, where T = num_frames or total frames if sample_all_frames=True
        frame_id_dict: dictionary mapping sampled frame indices to original frame indices
    '''
    print(f"Processing {vid_path}...")
    assert os.path.exists(vid_path), f"Video file {vid_path} does not exist"

    import decord
    from decord import VideoReader


    print("AV failed, trying decord")
    vr = VideoReader(vid_path, num_threads=1)

    total_frames = len(vr)
    original_fps = vr.get_avg_fps()

    if fps is not None:
        # Calculate frame indices based on desired fps
        interval = int(round(original_fps / fps))
        frame_indices = list(range(0, total_frames, interval))
    else:
        frame_indices = list(range(total_frames))

    if not sample_all_frames:
        # Ensure num_frames does not exceed available frames
        available_frames = len(frame_indices)
        if num_frames > available_frames:
            print(f"Warniqng: num_frames ({num_frames}) is greater than available frames ({available_frames}). Adjusting num_frames.")
            num_frames = available_frames

        # Calculate indices for uniformly sampled frames
        sample_indices = np.linspace(0, len(frame_indices)-1, num_frames, dtype=int)
        frame_indices = [frame_indices[i] for i in sample_indices]
        frame_id_dict = {i: idx for i, idx in enumerate(frame_indices)}
    else:
        frame_id_dict = None

    # Read frames
    frames = vr.get_batch(frame_indices).asnumpy()  # T,H,W,C

    # Convert to float if needed
    if use_float:
        frames = frames.astype(np.float32)

    # Add batch dimension and rearrange to B,T,C,H,W
    frames = frames[None]  # B,T,H,W,C
    frames = np.transpose(frames, (0, 1, 4, 2, 3))  # B,T,C,H,W

    if return_tensor:
        frames = torch.from_numpy(frames)
        if torch.isnan(frames).any() or torch.isinf(frames).any():
            breakpoint()
        if device is not None:
            frames = frames.to(device)
    return frames, frame_id_dict


def load_video(vid_path, return_tensor=False, device=None, use_float=False, num_frames=8, sample_all_frames=False, fps=None):
    '''
    load video from webm file with regular interval sampling.
    Args:
        vid_path: path to video file
        return_tensor: if True, return torch tensor, otherwise numpy array
        device: device to load tensor to
        num_frames: number of frames to sample (default=8)
        sample_all_frames: if True, return all frames without subsampling
        fps: if set, load video at this frame rate
    Returns:
        frames: (B, T, C, H, W) numpy array or tensor, where T = num_frames or total frames if sample_all_frames=True
    '''
    print(f"Processing {vid_path}...")
    assert os.path.exists(vid_path), f"Video file {vid_path} does not exist"

    # Option 2: Using PyAV
    import av
    try:
        container = av.open(vid_path)
    except:
        return False, None, None

    # Get video stream
    stream = container.streams.video[0]
    original_fps = float(stream.average_rate)

    frames = []
    if fps is not None:
        # Calculate frame interval based on desired fps
        interval = int(round(original_fps / fps))
        frame_count = 0
        for frame in container.decode(video=0):
            if frame_count % interval == 0:
                # Convert to RGB numpy array
                frame = frame.to_ndarray(format='rgb24')
                if use_float:  # for cotracker, we need to use float32; for dino feature extraction, we need to uint8
                    frame = frame.astype(np.float32)
                frames.append(frame)
            frame_count += 1
    else:
        # Original behavior without fps control
        for frame in container.decode(video=0):
            frame = frame.to_ndarray(format='rgb24')
            if use_float:
                frame = frame.astype(np.float32)
            frames.append(frame)

    total_frames = len(frames)
    container.close()

    # Stack frames into a single array and rearrange dimensions
    frames = np.stack(frames)[None]  # B,T,H,W,C
    frames = np.transpose(frames, (0, 1, 4, 2, 3))  # B,T,C,H,W
    frame_id_dict = None

    if not sample_all_frames:
        # Ensure num_frames does not exceed total_frames
        if num_frames > total_frames:
            print(f"Warning: num_frames ({num_frames}) is greater than total_frames ({total_frames}). Adjusting num_frames to total_frames.")
            num_frames = total_frames  # Set num_frames to total_frames
        # Calculate indices for uniformly sampled frames
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        frames = frames[:, frame_indices]  # (1, T, C, H, W)
        frame_id_dict = {i: frame_indices[i] for i in range(num_frames)}

    if return_tensor:
        frames = torch.from_numpy(frames).to(device)

    return True, frames, frame_id_dict


def vis_trail(video, kpts, kpts_vis, kpts_queries=None,point_to_take=None, fps=10, cluster_ids=None,
line_thickness=1, max_cluster_id=None):
    """
    This function calculates the median motion of the background, which is subsequently
    subtracted from the foreground motion. This subtraction process "stabilizes" the camera and
    improves the interpretability of the foreground motion trails.

    Args:
        video (np.ndarray): Video frames (T, H, W, C)
        kpts (np.ndarray): Keypoints (T, N, 2)
        kpts_vis (np.ndarray): Keypoint visibility (T, N)
        kpts_queries (np.ndarray): Frame at which the point was queried (N)
        fps (float): Frames per second
    """
    color_map = cm.get_cmap("jet")

    if cluster_ids is not None:
        if max_cluster_id is None:
            max_clusteter_id_to_use = max(cluster_ids)
        else:
            max_clusteter_id_to_use = max_cluster_id
    else:
        max_clusteter_id_to_use = 1

    images = video
    max_height = 200
    if point_to_take is None:
        point_to_take = np.ones(kpts.shape[1], dtype=bool)
    if kpts_queries is None:
        kpts_queries = np.zeros(kpts.shape[1], dtype=int)
    if kpts_vis is None:
        kpts_vis = np.ones(kpts.shape[:2], dtype=bool)

    frames = []
    back_history = int(fps // 2)

    #sample only the points that are needed to be taken
    kpts = kpts[:, point_to_take]
    kpts_vis = kpts_vis[:, point_to_take]
    kpts_queries = kpts_queries[point_to_take]
    point_to_take = point_to_take[point_to_take]

    num_imgs, num_pts = kpts.shape[:2]


    for i in range(num_imgs):

        img_curr = images[i]

        for t in range(i):


            img1 = img_curr.copy()
            # changing opacity
            if i - t < back_history:
                alpha = max(1 - 0.9 * ((i - t) / ((i + 1) * .99)), 0.1)
            else:
                alpha = 0
                # alpha = 0.6

            for j in range(num_pts):
                if (kpts_queries[j] > t) or (not point_to_take[j]):
                    continue
                if (kpts_vis[t:, j] == 0).all():
                    continue

                if cluster_ids is not None:
                    color  = np.array(color_map(cluster_ids[j]/ max_clusteter_id_to_use)[:3]) *255
                else:
                    color = np.array(color_map(j/max(1, float(num_pts - 1)))[:3]) * 255

                color_alpha = 1

                hsv = colorsys.rgb_to_hsv(color[0], color[1], color[2])
                color = colorsys.hsv_to_rgb(hsv[0], hsv[1]*color_alpha, hsv[2])

                pt1 = kpts[t, j]
                pt2 = kpts[t+1, j]
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))

                cv2.line(img1, p1, p2, color, thickness=line_thickness, lineType=16)

            img_curr = cv2.addWeighted(img1, alpha, img_curr, 1 - alpha, 0)

        for j in range(num_pts):
            if (kpts_queries[j] > i) or (not point_to_take[j]):
                    continue
            if (kpts_vis[i:, j] == 0).all():
                 continue
            if cluster_ids is not None:
                color  = np.array(color_map(cluster_ids[j]/ max_clusteter_id_to_use)[:3]) *255

            else:
                color = np.array(color_map(j/max(1, float(num_pts - 1)))[:3]) * 255
            pt1 = kpts[i, j]
            p1 = (int(round(pt1[0])), int(round(pt1[1])))
            cv2.circle(img_curr, p1, 2, color, -1, lineType=16)
            if cluster_ids is not None:
                text = str(cluster_ids[j])
                # Position text slightly above the point
                text_position = (p1[0], p1[1] - 3)  # 10 pixels above the point
                cv2.putText(img_curr, text, text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, # smaller font size
                           color.astype(int).tolist(), 1,
                           cv2.LINE_AA)


        # height, width, _ = img_curr.shape
        # if height > max_height:
        #     new_width = int(width * max_height / height)
        # else:
        #     new_width = width
        # img_curr = cv2.resize(img_curr, (new_width, max_height))

        frames.append(Image.fromarray(img_curr.astype(np.uint8)))

    return frames

def save_video(frames, save_path):
    frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=100, loop=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_splits", type=int, default=1)
    parser.add_argument("--split_id", type=int, default=0)
    parser.add_argument("--debug_mode", action="store_true", help="Enable debug mode")
    parser.add_argument("--missing", action="store_true", help="Run for missing videos")

    parser.add_argument("--use_vid_ids", action="store_true", help="Use specific video IDs")
    parser.add_argument("--use_connected_components", action="store_true", help="Use connected components")
    parser.add_argument("--num_frames_clustering", type=int, default=32, help="Number of frames to cluster")

    parser.add_argument("--dataset", type=str, default='somethingv2', help="Dataset to use")
    parser.add_argument("--merge_ratio", type=int, default=25, help="Merge ratio")
    parser.add_argument("--num_iters", type=int, default=11, help="Number of iterations")
    parser.add_argument("--clustering_method", type=str, default='bipartite', help="Clustering method to use")
    parser.add_argument("--n_clusters", type=int, default=32, help="Number of clusters")
    parser.add_argument("--num_smpl", type=int, default=16, help="Number of samples per mask")
    parser.add_argument("--use_grid", action="store_true", help="Use grid")
    parser.add_argument("--cotracker_grid_size", type=int, default=16, help="Cotracker grid size")
    args = parser.parse_args()
    debug_mode = args.debug_mode

    use_connected_components = args.use_connected_components
    num_points_per_entity = args.num_smpl
    merge_str = ''
    if args.merge_ratio != 25 or args.num_iters != 11:
        merge_str = f'_m{args.merge_ratio}_i{args.num_iters}'

    num_frames_clustering = args.num_frames_clustering
    if args.clustering_method == 'kmeans':
        n_clusters = args.n_clusters
        clust_str = f'kmeans_n{n_clusters}'
    elif args.clustering_method == 'bipartite':
        clust_str = f'bip'
        n_clusters = 32

    if not debug_mode:
        if args.use_vid_ids:
            vids_in_split = [137811, 121896, 55287, 113781, 39041]
            vids_in_split += [162071, 55287, 46229, 132398, 171127, 80575, 203917, 18052, 164009, 146475, 121896, 139051, 155570, 39041, 42780, 63166]
        else:
            if args.dataset == 'somethingv2':
                if args.missing:
                    df = pd.read_csv(f'/fs/cfar-projects/actionloc/bounce_back/point_tracking/dataset_csvs/somethingv2_sam_based_missing{merge_str}_{clust_str}.csv')
                else:
                    df = pd.read_csv(f'/fs/cfar-projects/actionloc/bounce_back/point_tracking/dataset_csvs/somethingv2_few_shot.csv')
                all_vid_ids  = df['id'].tolist()
                vids_in_split = np.array_split(all_vid_ids, args.num_splits)[args.split_id].tolist()
                fps = None
            elif args.dataset == 'kinetics':
                if args.missing:
                    raise ValueError("Missing is not supported for kinetics")
                df = pd.read_csv('/fs/cfar-projects/actionloc/bounce_back/point_tracking/dataset_csvs/kinetics.csv')
                all_vid_ids  = df['video_path'].tolist()
                vids_in_split = np.array_split(all_vid_ids, args.num_splits)[args.split_id].tolist()
                fps = 10
            elif args.dataset == 'kinetics_molo':
                if args.missing:
                    df = pd.read_csv(f'/fs/cfar-projects/actionloc/bounce_back/point_tracking/dataset_csvs/kinetics_molo_sam_based_missing{merge_str}.csv')
                else:
                    df = pd.read_csv('/fs/cfar-projects/actionloc/bounce_back/point_tracking/dataset_csvs/kinetics_molo_256.csv')
                all_vid_ids  = df['video_path'].tolist()
                vids_in_split = np.array_split(all_vid_ids, args.num_splits)[args.split_id].tolist()
                fps = 10
            elif args.dataset == 'ucf':
                if args.missing:
                    df = pd.read_csv(f'/fs/cfar-projects/actionloc/bounce_back/point_tracking/dataset_csvs/ucf_sam_based_missing{merge_str}.csv')
                else:
                    df = pd.read_csv('/fs/cfar-projects/actionloc/bounce_back/point_tracking/dataset_csvs/ucf_few_shot.csv')
                all_vid_ids  = df['video_path'].tolist()
                vids_in_split = np.array_split(all_vid_ids, args.num_splits)[args.split_id].tolist()
                fps = 10
            elif args.dataset == 'hmdb':
                if args.missing:
                    df = pd.read_csv(f'/fs/cfar-projects/actionloc/bounce_back/point_tracking/dataset_csvs/hmdb_sam_based_missing{merge_str}.csv')
                else:
                    df = pd.read_csv('/fs/cfar-projects/actionloc/bounce_back/point_tracking/dataset_csvs/hmdb_few_shot.csv')
                all_vid_ids  = df['video_path'].tolist()
                vids_in_split = np.array_split(all_vid_ids, args.num_splits)[args.split_id].tolist()
                fps = 10

            elif args.dataset == 'finegym':
                if args.missing:
                    df = pd.read_csv(f'/fs/cfar-projects/actionloc/bounce_back/point_tracking/dataset_csvs/finegym_sam_based_missing{merge_str}.csv')
                else:
                    df = pd.read_csv('/fs/cfar-projects/actionloc/bounce_back/point_tracking/dataset_csvs/finegym_few_shot.csv')
                all_vid_ids  = df['video_path'].tolist()
                vids_in_split = np.array_split(all_vid_ids, args.num_splits)[args.split_id].tolist()
                fps = 10
            else:
                raise ValueError(f"Dataset {args.dataset} not supported")
        vis_dino_cluster = False
        vis_mask = False
    else:
        if args.dataset == 'paper_vis':
            vids_in_split = ['/fs/cfar-projects/actionloc/bounce_back/DiST/seg/trimmed_butter.mp4']
            vis_dino_cluster = True
            vis_mask = True
            fps = 10
        else:
            #vids_in_split = [203917, 137811]
            vids_in_split = [113781]
            # vids_in_split = [137811, 121896, 55287, 113781, 39041]
            # vids_in_split += [162071, 55287, 46229, 132398, 171127, 80575, 203917, 18052, 164009, 146475, 121896, 139051, 155570, 39041, 42780, 63166]
            # vids_in_split = [88675, 36373]
            vis_dino_cluster = False
            vis_mask = False
            fps = None


    dump_name = 'cotracker3_{}_fr_{}'.format(clust_str, num_frames_clustering)
    if args.merge_ratio != 25 or args.num_iters != 11: # if not default then add to dump name
        dump_name += f'_m{args.merge_ratio}_i{args.num_iters}'
    if use_connected_components:
        dump_name += '_concomp'
    if fps is not None:
        dump_name += f'_fps_{fps}'
    base_featpath = f'/fs/cfar-projects/actionloc/bounce_back/point_tracking/feat_dumps/sam_based/{args.dataset}'


    use_nms = False
    nms_iou_threshold = 0.8

    points_per_mask_for_sampling = args.num_smpl
    sampling_method = 'grid'



    use_cotracker=True
    if 'umiacs' in socket.gethostname():
        if args.dataset == 'somethingv2':
            video_webm_root ="/fs/vulcan-datasets/SomethingV2/20bn-something-something-v2"
        if 'shirley' in os.getcwd():
            base_gif_path = '/fs/cfar-projects/actionloc/shirley/result_vis/sam_based/gif_dumps_numfclus{}'.format(num_frames_clustering)
        else:
            base_gif_path = '/fs/cfar-projects/actionloc/bounce_back/result_vis/sam_based/gif_dumps'

    else:
        if args.dataset == 'somethingv2':
            video_webm_root ="/mnt/amlfs-01/home/ruijiez/tmp/work/data/ssv2/20bn-something-something-v2"
            base_gif_path = './result_vis/sam_based/gif_dumps'

    cotracker_grid_size = args.cotracker_grid_size
    gif_dump_dir_path = os.path.join(base_gif_path, 'bipartite_temporal_sam2')
    os.makedirs(gif_dump_dir_path, exist_ok=True)

    #base_featpath = '/fs/cfar-projects/actionloc/shirley/sam_based_debug/somethingv2'
    featpath_dir = os.path.join(base_featpath, dump_name)
    os.makedirs(featpath_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    extractor = feature_extract()

    for video_index, video_uniq_stuff in enumerate(vids_in_split):
        if args.dataset == 'somethingv2':
            video_uniq_id = video_uniq_stuff
            cur_webm_path = os.path.join(video_webm_root, f"{video_uniq_id}.webm")
            feat_dump_name = f'{video_uniq_id}'
        elif args.dataset == 'kinetics':
            cur_webm_path = video_uniq_stuff
            video_uniq_id = video_uniq_stuff.split('/')[-1].split('.')[0]
            video_class = video_uniq_stuff.split('/')[-2]
            feat_dump_name = f'{video_class}/{video_uniq_id}'
        elif args.dataset == 'kinetics_molo':
            cur_webm_path = video_uniq_stuff
            video_uniq_id = video_uniq_stuff.split('/')[-1].split('.')[0]
            video_class = video_uniq_stuff.split('/')[-2]
            split_info = video_uniq_stuff.split('/')[-3].replace('_256_new', '').replace('_256', '')
            feat_dump_name = f'{split_info}/{video_class}/{video_uniq_id}'
        elif args.dataset == 'ucf':
            cur_webm_path = video_uniq_stuff
            video_uniq_id = video_uniq_stuff.split('/')[-1].split('.')[0]
            video_class = video_uniq_stuff.split('/')[-2]
            feat_dump_name = f'{video_uniq_id}'
        elif args.dataset == 'hmdb':
            cur_webm_path = video_uniq_stuff
            video_uniq_id = video_uniq_stuff.split('/')[-1].split('.')[0]
            video_class = video_uniq_stuff.split('/')[-2]
            feat_dump_name = f'{video_uniq_id}'
        elif args.dataset == 'finegym':
            cur_webm_path = video_uniq_stuff
            video_uniq_id = video_uniq_stuff.split('/')[-1].split('.')[0]
            video_class = video_uniq_stuff.split('/')[-2]
            feat_dump_name = f'{video_uniq_id}'
        elif args.dataset == 'paper_vis':
            cur_webm_path = video_uniq_stuff
            video_uniq_id = video_uniq_stuff.split('/')[-1].split('.')[0]
            feat_dump_name = f'{video_uniq_id}'
        else:

            raise ValueError(f"Dataset {args.dataset} not supported")
        feat_dump_path = os.path.join(featpath_dir, f"{feat_dump_name}.pkl")
        print(f"processing video {video_index} of {len(vids_in_split)} video {video_uniq_id}")


        if not debug_mode:
            if os.path.exists(feat_dump_path):
                continue
            save_root = None
        else:
            save_root = os.path.join(os.getcwd(), 'vis_seg_cluglobal{}_numfclu{}'.format(n_clusters, num_frames_clustering), str(video_uniq_id))
            os.makedirs(save_root, exist_ok=True)  # Create the directory if it doesn't exist


        video_loaded, video_frames, frames_id_dict = load_video(cur_webm_path, return_tensor=True, use_float=False, num_frames=num_frames_clustering, sample_all_frames=False, fps=fps) # (B, T, C, H, W)
        if not video_loaded:
            print(f"Video {video_uniq_id} not loaded")
            continue
        if args.dataset == 'paper_vis':
            print("Trimming video")
            video_frames = video_frames[..., 50:-50, 150:-200]

        video_frames = video_frames.permute(0, 1, 3, 4, 2).cpu().numpy() # (B, T, H, W, C)
        _,n_frames,h,w,_ = video_frames.shape
        if debug_mode:
            time_start = time.time()

        points_list, point_positive_labels_list, point_labels_list, component_labels_list = get_points_and_labels_from_global_cluster_main(video_frames, video_uniq_id, save_root, extractor, n_clusters=args.n_clusters, vis=vis_dino_cluster, use_connected_components=use_connected_components, num_points_per_entity=num_points_per_entity, merge_ratio=args.merge_ratio, num_iters=args.num_iters, clustering_method=args.clustering_method)

        queries_points, cluster_ids_all_frames = convert_points_for_tracking(points_list, point_labels_list, frames_id_dict=frames_id_dict, component_labels_list=component_labels_list, use_connected_components=use_connected_components)
        if debug_mode:
            time_end = time.time()
            print(f"Time taken to get points and labels: {time_end - time_start} seconds")
        # out_mask_tiny, queries_points, cluster_ids_all_frames = obtain_mask_and_queries_points_from_sam(video_frames, predictor_tiny, points_list, point_positive_labels_list, video_uniq_id, save_root, use_nms=use_nms, iou_threshold=nms_iou_threshold, points_per_mask=points_per_mask_for_sampling, sampling_method=sampling_method, vis=vis_mask, frames_id_dict=frames_id_dict)
        torch.cuda.empty_cache()
        if use_cotracker:

            _, video, _ = load_video(cur_webm_path, return_tensor=True, use_float=True, device=device, sample_all_frames=True, fps=fps) # B T C H W
            # Run Offline CoTracker:
            if args.dataset == 'paper_vis':
                video = video[..., 50:-50, 150:-200]


            if debug_mode:
                time_start = time.time()
            if args.use_grid:
                pred_tracks, pred_visibility = cotracker(video, grid_size=args.cotracker_grid_size, queries = None, backward_tracking=False) # Output B T N 2,  B T N 1, Input, queries M,2 -> B=1 M 2
            else:
                pred_tracks, pred_visibility = cotracker(video, queries = queries_points, backward_tracking=True) # Output B T N 2,  B T N 1, Input, queries M,2 -> B=1 M 2
            if debug_mode:
                time_end = time.time()
                print(f"Time taken to run cotracker: {time_end - time_start} seconds")
            point_queries = queries_points.cpu().squeeze(0).numpy()[:, 0]
            pred_tracks = pred_tracks.cpu().squeeze(0).numpy()
            pred_visibility = pred_visibility.cpu().squeeze(0).numpy()
            # pred_tracks = pred_tracks[:,filtered_masks]
            # pred_visibility = pred_visibility[:,filtered_masks]

            # cluster_ids_all_frames = np.array(cluster_ids_all_frames)[filtered_masks]
            video = video.cpu().squeeze(0).numpy()
            video = rearrange(video, 't c h w -> t h w c')
            pt_obj_cluster_dict = {}
            if not args.use_grid:

                for point_clusering_type in ['aggo', 'kmeans']:
                    for num_hod_clusters in [2, 4]:
                        dict_key = f'obj_ids_{point_clusering_type}_{num_hod_clusters}'
                        new_obj_ids = run_hod_obj_id_sampling_per_class(pred_tracks, cluster_ids_all_frames,
                            num_bins=32, num_clusters=num_hod_clusters,
                            clustering_method=point_clusering_type, temporal_rate=4)
                        pt_obj_cluster_dict[dict_key] = new_obj_ids.copy()
                        assert len(new_obj_ids) == len(cluster_ids_all_frames)

            if debug_mode:
                pred_visibility[pred_visibility==False] = True
                if args.use_grid:
                    frames = vis_trail(video, pred_tracks, pred_visibility, cluster_ids=None)
                    feat_dump_path = feat_dump_path.replace('.pkl', f'_grid_{args.cotracker_grid_size}.pkl')
                else:
                    frames = vis_trail(video, pred_tracks, pred_visibility, cluster_ids=cluster_ids_all_frames)

                gif_path = os.path.join(gif_dump_dir_path, f"{video_uniq_id}.gif")
                print(gif_path)
                save_video(frames, gif_path)

            dump_dict = {
                'pred_tracks': torch.tensor(pred_tracks).half(),
                'pred_visibility': torch.tensor(pred_visibility).bool(),
                'obj_ids': torch.tensor(cluster_ids_all_frames).long(),
                'point_queries': torch.tensor(point_queries).long(),
                **pt_obj_cluster_dict
            }
            if args.dataset == 'paper_vis':
                feat_dump_path = os.path.join('paper_vis', os.path.basename(feat_dump_path))

            else:
                os.makedirs(os.path.dirname(feat_dump_path), exist_ok=True)
                pickle.dump(dump_dict, open(feat_dump_path, "wb"))
            os.makedirs(os.path.dirname(feat_dump_path), exist_ok=True)
            pickle.dump(dump_dict, open(feat_dump_path, "wb"))
            torch.cuda.empty_cache()

            # vis = Visualizer(save_dir="./saved_videos_debug", pad_value=120, linewidth=3)
            # vis.visualize(video, pred_tracks, pred_visibility, filename=f"dinocluster{n_clusters}_points{points_per_mask_for_sampling}_{sampling_method}_{video_uniq_id}.mp4")


