"""
This file contains the functions to get the semantic points from the clustering.
"""
import os
import warnings
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from clustering import (get_temporal_bipartite_clusters,
                        cluster_coordinates,
                        cluster_coordinates_per_component)

from utils import get_cluster_peak_frames, find_connected_components
from utils import create_overlay_mask

def make_frame_cluster_vis(video_frames, frame_id, feat_cluster_labels,
                           debug_vis_root):
    """Make a visualisation of the cluster points for a given frame.

    Args:
        video_frames (torch.Tensor): Video frames
        frame_id (int): Frame id
        feat_cluster_labels (torch.Tensor): Feature cluster labels
        debug_vis_root (str): Debug visualisation path
    """
    original_frame = Image.fromarray(video_frames[0,frame_id])

    overlay_dino_global_torch = create_overlay_mask(original_frame,
                                                    feat_cluster_labels[frame_id])

    # Create a figure with one row and two columns
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Left subplot - Original frame
    ax1.imshow(video_frames[0,frame_id])
    ax1.set_title(f'Frame {frame_id}', fontsize=14)
    ax1.axis('off')

    # Right subplot - DINO Cluster with points
    ax2.imshow(overlay_dino_global_torch)
    ax2.set_title('DINO Global Cluster', fontsize=14)
    ax2.axis('off')

    # Adjust layout and save
    plt.tight_layout()
    os.makedirs(debug_vis_root, exist_ok=True)
    plt.savefig(os.path.join(debug_vis_root, f"debug_cluster_pts_{frame_id}.png"),
            bbox_inches='tight', dpi=150)
    plt.close()



def get_points_in_cluster(args,labels, image_size, clusters_to_consider=None):
    """Per cluster sample points such that the points are uniformly spread
    across the cluster.

    Args:
        args (argparse.Namespace): Arguments
        labels (np 2d array or torch tensor): Cluster labels spread across frame
        image_size (tuple): Size of the original image
        clusters_to_consider (list): cluster ids to consider

    Returns:
        points (np.ndarray): Points in the cluster.
        point_labels (np.ndarray): Point labels.
        component_labels (np.ndarray): Component labels.
    """
    img_height, img_width = image_size

    # Convert labels to tensor if it's numpy array
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    points_list = []
    labels_list = []
    component_labels_list = []
    for label in clusters_to_consider:
        mask = labels == label

        mask = torch.nn.functional.interpolate(
            mask[None, None, :, :].float(),  # Add batch and channel dims
            size=(img_height, img_width),
            mode='nearest'
        )[0, 0]  # Remove batch and channel dims

        y_indices, x_indices = torch.where(mask)
        if args.use_connected_components:
            components = find_connected_components(mask.cpu().numpy(),
                                                   connectivity=4)
            if not components:
                warnings.warn(f"No connected components found for label {label}")
                continue
            cluster_points, component_labels = cluster_coordinates_per_component(
                                    components,
                                    args.num_points_per_entity,
                                    cluster_coordinates_fn=cluster_coordinates)

            component_labels_list.extend(component_labels.tolist())

        else:
            stacked_indices = torch.stack([y_indices, x_indices], dim=1).cpu().numpy()
            cluster_points = cluster_coordinates(stacked_indices,
                                                 args.num_points_per_entity)

        scaled_x = cluster_points[:,1:2]
        scaled_y = cluster_points[:,0:1]

        # Optional: Clamp values to ensure they stay within image boundaries
        scaled_x = np.clip(scaled_x, 0, img_width - 1)
        scaled_y = np.clip(scaled_y, 0, img_height - 1)

        points_list.extend(np.concatenate([scaled_x, scaled_y], axis=1).tolist())
        labels_list.extend([label] * len(scaled_x))

    # Convert to numpy arrays
    points = np.array(points_list)  # Shape: (N, 2)
    point_labels = np.array(labels_list)  # Shape: (N,)
    component_labels = np.array(component_labels_list)

    return points, point_labels, component_labels




def get_points_from_clustering(args, video_frames, feat_extractor, debug_vis_root):
    """Get points from clustering.

    Args:
        args (argparse.Namespace): Arguments
        video_frames (torch.Tensor): Video frames
        feat_extractor (FeatureExtractor): Object feature extractor class.
        debug_vis_root (str): Debug visualisation path


    Returns:
        points_list (list): List of points.
        point_labels_list (list): List of point labels.
        component_labels_list (list): List of component labels.
    """
    _, n_frames, h, w, _ = video_frames.shape
    #extracting dino features
    dino_features = feat_extractor(video_frames, model_type='dino').squeeze() # (T, 16, 16, 768)

     # Apply clustering for all frame frame features
    if args.clustering_method == 'bipartite':
        feat_cluster_labels = get_temporal_bipartite_clusters(
                                        dino_features, merge_ratio=args.merge_ratio,
                                        num_iters=args.num_iters)
    elif args.clustering_method == 'kmeans':
        # vectorised kmeans implmented in feat_extractor class, using that.
        feat_cluster_labels, _ = feat_extractor.cluster_features(
                            dino_features, method='kmeans',
                            n_clusters=args.n_clusters, global_clustering=True,
                            use_torch=True
                        )
    else:
        raise ValueError(f"Clustering method not implemented {args.clustering_method}")

    points_list = []
    clusters_considerd = set()
    # where max cluster is visible, we make query points from there.
    peak_frames = get_cluster_peak_frames(feat_cluster_labels)
    point_labels_list = []
    component_labels_list = []

    for frame_id in range(n_frames):
        if frame_id not in peak_frames:
            continue
        clusters_to_consider = peak_frames[frame_id]
        # getting
        feat_cluster_labels_frame = torch.tensor(feat_cluster_labels[frame_id])
        points, cluster_labels, component_labels= get_points_in_cluster(
                    args, feat_cluster_labels_frame.to(dino_features.device),
                    image_size=(h,w), clusters_to_consider=clusters_to_consider)
        clusters_considerd.update(set(cluster_labels))
        points_list.append(points)
        point_labels_list.append(cluster_labels)
        if args.use_connected_components:
            component_labels_list.append(component_labels)


        if args.debug_mode:
            make_frame_cluster_vis(video_frames, frame_id, feat_cluster_labels,
                                   debug_vis_root)


    return points_list, point_labels_list, component_labels_list
