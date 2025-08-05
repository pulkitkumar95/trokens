"""Point sampler"""
from contextlib import contextmanager
import numpy as np
import torch
from einops import rearrange

from trokens.datasets.hod import hod_obj_id_sampling
# pylint: disable=bare-except

@contextmanager
def temp_seed(seed):
    """Set tempory seed"""
    if seed is not None:
        state = np.random.get_state()
        np.random.seed(seed)
    try:
        yield
    finally:
        if seed is not None:
            np.random.set_state(state)

def cluster_sample(pt_dict, points_to_sample, cluster_with_vis=False):
    """Cluster sample

    Args:
        pt_dict (dict): Point dictionary
        points_to_sample (int): Number of points to sample
        cluster_with_vis (bool, optional): Whether to cluster with visibility. Defaults to False.

    Returns:
        point_indices (np.ndarray): Point indices
        obj_ids (np.ndarray): Object IDs
    """
    # Get unique clusters and their counts
    obj_ids = pt_dict['obj_ids']  # shape: (p,)
    assert obj_ids.shape[0] == pt_dict['pred_tracks'].shape[1]

    # Filter by visibility first if needed
    if cluster_with_vis:
        pred_visibility = pt_dict['pred_visibility']  # shape: (t, p)
        pred_visibility = rearrange(pred_visibility, 't p -> p t')  # shape: (p, t)

        # Get points that are visible in at least one frame
        visible_in_any_frame = pred_visibility.sum(dim=1) > 0  # shape: (p,)
        valid_indices = visible_in_any_frame.nonzero().squeeze()  # shape: (num_valid,)

        # Filter obj_ids to only include visible points
        obj_ids = obj_ids[valid_indices]  # shape: (num_valid,)

    unique_clusters, cluster_counts = torch.unique(obj_ids, return_counts=True)
    num_clusters = len(unique_clusters)

    # Ensure at least one point from each cluster (minimum representation)
    min_points_per_cluster = 1
    remaining_points = points_to_sample - (min_points_per_cluster * num_clusters)

    if remaining_points < 0:
        # If we need to sample fewer points than clusters, randomly select N clusters
        # Weight the selection by cluster sizes to maintain some proportionality
        weights = cluster_counts.float() / cluster_counts.sum()
        selected_clusters = torch.multinomial(
                weights,
                num_samples=points_to_sample,
                replacement=False
            )
        points_per_cluster = [1 if i in selected_clusters else 0 for i in range(num_clusters)]

    else:
        # Distribute remaining points proportionally based on cluster sizes
        total_points = cluster_counts.sum().item()
        points_per_cluster = [
                min_points_per_cluster + int(remaining_points * count.item() / total_points)
                for count in cluster_counts
        ]

        # Handle remaining points due to rounding
        points_sum = sum(points_per_cluster)
        extra_needed = points_to_sample - points_sum

        if extra_needed > 0:
            # Randomly distribute remaining points, weighted by cluster sizes
            # to maintain approximate proportionality
            weights = cluster_counts.float() / total_points
            selected_clusters = torch.multinomial(
                    weights,
                    num_samples=extra_needed,
                    replacement=True
                )

            # Add one point to each selected cluster
            for cluster_idx in selected_clusters:
                points_per_cluster[cluster_idx] += 1


    # Sample from each cluster
    all_sampled_indices = []
    for i, cluster_id in enumerate(unique_clusters):
        cluster_indices = torch.where(obj_ids == cluster_id)[0]
        points_this_cluster = points_per_cluster[i]

        # If we filtered by visibility, map back to original indices
        if cluster_with_vis:
            cluster_indices = valid_indices[cluster_indices]

        # Handle case where cluster has fewer points than needed
        if len(cluster_indices) < points_this_cluster:
            sampled_indices = np.random.choice(
                    cluster_indices.numpy(),
                    points_this_cluster,
                    replace=True
                )
        else:
            sampled_indices = np.random.choice(
                    cluster_indices.numpy(),
                    points_this_cluster,
                    replace=False
                )

        all_sampled_indices.extend(sampled_indices)

    point_indices = np.array(all_sampled_indices)

    # Add assertions
    assert len(point_indices) == points_to_sample, \
            f"Expected {points_to_sample} points but got {len(point_indices)}"

    assert np.all(point_indices >= 0) and np.all(point_indices < len(pt_dict['obj_ids'])), \
            f"Point indices out of bounds. Should be in [0, {len(pt_dict['obj_ids'])})"

    return point_indices, obj_ids[point_indices]

def visible_point_distance(points, visible_points):
    """Visible point distance

    Args:
        points (np.ndarray): Points
        visible_points (np.ndarray): Visible points

    Returns:
        float: Total distance
    """
    # Filter out the visible points
    visible_indices = np.where(visible_points)[0]
    visible_points = points[visible_indices]

    # Calculate pairwise distances between consecutive visible points
    pairwise_distances = np.linalg.norm(np.diff(visible_points, axis=0), axis=1)
    # Sum up the distances to get the total distance
    total_dist = np.sum(pairwise_distances)
    return total_dist

def get_distance_of_all_points(pred_tracks, pred_visibility):
    """Get distance of all points

    Args:
        pred_tracks (np.ndarray): Predicted tracks
        pred_visibility (np.ndarray): Predicted visibility

    Returns:
        np.ndarray: Distance of all points
    """
    num_points = pred_tracks.shape[1]
    all_points_distance = []
    for i in range(num_points):
        points = pred_tracks[:,i]
        visible_points = pred_visibility[:,i]
        point_distance = visible_point_distance(points, visible_points)
        all_points_distance.append(point_distance)
    return np.array(all_points_distance)

def get_point_query_mask(point_queries, init_mask):
    """Get point query mask

    Args:
        point_queries (np.ndarray): Point queries
        init_mask (np.ndarray): Initial mask

    Returns:
        np.ndarray: Point query masks
    """
    temporal_length = init_mask.shape[0]
    point_query_masks = []
    for point_query in point_queries:
        point_mask = np.zeros(temporal_length, dtype=bool)
        point_mask[point_query:] = True
        point_query_masks.append(point_mask[:, None])
    point_query_masks = np.concatenate(point_query_masks, axis=1)

    return point_query_masks




def stratified_sampling(data, num_samples):
    """Stratified sampling

    Args:
        data (np.ndarray): Data
        num_samples (int): Number of samples

    Returns:
        np.ndarray: Sampled points
    """
    try:
        hist_counts, hist_bins = np.histogram(data, bins='auto')
    except:
        # if auto fails, use the default 10 bins
        hist_counts, hist_bins = np.histogram(data)
    points_per_bin = num_samples // len(hist_counts)
    sampled_points = []

    for bin_idx in range(len(hist_counts)):
        bin_start = hist_bins[bin_idx]
        bin_end = hist_bins[bin_idx + 1]
        bin_indices = np.where((data >= bin_start) & (data < bin_end))[0]
        sampled_indices = np.random.choice(bin_indices, min(points_per_bin,
                                            len(bin_indices)), replace=False)
        sampled_points.extend(sampled_indices)

    remaining_samples = num_samples - len(sampled_points)
    all_indices = np.arange(len(data))
    remaining_indices = np.setdiff1d(all_indices, sampled_points)
    additional_samples = np.random.choice(remaining_indices, remaining_samples, replace=False)
    sampled_points.extend(additional_samples)

    return np.array(sampled_points)


def fix_fixed_point_sampling(index_seed, current_points, points_to_sample):
    """Fix fixed point sampling

    Args:
        index_seed (int): Index seed
        current_points (np.ndarray): Current points
        points_to_sample (int): Number of points to sample
    """
    with temp_seed(index_seed):
        sampled_indices = np.random.permutation(current_points)[:points_to_sample]
    return sampled_indices



def point_sampler(cfg, pt_dict, pred_tracks, pred_visibility,
                   points_to_sample=256, sampling_type='random', index_select=None,
                   split='train', index_seed=None):
    """Take any number of points, but sample only N points

    Args:
        pred_tracks (np.ndarray, float): Predicted points (T x N x 2)
        pred_visibility (np.ndarray, bool): Predicited points visibility (T x N)
        per_point_queries (np.ndarray, int) : Frame at which the point was queried (N)
        points_to_sample (int): Number of points to sample.
        sampling_type (str): Type of sampling to use.
    Return:
        filtered_points (np.ndarray, bool): Points that are sampled.
        point_order (np.ndarray, int): Order of the points sampled. (points_to_sample)

    """
    num_points = pred_tracks.shape[1]
    filtered_points = np.zeros(num_points, dtype=bool) # All points are False initially.
    point_order = np.arange(points_to_sample, dtype=int)
    if sampling_type == 'hod':
        indices_of_points_to_sample, _ =  hod_obj_id_sampling(cfg, pt_dict,
                                            num_bins=cfg.POINT_INFO.HOD.NUM_BINS,
                                            num_clusters=cfg.POINT_INFO.HOD.NUM_CLUSTERS,
                                            points_to_sample=points_to_sample,
                                            pt_average=False)
        filtered_points[indices_of_points_to_sample] = True
        return filtered_points, point_order
    if sampling_type == 'hod_obj_id':
        indices_of_points_to_sample, _ =  hod_obj_id_sampling(cfg, pt_dict,
                                            num_bins=cfg.POINT_INFO.HOD.NUM_BINS,
                                            num_clusters=cfg.POINT_INFO.HOD.NUM_CLUSTERS,
                                            points_to_sample=points_to_sample)
        filtered_points[indices_of_points_to_sample] = True
        return filtered_points, point_order
    if sampling_type == 'cluster_sample':
        indices_of_points_to_sample, _ =  cluster_sample(
                                                    pt_dict=pt_dict,
                                                    points_to_sample=points_to_sample)
        filtered_points[indices_of_points_to_sample] = True
        return filtered_points, point_order

    if (split=='train' and cfg.POINT_INFO.PT_FIX_SAMPLING_TRAIN) or \
        (split=='test' and cfg.POINT_INFO.PT_FIX_SAMPLING_TEST):
        fix_sample_stratergy = sampling_type
        fix_sample_num_points = points_to_sample
        try:
            sampled_dict = pt_dict[fix_sample_stratergy][fix_sample_num_points]
            sampled_indices = sampled_dict['sampled_indices']
            ids_to_consider = sampled_dict['ids_to_consider']
        except:

            new_indices = fix_fixed_point_sampling(index_seed, num_points, fix_sample_num_points)
            sampled_indices = new_indices
            ids_to_consider = None

        if ids_to_consider is None:
            ids_to_consider = np.arange(num_points).astype(int)
        filtered_points[sampled_indices] = True

        return filtered_points, ids_to_consider


    if index_select is not None:
        pred_tracks = pred_tracks[index_select]
        pred_visibility = pred_visibility[index_select]

    if sampling_type == 'random':
        sampled_points = np.random.choice(num_points, points_to_sample, replace=False)
        filtered_points[sampled_points] = True
    elif sampling_type == 'stratified':

        pred_tracks_normalised = pred_tracks / pred_tracks.max()
        all_points_distance = get_distance_of_all_points(pred_tracks_normalised, pred_visibility)
        sampled_points = stratified_sampling(all_points_distance, points_to_sample)
        filtered_points[sampled_points] = True
        distances_sampled = all_points_distance[filtered_points]
        point_order = np.argsort(distances_sampled)
    else:
        raise NotImplementedError(f"Sampling type {sampling_type} not implemented")
    return filtered_points, point_order
