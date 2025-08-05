"""Helper functions for HOD based sampling."""
import numpy as np
import torch
from einops import rearrange
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering

def shake_off_extra_points(cluster_ids, num_samples, point_taken):
    """Shake off extra points

    Args:
        cluster_ids (np.ndarray): Cluster ids (N)
        num_samples (int): Number of samples to take
        point_taken (np.ndarray): Points to take

    Returns:
        np.ndarray: Points to take
    """
    bin_of_all_clusters = np.bincount(cluster_ids)

    while len(point_taken) > num_samples:
        extra_points = len(point_taken) - num_samples
        cluster_ids_of_current_points = cluster_ids[point_taken]
        cluster_bins = np.bincount(cluster_ids_of_current_points)
        max_cluster_bin_val = np.max(cluster_bins)
        # remove the points from the cluster with max frequency
        max_cluster_bin_val = np.max(cluster_bins)
        if (cluster_bins == max_cluster_bin_val).sum() == 1:
            # only one cluster has max frequency
            cluster_to_remove = np.argmax(cluster_bins)
            second_max_freq, max_freq = np.sort(cluster_bins)[-2:]
            num_remove_points = min(extra_points, max_freq - second_max_freq)
        else:
            clusters_with_max_freq = np.argwhere(cluster_bins == max_cluster_bin_val)[:,0]
            freq_of_clusters_in_all = bin_of_all_clusters[clusters_with_max_freq]
            cluster_to_remove = clusters_with_max_freq[np.argmax(freq_of_clusters_in_all)]
            num_remove_points = 1
        poss_points_to_remove_indices = np.argwhere(
                    cluster_ids_of_current_points == cluster_to_remove)[:, 0]
        poss_points_to_remove = np.array(point_taken)[poss_points_to_remove_indices]
        point_to_remove = np.random.choice(poss_points_to_remove,
                                            num_remove_points, replace=False)
        point_taken = np.setdiff1d(point_taken, point_to_remove)
    return point_taken

def cluster_equi_sampling(cluster_ids, num_samples, points_taken=None):
    """Sampling equally in each cluster.

    Args:
        cluster_ids (np.ndarray): Cluster ids (N)
        num_samples (int): Number of samples to take
        points_taken (list): Points already taken

    Returns:
        sampled_points: Sampled points (num_samples)
    """
    all_indices = np.arange(len(cluster_ids), dtype=int)
    remaining_indices = np.setdiff1d(all_indices, points_taken)
    unique_clusters = np.unique(cluster_ids[remaining_indices])
    points_per_cluster = max(num_samples // len(unique_clusters), 1)
    sampled_points = []
    for cluster_id in unique_clusters:
        cluster_indices = np.where(cluster_ids == cluster_id)[0]
        cluster_indices = np.setdiff1d(cluster_indices, points_taken)
        sampled_indices = np.random.choice(cluster_indices,
                    min(points_per_cluster, len(cluster_indices)), replace=False)
        sampled_points.extend(sampled_indices)
    return np.array(sampled_points)

def id_based_sampling(cluster_ids, num_samples):
    """Cluster ID based sampling

    Args:
        cluster_ids (np.ndarray): Cluster ids (N)
        num_samples (int): Number of samples to take

    Returns:
        np.ndarray: Sampled points (num_samples)
    """
    # try to sample equal number of points from each cluster
    cluster_ids = cluster_ids.astype(int)
    def sample_cluster(cluster_ids, num_samples):
        sampled_points = []
        while len(sampled_points) < num_samples:
            curr_sampled_points = cluster_equi_sampling(
                                    cluster_ids, num_samples, sampled_points)
            sampled_points.extend(curr_sampled_points)
        if len(sampled_points) > num_samples:
            # Remove the points that are extra
            # remove the points with max frequency in cluster ids
            sampled_points = shake_off_extra_points(cluster_ids, num_samples, sampled_points)


        return np.array(sampled_points)

    if isinstance(num_samples, int):
        if num_samples > len(cluster_ids):
            #sample points with replacement
            sampled_points = np.random.choice(len(cluster_ids), num_samples, replace=True)
        else:
            sampled_points = sample_cluster(cluster_ids, num_samples)

        return np.array(sampled_points)
    sampled_points = {}
    for num_sample in num_samples:
        if num_sample > len(cluster_ids):
            sampled_points[num_sample] = np.arange(len(cluster_ids), dtype=int)
        else:
            sampled_points[num_sample] = sample_cluster(cluster_ids, num_sample)

    return sampled_points



def estimate_cost_matrix(gt_labels, cluster_labels):
    """Estimate the cost matrix

    Args:
        gt_labels (np.ndarray): Ground truth labels
        cluster_labels (np.ndarray): Cluster labels

    Returns:
        np.ndarray: Cost matrix
    """
    # Make sure the lengths of the inputs match:
    if len(gt_labels) != len(cluster_labels):
        print('The dimensions of the gt_labls and the pred_labels do not match')
        return -1
    l_gt = np.unique(gt_labels)
    l_pr = np.unique(cluster_labels)
    nclass_pred = len(l_pr)
    dim_1 = max(nclass_pred, np.max(l_gt) + 1)
    profit_mat = np.zeros((nclass_pred, dim_1))
    for i in l_pr:
        idx = np.where(cluster_labels == i)
        gt_selected = gt_labels[idx]
        for j in l_gt:
            profit_mat[i][j] = np.count_nonzero(gt_selected == j)
    return -profit_mat

def hungarian_matching(gt_labels, req_c):
    """Hungarian matching

    Args:
        gt_labels (np.ndarray): Ground truth labels
        req_c (int): Number of clusters

    Returns:
        dict: Mapping from new to old cluster ids
    """
    gt_labels = gt_labels.astype(int)
    req_c = req_c.astype(int)
    cost_matrix = estimate_cost_matrix(gt_labels, req_c)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    new_to_old_mapping = {new: old for new, old in zip(col_ind, row_ind)}

    return new_to_old_mapping

def manage_cluster_ids(prev_cluster_id_info, current_cluster_id_info, current_point_indices):
    """Manage the cluster ids

    Args:
        prev_cluster_id_info (np.ndarray): Previous cluster ids (N)
        current_cluster_id_info (np.ndarray): Current cluster ids (N)

    Returns:
        final_cluster_ids: Final cluster ids (N)
    """
    prev_assigned_points = np.argwhere(prev_cluster_id_info != -1)[:,0]
    if len(prev_assigned_points) == 0:
        prev_cluster_id_info[current_point_indices] = current_cluster_id_info
        return prev_cluster_id_info

    previous_cluster_ids = prev_cluster_id_info[prev_assigned_points]
    curent_cluster_ids_of_prev_assigned = current_cluster_id_info[prev_assigned_points]
    new_to_old_mapping = hungarian_matching(previous_cluster_ids,
                                            curent_cluster_ids_of_prev_assigned)
    unqiue_new_ids = np.unique(current_cluster_id_info)
    cluster_ids_not_mapped = np.setdiff1d(unqiue_new_ids, list(new_to_old_mapping.keys()))
    new_cluster_id = np.max(max(new_to_old_mapping.values())) + 1
    for cluster_id in cluster_ids_not_mapped:
        new_to_old_mapping[cluster_id] = new_cluster_id
        new_cluster_id += 1
    mask = np.isin(current_cluster_id_info, list(new_to_old_mapping.keys()))
    assert np.all(mask)
    final_cluster_ids = np.where(mask, np.vectorize(new_to_old_mapping.get)(
                                                current_cluster_id_info), -1)
    prev_cluster_id_info[current_point_indices] = final_cluster_ids

    return prev_cluster_id_info


def get_feat_at_level(hod_feat, level):
    """Get feature at level

    Args:
        hod_feat (np.ndarray): HOD features
        level (int): Level

    Returns:
        np.ndarray: Feature at level
    """
    if level == 1:
        return hod_feat.sum(axis=1)
    else:
        num_splits = 2**(level-1)
        split_size = hod_feat.shape[1] // num_splits
        split_feats = []
        for i in range(num_splits):
            split_feats.append(hod_feat[:, i*split_size:(i+1)*split_size].sum(axis=1))
        split_feats = torch.cat(split_feats, axis=-1)
        return split_feats

def compute_temporal_pyramid(hod_feat, levels=3):
    """Compute temporal pyramid

    Args:
        hod_feat (np.ndarray): HOD features
        levels (int): Number of levels

    Returns:
        np.ndarray: Temporal pyramid features
    """
    pyramid_feats = []
    for level in range(levels):
        pyramid_feats.append(get_feat_at_level(hod_feat, level+1))
    pyramid_feats = torch.cat(pyramid_feats, axis=-1)
    return pyramid_feats



def get_hist_vector(bin_indices, bin_weigthts, num_bins, preserve_temporal=False):
    """Get histogram vector

    Args:
        bin_indices (np.ndarray): Bin indices
        bin_weigthts (np.ndarray): Bin weights
        num_bins (int): Number of bins
        preserve_temporal (bool): Whether to preserve temporal information

    Returns:
        np.ndarray: Histogram vector
    """
    final_vec = []
    bin_indices = bin_indices.astype(int)
    for pt_idx in range(bin_indices.shape[0]):
        pt_bin_indices = bin_indices[pt_idx]
        pt_bin_weights = bin_weigthts[pt_idx]
        if preserve_temporal:
            pt_vec = np.zeros((pt_bin_indices.shape[0] + 1 , num_bins))
            for idx in range(pt_bin_indices.shape[0]):
                pt_vec[idx + 1, pt_bin_indices[idx]] += pt_bin_weights[idx]
        else:
            pt_vec = np.bincount(pt_bin_indices, weights=pt_bin_weights,
                                                        minlength=num_bins)



        final_vec.append(pt_vec[None])
    final_vec = np.vstack(final_vec)
    return final_vec

def disp_wth_delta(points, delta):
    """Compute displacement with time delta if not 1

    Args:
        points (np.ndarray): Points
        delta (int): Time delta

    Returns:
        np.ndarray: Displacement
    """
    if delta == 1:
        return points[:, 1:] - points[:, :-1]
    else:
        # For t ≤ delta: calculate displacement from t=0
        # For t > delta: calculate displacement from t-delta
        temporal_length = points.shape[1]
        disp = np.zeros((points.shape[0], temporal_length-1, 2))

        # First delta steps: displacement from t=0
        disp[:, :delta] = points[:, 1:delta+1] - points[:, 0:1]
        for t in range(delta, temporal_length-1):
            disp[:, t] = points[:, t] - points[:, t-delta]
        return disp


def get_orientation_hist(points, num_bins, delta=1, preserve_temporal=False):
    """Get histogram of oriented displacements for the points

    Args:
        points (np.ndarray ): Point to use (N, T, 2)
        num_bins (int): Number of bins to use for the histogram

    Returns:
        final_hist: Histogram of oriented displacements (N, num_bins)
    """

    disp = disp_wth_delta(points, delta)
    magnitude = np.linalg.norm(disp, axis=-1)
    angle = np.arctan2(disp[:, :,1], disp[:, :,0]) / np.pi
    angle = (angle+1)/2
    angle_bins = angle * num_bins
    frac, mod_bin = np.modf(angle_bins)

    back_mag_to_take = magnitude * frac
    back_bin = mod_bin % num_bins
    back_hist_vec = get_hist_vector(back_bin, back_mag_to_take,
                                                    num_bins, preserve_temporal)

    forward_mag_to_take = magnitude * ( 1 - frac)
    forward_bin = (mod_bin + 1) % num_bins
    forward_hist_vec = get_hist_vector(forward_bin, forward_mag_to_take,
                                                    num_bins, preserve_temporal)
    final_hist = forward_hist_vec + back_hist_vec
    return final_hist

def get_l1_distance_matrix(matrix):
    """Calculate L1 distance between all pairs of samples in matrix using vectorization

    Args:
        matrix (np.ndarray): Input matrix of shape (n, D) containing n samples of D dimensions

    Returns:
        dist_matrix (np.ndarray): Distance matrix of shape (n, n) containing L1 distances
    """
    # Expand dimensions to enable broadcasting
    # Shape: (n, 1, D) - (1, n, D) = (n, n, D)
    diff = matrix[:, np.newaxis, :] - matrix[np.newaxis, :, :]

    # Calculate L1 distances by taking absolute values and summing along last axis
    dist_matrix = np.abs(diff).sum(axis=-1)

    return dist_matrix

def get_per_cluster_dist(feat, cluster_ids, min_samples_per_cluster=1):
    """Get per cluster distance

    Args:
        feat (np.ndarray): Features
        cluster_ids (np.ndarray): Cluster ids
        min_samples_per_cluster (int): Minimum samples per cluster
    """
    cluster_metrics = []

    for cluster_id in np.unique(cluster_ids):
        cluster_indices = np.where(cluster_ids == cluster_id)[0]
        # if the cluster has only one point, skip it
        if len(cluster_indices) <= min_samples_per_cluster:
            continue
        cluster_points = feat[cluster_indices]

        # Calculate multiple metrics
        l1_max_dist = get_l1_distance_matrix(cluster_points).max()

        # Calculate variance within cluster
        cluster_variance = np.var(cluster_points, axis=0).sum()

        # Calculate silhouette-like score using mean intra vs inter distances
        dist_matrix = get_l1_distance_matrix(cluster_points)
        intra_dist = np.mean(dist_matrix)  # average distance within cluster

        # Combine metrics (you can adjust weights)
        split_score = l1_max_dist * 0.4 + cluster_variance * 0.4 + intra_dist * 0.2

        cluster_metrics.append([cluster_id, split_score])
    return np.array(cluster_metrics)

def cluster_matrix_l1_agglom(matrix, num_clusters=2):
    """Cluster matrix into 2 clusters using agglomerative clustering with L1 distance

    Args:
        matrix (np.ndarray): Input matrix of shape (n_samples, n_features)

    Returns:
        cluster_ids (np.ndarray): Cluster assignments (0 or 1) for each sample
    """
    clustering = AgglomerativeClustering(
        n_clusters=num_clusters,
        metric='euclidean',  # L1 distance
        linkage='average'    # Use average linkage
    )

    cluster_ids = clustering.fit_predict(matrix)
    return cluster_ids


def aggo_top_down(feat, req_c, min_samples_per_cluster=1):
    """Agglomerative top down clustering

    Args:
        feat (np.ndarray): Features
        req_c (int): Number of clusters
        min_samples_per_cluster (int): Minimum samples per cluster
    """
    cluster_ids = np.zeros(feat.shape[0], dtype=int)
    while len(np.unique(cluster_ids)) < req_c:
        new_cluster_id = int(np.max(cluster_ids) + 1)
        cluter_id_dist = get_per_cluster_dist(feat, cluster_ids,
                                min_samples_per_cluster=min_samples_per_cluster)
        max_dist_cluster_id = cluter_id_dist[np.argmax(cluter_id_dist[:, 1])][0]
        # split the cluster with max distance
        cluster_indices = np.where(cluster_ids == max_dist_cluster_id)[0]
        feats_to_cluster = feat[cluster_indices]
        req_c_clusters = cluster_matrix_l1_agglom(feats_to_cluster)
        cluster_ids[cluster_indices[req_c_clusters==1]] = new_cluster_id
    return cluster_ids


def get_hod_based_clustering(points, num_bins, req_c=None,
                            temporal_rate=4, min_samples_per_cluster=1):
    """Get histogram of oriented displacements for the points

    Args:
        points (np.ndarray ): Point to use (N, T, 2)
        num_bins (int): Number of bins to use for the histogram

    Returns:
        final_hist: Histogram of oriented displacements (N, num_bins)
    """
    points = points[:, ::temporal_rate]

    hist = get_orientation_hist(points, num_bins)
    req_c_clusters = aggo_top_down(hist, req_c,
                                min_samples_per_cluster=min_samples_per_cluster)


    return req_c_clusters



def hod_based_sampling(points, points_to_sample=None, num_bins=32,
                        num_clusters=None, id_based_sampling_on=True,
                        min_samples_per_cluster=1):
    """HOD based sampling

    Args:
        points (np.ndarray): Points
        points_to_sample (int, optional): Number of points to sample. Defaults to None.
        num_bins (int, optional): Number of bins. Defaults to 32.
        num_clusters (int, optional): Number of clusters. Defaults to None.
        id_based_sampling_on (bool, optional): Whether to use id based sampling. Defaults to True.
        min_samples_per_cluster (int, optional): Minimum samples per cluster. Defaults to 1.

    Returns:
        np.ndarray: Sampled points
        np.ndarray: Sampled object ids
    """
    if num_clusters is None:
        num_clusters = points_to_sample // 2

    points_normalised = points / points.max()
    points_to_use = rearrange(points_normalised, 't n d -> n t d').numpy()
    if len(points_to_use) > num_clusters:
        ids_to_consider = get_hod_based_clustering(
                            points_to_use, num_bins, req_c=num_clusters,
                            min_samples_per_cluster=min_samples_per_cluster)
    else:
        ids_to_consider = np.arange(points_to_use.shape[0])
    if id_based_sampling_on:
        sampled_points = id_based_sampling(ids_to_consider, points_to_sample)
    else:
        sampled_points = None
    return sampled_points, ids_to_consider


def sample_n_per_cluster_with_old_ids(cluster_ids, old_cluster_ids, n_per_cluster=10):
    """Sample exactly n points from each cluster, trying to maintain equal representation
    from old cluster IDs within each cluster.

    Args:
        cluster_ids (np.ndarray): Array of current cluster IDs
        old_cluster_ids (np.ndarray): Array of old cluster IDs for each point
        n_per_cluster (int): Number of points to sample from each cluster

    Returns:
        np.ndarray: Array of sampled indices
    """
    unique_clusters = np.unique(cluster_ids)
    sampled_indices = []

    for cluster in unique_clusters:
        # Get indices for current cluster
        cluster_mask = cluster_ids == cluster
        cluster_indices = np.where(cluster_mask)[0]
        old_ids_in_cluster = old_cluster_ids[cluster_mask]
        cluster_points_ids_sampled = id_based_sampling(old_ids_in_cluster, n_per_cluster)

        #sampled_indices.extend(cluster_indices[cluster_points_ids_sampled])

        try:
            sampled_indices.extend(cluster_indices[cluster_points_ids_sampled])
        except IndexError as e:
            print('debug!!!!cluster_points_ids_sampled', cluster_points_ids_sampled)
            print('debug!!!!cluster_points_ids_sampled dtype', cluster_points_ids_sampled.dtype)
            print("IndexError occurred:", e)
            print("Data type of cluster_points_ids_sampled:", cluster_points_ids_sampled.dtype)
            print("Contents of cluster_points_ids_sampled:", cluster_points_ids_sampled)
            print("Cluster indices:", cluster_indices)
            print("points per clusters IS:", n_per_cluster)
            raise  # Re-raise the exception after logging for further handling if needed

    return np.array(sampled_indices)


def average_cluster_points(points, cluster_ids):
    """
    Efficiently average points within each cluster at each timestep using vectorized operations.

    Args:
        points: Tensor/Array of shape (T, N, 2) where:
            T = number of timesteps
            N = number of points
            2 = x,y coordinates
        cluster_ids: Array of shape (N,) containing cluster IDs from 0 to M-1
            where M is number of clusters

    Returns:
        averaged_points: Tensor/Array of shape (T, M, 2) containing averaged positions
            where M is number of unique clusters
    """
    max_clusters = max(cluster_ids) + 1

    # Create a one-hot encoding matrix for cluster assignments
    one_hot = np.eye(max_clusters)[cluster_ids]  # Shape: (N, M)

    # Compute denominator (count of points in each cluster)
    cluster_sizes = one_hot.sum(axis=0)  # Shape: (M,)
    # Add small epsilon to avoid division by zero
    cluster_sizes = cluster_sizes[:, None] + 1e-8  # Shape: (M, 1)

    # Compute the sum for each cluster and divide by cluster sizes
    # Using einsum for efficient matrix multiplication
    averaged_points = np.einsum('tnd,nm->tmd', points, one_hot) / cluster_sizes

    return averaged_points



def hod_obj_id_sampling(cfg, pt_dict, num_bins=32, num_clusters=16,
                                        points_to_sample=80, pt_average=True):
    """HOD based object id sampling

    Args:
        cfg (CfgNode): Config node
        pt_dict (dict): Point dictionary
        num_bins (int): Number of bins
        num_clusters (int): Number of clusters
        points_to_sample (int): Number of points to sample
        pt_average (bool): Whether to average points

    Returns:
        np.ndarray: Sampled points
        torch.Tensor: Sampled object ids
    """
    points = pt_dict['pred_tracks']
    pred_visibility = pt_dict['pred_visibility'].numpy().astype(bool)
    pred_visibility[pred_visibility==False] = True # pylint: disable=singleton-comparison
    obj_ids = pt_dict[cfg.POINT_INFO.OBJ_ID_KEY]
    if torch.is_tensor(obj_ids):
        obj_ids = obj_ids.numpy()
    if pt_average:
        # average points within each each object id
        points_averaged = average_cluster_points(points.numpy(), obj_ids)
        unique_obj_ids = np.unique(obj_ids)
        unique_obj_ids.sort()
        new_avg_points = []
        obj_ids_mapping = {}

        for idx, obj_id in enumerate(unique_obj_ids):
            feat_to_take = points_averaged[:, obj_id:obj_id+1]
            obj_ids_mapping[obj_id] = idx
            new_avg_points.append(feat_to_take)
        new_avg_points = torch.from_numpy(np.concatenate(new_avg_points, axis=1))
        new_obj_ids = np.array([obj_ids_mapping[obj_id] for obj_id in obj_ids])
        min_samples_per_cluster = 1
    else:
        new_avg_points = points
        new_obj_ids = obj_ids
        num_points = points.shape[1]
        if cfg.POINT_INFO.HOD_MIN:
            min_samples_per_cluster = max((num_points // len(np.unique(obj_ids))) // 2, 1)
        else:
            min_samples_per_cluster = 1

    _, ids_to_consider = hod_based_sampling(new_avg_points,
                                            num_bins=num_bins,
                                            num_clusters=num_clusters,
                                            id_based_sampling_on=False,
                                            min_samples_per_cluster=min_samples_per_cluster)
    if pt_average:
        ids_to_consider = np.array([ids_to_consider[new_obj_ids[idx]]
                                        for idx in range(new_obj_ids.shape[0])])


    points_per_cluster = points_to_sample // len(np.unique(ids_to_consider))


    # clip points per cluster minimal is 1
    points_per_cluster = max(points_per_cluster, 1)

    sampled_points = sample_n_per_cluster_with_old_ids(ids_to_consider,
                                                    obj_ids,
                                                    n_per_cluster=points_per_cluster)
    if len(sampled_points) < points_to_sample:
        num_extra_points = points_to_sample - len(sampled_points)
        extra_points = np.random.choice(np.arange(points.shape[1]),
                                            num_extra_points, replace=False)
        sampled_points = np.concatenate([sampled_points, extra_points])
    else:
        sampled_points = np.random.choice(sampled_points,
                                          points_to_sample, replace=False)

    return sampled_points, torch.from_numpy(ids_to_consider[sampled_points]).long()
