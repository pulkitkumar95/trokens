"""
Point tracking module for extracting and tracking semantic points in videos.

This module provides functionality to:
- Extract semantic points from videos using clustering methods
- Track points across video frames using CoTracker
- Save tracking results and generate visualizations
"""
import sys
import os
import time
import random
import argparse
import pickle
import torch
import numpy as np
from einops import rearrange
import pandas as pd
from utils import convert_points_for_tracking, save_video
from feat_extractor import feature_extract
from get_semantic_points import get_points_from_clustering
from new_video_loader import load_video
from omni_vis import vis_trail

# set seeds
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE_PATH = '/fs/cfar-projects/actionloc/camera_ready/tats_v2/dumps'

os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
# pylint: disable=redefined-outer-name


def check_columns_in_df(df):
    """Check if the dataframe has the required columns.

    Args:
        df (pd.DataFrame): Dataframe to check

    Raises:
        ValueError: If the dataframe does not have the required columns
    """
    required_columns = ['video_path', 'dataset']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in the dataframe")


def extract_points(args, cotracker, feat_extractor, video_path, ds_dump_path):
    """Extract points from a video and save them to a pickle file.

    Args:
        args (argparse.Namespace): Arguments
        cotracker (torch.nn.Module): Cotracker model
        feat_extractor (torch.nn.Module): Feature extractor model
        video_path (str): Path to the video
        ds_dump_path (str): Path to the directory where the pickle file will be saved

    Returns:
        bool: True if the points were extracted, False otherwise
    """
    # load video for DINO feat extractor
    vid_name = video_path.split('/')[-1].split('.')[0]
    debug_vis_dump_root = os.path.join(ds_dump_path, 'debug_vis', vid_name)
    feat_dump_path = os.path.join(ds_dump_path, 'feat_dump', f'{vid_name}.pkl')
    gif_dump_path = os.path.join(ds_dump_path, 'gif_dump', f'{vid_name}.gif')
    if os.path.exists(feat_dump_path) and not args.rerun:
        return True

    video_loaded, video_frames, frames_id_dict = load_video(
        video_path, return_tensor=True, use_float=False,
        num_frames=args.num_frames_clustering, sample_all_frames=False,
        fps=args.fps)  # (B, T, C, H, W)
    if not video_loaded:
        print(f"Video {vid_name} not loaded")
        return None
    video_frames = rearrange(video_frames, 'b t c h w -> b t h w c')
    video_frames = video_frames.cpu().numpy()

    if args.debug_mode:
        time_start = time.time()

    base_point_info = get_points_from_clustering(
        args, video_frames, feat_extractor, debug_vis_dump_root)
    points_list, point_labels_list, component_labels_list = base_point_info
    queries_points, cluster_ids_all_frames = convert_points_for_tracking(
        points_list, point_labels_list, frames_id_dict=frames_id_dict,
        component_labels_list=component_labels_list,
        use_connected_components=args.use_connected_components, device=args.device)
    if args.debug_mode:
        os.makedirs(debug_vis_dump_root, exist_ok=True)
        time_end = time.time()
        print(f"Time taken to get points and labels: {time_end - time_start} seconds")
    torch.cuda.empty_cache()

    _, video, _ = load_video(video_path, return_tensor=True, use_float=True,
                             device=args.device, sample_all_frames=True, fps=args.fps)  # B T C H W

    if args.debug_mode:
        time_start = time.time()
    if args.use_grid:
        pred_tracks, pred_visibility = cotracker(
            video, grid_size=args.cotracker_grid_size,
            queries=None, backward_tracking=False)
    else:
        pred_tracks, pred_visibility = cotracker(video,
                                                 queries=queries_points,
                                                 backward_tracking=True)
    if args.debug_mode:
        time_end = time.time()
        print(f"Time taken to run cotracker: {time_end - time_start} seconds")
    point_queries = queries_points.cpu().squeeze(0).numpy()[:, 0]
    pred_tracks = pred_tracks.cpu().squeeze(0).numpy()
    pred_visibility = pred_visibility.cpu().squeeze(0).numpy()
    video = video.cpu().squeeze(0).numpy()
    video = rearrange(video, 't c h w -> t h w c')
    pt_obj_cluster_dict = {}

    dump_dict = {
        'pred_tracks': torch.tensor(pred_tracks).half(),
        'pred_visibility': torch.tensor(pred_visibility).bool(),
        'obj_ids': torch.tensor(cluster_ids_all_frames).long(),
        'point_queries': torch.tensor(point_queries).long(),
        **pt_obj_cluster_dict
    }

    os.makedirs(os.path.dirname(feat_dump_path), exist_ok=True)
    pickle.dump(dump_dict, open(feat_dump_path, "wb"))
    torch.cuda.empty_cache()

    if args.debug_mode or args.make_vis:
        frames = vis_trail(video, pred_tracks, pred_visibility,
                           cluster_ids=cluster_ids_all_frames)
        os.makedirs(os.path.dirname(gif_dump_path), exist_ok=True)
        save_video(frames, gif_dump_path)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_mode", action="store_true",
                        help="Enable debug mode")

    parser.add_argument("--use_connected_components", action="store_true",
                        help="Use connected components")

    parser.add_argument("--num_frames_clustering", type=int, default=32,
                        help="Number of frames to cluster")

    parser.add_argument("--merge_ratio", type=int, default=25,
                        help="Merge ratio")

    parser.add_argument("--num_iters", type=int, default=11,
                        help="Number of iterations")

    parser.add_argument("--clustering_method", type=str, default='bipartite',
                        help="Clustering method to use")

    parser.add_argument("--n_clusters", type=int, default=32,
                        help="Number of clusters")

    parser.add_argument("--num_points_per_entity", type=int, default=16,
                        help="Number of samples per mask")

    parser.add_argument("--use_grid", action="store_true",
                        help="Use grid")

    parser.add_argument("--cotracker_grid_size", type=int, default=16,
                        help="Cotracker grid size")

    parser.add_argument("--csv_path", type=str, default='sample.csv',
                        help='Path to csv file')

    parser.add_argument("--fps", type=int, default=None,
                        help="FPS for point tracking")

    parser.add_argument("--base_feat_path", type=str, default=BASE_PATH,
                        help="Base path for feature dumps")

    parser.add_argument("--make_vis", action="store_true",
                        help="Make gifs")
    parser.add_argument("--rerun", action="store_true",
                        help="Rerun the point tracking")

    args = parser.parse_args()

    use_connected_components = args.use_connected_components

    if args.clustering_method == 'kmeans':
        CLUSTER_STR = f'kmeans_n{args.n_clusters}'
    elif args.clustering_method == 'bipartite':
        CLUSTER_STR = 'bip'
    else:
        raise ValueError(f"Invalid clustering method: {args.clustering_method}")
    df = pd.read_csv(args.csv_path)
    check_columns_in_df(df)
    if args.debug_mode:
        df = df.iloc[:1]  # just running on the first sample for debugging

    dump_name = f'cotracker3_{CLUSTER_STR}_fr_{args.num_frames_clustering}'
    if args.merge_ratio != 25 or args.num_iters != 11:  # if not default then add to dump name
        dump_name += f'_m{args.merge_ratio}_i{args.num_iters}'
    if use_connected_components:
        dump_name += '_concomp'
    if args.fps is not None:
        dump_name += f'_fps_{args.fps}'

    # base_featpath = '/fs/cfar-projects/actionloc/shirley/sam_based_debug/somethingv2'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setattr(args, 'device', device)
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    feat_extractor = feature_extract()

    for video_index, vid_info_row in df.iterrows():
        dataset = vid_info_row['dataset']
        video_path = vid_info_row['video_path']
        video_uniq_id = video_path.split('/')[-1].split('.')[0]
        feat_dump_name = f'{video_uniq_id}'
        ds_dump_path = os.path.join(args.base_feat_path, dump_name, dataset)
        extract_points(args, cotracker, feat_extractor, video_path, ds_dump_path)
