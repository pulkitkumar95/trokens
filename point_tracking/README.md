# Semantic Point Extraction and Tracking using CoTracker and DINOv2

This code implements semantic point tracking and extraction using the CoTracker. The implementation uses CoTracker3 and focuses on extracting semantic points from videos using clustering methods on DINOv2 features rather than simple grid-based sampling.

## Overview

The code provides functionality to:
- Extract semantic points from videos using clustering methods (bipartite or k-means) on DINOv2 features
- Track points across video frames using CoTracker3
- Filter redundant points based on spatial proximity
- Support different clustering strategies and parameters
- Generate visualizations of tracked points
- Save tracking results in pickle format

## Key Features

- Uses CoTracker3 for robust point tracking
- Supports semantic point extraction through clustering on DINOv2 features
- Includes point filtering to remove redundant tracks
- Optional visualization of tracked points with trails
- Configurable clustering parameters and methods
- GPU acceleration support (CUDA)
- Debug mode for development and testing

## Dependencies

Please refer to the [CoTracker repository](https://github.com/facebookresearch/co-tracker) for complete dependency information. The implementation also requires:
- PyTorch
- NumPy
- Pandas
- Einops

## Input Format

The script expects a CSV file containing video information. The CSV must include:
- `video_path`: Full path to each video file that needs to be processed
- `dataset`: Dataset name for organizing output files

## Usage

The script can be run from the command line with various arguments to control the point extraction process. Key parameters include:

### Clustering Parameters
- `--clustering_method`: Method for clustering ('bipartite' or 'kmeans') on DINOv2 features
- `--n_clusters`: Number of clusters for k-means clustering
- `--num_frames_clustering`: Number of frames to use for clustering
- `--num_points_per_entity`: Number of points to sample per semantic entity
- `--merge_ratio`: Merge ratio for clustering (default: 25)
- `--num_iters`: Number of clustering iterations (default: 11)

### Tracking Parameters
- `--use_grid`: Use grid-based tracking instead of semantic points
- `--cotracker_grid_size`: Grid size for CoTracker (default: 16)
- `--use_connected_components`: Use connected components analysis
- `--fps`: FPS for video processing (optional)

### Output and Debug
- `--base_feat_path`: Base directory path for dumping extracted features
- `--make_vis`: Generate visualization GIFs
- `--debug_mode`: Enable debug mode (processes only first video)
- `--rerun`: Rerun point tracking even if results exist

### Input
- `--csv_path`: Path to the CSV file containing video information

For example, the following command shows a typical configuration:

```bash
python new_point_tracking.py --clustering_method bipartite --csv_path /path/to/videos.csv 
```

### Output Structure

The script creates the following directory structure:
```
base_feat_path/
└── cotracker3_bip_fr_32/
    └── dataset_name/
        ├── feat_dump/
        │   └── video_name.pkl
        ├── gif_dump/
        │   └── video_name.gif
        └── debug_vis/
            └── video_name/
```

The pickle files contain:
- `pred_tracks`: Tracked point coordinates across frames
- `pred_visibility`: Visibility mask for each point
- `obj_ids`: Object/cluster IDs for each point
- `point_queries`: Original query point indices

## Important note
- All datasets apart from Something Something V2 (SSV2) were extracted at 10fps. SSV2 was extracted at default fps of the videos(12fps).