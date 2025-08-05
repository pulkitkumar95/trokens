"""
This file contains the functions to load videos.
"""
import os
import av
import torch
import numpy as np
from decord import VideoReader


def load_video_decord(vid_path, return_tensor=False,
                      use_float=False, num_frames=8,
                      sample_all_frames=False, fps=None, device=None):
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
        frames: (B, T, C, H, W) numpy array or tensor, where T = num_frames
        frame_id_dict: dictionary mapping sampled frame indices to original  indices
    '''
    print(f"Processing {vid_path}...")
    assert os.path.exists(vid_path), f"Video file {vid_path} does not exist"


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
            print(f"Warning: num_frames ({num_frames}) is greater than "
                  f"available frames ({available_frames})")
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
            raise ValueError("Frames contain NaNs or Infs")
        if device is not None:
            frames = frames.to(device)
    return frames, frame_id_dict


def load_video(vid_path, return_tensor=False, device=None, use_float=False,
               num_frames=8, sample_all_frames=False, fps=None):
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
        frames: (B, T, C, H, W) numpy array or tensor, where T = num_frames
    '''
    print(f"Processing {vid_path}...")
    assert os.path.exists(vid_path), f"Video file {vid_path} does not exist"

    # Option 2: Using PyAV
    # pylint: disable=broad-exception-caught
    try:
        container = av.open(vid_path)
    except (OSError, ValueError, Exception):
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
                if use_float:
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
            print(f"Warning: num_frames ({num_frames}) is greater than "
                  f"total_frames ({total_frames}). Adjusting num_frames to total_frames.")
            num_frames = total_frames  # Set num_frames to total_frames
        # Calculate indices for uniformly sampled frames
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        frames = frames[:, frame_indices]  # (1, T, C, H, W)
        frame_id_dict = {i: frame_indices[i] for i in range(num_frames)}

    if return_tensor:
        frames = torch.from_numpy(frames).to(device)

    return True, frames, frame_id_dict
