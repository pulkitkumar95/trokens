<div align="center">
<h1>Trokens: Semantic-Aware Relational Trajectory Tokens for Few-Shot Action Recognition</h1>

[**Pulkit Kumar***](https://www.cs.umd.edu/~pulkit/)<sup>1</sup> · [**Shuaiyi Huang***](https://shuaiyihuang.github.io/)<sup>1</sup> · [**Matthew Walmer**](https://www.cs.umd.edu/~mwalmer/)<sup>1</sup> · [**Sai Saketh Rambhatla**](https://rssaketh.github.io)<sup>1,2</sup> · [**Abhinav Shrivastava**](http://www.cs.umd.edu/~abhinav/)<sup>1</sup>

<sup>1</sup>University of Maryland, College Park&emsp;&emsp;&emsp;&emsp;<sup>2</sup>GenAI, Meta <br>
**ICCV 2025** <br>
<sup>*Equal contribution</sup>

<a href='https://arxiv.org/abs/2508.03695'><img src='https://img.shields.io/badge/arXiv-Trokens-red' alt='Paper PDF'></a>
<a href='https://trokens-iccv25.github.io'><img src='https://img.shields.io/badge/Project_Page-Trokens-green' alt='Project Page'></a>
<a href='https://huggingface.co/datasets/pulkitkumar95/trokens_pt_data'><img src='https://img.shields.io/badge/Hugging_Face-Data-blue' alt='Hugging Face'></a>

<!-- <p float='center'><img src="assets/teaser.png" width="80%" /></p>
<span style="color: green; font-size: 1.3em; font-weight: bold;">LocoTrack is an incredibly efficient model,</span> enabling near-dense point tracking in real-time. It is <span style="color: red; font-size: 1.3em; font-weight: bold;">6x faster</span> than the previous state-of-the-art models. -->
</div>

This repository contains the official code for the paper "Trokens: Semantic-Aware Relational Trajectory Tokens for Few-Shot Action Recognition".

</div>

## Installation

1. Create a environment using either conda or venv:
```bash
conda create -n trokens python=3.10
```

2. Activate the environment:
```bash
conda activate trokens
```

3. Install all dependencies:
```bash
pip install -r requirements.txt
```

## Setting Up Trokens Point Tracking Data

The pre-computed Trokens point tracking data is available on Hugging Face at: [https://huggingface.co/datasets/pulkitkumar95/trokens_pt_data](https://huggingface.co/datasets/pulkitkumar95/trokens_pt_data)

To download and set up the point tracking data:

```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Download the dataset
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="pulkitkumar95/trokens_pt_data",
    repo_type="dataset",
    local_dir=<LOCAL_PATH_TO_SAVE_TROKENS_PT_DATA>
)
```

Or using the command line:
```bash
huggingface-cli download pulkitkumar95/trokens_pt_data --repo-type dataset --local-dir <LOCAL_PATH_TO_SAVE_TROKENS_PT_DATA>
```
Once the dataset is downlaoded, unzip the individual dataset directory by running:
```bash
cd <LOCAL_PATH_TO_SAVE_TROKENS_PT_DATA>/cotracker3_bip_fr_32
unzip *zip 
```


### Extracting Point Tracking Data for Custom Datasets

All the point tracking data provided was extracted using the scripts available in the `point_tracking/` directory. For details on the extraction process and how to extract point tracking data for new custom datasets, please refer to `point_tracking/README.md`.

## Training and Testing

Before running the training, set the following environment variables:
```bash
# Set Config name (e.g., ssv2_small,ssv2_full, hmdb, k400, finegym)
export CONFIG_TO_USE=ssv2_small
export DATASET=ssv2
export EXP_NAME=trokens_release
export SECONDAY_EXP_NAME=sample_exp

# Path to store PyTorch models and weights
export TORCH_HOME=<LOCAL_PATH_TO_SAVE_PYTORCH_MODELS>

# Path to dataset directory containing videos
export DATA_DIR=<LOCAL_PATH_TO_SAVE_DATASET>

# Path to pre-computed Trokens point tracking data and few shot info from huggingface.
export TROKENS_PT_DATA=<LOCAL_PATH_TO_SAVE_TROKENS_PT_DATA>

# Base output directory for experiments
export BASE_OUTPUT_DIR=<LOCAL_PATH_TO_SAVE_EXPERIMENTS>
```

### Using the Sample Script

A sample training script is provided in `scripts/trokens.sh`. After setting the above environment variables and configuring the paths, you can run:

```bash
bash scripts/trokens.sh <config_name_to_use>
```

For example:
```bash
bash scripts/trokens.sh ssv2_small
```

### Manual Training Command

Alternatively, to train the model manually, you can use the following command:

```bash
torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    tools/run_net.py --init_method env:// --new_dist_init \
    --cfg configs/trokens/$CONFIG_TO_USE.yaml \
    WANDB.ID $WANDB_ID \
    WANDB.EXP_NAME $EXP_NAME \
    MASTER_PORT $MASTER_PORT \
    OUTPUT_DIR $BASE_OUTPUT_DIR/$CONFIG_TO_USE/$EXP_NAME/$SECONDAY_EXP_NAME \
    NUM_GPUS $NUM_GPUS \
    DATA_LOADER.NUM_WORKERS $NUM_WORKERS \
    DATA.USE_RAND_AUGMENT True \
    DATA.PATH_TO_DATA_DIR $DATA_DIR \
    DATA.PATH_TO_TROKEN_PT_DATA $TROKENS_PT_DATA \
    FEW_SHOT.K_SHOT $K_SHOT \
    FEW_SHOT.TRAIN_QUERY_PER_CLASS 6 \
    FEW_SHOT.N_WAY $N_WAY \
    POINT_INFO.NAME $POINT_INFO_NAME \
    POINT_INFO.SAMPLING_TYPE cluster_sample \
    POINT_INFO.NUM_POINTS_TO_SAMPLE $NUM_POINTS_TO_SAMPLE \
    MODEL.FEAT_EXTRACTOR dino \
    MODEL.DINO_CONFIG dinov2_vitb14 \
    MODEL.MOTION_MODULE.USE_CROSS_MOTION_MODULE True \
    MODEL.MOTION_MODULE.USE_HOD_MOTION_MODULE True
```

Key parameters:
- `CONFIG_TO_USE`: Configuration file to use (e.g., ssv2_full, hmdb, k400, finegym)
- `NUM_GPUS`: Number of GPUs to use (e.g., 1)
- `NUM_WORKERS`: Number of data loader workers (e.g., 16)
- `K_SHOT`: Number of support examples per class (e.g., 1)
- `N_WAY`: Number of classes per episode (e.g., 5)
- `POINT_INFO_NAME`: Point tracking method name
- `NUM_POINTS_TO_SAMPLE`: Number of trajectory points to sample
- `WANDB_ID`: Weights & Biases experiment ID
- `EXP_NAME`: Experiment name for wandb tracking
- `OUTPUT_DIR`: Output directory (typically derived as `$BASE_OUTPUT_DIR/$CONFIG_TO_USE/$EXP_NAME/$SECONDARY_EXP_NAME`)

## Development

This codebase is under active development. If you encounter any issues or have questions, please feel free to:
- Open an issue in this repository
- Contact Pulkit at pulkit[at]umd[dot]edu

## Acknowledgments

This codebase is built upon two excellent repositories:
- [TATs](https://github.com/pulkitkumar95/tats): Trajectory-aligned Space-time Tokens for Few-shot Action Recognition
- [MoLo](https://github.com/alibaba-mmai-research/MoLo): Motion-augmented Long-form Video Understanding
- [ORViT](https://github.com/eladb3/ORViT): Object-Regions for Video Instance Recognition and Tracking

We thank the authors for making their code publicly available.

## Citation

If you find this code and out paper useful for your research, please cite our papers:

```bibtex
@inproceedings{kumar2025trokens,
  title={Trokens: Semantic-Aware Relational Trajectory Tokens for Few-Shot Action Recognition},
  author={Kumar, Pulkit and Huang, Shuaiyi and Walmer, Matthew and Rambhatla, Sai Saketh and Shrivastava, Abhinav},
  booktitle={International Conference on Computer Vision},
  year={2025}
}

@inproceedings{kumar2024trajectory,
  title={Trajectory-aligned Space-time Tokens for Few-shot Action Recognition},
  author={Kumar, Pulkit and Padmanabhan, Namitha and Luo, Luke and Rambhatla, Sai Saketh and Shrivastava, Abhinav},
  booktitle={European Conference on Computer Vision},
  pages={474--493},
  year={2024},
  organization={Springer}
}