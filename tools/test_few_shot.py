#!/usr/bin/env python3

"""Test a few shot classification model."""
# pylint: disable=wrong-import-position,import-error,wrong-import-order
import os
import sys
import pprint
import torch
import torch.nn.functional as F
import wandb
import numpy as np
import pandas as pd
from einops import rearrange
import trokens.utils.checkpoint as cu
import trokens.utils.distributed as du
import trokens.utils.logging as logging
import trokens.utils.metrics as metrics
import trokens.utils.misc as misc
from trokens.datasets import loader
from trokens.utils.meters import ValMeter
from trokens.models import build_model
from fvcore.common.config import CfgNode
from fvcore.nn.precise_bn import update_bn_stats

def wandb_init_dict(cfg_node):
    """Convert a config node to dictionary.
    """
    if not isinstance(cfg_node, CfgNode):
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = wandb_init_dict(v)
        return cfg_dict
# pylint: disable=line-too-long
def process_patch_tokens(cfg, support_tokens, query_tokens):
    """
    Process the patch tokens for few shot learning.
    Ref: https://github.com/alibaba-mmai-research/MoLo/blob/f7f73b6dd8cba446b414b1c47652ab26033bc88e/models/base/few_shot.py#L2552
    args:
        cfg: config
        support_tokens: (num_support, temp_len, num_patches, embed_dim)
        query_tokens: (num_query, temp_len, num_patches, embed_dim)
    """
    #Putting an activation here, may be not needed
    support_tokens = F.relu(support_tokens)
    query_tokens = F.relu(query_tokens)
    num_supports = support_tokens.shape[0]
    num_querries = query_tokens.shape[0]
    if not cfg.MODEL.USE_EXTRA_ENCODER:
        if cfg.FEW_SHOT.PATCH_TOKENS_AGG == 'temporal':
            support_tokens = support_tokens.mean(dim=1)
            query_tokens = query_tokens.mean(dim=1)
        elif cfg.FEW_SHOT.PATCH_TOKENS_AGG == 'spatial':
            support_tokens = support_tokens.mean(dim=2)
            query_tokens = query_tokens.mean(dim=2)
        elif cfg.FEW_SHOT.PATCH_TOKENS_AGG == 'no_agg':
            support_tokens = rearrange(support_tokens, 'b t p e -> b (t p) e')
            query_tokens = rearrange(query_tokens, 'b t p e -> b (t p) e')
        else:
            raise NotImplementedError(
                f"Aggregation method {cfg.FEW_SHOT.PATCH_TOKENS_AGG} not implemented")

    support_tokens = rearrange(support_tokens, 'b p e -> (b p) e')
    query_tokens = rearrange(query_tokens, 'b p e -> (b p) e')
    sim_matrix = cos_sim(query_tokens, support_tokens)
    dist_matrix = 1 - sim_matrix

    dist_rearranged = rearrange(dist_matrix, '(q qt) (s st) -> q s qt st',
                                q=num_querries, s=num_supports)
    # Take the minimum distance for each query token
    dist_logits = dist_rearranged.min(3)[0].sum(2) + dist_rearranged.min(2)[0].sum(2)
    if cfg.FEW_SHOT.DIST_NORM == 'max_div':
        max_dist = dist_logits.max(dim=1, keepdim=True)[0]
        dist_logits = dist_logits / max_dist
    elif cfg.FEW_SHOT.DIST_NORM == 'max_sub':
        max_dist = dist_logits.max(dim=1, keepdim=True)[0]
        dist_logits = max_dist - dist_logits
    return - dist_logits

def cos_sim(x, y, epsilon=0.01):
    """
    Calculates the cosine similarity between the last dimension of two tensors.
    """
    numerator = torch.matmul(x, y.transpose(-1,-2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1,-2)) + epsilon
    dists = torch.div(numerator, denominator)
    return dists


def support_query_split(preds, labels, metadata):
    """
    Split the preds and labels into support and query.
    """

    device = preds.device
    sample_info = np.array(metadata['sample_type'])
    batch_labels = metadata['batch_label']
    support_condition = sample_info=='support'
    support_labels = labels[support_condition]
    support_preds = preds[support_condition]
    support_batch_labels = batch_labels[support_condition]

    # average the support preds for each class
    support_to_take = []
    support_main_label_to_take = []
    support_batch_label_to_take = []
    for label in np.unique(support_batch_labels.cpu().numpy()):
        label_condition = support_batch_labels==label
        label_mean_support = support_preds[label_condition].mean(dim=0,
                                                                keepdims=True)
        support_main_label = support_labels[label_condition][0]
        support_main_label_to_take.append(support_main_label)
        support_batch_label_to_take.append(label)
        support_to_take.append(label_mean_support)
    support_labels = torch.tensor(support_main_label_to_take, device=device)
    support_batch_labels = torch.tensor(support_batch_label_to_take, device=device)
    support_preds = torch.cat(support_to_take, dim=0)


    query_labels = labels[~support_condition]
    query_preds = preds[~support_condition]
    query_batch_labels = batch_labels[~support_condition]
    return_dict = {
        'query_labels':query_labels,
        'query_batch_labels':query_batch_labels,
        'support_labels':support_labels,
        'support_batch_labels':support_batch_labels,
        'support_preds':support_preds,
        'query_preds':query_preds
    }
    return return_dict

logger = logging.get_logger(__name__)

def conv_fp16(var):
    """Convert to float16.
    """
    return np.float16(np.around(var, 4))






@torch.no_grad()
def test_epoch(val_loader, model, val_meter, cur_epoch, cfg):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            trokens/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    epoch_top_1_acc_few_shot = []
    epoch_q2s_loss = []
    num_test_classes = len(val_loader.batch_sampler.class_ids)
    if cfg.TRAIN.DATASET == 'FINEGYM':
        num_test_classes = 100
    confusion_matrix = np.zeros((num_test_classes, num_test_classes))
    all_df = []

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        if cur_iter > len(val_loader):
            break
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            inputs, labels, meta = misc.iter_to_cuda([inputs, labels, meta])

        val_meter.data_toc()
        input_dict = {'video':inputs, 'metadata':meta}
        # for few shot, patch tokens are also returning
        preds, patch_tokens = model(input_dict)

        patch_support_query_dict = support_query_split(patch_tokens, labels, meta)
        patch_q2s_logits = process_patch_tokens(
                                    cfg,
                                    patch_support_query_dict['support_preds'],
                                    patch_support_query_dict['query_preds'])
        q2s_labels = patch_support_query_dict['query_batch_labels']
        q2s_loss = F.cross_entropy(patch_q2s_logits, q2s_labels)

        # Explicitly declare reduction to mean.
        few_shotk_correct = metrics.topks_correct(patch_q2s_logits,
                                                    q2s_labels, (1, 5))
        few_shot_top1_acc, _ = [
            (x / patch_q2s_logits.size(0)) * 100.0 for x in few_shotk_correct
        ]
        cfg['wandb'].log({
            'iteration': cur_iter,
            'iter_top_1_acc': few_shot_top1_acc.item(),
        })

        if cfg.NUM_GPUS > 1:
            few_shot_top1_acc, q2s_loss = du.all_reduce([few_shot_top1_acc, q2s_loss])
            q2s_loss = du.all_reduce([q2s_loss])[0]

        # Copy the errors from GPU to CPU (sync point).
        few_shot_top1_acc = few_shot_top1_acc.item()
        q2s_loss = q2s_loss.item()
        epoch_q2s_loss.append(q2s_loss)
        epoch_top_1_acc_few_shot.append(few_shot_top1_acc)

        support_labels = patch_support_query_dict['support_labels']
        query_labels = patch_support_query_dict['query_labels']
        # pylint: disable=unbalanced-tuple-unpacking
        if cfg.NUM_GPUS > 1:
            patch_q2s_logits, support_labels, query_labels = du.all_gather(
                [patch_q2s_logits, support_labels, query_labels]
            )
        patch_q2s_logits = patch_q2s_logits.cpu().numpy()
        support_labels = support_labels.cpu().numpy()
        query_labels = query_labels.cpu().numpy()
        pred_query_batch_labels = patch_q2s_logits.argmax(axis=1)
        pred_query_labels = support_labels[pred_query_batch_labels]
        confusion_matrix[query_labels, pred_query_labels] += 1
        batch_df = pd.DataFrame({'y_true':query_labels, 'y_preds':pred_query_labels})
        all_df.append(batch_df)

        val_meter.iter_toc()
        # Update and log stats.
        val_meter.update_stats(
            q2s_loss,
            few_shot_top1_acc,
            inputs[0].size(0)
            * max(
                cfg.NUM_GPUS, 1
            ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
        )


        val_meter.update_predictions(preds, labels)
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    log_dict = {
        'test_q2s_loss': np.mean(epoch_q2s_loss),
        'test_top1_acc_few_shot': np.mean(epoch_top_1_acc_few_shot),
        'epoch': cur_epoch}
    if cfg['wandb']:
        cfg['wandb'].log(log_dict)
    all_df = pd.concat(all_df)
    all_df.to_csv(os.path.join(cfg.OUTPUT_DIR,cfg['csv_dump_name']))

    val_meter.reset()

# pylint: disable=redefined-outer-name
def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i, _ in enumerate(inputs):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)



def test_few_shot(cfg, args, wandb_run=None):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            trokens/config/defaults.py
    """
    # Set up environment.
    if not args.new_dist_init:
        du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed_all(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = True

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    if wandb_run is not None:
        wandb_instance = wandb_run
        wandb_instance.define_metric("test*", step_metric="epoch")
        wandb_instance.define_metric("test_top1_acc_few_shot", summary="max")
    else:
        if du.get_rank() == 0:
            wandb_config_dict = wandb_init_dict(cfg)
            wandb_instance = wandb.init(project=cfg.WANDB.PROJECT,config=wandb_config_dict,
                                        entity=cfg.WANDB.ENTITY)
            wandb_instance.define_metric("epoch")
            wandb_instance.define_metric("iteration")

            wandb_instance.define_metric("iter*", step_metric="iteration")

            wandb_instance.define_metric("train*", step_metric="epoch")
            wandb_instance.define_metric("val*", step_metric="epoch")
            wandb_instance.define_metric("test*", step_metric="epoch")

            wandb_instance.define_metric("train_loss", summary="min")
            wandb_instance.define_metric("val_loss", summary="min")
            wandb_instance.define_metric("test_loss", summary="min")
            wandb_instance.define_metric("val_top5_acc", summary="max")
            wandb_instance.define_metric("val_top1_acc", summary="max")
            wandb_instance.define_metric("test_top1_acc_few_shot", summary="max")
        else:
            wandb_instance = None
    cfg['wandb'] = wandb_instance
    cfg['csv_dump_name'] = 'confusion_matrix.csv'

    # Init multigrid.
    logger.info("Test with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    cur_epoch = cu.load_test_checkpoint(cfg, model)
    val_loader = loader.construct_loader(cfg, "test") # MOLO uses test set for validation
    val_meter = ValMeter(len(val_loader), cfg)

    test_epoch(val_loader, model, val_meter, cur_epoch, cfg)
    # Close wandb logging
    if wandb_instance is not None:
        wandb_instance.finish()

    # Exit
    sys.exit()
