#!/usr/bin/env python3

"""Train a few shot video classification model."""
import os
import pprint
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from fvcore.common.config import CfgNode
import wandb
import trokens.models.losses as losses
import trokens.models.optimizer as optim
import trokens.utils.checkpoint as cu
import trokens.utils.distributed as du
import trokens.utils.logging as logging
import trokens.utils.metrics as metrics
import trokens.utils.misc as misc
from trokens.datasets import loader
from trokens.datasets.mixup import MixUp
from trokens.models import build_model
from trokens.utils.meters import EpochTimer, TrainMeter, ValMeter
from trokens.utils.multigrid import MultigridSchedule

warnings.filterwarnings('ignore')


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

logger = logging.get_logger(__name__)

def count_parameters(model):
    """Count the number of parameters in the model.

    Args:
        model (torch.nn.Module): The model

    """
    count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'dino' in name or 'resnet' in name:
                print(name)
                continue
            count += param.numel()
    return count


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
    """Calculates the cosine similarity between the last dimension of two tensors.
    """
    numerator = torch.matmul(x, y.transpose(-1,-2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1,-2)) + epsilon
    dists = torch.div(numerator, denominator)
    return dists


def support_query_split(preds, labels, metadata):
    """Split the preds and labels into support and query
    Args:
        preds (torch.Tensor): The predictions
        labels (torch.Tensor): The labels
        metadata (dict): The metadata
    Returns:
        dict: The dictionary containing the support and query labels and preds
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

def conv_fp16(inputs):
    """Convert to float 16
    """
    return np.float16(np.around(inputs, 4))



def train_epoch(
    train_loader,
    model,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
    wandb_run=None
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            trokens/config/defaults.py
        wandb_run (wandb.run): wandb run object
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    epoch_top_1_err = []
    epoch_top_5_err = []
    epoch_top_1_acc_few_shot = []
    epoch_cls_loss = []
    epoch_q2s_loss = []

    if cfg.MIXUP.ENABLE:
        mixup_fn = MixUp(
            mixup_alpha=cfg.MIXUP.ALPHA,
            cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
            mix_prob=cfg.MIXUP.PROB,
            switch_prob=cfg.MIXUP.SWITCH_PROB,
            label_smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE,
            num_classes=misc.get_num_classes(cfg)
        )
    lr = optim.get_epoch_lr(cur_epoch, cfg)
    optim.set_lr(optimizer, lr, log=True)
    for cur_iter, (inputs, labels, _vid_idx, meta) in enumerate(train_loader):
        if cur_iter > len(train_loader):
            break
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            inputs, labels, meta = misc.iter_to_cuda([inputs, labels, meta])
        # Update the learning rate.


        train_meter.data_toc()
        if cfg.MIXUP.ENABLE:
            samples, labels = mixup_fn(inputs[0], labels)
            inputs[0] = samples


        with torch.amp.autocast('cuda', enabled=cfg.TRAIN.MIXED_PRECISION):
            input_dict = {'video':inputs, 'metadata':meta}

            preds, patch_tokens = model(input_dict)

            preds = preds / cfg.SOLVER.TEMPRATURE

            if isinstance(preds, tuple):
                preds, _ = preds

            # Explicitly declare reduction to mean.
            loss_fun = losses.get_loss_func(cfg)(
                reduction="mean"
            )

            classfication_loss = loss_fun(preds, labels)
            loss_dict = {'classfication_loss':classfication_loss}
            patch_support_query_dict = support_query_split(patch_tokens, labels, meta)
            patch_q2s_logits = process_patch_tokens(
                                        cfg,
                                        patch_support_query_dict['support_preds'],
                                        patch_support_query_dict['query_preds'])
            q2s_labels = patch_support_query_dict['query_batch_labels']
            patch_q2s_logits = patch_q2s_logits / cfg.SOLVER.TEMPRATURE
            q2s_loss = F.cross_entropy(patch_q2s_logits, q2s_labels)
            loss_dict['q2s_loss'] = q2s_loss
        loss = (cfg.FEW_SHOT.CLASS_LOSS_LAMBDA * classfication_loss +
                cfg.FEW_SHOT.Q2S_LOSS_LAMBDA * q2s_loss)

        misc.check_nan_losses(loss)
        # Perform the backward pass.
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        # Clip gradients if necessary
        if cfg.SOLVER.CLIP_GRAD_VAL:
            torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
            )
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )

        # Update the parameters.
        scaler.step(optimizer)
        scaler.update()

        top1_err, top5_err = None, None
        top1_err, top5_err = None, None


        if cfg.DATA.MULTI_LABEL:
            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                [loss] = du.all_reduce([loss])
            loss = loss.item()

        else:
            # Compute the errors.
            num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
            top1_err, top5_err = [
                (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
            ]
            # for few shot
            few_shotk_correct = metrics.topks_correct(patch_q2s_logits, q2s_labels, (1, 5))
            few_shot_top1_acc, _ = [
                (x / patch_q2s_logits.size(0)) * 100.0 for x in few_shotk_correct
            ]

            # Gather all the predictions across all the devices.
            classification_loss = loss_dict['classfication_loss']
            q2s_loss = loss_dict['q2s_loss']
            if cfg.NUM_GPUS > 1:
                classification_loss, top1_err, top5_err, few_shot_top1_acc = du.all_reduce(
                    [classification_loss, top1_err, top5_err, few_shot_top1_acc]
                )
                q2s_loss = du.all_reduce([q2s_loss])[0]

            # Copy the stats from GPU to CPU (sync point).
            classification_loss, top1_err, top5_err = (
                classification_loss.item(),
                top1_err.item(),
                top5_err.item(),
            )
            q2s_loss = q2s_loss.item()

            few_shot_top1_acc = few_shot_top1_acc.item()

            epoch_cls_loss.append(classification_loss)
            epoch_q2s_loss.append(q2s_loss)

            epoch_top_1_err.append(top1_err)
            epoch_top_5_err.append(top5_err)
            epoch_top_1_acc_few_shot.append(few_shot_top1_acc)
            global_iter = data_size * cur_epoch + cur_iter
            wandb_iter_dict = {'iter_cls_loss':classification_loss,
                                'iter_q2s_loss':q2s_loss,
                                'iter_top1_err':top1_err,
                                'iter_top5_err':top5_err,
                            'iteration':global_iter,
                            'iter_top1_acc_few_shot':few_shot_top1_acc}
            if wandb_run:
                wandb_run.log(wandb_iter_dict)

        # Update and log stats.
        train_meter.update_stats(
            top1_err,
            top5_err,
            loss_dict,
            lr,
            inputs[0].size(0)
            * max(
                cfg.NUM_GPUS, 1
            ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
        )
        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()

    wandb_iter_dict = {'train_cls_loss':np.mean(epoch_cls_loss),
                    'train_q2s_loss':np.mean(epoch_q2s_loss),
                    'train_top1_err':np.mean(epoch_top_1_err),
                    'train_top5_err':np.mean(epoch_top_5_err),
                    'train_top5_acc':100-np.mean(epoch_top_5_err),
                    'train_top1_acc':100-np.mean(epoch_top_1_err),
                    'train_top1_acc_few_shot':np.mean(epoch_top_1_acc_few_shot),
                    'epoch':cur_epoch}

    if wandb_run:
        wandb_run.log(wandb_iter_dict)


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, wandb_run=None):
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
        wandb_run (wandb.run): wandb run object
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    epoch_top_1_acc_few_shot = []
    epoch_q2s_loss = []

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        if cur_iter > len(val_loader):
            break
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            inputs, labels, meta = misc.iter_to_cuda([inputs, labels, meta])

        val_meter.data_toc()
        input_dict = {'video':inputs, 'metadata':meta}
        preds, patch_tokens = model(input_dict)

        patch_support_query_dict = support_query_split(patch_tokens, labels, meta)
        patch_q2s_logits = process_patch_tokens(
                                    cfg,
                                    patch_support_query_dict['support_preds'],
                                    patch_support_query_dict['query_preds'])
        q2s_labels = patch_support_query_dict['query_batch_labels']
        q2s_loss = F.cross_entropy(patch_q2s_logits, q2s_labels)

        if isinstance(preds, tuple):
            preds, _ = preds


        few_shotk_correct = metrics.topks_correct(patch_q2s_logits,
                                                    q2s_labels, (1, 5))
        few_shot_top1_acc, _ = [
            (x / patch_q2s_logits.size(0)) * 100.0 for x in few_shotk_correct
        ]


        if cfg.NUM_GPUS > 1:
            few_shot_top1_acc, q2s_loss = du.all_reduce([few_shot_top1_acc, q2s_loss])

            q2s_loss = du.all_reduce([q2s_loss])[0]

        # Copy the errors from GPU to CPU (sync point).
        few_shot_top1_acc = few_shot_top1_acc.item()
        q2s_loss = q2s_loss.item()
        epoch_q2s_loss.append(q2s_loss)
        epoch_top_1_acc_few_shot.append(few_shot_top1_acc)

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
        # write to tensorboard format if available.


        val_meter.update_predictions(preds, labels)
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)

    log_dict = {
        'val_q2s_loss': np.mean(epoch_q2s_loss),
        'val_top1_acc_few_shot': np.mean(epoch_top_1_acc_few_shot),
        'epoch': cur_epoch}
    if wandb_run:
        wandb_run.log(log_dict)
    epoch_mean_acc = np.mean(epoch_top_1_acc_few_shot)
    val_meter.reset()
    return epoch_mean_acc



def train_few_shot(cfg, args, wandb_run=None):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            trokens/config/defaults.py
        args (argparse.Namespace): arguments
        wandb_run (wandb.run): wandb run object
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

    if du.get_rank() == 0:
        wandb_config_dict = wandb_init_dict(cfg)
        wandb_config_dict['slurm_id'] = os.environ.get('SLURM_JOB_ID')
        wandb_run = wandb.init(project=cfg.WANDB.PROJECT,config=wandb_config_dict,
                                    entity=cfg.WANDB.ENTITY, name=cfg.WANDB.EXP_NAME)
        wandb_run.define_metric("epoch")
        wandb_run.define_metric("iteration")

        wandb_run.define_metric("iter*", step_metric="iteration")

        wandb_run.define_metric("train*", step_metric="epoch")
        wandb_run.define_metric("val*", step_metric="epoch")
        wandb_run.define_metric("train_loss", summary="min")
        wandb_run.define_metric("val_loss", summary="min")
        wandb_run.define_metric("val_top5_acc", summary="max")
        wandb_run.define_metric("val_top1_acc", summary="max")
    else:
        wandb_run = None


    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if cfg.NUM_GPUS>1:
        cfg['num_patches'] = model.module.num_patches
    else:
        cfg['num_patches'] = model.num_patches

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    # Create a GradScaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.TRAIN.MIXED_PRECISION)

    # Load a checkpoint to resume training if applicable.
    # start_epoch = cu.load_train_checkpoint(
    #     cfg, model, optimizer, scaler if cfg.TRAIN.MIXED_PRECISION else None
    # )

    start_epoch = 0
    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    # MOLO uses test set for validation
    val_loader = loader.construct_loader(cfg, "test", less_iters=True)
    if du.is_master_proc():
        model_info = misc.log_model_info(model, cfg, train_loader)
        # log in wandb as a summary
        wandb_run.summary.update(model_info)

    # Create meters.


    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)



    # Perform the training loop.
    logger.info("Start epoch: %s", start_epoch + 1)

    epoch_timer = EpochTimer()
    best_val_acc = 0
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):

        # Train for one epoch.
        epoch_timer.epoch_tic()
        if not cfg.TRAIN.VAL_ONLY:
            train_epoch(
                train_loader,
                model,
                optimizer,
                scaler,
                train_meter,
                cur_epoch,
                cfg,
                wandb_run
            )
        epoch_timer.epoch_toc()
        logger.info(
            "Epoch %s takes %.2fs. Epochs "
            "from %s to %s take "
            "%.2fs in average and "
            "%.2fs in median.",
            cur_epoch, epoch_timer.last_epoch_time(),
            start_epoch, cur_epoch,
            epoch_timer.avg_epoch_time(),
            epoch_timer.median_epoch_time()
        )
        logger.info(
            "For epoch %s, each iteraction takes "
            "%.2fs in average. "
            "From epoch %s to %s, each iteraction takes "
            "%.2fs in average.",
            cur_epoch, epoch_timer.last_epoch_time()/len(train_loader),
            start_epoch, cur_epoch,
            epoch_timer.avg_epoch_time()/len(train_loader)
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None if multigrid is None else multigrid.schedule,
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        )


        # Save a checkpoint.
        cfg_to_save = cfg.clone()
        if is_checkp_epoch:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                cur_epoch,
                cfg_to_save,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
        # Evaluate the model on validation set.1
        if is_eval_epoch:
            val_acc = eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, wandb_run)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                cu.save_checkpoint(
                    cfg.OUTPUT_DIR,
                    model,
                    optimizer,
                    cur_epoch,
                    cfg_to_save,
                    scaler if cfg.TRAIN.MIXED_PRECISION else None,
                    best=True
                )
        if cfg.TRAIN.VAL_ONLY:
            break


    return wandb_run
