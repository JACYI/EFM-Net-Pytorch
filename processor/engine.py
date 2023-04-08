# Copyright © Beijing University of Posts and Telecommunications,
# School of Artificial Intelligence.


import math
import time
import datetime
from typing import Iterable, Optional
import numpy as np
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma, AverageMeter
from util.utils import reduce_tensor, MetricLogger, SmoothedValue, get_detail_images, get_mask_images
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False, logger=None,
                    ):
    model.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    # counter
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    start = time.time()
    end = time.time()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                input_pred, attention_map = model(samples)
        else: # full precision
            input_pred, attention_map = model(samples)

        if attention_map != []:
            with torch.no_grad():
                detail_images = get_detail_images(samples, attention_map[:, :1, :, :], theta_detail=(0.4, 0.6), padding=0.1)

            # detail-images forward
            detail_pred, _ = model(detail_images)

            with torch.no_grad():
                mask_images = get_mask_images(samples, attention_map[:, 1:, :, :], theta_mask=(0.2, 0.5))

            mask_pred, _ = model(mask_images)

            output = (input_pred + detail_pred + mask_pred)/3.
            loss = criterion(input_pred, targets)/3. + \
                criterion(detail_pred, targets)/3. + \
                criterion(mask_pred, targets)/3.
        else:
            loss = criterion(input_pred, targets)
            output = input_pred

        loss_value = loss.item()

        if not math.isfinite(loss_value): # this could trigger if using AMP
            logger.info("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            if class_acc:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
            if use_amp:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})

        loss_meter.update(loss.item(), targets.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if data_iter_step % update_freq == 0:
            lr = max_lr
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - data_iter_step)
            logger.info(
                f'Train: [{epoch}][{data_iter_step}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    train_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    return train_stat

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False, logger=None, update_freq=1):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    # statistics
    record_truth = np.array([])
    record_pred = np.array([])
    record_feature = torch.tensor([])

    end = time.time()
    idx = 0
    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, 10, header):
            images = batch[0]
            target = batch[-1]
            label_true = target.numpy().squeeze()
            record_truth = np.concatenate((record_truth, label_true))

            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            if use_amp:
                with torch.cuda.amp.autocast():
                    input_pred, attention_map = model(images)
            else:
                input_pred, attention_map = model(images)

            if attention_map != []:
                detail_images = get_detail_images(images, attention_map, theta_detail=0.1, padding=0.05)
                detail_pred, _ = model(detail_images)
                output = (input_pred + detail_pred)/2.
                loss = criterion(output, target)
            else:
                loss = criterion(input_pred, target)
                output = input_pred

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            # record
            _, pred = torch.max(output, dim=1)
            pred = pred.cpu().numpy().squeeze()
            record_pred = np.concatenate((record_pred, pred))

            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)
            loss = reduce_tensor(loss)

            loss_meter.update(loss.item(), target.size(0))
            acc1_meter.update(acc1.item(), target.size(0))
            acc5_meter.update(acc5.item(), target.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % update_freq == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                if logger:
                    logger.info(
                        f'Test: [{idx}/{len(data_loader)}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                        f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                        f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                        f'Mem {memory_used:.0f}MB')
            idx += 1

            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # metric
    acc = accuracy_score(record_truth, record_pred)
    precision = precision_score(record_truth, record_pred, average='weighted')
    recall = recall_score(record_truth, record_pred, average='weighted')
    f1 = f1_score(record_truth, record_pred, average='weighted')
    logger.info(f'[Info] acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
