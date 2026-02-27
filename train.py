import argparse
import os
import time
from collections import defaultdict

import torch
import numpy as np
from contextlib import redirect_stdout
from config import _C as config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

from recorder import Recorder
from data_utils import RMS
from logger import Logger
from criterion import get_loss_values
from model import Video2Sound
from util import prepare_dataloaders, get_criterions, set_seed


def test_model(model, criterions, test_loader, epoch, discretized=False, visualization=False):
    model.eval()
    reduced_losses = defaultdict(list)
    if discretized:
        mu_bins = RMS.get_mu_bins(config.data.rms_mu, config.data.rms_num_bins, config.data.rms_min)
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            model.parse_batch(batch)
            model.forward()
            if visualization:
                for j in range(len(model.pred_rms)):
                    plt.figure(figsize=(10, 8))
                    plt.subplot(311)
                    plt.plot(model.gt_rms[j].detach().cpu().numpy()) 
                    plt.title(model.video_class[j]+'_'+model.video_name[j]+"_ground_truth_rms")
                    plt.subplot(312)
                    pred = model.pred_rms[j].detach().cpu().numpy()
                    if discretized:
                        pred = pred.argmax(axis=0) # (C, rms_len) -> (rms_len)
                        plt.plot(pred)
                    else:
                        plt.plot(pred)
                    plt.title(model.video_class[j]+'_'+model.video_name[j]+"_predicted_rms")
                    plt.subplot(313)
                    if discretized:
                        plt.plot(RMS.undiscretize_rms(torch.tensor(pred), mu_bins, ignore_min=True).numpy())
                    else:
                        plt.plot(RMS.zero_phased_filter(pred))
                    plt.title(model.video_class[j]+'_'+model.video_name[j]+"_predicted_rms_zero-phased-filter")
                    plt.tight_layout()
                    viz_dir = os.path.join(config.log.save_dir, "viz", f'epoch_{epoch:05d}')
                    os.makedirs(viz_dir, exist_ok=True)
                    plt.savefig(os.path.join(viz_dir, model.video_class[j]+'_'+model.video_name[j]+".jpg"))
                    plt.close()
            
            _gt = (model.gt_rms, model.gt_rms_continuous) if discretized else model.gt_rms
            kwargs = defaultdict(lambda: None)
            if model.onset_supervision:
                kwargs = {'model_output_onset': model.pred_onset, 'targets_onset': model.gt_onset}
            for loss_type, losses in get_loss_values(model.pred_rms, _gt, criterions, **kwargs).items():
                batch_size = model.pred_rms.shape[0]
                reduced_losses[loss_type].append(losses * batch_size)
        print("Test loss epoch:{} loss {:.6f} ".format(epoch, np.sum(reduced_losses[config.train.loss.type])/len(test_loader.dataset)))
    model.train()
    
    return {loss_type: np.sum(losses)/len(test_loader.dataset)
            for loss_type, losses in reduced_losses.items()}


def train():
    set_seed(config.train.seed)

    model = Video2Sound(config)

    criterions = get_criterions(config, config.log.loss.types)

    logger = Logger(config.log)

    train_loader, test_loader = prepare_dataloaders(config=config.data, batch_size=config.train.batch_size, 
                                                    train=True, test=True)
    print('Dataset prepared.')

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0

    if config.train.checkpoint_path != '':
        model.load_checkpoint(config.train.checkpoint_path)
        iteration = model.epoch * len(train_loader)
        iteration += 1  # next iteration is iteration + 1
        epoch_offset = max(0, int(iteration / len(train_loader)))
    config.train.epoch_count = epoch_offset
    model.setup()

    model.train()
    prev_val_loss = np.inf
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in tqdm(range(epoch_offset, config.train.epochs)):
        print("Epoch: {}".format(epoch))
        reduced_losses = defaultdict(list)
        start = time.perf_counter()
        for i, batch in enumerate(train_loader):
            model.zero_grad()
            model.parse_batch(batch)
            model.optimize_parameters()
            learning_rate = model.optimizers[0].param_groups[0]['lr']
            
            if config.data.rms_discretize:
                _gt = (model.gt_rms, model.gt_rms_continuous)
            else:
                _gt = model.gt_rms
            kwargs = defaultdict(lambda: None)
            if model.onset_supervision:
                kwargs = {'model_output_onset': model.pred_onset, 'targets_onset': model.gt_onset}
            for loss_type, losses in get_loss_values(model.pred_rms, _gt, criterions, **kwargs).items():
                reduced_losses[loss_type].append(losses)

            iteration += 1
        
        duration = time.perf_counter() - start
        print("epoch:{} loss:{:.6f}".format(epoch, np.mean(reduced_losses[config.train.loss.type])))
        
        if epoch % config.train.num_epoch_save != 0:
            test_losses = test_model(model, criterions, test_loader, epoch, discretized=config.data.rms_discretize)
        if epoch % config.train.num_epoch_save == 0:
            print(f"evaluation and save model")
            test_losses = test_model(model, criterions, test_loader, epoch, discretized=config.data.rms_discretize, 
                                   visualization=True)
            if prev_val_loss > test_losses[config.train.loss.type]:
                prev_val_loss = test_losses[config.train.loss.type]
            logger.save_checkpoint(model, epoch)
        logger.log_training({loss_type: np.mean(losses) for loss_type, losses in reduced_losses.items()},
                            test_losses, 
                            learning_rate, duration, epoch)

        model.update_learning_rate(test_losses[config.train.loss.type])
    
    if 'wandb' in config.log.logger:
        wandb.save(os.path.join(config.log.save_dir, "checkpoint_*"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str, default='',
                        help='file for configuration')
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        config.merge_from_file(args.config_file)
 
    config.merge_from_list(args.opts)

    os.makedirs(config.log.save_dir, exist_ok=True)
    with open(os.path.join(config.log.save_dir, 'opts.yml'), 'w') as f:
        with redirect_stdout(f): 
            print(config.dump())
    f.close()

    recorder = Recorder(config.log.save_dir, config.log.exclude_dirs)

    torch.backends.cudnn.enabled = config.train.cudnn_enabled
    torch.backends.cudnn.benchmark = config.train.cudnn_benchmark
    print("cuDNN Enabled:", config.train.cudnn_enabled)
    print("cuDNN Benchmark:", config.train.cudnn_benchmark)

    train()
