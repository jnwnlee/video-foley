import argparse
import os
from collections import defaultdict
from time import time
from typing import Dict

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from yacs.config import CfgNode as CN

from config import _C as config
from criterion import get_loss_values
from util import load_config, load_model, get_criterions, prepare_dataloaders, set_seed
    

@torch.no_grad
def evaluate(epoch:int = 500, ckpt_dir:str = '', config:CN=None, drop_correct_0:bool=True, average:str=None)\
    -> Dict[str, float]:
    if ckpt_dir == '':
        raise ValueError("ckpt_dir is empty")
    if config is None:
        raise ValueError("config is None")
    
    print(f"Evaluating epoch {epoch}...")
    
    print(f'Setting seed: {config.train.seed}')
    set_seed(config.train.seed)

    # load ckpt and prepare model
    print('Loading model...')
    model = load_model(epoch, ckpt_dir, config)

    # prepare loss functions
    print('Preparing loss functions...')
    criterions = get_criterions(config, config.log.loss.types, average=average)

    # prepare dataloader
    print('Preparing dataset...')
    test_loader = prepare_dataloaders(config.data, batch_size=1, train=False, test=True)

    # evaluate
    print('Evaluating...')
    model.eval()
    reduced_losses = defaultdict(list) # {} # None
    
    reduced_losses['params'].append(sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader)): # batch size 1
            model.parse_batch(batch)
            
            # make inference, and measure inference time
            start_time = time()
            model.forward()
            infer_time = time() - start_time
            reduced_losses['infer_time'].append(infer_time)
            
            if drop_correct_0:
                # if pred_rms class and gt_rms class are both 0, skip that index
                # get indices where pred_rms class 0
                pred_0_indices = torch.where(model.pred_rms.argmax(dim=1) == 0)[1]
                gt_0_indices = torch.where(model.gt_rms == 0)[1]
                # find intersection
                drop_indices = pred_0_indices[torch.where(torch.isin(pred_0_indices, gt_0_indices))]
                drop_mask = torch.ones(model.pred_rms.shape[2], dtype=torch.bool)
                drop_mask[drop_indices] = False
                if len(drop_indices) > 0:
                    # drop indices (axis 2 of model.pred_rms, axis 1 of model.gt_rms)
                    model.pred_rms = model.pred_rms[:, :, drop_mask]
                    model.gt_rms = model.gt_rms[:, drop_mask]
                    model.gt_rms_continuous = model.gt_rms_continuous[:, drop_mask]
            assert model.pred_rms.shape[2] == model.gt_rms.shape[1] == model.gt_rms_continuous.shape[1]
            if model.pred_rms.shape[2] == 0:
                continue
                
            _reduced_losses = get_loss_values(model.pred_rms, (model.gt_rms, model.gt_rms_continuous), criterions,
                                              average=average if average != 'macro' else None)
            if idx == 0:
                print(_reduced_losses)
            nan_indices = []
            for loss_type, loss in _reduced_losses.items():
                if loss_type in ['CE', 'CE_GLS', 'MSE', 'MAE', 'PRAUC', 'ROCAUC']:
                    reduced_losses[loss_type].append(loss)
                else:
                    if average in ['macro', None]:
                        if loss_type == 'ACC':
                            nan_indices = [i for i, l in enumerate(loss) if np.isnan(l)]
                        reduced_losses[loss_type].append(np.mean([l for idx, l in enumerate(loss) if idx not in nan_indices])) # macro
                        if average is None:
                            for i in range(len(loss)):
                                if i not in nan_indices:
                                    reduced_losses[f'{loss_type}_{i}'].append(loss[i])
                    elif average == 'micro':
                        reduced_losses[loss_type].append(loss)
                    
        
    reduced_losses = {loss_type: np.mean(losses) for loss_type, losses in reduced_losses.items()}
    
    return reduced_losses
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt_dir', type=str, default='', required=True,
                        help='directory for model checkpoint')
    parser.add_argument('-e', '--epoch', type=int, default=500, required=True,
                        help='number of epochs to evaluate')
    parser.add_argument('-o', '--output_file', type=str, default='./evaluate.csv', required=False,
                        help='path to csv file to save scores')
    parser.add_argument('-a', '--average', default='micro', required=False, type=str, 
                        choices=['micro', 'macro', None],
                        help='average type for ACC, PREC, RECALL, F1')
    parser.add_argument('-0', '--preserve_correct_0', action='store_true', default=False, required=False,
                        help='preserve correct predictions of class 0')
    args = parser.parse_args()
    
    config = load_config(os.path.join(args.ckpt_dir, 'opts.yml'))
    config.freeze()

    scores_dict = evaluate(epoch=args.epoch, ckpt_dir=args.ckpt_dir, config=config,
                           drop_correct_0=(not args.preserve_correct_0), average=args.average)
    scores_dict = {key: float(f'{value:.5f}') for key, value in scores_dict.items()}
    
    ### append and save score values to existing csv file
    # get values
    save_dir = config.log.save_dir
    loss_type = config.train.loss.type
    rms_num_bins = config.data.rms_num_bins
    gls_blur_range = config.log.loss.gls_blur_range if 'CE_GLS' in config.log.loss.types else None
    epoch = args.epoch
    
    # create dataframe
    df = pd.DataFrame({
        'save_dir': [save_dir],
        'loss_type': [loss_type],
        'rms_num_bins': [rms_num_bins],
        'gls_blur_range': [gls_blur_range],
        'epoch': [epoch],
        **{key: [value] for key, value in scores_dict.items()}
    })
    
    # if file does not exist write header
    if not os.path.isfile(args.output_file):
        df.to_csv(args.output_file, index=False)
    else:
        # read existing file
        existing_df = pd.read_csv(args.output_file)
        # concatenate new data with existing data
        df = pd.concat([existing_df, df], ignore_index=True)
        # save the concatenated dataframe
        df.to_csv(args.output_file, index=False)
    
    print(scores_dict)
    print(f"Scores saved to {args.output_file}")
    