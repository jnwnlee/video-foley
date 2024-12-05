import cv2 # 3.4.18.65
from glob import glob
import os
import argparse
from multiprocessing import Pool
from functools import partial
import numpy as np
from tqdm import tqdm
import torch
import torchvision
from torchvision.utils import flow_to_image
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
import torchvision.transforms.functional as F

def load_model_Raft(device_id=0):
    assert torch.cuda.is_available(), 'CUDA is not available.'
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(f'cuda:{device_id}')
    
    return transforms, model

def preprocess_frames(prev_frames, curr_frames, transforms, height, width):
    prev_frames_b4_transform = F.resize(prev_frames, size=[height, width], antialias=False)
    curr_frames_b4_transform = F.resize(curr_frames, size=[height, width], antialias=False)
    prev_frames, curr_frames = transforms(prev_frames_b4_transform, curr_frames_b4_transform)
    return prev_frames, curr_frames, curr_frames_b4_transform

def compute_RAFT_optical_flow(prev_batch, curr_batch, model, device_id=0, bound=20):
    """Compute the RAFT optical flow."""
    # refers to RegNet, https://pytorch.org/vision/0.13/models/raft.html
    # and https://pytorch.org/vision/0.13/auto_examples/plot_optical_flow.html
    # and https://pytorch.org/vision/main/_modules/torchvision/utils.html#flow_to_image
    device = f"cuda:{device_id}"

    model = model.eval()
    
    list_of_flows = model(prev_batch.to(device), curr_batch.to(device))
    flows = list_of_flows[-1].detach().cpu()
    
    # rescale flow to [0, 255]
    if bound == 'max':
        max_norm = torch.sum(flows**2, dim=1).sqrt().max()
        epsilon = torch.finfo((flows).dtype).eps
        normalized_flow = flows.clone() / (max_norm + epsilon)
        rescaled_flows = (normalized_flow+1) * 255 // 2
    else:
        assert isinstance(bound, int), f'bound should be int or "max", but got {bound}'
        rescaled_flows = flows.clone()
        rescaled_flows[rescaled_flows>bound] = bound
        rescaled_flows[rescaled_flows<-bound] = -bound
        rescaled_flows += bound #[0, 2*bound]
        rescaled_flows *= (255/float(2*bound))
    return flows, rescaled_flows.to(dtype=torch.uint8)

def cal_for_frames(video_path, output_dir, n_frames, width, height, batch_size, device_id=0):
    transforms, model = load_model_Raft(device_id)
    save_dir = os.path.join(output_dir, os.path.basename(video_path).split('.')[0])
    os.makedirs(save_dir, exist_ok=True)
    frames, _, _ = torchvision.io.read_video(str(video_path), output_format="TCHW")
    
    # save 1st frame
    # frame_0_resized = F.resize(frames[0], size=[height, width], antialias=False)
    # torchvision.io.write_jpeg(frame_0_resized, os.path.join(save_dir, f"img_{0:05d}.jpg")) # .to('cpu')
    
    # compute optical flow
    for idx in range(0, len(frames)-1, batch_size):
        start_idx = idx
        end_idx = idx+batch_size if idx+batch_size < len(frames)-1 else len(frames)-1
        prev_frames = frames[start_idx : end_idx].clone()
        curr_frames = frames[start_idx+1 : end_idx+1].clone()
        assert len(prev_frames) == len(curr_frames), f'{len(prev_frames)} != {len(curr_frames)}, {start_idx}, {end_idx}'
        
        prev_frames, curr_frames, curr_frames_b4_transform = preprocess_frames(prev_frames, curr_frames, transforms, height, width)
        for i in range(len(curr_frames)): # save frames
            torchvision.io.write_jpeg(curr_frames_b4_transform[i], os.path.join(save_dir, f"img_{start_idx+1+i:05d}.jpg"))
        
        flows, rescaled_flows = compute_RAFT_optical_flow(prev_frames, curr_frames, model, device_id=0) # bound='max'
        assert len(flows) == len(curr_frames)
        
        for i in range(len(flows)): # save flows
            torchvision.io.write_jpeg(rescaled_flows[i, 0, :, :].unsqueeze(0).repeat(3,1,1), 
                                      os.path.join(save_dir, f"flow_x_{start_idx+1+i:05d}.jpg"))
            torchvision.io.write_jpeg(rescaled_flows[i, 1, :, :].unsqueeze(0).repeat(3,1,1), 
                                      os.path.join(save_dir, f"flow_y_{start_idx+1+i:05d}.jpg"))
            # torchvision.io.write_jpeg(flow_to_image(flows[i]), 
            #                           os.path.join(save_dir, f"flow_{start_idx+1+i:05d}.jpg"))

    if len(frames) < n_frames:
        print(f'less than {n_frames} frames:', video_path)
        return video_path
    else:
        return None
    
 
if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument("-i", "--input_dir", default="/home/junwon/video2foley/data/features/dog/videos_10s_21.5fps")
    paser.add_argument("-o", "--output_dir", default="/home/junwon/video2foley/data/features/dog/OF_10s_21.5fps")
    paser.add_argument("-f", "--video_fps", type=float, default=30)
    paser.add_argument("-l", "--length", type=int, default=10) # seconds
    paser.add_argument("-w", "--width", type=int, default=344) # 340
    paser.add_argument("-g", "--height", type=int, default=256)
    paser.add_argument("-n", '--num_worker', type=int, default=16)
    paser.add_argument("-d", '--device_id', type=int, default=0)
    paser.add_argument("-b", '--batch_size', type=int, default=64)

    args = paser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    fps = args.video_fps
    length = args.length
    width = args.width
    height = args.height
    device_id = args.device_id
    batch_size = args.batch_size
    n_frames = fps * length

    video_paths = glob(os.path.join(input_dir, "*.mp4"))
    video_paths.sort()
    print(f"{input_dir} -> {output_dir}")
    
    less_than_n_frames = []
    for idx in tqdm(range(len(video_paths))):
        video_path = cal_for_frames(video_paths[idx], output_dir, n_frames, width, height, batch_size, device_id)
        if video_path is not None:
            less_than_n_frames.append(video_path)
    
    with open(f"less_than_{n_frames}_{os.path.dirname(output_dir).split('/')[-1]}.txt", 'w') as f:
        for item in less_than_n_frames:
            f.write("%s\n" % item)
