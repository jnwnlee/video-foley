import argparse
import time
import os
import numpy as np
import pickle as pkl
import torch
from PIL import Image
from torch.utils.data import Dataset
import torch.optim
import torchvision
from glob import glob
from tqdm import tqdm

from tsn.models import TSN


class GroupScale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, torchvision.transforms.InterpolationMode.BILINEAR)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0]//len(self.mean))
        rep_std = self.std * (tensor.size()[0]//len(self.std))

        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor

class Stack(object):
    
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)

class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class TSNDataSet(Dataset):
    def __init__(self, root_path, list_file, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None):

        self.root_path = root_path
        self.list_file = list_file
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        with open(list_file) as f:
            self.video_list = [line.strip() for line in f]
        f.close()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')
            return [x_img, y_img]

    def __getitem__(self, index):
        video_path = os.path.join(self.root_path, self.video_list[index])
        images = list()
        if self.modality == 'RGB':          
            num_frames = len(glob(os.path.join(video_path, "img*.jpg")))
        elif self.modality == 'Flow':
            num_frames = len(glob(os.path.join(video_path, "flow_x*.jpg")))
        for ind in (np.arange(num_frames)+1):            
            images.extend(self._load_image(video_path, ind))
        process_data = self.transform(images)
        return process_data, video_path

    def __len__(self):
        return len(self.video_list)

# def eval_video(data):
#     if args.modality == 'RGB':
#         length = 3
#     elif args.modality == 'Flow':
#         length = 2
#     else:
#         raise ValueError("Unknown modality "+args.modality)
#     input_var = torch.autograd.Variable(data.view(-1, length, data.size(2), data.size(3)),
#                                         volatile=True)
#     baseout = np.squeeze(net(input_var).data.cpu().numpy().copy())
#     return baseout


@torch.no_grad()
def extract_bn_inception_feature(input_dir, output_dir, modality, test_list, input_size=224, 
                                 crop_fusion_type='avg', dropout=0.7, workers=4, flow_prefix='', device_id=0):
    net = TSN(modality,            
              consensus_type=crop_fusion_type,
              dropout=dropout)

    cropping = torchvision.transforms.Compose([
        GroupScale((net.input_size, net.input_size)),
    ])
    
    print(f"input_dir {input_dir} -> output_dir {output_dir}")
    # print(f"model input size: {net.input_size}, given input size: {input_size}") # 224
    assert net.input_size == input_size, "given input size is not equal to model input size"
    
    data_loader = torch.utils.data.DataLoader(
            TSNDataSet(input_dir, test_list,
                    modality=modality,
                    image_tmpl="img_{:05d}.jpg" if modality == 'RGB' else flow_prefix+"flow_{}_{:05d}.jpg",
                    transform=torchvision.transforms.Compose([
                        cropping, Stack(roll=True),
                        ToTorchFormatTensor(div=False),
                        GroupNormalize(net.input_mean, net.input_std),
                    ])),
            batch_size=1, shuffle=False,
            num_workers=workers, pin_memory=True)

    net = torch.nn.DataParallel(net).cuda(device_id)
    net.eval()
    for i, (data, video_path) in tqdm(enumerate(data_loader), total=len(data_loader)):
        try:
            os.makedirs(output_dir, exist_ok=True)
            ft_path = os.path.join(output_dir, video_path[0].split(os.sep)[-1]+".pkl")
            length = 3 if modality == 'RGB' else 2
            input_var = torch.autograd.Variable(data.view(-1, length, data.size(2), data.size(3)))
            rst = np.squeeze(net(input_var).data.cpu().numpy().copy()) # shape: (n_frames, 1024->dim of last pooling layer in BN-Inception)
            pkl.dump(rst, open(ft_path, "wb"))
        except Exception as e:
            with open(os.path.join('./', f"error_{modality}_{'_'.join(ft_path.split('/')[-3:])}.txt"), "w") as f:
                print(f"error: {e} video_path: {video_path} ft_path: {ft_path}")
                f.write(f"error: {e}\n")
                f.write(f"video_path: {video_path}\n")
                f.write(f"ft_path: {ft_path}\n")
            break
        data.cpu()
        input_var.cpu()
        del data, input_var, rst
        
    net.cpu()
    del net
    torch.cuda.empty_cache()

if __name__ == '__main__':    
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str)
    parser.add_argument('-o', '--output_dir', type=str)
    parser.add_argument('-m', '--modality', type=str, choices=['RGB', 'Flow'])
    parser.add_argument('-t', '--test_list', type=str)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--crop_fusion_type', type=str, default='avg',
                        choices=['avg', 'max', 'topk'])
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--flow_prefix', type=str, default='')
    parser.add_argument('--device_id', type=int, default=0)

    args = parser.parse_args()
    
    extract_bn_inception_feature(args.input_dir, args.output_dir, args.modality, args.test_list,
                                 args.input_size, args.crop_fusion_type, args.dropout,
                                 args.workers, args.flow_prefix, args.device_id)