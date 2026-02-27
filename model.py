import glob
import math
import os
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.optim import lr_scheduler

from criterion import RMSLoss
from data_utils import VideoAnnotation


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Encoder_MLP(nn.Module):
    def __init__(self, config, config_data):
        super(Encoder_MLP, self).__init__()
        self.rms_discretize = config_data.rms_discretize
        
        self.MLP = nn.ModuleList([
            nn.Linear(in_features=int(config_data.video_samples), 
                      out_features=int(config_data.rms_samples)),
            nn.Linear(in_features=int(config.encoder_embedding_dim),
                      out_features=int(config_data.rms_num_bins) if config_data.rms_discretize else 1)
        ])

    def forward(self, x):
        for proj in self.MLP:
            x = x.transpose(1, 2) 
            x = proj(x) # (batch, len, feature_dim) -> (batch, feature_dim, len_rms) -> (batch, len_rms, rms_num_bins)
            x = F.relu(x)
        
        if self.rms_discretize:
            x = x.transpose(1, 2) # (batch, len_rms, rms_num_bins) -> (batch, rms_num_bins, len_rms)
        else:
            x = x.squeeze(-1) # (batch, len_rms, 1) -> (batch, len_rms)
        
        return x


class Encoder(nn.Module):

    def __init__(self, config, config_data):
        super(Encoder, self).__init__()
        self.rms_discretize = config_data.rms_discretize
        self.onset_supervision = config_data.onset_supervision
        
        convolutions = []
        for _ in range(config.encoder_n_convolutions):
            conv_input_dim = config.encoder_embedding_dim
            conv_layer = nn.Sequential(
                ConvNorm(conv_input_dim,
                         config.encoder_embedding_dim,
                         kernel_size=config.encoder_kernel_size, stride=1,
                         padding=int((config.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(config.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.BiLSTM = nn.LSTM(input_size=config.encoder_embedding_dim,
                            hidden_size=int(config.encoder_embedding_dim / 4),
                            num_layers=config.encoder_n_lstm,
                            batch_first=True, bidirectional=True)
        
        # RMS_head
        self.BiLSTM_projs = nn.ModuleList([
            nn.Linear(in_features=int(config_data.video_samples), 
                      out_features=int(config_data.rms_samples)),
            nn.Linear(in_features=int(config.encoder_embedding_dim / 2),
                      out_features=int(config_data.rms_num_bins) if config_data.rms_discretize else 1)
        ])
        
        if self.onset_supervision:
            self.onset_head = nn.ModuleList([
                nn.Linear(in_features=int(config_data.video_samples), 
                        out_features=int(config_data.video_samples)),
                nn.Linear(in_features=int(config.encoder_embedding_dim / 2),
                          out_features=1) # 1-dim: Onset or not
            ])

    def forward(self, x):
        x = x.transpose(1, 2) # x: (batch, video_frames, feature_dim) -> (batch, feature_dim, video_frames)
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        x = x.transpose(1, 2) # (batch, emb_dim, video_frames) -> (batch, video_frames, emb_dim)
        if type(self.BiLSTM) in [nn.LSTM, nn.GRU]:
            x, _ = self.BiLSTM(x) # (batch, video_frames, emb_dim/2)
        else:
            x = self.BiLSTM(x) # (batch, video_frames, emb_dim/2)
        
        rms = x
        
        for proj in self.BiLSTM_projs:
            rms = rms.transpose(1, 2) 
            rms = proj(rms) # (batch, video_frames, emb_dim/2) -> (batch, emb_dim/2, rms_frames) -> (batch, rms_frames, rms_num_bins)
        rms = F.relu(rms)
        
        if self.onset_supervision:
            onset = x
            for proj in self.onset_head:
                onset = onset.transpose(1, 2) 
                onset = proj(onset) # (batch, len, emb_dim/2) -> (batch, emb_dim/2, rms_frames) -> (batch, rms_frames, 1)
        
        if self.rms_discretize:
            rms = rms.transpose(1, 2) # (batch, rms_frames, rms_num_bins) -> (batch, rms_num_bins, rms_frames)
        else:
            rms = rms.squeeze(-1) # (batch, rms_frames, 1) -> (batch, rms_frames)
        
        if self.onset_supervision:
            return rms, onset
        return rms
    
    
class Video2RMS(nn.Module):
    def __init__(self, config, config_data):
        super(Video2RMS, self).__init__()
        self.encoder = Encoder(config, config_data)

    def forward(self, inputs):
        encoder_output = self.encoder(inputs)
        
        return encoder_output
    

def init_net(net, device, init_type='normal', init_gain=0.02):
    assert (torch.cuda.is_available())
    net.to(device)
    net = torch.nn.DataParallel(net, range(torch.cuda.device_count()))
    init_weights(net, init_type, gain=init_gain)
    return net


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
    

class Video2Sound(nn.Module):
    ''' 
    Video2Sound model for training and inference, for video2rms task.
    Does not include the video feature extraction model, and RMS2Sound model.
    '''
    def __init__(self, config):
        super(Video2Sound, self).__init__()
        self.config = config
        
        self.model_names = ['Video2RMS']
        self.device = torch.device('cuda:0')
        self.Video2RMS = init_net(Video2RMS(config.model, config.data), self.device)
        
        self.RMSLoss = RMSLoss(config.train.loss, config.data.rms_discretize, 
                               config.data.rms_mu, config.data.rms_num_bins, config.data.rms_min).to(self.device)
        self.onset_supervision = config.train.onset_supervision
        if config.train.onset_supervision:
            self.onset_annotation_dir = config.data.onset_annotation_dir
            if not os.path.isdir(self.onset_annotation_dir):
                raise FileNotFoundError(f"Onset annotation directory '{self.onset_annotation_dir}' not found")
            self.get_onset_label = partial(VideoAnnotation.get_onset_label, 
                                           annot_dir=self.onset_annotation_dir,
                                           length=config.data.audio_samples,
                                           sample_rate=config.data.video_samples//config.data.audio_samples)
            self.onsetLoss = nn.BCEWithLogitsLoss()
            self.onsetLoss_lambda = self.config.train.onset_loss_lambda

        self.optimizers = []
        self.optimizer_Video2RMS = torch.optim.Adam(self.Video2RMS.parameters(),
                                            lr=config.train.lr, betas=(config.train.beta1, 0.999))
        self.optimizers.append(self.optimizer_Video2RMS)
        self.n_iter = -1

    def parse_batch(self, batch):
        feature, rms, video_name, video_class = batch
        self.feature = feature.to(self.device).float()
        
        if type(rms) is not torch.Tensor and len(rms) == 2:
            self.gt_rms, self.gt_rms_continuous = rms
            self.gt_rms = self.gt_rms.to(self.device)
            self.gt_rms_continuous = self.gt_rms_continuous.to(self.device)
        else:
            self.gt_rms = rms.to(self.device)
        self.video_name = video_name
        self.video_class = video_class
        
        if self.onset_supervision:
            # call get_onset_label for each sample in batch and stack them
            self.gt_onset = torch.stack([self.get_onset_label(videoname=video_id.split('_')[0], index=int(video_id.split('_')[1])) 
                                         for video_id in video_name], dim=0).unsqueeze(2).to(self.device)

    def forward(self):
        self.pred_rms = self.Video2RMS(self.feature)
        if self.onset_supervision:
            self.pred_rms, self.pred_onset = self.pred_rms

    def get_scheduler(self, optimizer, config):
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 2 + config.epoch_count - config.niter) / float(config.epochs - config.niter + 1)
            return lr_l

        if config.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.weight_decay, 
                                                    patience=config.niter, threshold=config.loss_threshold, 
                                                    threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        elif config.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=config.niter, gamma=config.weight_decay)
        else:
            raise NotImplementedError('learning rate policy [%s] is not implemented', config.lr_policy)
        return scheduler

    def setup(self):
        self.schedulers = [self.get_scheduler(optimizer, self.config.train) for optimizer in self.optimizers]
    
    def load_checkpoint(self, checkpoint_path):
        for name in self.model_names:
            filepath = "{}_{}.pt".format(checkpoint_path, name)
            print("Loading {}_module from checkpoint '{}'".format(name, filepath))
            state_dict = torch.load(filepath, map_location='cpu')
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata

            net = getattr(self, name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            checkpoint_state = state_dict["optimizer_{}".format(name)]
            net.load_state_dict(checkpoint_state)
            self.epoch = state_dict["epoch"]

            learning_rate = state_dict["learning_rate"]
        for index in range(len(self.optimizers)):
            for param_group in self.optimizers[index].param_groups:
                param_group['lr'] = learning_rate

    def update_learning_rate(self, val_loss):
        for scheduler in self.schedulers:
            scheduler.step(val_loss) if self.config.train.lr_policy == 'plateau' else scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
                    
    def backward_RMS(self):
        self.loss_RMS = self.RMSLoss(self.pred_rms, self.gt_rms)
        if self.onset_supervision:
            self.loss_RMS += self.onsetLoss_lambda * self.onsetLoss(self.pred_onset, self.gt_onset)
        # for onset-only training           
        # self.loss_RMS = self.onsetLoss_lambda * self.onsetLoss(self.pred_onset, self.gt_onset)
        
        self.loss_RMS.backward()

    def optimize_parameters(self):
        self.n_iter += 1
        self.forward()

        # update Video2RMS
        self.set_requires_grad(self.Video2RMS, True)
        self.optimizer_Video2RMS.zero_grad()
        self.backward_RMS()
        self.optimizer_Video2RMS.step()
    