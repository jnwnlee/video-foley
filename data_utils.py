import os
import pickle
from typing import Dict, Tuple

import numpy as np
import torch
import torch.utils.data
import librosa
from scipy.signal import ellip, filtfilt


class RMS:
    @staticmethod
    def zero_phased_filter(x:np.ndarray):
        '''Zero-phased low-pass filtering'''
        b, a = ellip(4, 0.01, 120, 0.125) 
        x = filtfilt(b, a, x, method="gust")
        return x

    @staticmethod
    def mu_law(rms:torch.Tensor, mu:int=255):
        '''Mu-law companding transformation'''
        # assert if all values of rms are non-negative
        assert torch.all(rms >= 0), f'All values of rms must be non-negative: {rms}'
        mu = torch.tensor(mu)
        mu_rms = torch.sign(rms) * torch.log(1 + mu * torch.abs(rms)) / torch.log(1 + mu)
        return mu_rms

    @staticmethod
    def inverse_mu_law(mu_rms:torch.Tensor, mu:int=255):
        '''Inverse mu-law companding transformation'''
        assert torch.all(mu_rms >= 0), f'All values of rms must be non-negative: {mu_rms}'
        mu = torch.tensor(mu)
        rms = torch.sign(mu_rms) * (torch.exp(mu_rms * torch.log(1 + mu)) - 1) / mu
        return rms


    @staticmethod
    def get_mu_bins(mu, num_bins, rms_min):
        mu_bins = torch.linspace(RMS.mu_law(torch.tensor(rms_min), mu), 1, steps=num_bins)
        mu_bins = RMS.inverse_mu_law(mu_bins, mu)
        return mu_bins
        
    @staticmethod
    def discretize_rms(rms, mu_bins):
        rms = torch.maximum(rms, torch.tensor(0.0)) # change negative values to zero
        rms_inds = torch.bucketize(rms, mu_bins, right=True) # discretize
        return rms_inds

    @staticmethod
    def undiscretize_rms(rms_inds, mu_bins, ignore_min=True):
        if ignore_min and mu_bins[0] > 0.0:
            mu_bins[0] = 0.0
        
        rms_inds_is_cuda = rms_inds.is_cuda
        if rms_inds_is_cuda:
            device = rms_inds.device
            rms_inds = rms_inds.detach().cpu()
        rms = mu_bins[rms_inds]
        if rms_inds_is_cuda:
            rms = rms.to(device)
        return rms
    
    @staticmethod
    def get_rms(waveform:np.ndarray, nframes:int, hop:int) -> np.ndarray:
        waveform = np.pad(waveform, (int((nframes - hop) / 2), int((nframes - hop) / 2), ), mode="reflect")
        rms = librosa.feature.rms(y=waveform, frame_length=nframes, hop_length=hop, 
                                  center=False, pad_mode="reflect")
        rms = rms[0]
        return rms


def pad_or_truncate_feature(feature: np.ndarray, video_samples: int) -> np.ndarray:
    if feature.shape[0] < video_samples:
        padded = np.zeros((video_samples, feature.shape[1]))
        padded[0:feature.shape[0], :] = feature
        return padded
    else:
        return feature[0:video_samples, :]


class VideoAnnotation:
    @staticmethod
    def _get_onset_annotation(annot_dir:str, videoname:str, index:int, length:int=10) \
        -> Dict[float, Tuple[str, str, str]]:
        '''Get onset annotation from txt file, within index*length <= time < (index+1)*length
        txt file should be in format:
        onset_time material action reaction
        e.g.,
        2.677438 ceramic hit static
        3.675198 ceramic hit static
        
        Parameters:
        annot_dir (str): Directory of annotation file
        videoname (str): video name(id) without index
        index (int): Index of requested onset range
        length (int): Length of requested onset range
        
        Returns:
        onset_dict (dict): Dictionary of onset time and corresponding material, action, reaction
        '''
        with open(os.path.join(annot_dir, videoname+'_times.txt'), 'r') as f:
            lines = f.readlines()
        start_time = index*length
        end_time = (index+1)*length
        onset_dict = {}
        for line in lines:
            time, material, action, reaction = line.split()
            time = float(time)
            if start_time <= time < end_time:
                onset_dict[time - start_time] = (material, action, reaction)
        
        return onset_dict
    
    @staticmethod
    def get_text_prompt(annot_dir:str, videoname:str, index:int, length:int=10,
                        template:str='a person {action} {material} with a wooden stick.') -> str:
        # sort onset_dict by time
        onset_dict = VideoAnnotation._get_onset_annotation(annot_dir, videoname, index, length)
        onset_dict = dict(sorted(onset_dict.items(), key=lambda x: x[0]))
        
        text_prompt_list = []
        for time, (material, action, reaction) in onset_dict.items():
            if material is None or action is None:
                continue
            text_prompt_list.append(template.format(material=material, action=action))
        if len(text_prompt_list) == 0:
            text_prompt_list.append(template.format(material='something', action='hit'))
        text_prompt = ' After that, '.join(text_prompt_list)
        text_prompt = text_prompt[0].upper() + text_prompt[1:]
        
        return text_prompt
    
    @staticmethod
    def get_onset_label(annot_dir:str, videoname:str, index:int, length:int=10, sample_rate:int=30) \
        -> torch.Tensor:
        onset_dict = VideoAnnotation._get_onset_annotation(annot_dir, videoname, index, length)
        onset_times = list(onset_dict.keys())
        
        onset_frames = librosa.time_to_samples(onset_times, sr=sample_rate)
        
        onset_label = torch.zeros(sample_rate*length, dtype=torch.float32)
        onset_label[onset_frames] = 1
        
        assert sum(onset_label) == len(onset_frames), f"Error while creating onset label, got sum {sum(onset_label)} instead of {len(onset_frames)}."
        
        return onset_label


class VideoAudioDataset(torch.utils.data.Dataset):
    """
    loads image, flow feature, audio files
    """

    def __init__(self, list_file, rgb_feature_dir, flow_feature_dir, mel_dir, config, max_sample=-1):
        self.video_samples = config.video_samples
        self.audio_samples = config.audio_samples
        self.audio_len = config.audio_samples # seconds
        self.rms_samples = config.rms_samples
        self.rms_nframes = config.rms_nframes
        self.rms_hop = config.rms_hop
        self.rms_discretize = config.rms_discretize
        if self.rms_discretize:
            self.rms_mu = config.rms_mu
            self.rms_num_bins = config.rms_num_bins
            self.rms_min = config.rms_min
            self.mu_bins = RMS.get_mu_bins(self.rms_mu, self.rms_num_bins, self.rms_min)
        self.rgb_feature_dir = rgb_feature_dir
        self.flow_feature_dir = flow_feature_dir
        self.mel_dir = mel_dir

        with open(list_file, encoding='utf-8') as f:
            self.video_ids = [line.strip() for line in f]
        self.video_class = os.path.basename(list_file).split("_")[0]

    def get_data_pair(self, video_id):
        im_path = os.path.join(self.rgb_feature_dir, video_id+".pkl")
        flow_path = os.path.join(self.flow_feature_dir, video_id+".pkl")
        audio_path = os.path.join(self.mel_dir, video_id+"_audio.npy")
        im = self.get_im(im_path)
        flow = self.get_flow(flow_path)
        rms = self.get_rms(audio_path)
        if self.rms_discretize:
            rms_continuous = torch.tensor(rms.copy(), dtype=torch.float32)
            with torch.no_grad():
                rms = RMS.discretize_rms(torch.tensor(rms.copy()), self.mu_bins)
            rms = rms.long()
            rms_data = (rms, rms_continuous)
        else:
            rms_data = torch.tensor(rms.copy(), dtype=torch.float32)
        
        feature = np.concatenate((im, flow), 1)
        feature = torch.FloatTensor(feature.astype(np.float32))
        return (feature, rms_data, video_id, self.video_class)

    def get_im(self, im_path):
        with open(im_path, 'rb') as f:
            im = pickle.load(f, encoding='bytes')
        f.close()
        
        im_padded = pad_or_truncate_feature(im, self.video_samples)
        assert im_padded.shape[0] == self.video_samples
        return im_padded

    def get_flow(self, flow_path):
        with open(flow_path, 'rb') as f:
            flow = pickle.load(f, encoding='bytes')
        f.close()
        
        flow_padded = pad_or_truncate_feature(flow, self.video_samples)
        assert flow_padded.shape[0] == self.video_samples
        return flow_padded
    
    def get_rms(self, audio_path):
        waveform = np.load(audio_path)
        rms = RMS.get_rms(waveform=waveform, nframes=self.rms_nframes, hop=self.rms_hop)
        if not len(rms) == self.rms_samples:
            raise RuntimeError(f"Error while calculating RMS, got length {rms.shape} instead of {self.rms_samples}.") 
        rms = RMS.zero_phased_filter(rms)
        return rms
    
    def __getitem__(self, index):
        return self.get_data_pair(self.video_ids[index])

    def __len__(self):
        return len(self.video_ids)