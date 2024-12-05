from typing import Dict, Union, Tuple
import math
from functools import partial

import librosa
import numpy as np
import torch
from torch import nn
from torcheval.metrics.functional import multiclass_accuracy, multiclass_precision, multiclass_recall, multiclass_f1_score, multiclass_auprc, multiclass_auroc
from sklearn.metrics import accuracy_score, average_precision_score, top_k_accuracy_score

from data_utils import RMS

@torch.no_grad()
def get_loss_values(model_output, targets, criterions: Dict[str, nn.Module], average='micro', 
                    model_output_onset=None, targets_onset=None) -> Dict[str, float]:
    ''' Calculate loss or evaluation metric values for given criterions.
    Args:
        model_output: torch.Tensor, model output
        targets: torch.Tensor, target values
        criterions: Dict[str, nn.Module], loss functions
        average: str, average type for multiclass metrics. One of ['micro', 'macro', None].
        (optional, for onset prediction)
        model_output_onset: torch.Tensor, model output for onset detection
        targets_onset: torch.Tensor, target values for onset detection
    Returns:
        reduced_losses: Dict[str, float], reduced loss / averaged metric values
    '''
    if average not in ['micro', 'macro', None]:
        raise ValueError(f"Invalid average type: {average}. Should be one of ['micro', 'macro', None].")
    reduced_losses = {}
    if type(targets) == tuple:
        target = targets[0]
        target_continuous = targets[1]
    else:
        target = targets
        target_continuous = None

    for loss_type, criterion in criterions.items():
        if loss_type in ["CE", "CE_GLS"]: # outputs mean value within the batch
            loss = criterion(model_output, target)
        elif loss_type in ["MSE", "MAE"]: # outputs mean value within the batch
            if target_continuous is not None: # test phase
                loss = criterion(model_output, target_continuous)
            else:
                loss = criterion(model_output, target)
        elif loss_type in ["ACC", "PREC", "RECALL", "F1", "PRAUC", "ROCAUC", "ACC@3", "ACC@5", "ACC@10"]:
            # calculate for each instance in batch, and average
            assert model_output.shape[0] == target.shape[0]
            losses = []
            for i in range(model_output.shape[0]):
                n_classes, n_samples = model_output[i].shape
                assert n_samples == len(target[i])
                losses.append(criterion(model_output[i].transpose(0, 1),
                                        target[i], average=average))
            if average in ['micro', 'macro']:
                loss = sum(losses) / len(losses)
            elif average is None:
                loss = losses
        elif loss_type in ["OnsetACC", "OnsetAP"]:
            # calculate for each instance in batch, and average
            assert model_output_onset.shape[0] == targets_onset.shape[0], f"{model_output_onset.shape[0]} != {targets_onset.shape[0]}"
            losses = []
            for i in range(model_output_onset.shape[0]):
                assert model_output_onset[i].shape == targets_onset[i].shape
                _target, _model_output, _ = OnsetPostProcess.get_onset_vectors(targets_onset[i].detach().cpu().squeeze(-1),
                                                                            model_output_onset[i].detach().cpu().squeeze(-1))
                losses.append(criterion(_model_output, _target))
            if average in ['micro', 'macro']:
                loss = sum(losses) / len(losses)
            elif average is None:
                loss = losses
        else:
            raise ValueError(f"Invalid loss type: {loss_type}.")
        reduced_loss = loss.item() if isinstance(loss, torch.Tensor) else loss[0].tolist()
        # if not math.isnan(reduced_loss):
        reduced_losses[loss_type] = reduced_loss
    
    return reduced_losses

def empty_onehot(target: torch.Tensor, num_classes: int):
    # target_size = (batch, dim1, dim2, ...)
    # one_hot size = (batch, dim1, dim2, ..., num_classes)
    onehot_size = target.size() + (num_classes, )
    return torch.FloatTensor(*onehot_size).zero_()


def to_onehot(target: torch.Tensor, num_classes: int, src_onehot: torch.Tensor = None):
    if src_onehot is None:
        one_hot = empty_onehot(target, num_classes)
    else:
        one_hot = src_onehot

    last_dim = len(one_hot.size()) - 1

    # creates a one hot vector provided the target indices
    # and the Tensor that holds the one-hot vector
    with torch.no_grad():
        one_hot = one_hot.scatter_(
            dim=last_dim, index=torch.unsqueeze(target, dim=last_dim), value=1.0)
    return one_hot

class CrossEntropyLossWithGaussianSmoothedLabels(nn.Module):
    """
    https://github.com/dansuh17/jdcnet-pytorch/blob/c3e12964228ff35a7f452c8c4aea95a0027234ed/jdc/loss.py#L7
    """
    def __init__(self, num_classes=16, blur_range=3):
        super().__init__()
        self.dim = -1
        self.num_classes = num_classes
        self.blur_range = blur_range

        # pre-calculate decayed values following Gaussian distribution
        # up to distance of three (== blur_range)
        self.gaussian_decays = [self.gaussian_val(dist=d, sigma=1) for d in range(blur_range + 1)]

    @staticmethod
    def gaussian_val(dist: int, sigma=1):
        return math.exp(-math.pow(2, dist) / (2 * math.pow(2, sigma)))

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # pred: (b, C, d)
        # target: (b, d)
        
        pred = pred.transpose(1, 2) # (b, C, d) -> (b, d, C)
        pred_logit = torch.log_softmax(pred, dim=self.dim)

        # out: (b, d, C)
        target_smoothed = self.smoothed_label(target)

        # calculate the 'cross entropy' for each of 31 features
        target_loss_sum = -(pred_logit * target_smoothed).sum(dim=self.dim)
        return target_loss_sum.mean()  # and then take their mean

    def smoothed_label(self, target: torch.Tensor):
        # out: (b, d, C)
        target_onehot = empty_onehot(target, self.num_classes).to(target.device)

        # apply gaussian smoothing
        target_smoothed = self.gaussian_blur(target, target_onehot)
        # insert 1 at the target ground-truth index
        target_smoothed = to_onehot(target, self.num_classes, target_smoothed)
        return target_smoothed

    def gaussian_blur(self, target: torch.Tensor, one_hot: torch.Tensor):
        # blur the one-hot vector with gaussian decay
        with torch.no_grad():
            # Going in the reverse direction from 3 -> 0 since the value on the clamped index
            # will override the previous value
            # when the class index is less than 4 or greater then (num_class - 4).
            for dist in range(self.blur_range, -1, -1):
                # one_hot = self.set_decayed_values(dist, target, one_hot)
                one_hot = self.set_decayed_values_except_0(dist, target, one_hot)
        return one_hot

    def set_decayed_values(self, dist: int, target_idx: torch.Tensor, one_hot: torch.Tensor):
        # size of target_idx: (batch, num_seq) = (batch, 31)
        # size of one_hot: (batch, num_seq, num_classes) = (batch, 31, 722)
        for direction in [1, -1]:  # apply at both positive / negative directions
            # used `clamp` to prevent index from underflowing / overflowing
            blur_idx = torch.clamp(
                target_idx + (direction * dist), min=0, max=self.num_classes - 1)
            # set decayed values at indices represented by blur_idx
            decayed_val = self.gaussian_decays[dist]
            one_hot = one_hot.scatter_(
                dim=2, index=torch.unsqueeze(blur_idx, dim=2), value=decayed_val)
        return one_hot
    
    def set_decayed_values_except_0(self, dist: int, target_idx: torch.Tensor, one_hot: torch.Tensor):
        # size of target_idx: (batch, num_seq) = (batch, 31)
        # size of one_hot: (batch, num_seq, num_classes) = (batch, 31, 722)
        # for 0 value in target_idx, do not apply gaussian blur
        for direction in [1, -1]:  # apply at both positive / negative directions
            # used `clamp` to prevent index from underflowing / overflowing
            blur_idx = torch.clamp(
                target_idx + (direction * dist), min=1, max=self.num_classes - 1)
            # set decayed values at indices represented by blur_idx
            decayed_val = self.gaussian_decays[dist]
            one_hot = one_hot.scatter_(
                dim=2, index=torch.unsqueeze(blur_idx, dim=2), value=decayed_val)
            
            zero_mask = target_idx == 0
            one_hot = one_hot.masked_fill(zero_mask.unsqueeze(2), 0.0)
        return one_hot


class RMSLoss(nn.Module):
    '''For RMS loss and evaluation metrics, the input should be the output of Video2RMS and the target RMS.'''
    def __init__(self, loss_config: Dict = {"type":"MSE"}, rms_discretize=False, rms_mu=255, rms_num_bins=16, rms_min=0.01):
        super(RMSLoss, self).__init__()
        assert 'type' in loss_config.keys(), "Loss type not found in loss_config."
        self.LOSS_TYPES = ["MSE", "MAE", "CE", "CE_GLS", "ACC", "PREC", "RECALL", "F1", "PRAUC", "ROCAUC", "ACC@3", "ACC@5", "ACC@10"]
        self.loss_type = loss_config['type']
        if self.loss_type not in self.LOSS_TYPES:
            raise ValueError(f"Invalid loss type: {self.loss_type}. Should be one of {self.LOSS_TYPES}.")
        
        if self.loss_type == "CE_GLS":
            assert "gls_num_classes" in loss_config.keys() and "gls_blur_range" in loss_config.keys()
            self.loss_config = loss_config
        self.rms_discretize = rms_discretize
        
        if rms_discretize:
            self.rms_mu = rms_mu
            self.rms_num_bins = rms_num_bins
            self.rms_min = rms_min
        else:
            if self.loss_type in ["CE", "CE_GLS"]:
                raise ValueError("Discretization is required for CE loss.")
            
        print("Loss type: {}".format(self.loss_type))

    def forward(self, model_output, targets, average='micro'):
        rms_target = targets
        rms_target.requires_grad = False
        rms_out = model_output

        if self.rms_discretize and self.loss_type in ["MSE", "MAE"]:
            mu_bins = RMS.get_mu_bins(self.rms_mu, self.rms_num_bins, self.rms_min)
            assert len(rms_out.shape) == 3 # assure batched input
            rms_out = rms_out.argmax(axis=1)
            rms_out = RMS.undiscretize_rms(rms_out, mu_bins, ignore_min=True)
            # rms_target = undiscretize_rms(rms_target, mu_bins, ignore_min=True)
        
        if self.loss_type == "MSE":
            loss_fn = nn.MSELoss()            
        elif self.loss_type == "MAE":
            loss_fn = nn.L1Loss()
        elif self.loss_type == "CE":
            loss_fn = nn.CrossEntropyLoss()
        elif self.loss_type == "CE_GLS":
            loss_fn = CrossEntropyLossWithGaussianSmoothedLabels(num_classes=self.loss_config['gls_num_classes'], 
                                                                 blur_range=self.loss_config['gls_blur_range'])
        elif self.loss_type == "ACC":
            loss_fn = partial(multiclass_accuracy, average=average, num_classes=self.rms_num_bins)
        elif self.loss_type == "ACC@3":
            _loss_value = top_k_accuracy_score(rms_target.detach().cpu().numpy(),
                                        rms_out.detach().cpu().numpy(), 
                                        k=3, labels=list(range(self.rms_num_bins)))
            return torch.tensor(_loss_value)
        elif self.loss_type == "ACC@5":
            _loss_value = top_k_accuracy_score(rms_target.detach().cpu().numpy(),
                                        rms_out.detach().cpu().numpy(), 
                                        k=5, labels=list(range(self.rms_num_bins)))
            return torch.tensor(_loss_value)
        elif self.loss_type == "ACC@10":
            _loss_value = top_k_accuracy_score(rms_target.detach().cpu().numpy(), 
                                        rms_out.detach().cpu().numpy(), 
                                        k=10, labels=list(range(self.rms_num_bins)))
            return torch.tensor(_loss_value)
        elif self.loss_type == "PREC":
            loss_fn = partial(multiclass_precision, average=average, num_classes=self.rms_num_bins)
        elif self.loss_type == "RECALL":
            loss_fn = partial(multiclass_recall, average=average, num_classes=self.rms_num_bins)
        elif self.loss_type == "F1":
            loss_fn = partial(multiclass_f1_score, average=average, num_classes=self.rms_num_bins)
        elif self.loss_type == "PRAUC":
            loss_fn = partial(multiclass_auprc, average='macro', num_classes=self.rms_num_bins)
        elif self.loss_type == "ROCAUC":
            loss_fn = partial(multiclass_auroc, average='macro', num_classes=self.rms_num_bins)
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}. Should be one of {self.LOSS_TYPES}.")
                
        loss = loss_fn(rms_out, rms_target)

        return loss
    
    
class OnsetPostProcess:
    # Reference: https://github.com/XYPB/CondFoleyGen/blob/c565db7a7c84c32e8bf53d3a33c095803f6165b9/predict_onset.py
    @staticmethod
    def _onset_nms(onset:torch.Tensor, confidence:torch.Tensor, 
                window_length:Union[int,float]=librosa.time_to_samples(0.05, sr=30).item())\
                    ->torch.Tensor:
        '''Onset non-maximum suppression
        (Get most confident (max-amplitude in wav within conf_interval) onset within a window around each onset)
        '''
        onset_indices = torch.where(onset == 1)[0].tolist()
        onset_remain = onset_indices[:]
        output = []
        sorted_idx = np.argsort(np.array(confidence)[onset_indices])[::-1] # descending order
        for idx in sorted_idx:
            cur = onset_indices[idx]
            if cur not in onset_remain:
                continue
            output.append(cur)
            onset_remain.remove(cur)
            for o in onset_remain:
                if abs(cur - o) < window_length:
                    onset_remain.remove(o)
        
        onset_return = torch.zeros_like(onset, dtype=torch.int)
        onset_return[output] = 1
        return onset_return
    
    @staticmethod
    def get_onset_vectors(onset_gt:torch.Tensor, logit:torch.Tensor, conf_interval:float=0.05,
                        sr:int=30) -> Tuple[list,list,torch.Tensor] :
        '''Get onset vectors for evaluation
        Params:
            onset_gt: ground truth onsets in one-hot vector. torch.Tensor of shape (n_frames).
            logit: predicted logit. torch.Tensor of shape (n_frames).
            conf_interval: confidence interval. time in seconds.
            sr: sample rate. int number of samples per second.
            window_length: window length. int number of samples.
            hop_length: hop length. int number of samples.
        Return:
            y_gt: ground truth onset vector. list of one-hot vector.
            y_pred: predicted onset vector. list of confidence.
        '''
        if len(onset_gt.shape) != 1: raise ValueError(f'onset_gt should be 1D tensor, but got {onset_gt.shape}.')
        if len(logit.shape) != 1: raise ValueError(f'logit should be 1D tensor, but got {logit.shape}.')
        
        onset_pred = torch.zeros_like(logit, dtype=torch.int)
        onset_pred[logit > 0.5] = 1
        
        # normalize logit in [0, 1]
        if torch.max(logit) > 1 or torch.min(logit) < 0:
            logit = 1 / (1 + np.exp(-logit)) # apply sigmoid func
        # get confidence of each onset_pred from logit
        conf_interval_frames = librosa.time_to_samples(conf_interval, sr=sr).item()
        confidence = [torch.max(logit[ max(0, i-int(conf_interval_frames)) : min(len(logit)-1, i+int(conf_interval_frames)) ]).item() \
                    for i in range(len(onset_pred))]
        # onset non-maximum suppression
        onset_pred = OnsetPostProcess._onset_nms(onset_pred, confidence, window_length=conf_interval_frames)
        
        onset_pred_working_indices = torch.where(onset_pred == 1)[0].tolist()
        onset_pred_result = torch.zeros(onset_pred.shape, dtype=torch.long)
        
        y_gt = []
        y_pred = []
        delta = librosa.time_to_samples(conf_interval*2, sr=sr).item()
        
        for o_gt in torch.where(onset_gt == 1)[0]:
            # find the onset_pred within delta distance from o_gt
            diff = [abs(int(o_pred - o_gt)) for o_pred in onset_pred_working_indices]
            idx_in_window = [idx for idx in range(len(onset_pred_working_indices)) if diff[idx] < delta]
            if len(idx_in_window) == 0: # False Negative: no onset_pred within delta distance
                y_gt.append(1)
                conf = confidence[o_gt]
                y_pred.append(conf)
            else: # True Positive: at least one onset_pred within delta distance
                # find the most confident (highest confidence) onset_pred within delta distance
                conf_in_window = [confidence[onset_pred_working_indices[idx]] for idx in idx_in_window]
                max_conf_idx = np.argsort(conf_in_window)[-1]
                match_idx = idx_in_window[max_conf_idx]
                
                y_gt.append(1)
                # get the confidence of the matched onset_pred
                conf = confidence[onset_pred_working_indices[match_idx]]
                onset_pred_result[onset_pred_working_indices[match_idx]] = 1
                y_pred.append(conf)
                # remove the matched onset_pred from onset_pred_working_indices
                del onset_pred_working_indices[match_idx]
                if len(onset_pred_working_indices) == 0:
                    break
        
        for o_pred in onset_pred_working_indices: # False Positive: no onset_gt within delta distance
            y_gt.append(0)
            y_pred.append(confidence[o_pred])
            onset_pred_result[o_pred] = 1
            
        return y_gt, y_pred, onset_pred_result


class OnsetLoss(nn.Module):
    def __init__(self, loss_config: Dict = {"type":"OnsetACC", "tolerance":0.1}):
        super(OnsetLoss, self).__init__()
        assert 'type' in loss_config.keys(), "Loss type not found in loss_config."
        assert 'tolerance' in loss_config.keys(), "Tolerance not found in loss_config."
        self.LOSS_TYPES = ["OnsetACC", "OnsetAP"]
        self.loss_type = loss_config['type']
        if self.loss_type not in self.LOSS_TYPES:
            raise ValueError(f"Invalid loss type: {self.loss_type}. Should be one of {self.LOSS_TYPES}.")
        self.tolerance = loss_config['tolerance']
        if not isinstance(self.tolerance, float):
            raise ValueError(f"Tolerance should be a float value, but got {self.tolerance}.")
        
        print("Loss type: {}".format(self.loss_type))
    
    def forward(self, model_output:Union[np.ndarray,list], targets:Union[np.ndarray,list]):
        # assert shape is 1-dimensional
        if isinstance(model_output, np.ndarray) and len(model_output.shape) > 1:
            raise ValueError(f"model_output should be 1-dimensional, but got {model_output.shape}.")
        if isinstance(targets, np.ndarray) and len(targets.shape) > 1:
            raise ValueError(f"targets should be 1-dimensional, but got {targets.shape}.")
        
        if self.loss_type == "OnsetACC":
            model_output_binary = [1 if y >= 0.5 else 0 for y in model_output]
            loss = accuracy_score(targets, model_output_binary)
        elif self.loss_type == "OnsetAP":
            loss = average_precision_score(targets, model_output)
        
        return torch.tensor(loss, dtype=torch.float32)