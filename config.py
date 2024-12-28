import os
from yacs.config import CfgNode as CN
import librosa
import numpy as np

_C  =  CN()

### Logging Config
_C.log = CN()
_C.log.save_dir = 'ckpt/rms_64_GLS_2'
_C.log.exclude_dirs = ['ckpt', 'data', 'opencv-python']
_C.log.logger = ['tensorboard', 'wandb']
_C.log.loss = CN()
_C.log.loss.types = ["CE_GLS", "MAE", "ACC", "ACC+-2", "ACC+-5", "ACC+-8"] # MAE == E-L1
if "CE_GLS" in _C.log.loss.types:
    _C.log.loss.gls_num_classes = 64
    _C.log.loss.gls_blur_range = 2


### Training Config
_C.train = CN()
_C.train.epochs = 600
_C.train.num_epoch_save = 50
_C.train.seed = 123
_C.train.cudnn_enabled = True
_C.train.cudnn_benchmark = False
_C.train.checkpoint_path = ''
_C.train.epoch_count = 0

_C.train.loss = CN()
_C.train.loss.type = "CE_GLS"
if _C.train.loss.type == "CE_GLS":
    _C.train.loss.gls_num_classes = 64
    _C.train.loss.gls_blur_range = 2
    if "CE_GLS" in _C.log.loss.types and \
    (_C.train.loss.gls_num_classes != _C.log.loss.gls_num_classes 
     or _C.train.loss.gls_blur_range != _C.log.loss.gls_blur_range):
        print(f"Warning: train.loss.gls_num_classes {_C.train.loss.gls_num_classes} != log.loss.gls_num_classes {_C.log.loss.gls_num_classes}")
        print(f"Warning: train.loss.gls_blur_range {_C.train.loss.gls_blur_range} != log.loss.gls_blur_range {_C.log.loss.gls_blur_range}")
_C.train.batch_size = 512
_C.train.lr = 1e-3
_C.train.beta1 = 0.5
_C.train.continue_train = False
_C.train.lr_policy = "step"
_C.train.weight_decay = 0.5
_C.train.niter = 100
_C.train.loss_threshold = 0.0001
_C.train.onset_supervision = False
if _C.train.onset_supervision: # BCE loss
    _C.train.onset_loss_lambda = 0.001


### Data Config
_C.data = CN()
data_dir = '/mnt/GreatestHits'
materials = os.listdir(os.path.join(data_dir, 'features')) if os.path.isdir(os.path.join(data_dir, 'features')) else []
            # ['carpet', 'ceramic', 'cloth', 'dirt', 'drywall', 'glass', 'grass', 'gravel', 'leaf', 'metal', 
            #  'multiple', 'None', 'paper', 'plastic', 'plastic-bag', 'rock', 'tile', 'water', 'wood']
if materials == []: print(f"Warning: data_dir {data_dir} seems empty (while loading config)")
materials.sort()
_C.data.training_files = [os.path.join(data_dir, f'filelists/{material}_train.txt') for material in materials]
_C.data.test_files = [os.path.join(data_dir, f'filelists/{material}_test.txt') for material in materials]
_C.data.rgb_feature_dirs = [os.path.join(data_dir, f"features/{material}/feature_rgb_bninception_dim1024_30fps") 
                            for material in materials]
_C.data.flow_feature_dirs = [os.path.join(data_dir, f"features/{material}/feature_flow_bninception_dim1024_30fps") 
                             for material in materials]
_C.data.mel_dirs = [os.path.join(data_dir, f"features/{material}/melspec_10s_16000hz") for material in materials]
_C.data.video_samples = 300
_C.data.audio_samples = 10
_C.data.audio_sample_rate = 16000
_C.data.rms_nframes = 512
_C.data.rms_hop = 128
_dummy_audio =np.pad(np.zeros(_C.data.audio_samples*_C.data.audio_sample_rate),
                    (int((_C.data.rms_nframes - _C.data.rms_hop) / 2), int((_C.data.rms_nframes - _C.data.rms_hop) / 2), ), 
                    mode="reflect")
_C.data.rms_samples = int(librosa.feature.rms(y=_dummy_audio, \
                                            frame_length=_C.data.rms_nframes, hop_length=_C.data.rms_hop, \
                                            center=False, pad_mode="reflect").shape[1])
_C.data.rms_discretize = True
if _C.data.rms_discretize:
    _C.data.rms_num_bins = 64
    
    if _C.train.loss.type == "CE_GLS":
        assert _C.train.loss.gls_num_classes == _C.data.rms_num_bins
    if "CE_GLS" in _C.log.loss.types:
        assert _C.log.loss.gls_num_classes == _C.data.rms_num_bins
    
    _C.data.rms_mu = _C.data.rms_num_bins - 1
    _C.data.rms_min = 0.01

_C.data.onset_supervision = _C.train.onset_supervision
if _C.data.onset_supervision:
    _C.data.onset_annotation_dir = os.path.join(data_dir, 'vis-data')
    _C.data.onset_tolerance = 0.1


### Model Config
_C.model = CN()
_C.model.encoder_embedding_dim = 2048
_C.model.encoder_kernel_size = 5
_C.model.encoder_n_convolutions = 3
_C.model.encoder_n_lstm = 2
