import argparse
import os

import torch
import numpy as np
from yacs.config import CfgNode as CN

from config import _C as config
from data_utils import RMS, VideoAnnotation
from util import load_config, load_models, prepare_dataloaders, get_audio, save_audio, save_video_with_audio, interpolate_rms_for_rms2sound, set_seed

    
@torch.no_grad
def infer(epoch:int=500, video2rms_ckpt_dir:str='', rms2sound_ckpt_dir:str = '', 
          prompt_type:str='audio', config:CN=None, output_dir:str='./infer') -> None:
    if video2rms_ckpt_dir == '' or rms2sound_ckpt_dir == '':
        raise ValueError("ckpt_dir is empty")
    if config is None:
        raise ValueError("config is None")
    if prompt_type not in ['audio', 'text']:
        raise ValueError("prompt_type must be 'audio' or 'text'")
    
    print(f"Inference with epoch {epoch}...")
    
    print(f'Setting seed: {config.train.seed}')
    set_seed(config.train.seed)

    # load ckpt and prepare model
    print('Loading model...')
    video2rms_model, audio_ldm_controlnet = load_models(epoch, video2rms_ckpt_dir, rms2sound_ckpt_dir, config,
                                                        torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    # prepare dataloader
    print('Preparing dataset...')
    test_loader = prepare_dataloaders(config.data, batch_size=16, train=False, test=True)

    # evaluate
    print('Inference...')
    os.makedirs(os.path.join(output_dir, f'{prompt_type}_prompt', 'audio'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, f'{prompt_type}_prompt', 'video'), exist_ok=True)
    
    video2rms_model.eval()
    assert config.data.rms_discretize, 'RMS must be discretized'
    mu_bins:torch.tensor = RMS.get_mu_bins(config.data.rms_mu, config.data.rms_num_bins, config.data.rms_min)
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            video2rms_model.parse_batch(batch)
            video2rms_model.forward()
            for j in range(len(video2rms_model.pred_rms)):
                pred_rms = video2rms_model.pred_rms[j].detach().cpu().numpy()
                pred_rms_undiscretized:torch.tensor =RMS.undiscretize_rms(torch.from_numpy(pred_rms.argmax(axis=0)),
                                                                       mu_bins, ignore_min=True)
                pred_rms_undiscretized = pred_rms_undiscretized.detach().cpu().unsqueeze(0)
                pred_rms_undiscretized = interpolate_rms_for_rms2sound(pred_rms_undiscretized,
                                                                       audio_len=config.data.audio_samples , 
                                                                       sr=config.data.audio_sample_rate,
                                                                       frame_len=1024, 
                                                                       hop_len=160)
                
                if prompt_type == 'audio':
                    gt_audio:np.ndarray = get_audio(os.path.join(config.data.audio_src_dir.replace('*', video2rms_model.video_class[j]),
                                                    video2rms_model.video_name[j] + '.wav'), 
                                        sr=config.data.audio_sample_rate)
                    gt_audio = torch.from_numpy(gt_audio).unsqueeze(0)
                    
                    generated_audio:np.ndarray = audio_ldm_controlnet.generate(
                        waveform=gt_audio,
                        rms=pred_rms_undiscretized
                    )
                else:
                    videoname, index = video2rms_model.video_name[j].split('_')
                    index = int(index)
                    text_prompt:str = VideoAnnotation.get_text_prompt(
                        annot_dir=config.data.annotation_dir, 
                        videoname=videoname, 
                        index=index, 
                        length=config.data.audio_samples
                    )
                    
                    generated_audio:np.ndarray = audio_ldm_controlnet.generate(
                        text_prompt=text_prompt,
                        rms=pred_rms_undiscretized
                    )
                
                save_audio(audio=generated_audio,
                            output_path=os.path.join(output_dir, f'{prompt_type}_prompt', 'audio', 
                                                     video2rms_model.video_name[j] + '.wav'),
                            sr=config.data.audio_sample_rate)
                
                save_video_with_audio(video_path=os.path.join(config.data.video_src_dir.replace('*', video2rms_model.video_class[j]),
                                                   video2rms_model.video_name[j] + '.mp4'),
                                      audio=generated_audio, 
                                      output_path=os.path.join(output_dir, f'{prompt_type}_prompt', 'video', 
                                                               video2rms_model.video_name[j] + '.mp4'),
                                      sr=config.data.audio_sample_rate)

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video2rms_ckpt_dir', type=str, 
                        default='./ckpt/video-foley-model', 
                        help='directory for model checkpoint')
    parser.add_argument('-r', '--rms2sound_ckpt_dir', type=str, 
                        default='./ckpt/video-foley-model', 
                        help='directory for model checkpoint')
    parser.add_argument('-e', '--epoch', type=int, default=500, 
                        help='number of epochs of Video2RMS model')
    parser.add_argument('-d', '--data_dir', type=str, default=None,
                        help='input directory for dataset')
    parser.add_argument('-o', '--output_dir', type=str, 
                        default='./video2sound/Video-Foley', 
                        help='output directory to save generated results')
    parser.add_argument('-p', '--prompt_type', type=str, 
                        default='text',
                        choices=['text', 'audio'], 
                        help='prompt type for audio generation')
    args = parser.parse_args()
    
    
    config = load_config(os.path.join(args.video2rms_ckpt_dir, 'opts.yml'))
    config_data_dir = config.data.rgb_feature_dirs[0].split('/features/')[0]
    if args.data_dir is None:
        args.data_dir = config_data_dir
    else:
        if not (os.path.exists(os.path.join(args.data_dir, 'features')) 
                and os.path.exists(os.path.join(args.data_dir, 'features'))):
            raise FileNotFoundError(f"Data directory {args.data_dir} not found")
        if not os.path.abspath(args.data_dir) == os.path.abspath(config_data_dir):
            print(f"Warning: data directory {args.data_dir} is different from data directory in config {config_data_dir}")
    
    config.data.audio_src_dir = os.path.join(args.data_dir, 'features/*/audio_10s_16000hz_muted')
    config.data.video_src_dir = os.path.join(args.data_dir, 'features/*/videos_10s_30fps')
    config.data.annotation_dir = '/media/daftpunk4/dataset/GreatestHits/vis-data' # os.path.join(args.data_dir, 'vis-data')
    config.freeze()
    
    infer(epoch=args.epoch,
        video2rms_ckpt_dir=args.video2rms_ckpt_dir,
        rms2sound_ckpt_dir=args.rms2sound_ckpt_dir,
        prompt_type=args.prompt_type,
        config=config,
        output_dir=args.output_dir)
    
    
    
    