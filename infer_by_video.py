import argparse
import os
from typing import List

import torch
import numpy as np
import librosa
from glob import glob
import subprocess
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from yacs.config import CfgNode as CN

from config import _C as config
from data_utils import RMS, pad_or_truncate_feature
from util import load_config, load_models, save_audio, save_video_with_audio, interpolate_rms_for_rms2sound, set_seed

# Import preprocessing functions
from preprocess.extract_audio_and_video import pipline_align, pipline_cut
from preprocess.extract_rgb_flow_raft import cal_for_frames
from preprocess.extract_feature import extract_bn_inception_feature


def add_silent_audio_if_needed(video_path:str) -> None:
    # Get video information using ffprobe
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a:0',
        '-count_packets',
        '-show_entries', 'stream=codec_type,nb_read_packets',
        '-of', 'csv=p=0',
        video_path
    ]
    output = subprocess.check_output(cmd).decode('utf-8').strip()
    
    # If no audio stream is found or no audio packets are present
    if not (output == '' or output.endswith(',0')): return None
    
    print(f"No audio found in {video_path}. Adding silent audio track...")
    
    # Create a temporary file for the video with silent audio
    temp_output = os.path.join(os.path.dirname(video_path), f"temp_{os.path.basename(video_path)}")
    
    # Get video duration
    duration_cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    duration = float(subprocess.check_output(duration_cmd).decode('utf-8').strip())
    
    # Add silent audio to the video
    silent_cmd = [
        'ffmpeg',
        '-i', video_path,
        '-f', 'lavfi',
        '-i', f'anullsrc=channel_layout=stereo:sample_rate=44100:duration={duration}',
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
        temp_output
    ]
    subprocess.call(silent_cmd)
    
    # Replace the original video with the new one containing silent audio
    os.replace(temp_output, video_path)
    print(f"Silent audio track added to {video_path}")


@torch.no_grad()
def preprocess_videos(video_dir: str, config: CN, output_dir: str, device_id: int, num_worker: int, batch_size: int) -> List[str]:
    assert glob(os.path.join(args.video_dir, '*.mp4')) + glob(os.path.join(args.video_dir, '*.avi')) != [], "No video files found in video_dir (must be .mp4 or .avi)"
    
    processed_video_paths = []
    
    preproc_dir = os.path.join(output_dir, 'preprocess')
    os.makedirs(preproc_dir, exist_ok=True)
    
    feature_dir = os.path.join(output_dir, 'features')
    os.makedirs(feature_dir, exist_ok=True)
    
    # Inspect video files in video_dir
    video_paths = glob(os.path.join(video_dir, '*.mp4')) + glob(os.path.join(video_dir, '*.avi'))
    video_paths = sorted(video_paths)
    
    for video_path in video_paths:
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            continue
    
    print("Preprocess: cut videos...")

    # Add silent audio if video has no audio
    for video_path in video_paths:
        add_silent_audio_if_needed(video_path)
        
    # Align video and audio
    with Pool(num_worker) as p:
        for _ in tqdm(p.imap_unordered(partial(pipline_align, output_dir=preproc_dir), 
                                       video_paths), total=len(video_paths)):
            pass
            
    segment_ids = []
    
    for video_path in video_paths:
        # Cut video and audio into segments
        video_name = os.path.basename(video_path).split('.')[0]
        aligned_audio_path = os.path.join(preproc_dir, 'audio_ori', f"{video_name}.wav")
        
        # Get audio duration
        _audio_align, _sr_align = librosa.load(aligned_audio_path, sr=None)
        audio_duration = librosa.get_duration(y=_audio_align, sr=_sr_align)
        num_segments = int(np.floor(audio_duration / config.data.audio_samples))
        
        segment_ids.extend([f"{video_name}_{onset_idx}_" for onset_idx in range(num_segments)])
        
        # make dummy annotation file
        with open(os.path.join(preproc_dir, f"{video_name}_times.txt"), "w") as f:
            for i in range(num_segments): f.write(f"{i * 10} \n")
    
    # Cut video & resample audio
    with Pool(num_worker) as p:
        for _ in tqdm(p.imap_unordered(partial(pipline_cut, metadata_dir=preproc_dir, preproc_dir=preproc_dir, 
                                            output_dir=feature_dir, sr=config.data.audio_sample_rate, 
                                            fps=config.data.video_fps, duration_target=config.data.audio_samples), 
                                        segment_ids),
                    total=len(segment_ids)):
            pass
    
    segment_ids = [segment_id[:-1] for segment_id in segment_ids] # eliminate last character ('_')
    
    # Extract optical flow
    for segment_id in segment_ids:
        video_segment_path = os.path.join(feature_dir, f"videos_{config.data.audio_samples}s_{config.data.video_fps}fps", 
                                          f"{segment_id}.mp4")
        of_dir = os.path.join(feature_dir, f'OF_{config.data.audio_samples}s_{config.data.video_fps}fps')
        os.makedirs(of_dir, exist_ok=True)
        
        cal_for_frames(
            video_path=video_segment_path,
            output_dir=of_dir,
            n_frames=int(config.data.video_fps * config.data.audio_samples),
            width=config.data.video_width,
            height=config.data.video_height,
            batch_size=batch_size,
            device_id=device_id
        )
        
        processed_video_paths.append(video_segment_path)
    
    with open(os.path.join(feature_dir, 'temp_file_list.txt'), 'w') as f:
        for segment_id in segment_ids: f.write(f"{segment_id}\n")
            
    # Extract RGB / Optical Flow features
    for modality in ['RGB', 'Flow']:
        extract_bn_inception_feature(
            input_dir=of_dir,
            output_dir=os.path.join(feature_dir, f"feature_{modality}"),
            modality=modality,
            test_list=os.path.join(feature_dir, 'temp_file_list.txt'),
            workers=num_worker,
            device_id=device_id
        )
                
    return processed_video_paths


@torch.no_grad()
def generate_audio(processed_video_paths: List[str], prompts: List[str], prompt_type: str, epoch: int, 
                   video2rms_ckpt_dir: str, rms2sound_ckpt_dir: str, config: CN, output_dir: str, 
                   device: torch.device, batch_size: int = 1) -> None:
    print(f"Inference with epoch {epoch}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f'Setting seed: {config.train.seed}')
    set_seed(config.train.seed)

    # Load models
    print('Loading models...')
    video2rms_model, audio_ldm_controlnet = load_models(epoch, video2rms_ckpt_dir, rms2sound_ckpt_dir, config, device)

    if prompt_type == 'text' and (prompts is None or len(prompts) == 0):
        raise ValueError("Text prompts are required when prompt_type is 'text'")
    if prompts is not None and len(prompts) != len(processed_video_paths):
        raise ValueError("Number of prompts must match the number of processed video paths")

    # Process videos in batches
    for i in range(0, len(processed_video_paths), batch_size):
        batch_video_paths = processed_video_paths[i:i+batch_size]
        batch_prompts = prompts[i:i+batch_size] if prompts else None

        # Prepare batch features
        batch_features = []
        for processed_video_path in batch_video_paths:
            feature_dir = '/'.join(os.path.dirname(processed_video_path).split('/')[:-1])
            rgb_feature = np.load(os.path.join(feature_dir, f"feature_RGB", f"{os.path.basename(processed_video_path).split('.')[0]}.pkl"), allow_pickle=True)
            flow_feature = np.load(os.path.join(feature_dir, f"feature_Flow", f"{os.path.basename(processed_video_path).split('.')[0]}.pkl"), allow_pickle=True)
            
            rgb_feature = pad_or_truncate_feature(rgb_feature, config.data.video_samples)
            flow_feature = pad_or_truncate_feature(flow_feature, config.data.video_samples)
            
            combined_feature = np.concatenate([rgb_feature, flow_feature], axis=1)
            batch_features.append(combined_feature)

        batch_features_tensor = torch.from_numpy(np.stack(batch_features)).to(device)

        print('Generating RMS from video features...')
        video2rms_model.eval()
        assert config.data.rms_discretize, 'RMS must be discretized'
        mu_bins = RMS.get_mu_bins(config.data.rms_mu, config.data.rms_num_bins, config.data.rms_min)
        
        with torch.no_grad():
            video2rms_model.parse_batch( (batch_features_tensor, 
                                          torch.zeros(batch_size, config.data.rms_samples).to(device), 
                                          None, None) )
            video2rms_model.forward()

        print('Generating audio...')
        for j, video_path in enumerate(batch_video_paths):
            pred_rms = video2rms_model.pred_rms[j].detach().cpu().numpy()
            pred_rms_undiscretized = RMS.undiscretize_rms(torch.from_numpy(pred_rms.argmax(axis=0)),
                                                        mu_bins, ignore_min=True)
            pred_rms_undiscretized = pred_rms_undiscretized.detach().cpu().unsqueeze(0)
            pred_rms_undiscretized = interpolate_rms_for_rms2sound(pred_rms_undiscretized,
                                                                   audio_len=config.data.audio_samples,
                                                                   sr=config.data.audio_sample_rate,
                                                                   frame_len=1024,
                                                                   hop_len=160)
            
            if prompt_type == 'audio':
                if batch_prompts is None or len(batch_prompts) == 0:
                    # Use ground truth audio as prompt
                    feature_dir = os.path.dirname(os.path.dirname(video_path))
                    gt_audio_path = os.path.join(feature_dir, f'audio_{config.data.audio_samples}s_{config.data.audio_sample_rate}hz_muted', 
                                                 f"{os.path.basename(video_path).replace('.mp4', '.wav')}")
                    prompt_audio, _ = librosa.load(gt_audio_path, sr=config.data.audio_sample_rate)
                else:
                    prompt_audio, _ = librosa.load(batch_prompts[j], sr=config.data.audio_sample_rate)
                prompt_audio = torch.from_numpy(prompt_audio).unsqueeze(0).to(device)
                
                generated_audio = audio_ldm_controlnet.generate(
                    waveform=prompt_audio,
                    rms=pred_rms_undiscretized.to(device)
                )
            else:  # text prompt
                generated_audio = audio_ldm_controlnet.generate(
                    text_prompt=batch_prompts[j],
                    rms=pred_rms_undiscretized.to(device)
                )
            
            # Save results
            video_name = os.path.basename(video_path).split('.')[0]
            audio_output_path = os.path.join(output_dir, 'audio', f'{video_name}_generated_audio.wav')
            save_audio(generated_audio, audio_output_path, sr=config.data.audio_sample_rate)
            src_video_path = os.path.join(feature_dir, f"videos_{config.data.audio_samples}s", 
                                          os.path.basename(video_path))
            video_output_path = os.path.join(output_dir, 'video', f'{video_name}_with_generated_audio.mp4')
            save_video_with_audio(src_video_path, generated_audio, video_output_path, sr=config.data.audio_sample_rate)


    # Free up GPU memory
    del video2rms_model
    del audio_ldm_controlnet
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Infer audio from video files using Video2RMS and AudioLDM models")
    parser.add_argument('-i', '--video_dir', type=str, help='Directory containing input video files',
                        default='./dummy_input')
    parser.add_argument('--prompt', nargs='*', help='List of text prompts or paths to audio prompt files, in order corresponding to sorted video filenames')
    parser.add_argument('-p', '--prompt_type', type=str, choices=['text', 'audio'], required=True, help='Type of prompt (text or audio)')
    parser.add_argument('-v', '--video2rms_ckpt_dir', type=str, help='Directory for Video2RMS model checkpoint',
                        default='./ckpt/video-foley-model')
    parser.add_argument('-r', '--rms2sound_ckpt_dir', type=str, help='Directory for AudioLDM model checkpoint',
                        default='./ckpt/video-foley-model')
    parser.add_argument('-e', '--epoch', type=int, default=500, help='Number of epochs of Video2RMS model')
    parser.add_argument('-o', '--output_dir', type=str, default='./output', help='Directory to save generated results')
    parser.add_argument('-d', '--device_id', type=int, default=0 if torch.cuda.is_available() else -1, help='Device ID to use for computation (-1 for CPU, 0 or positive integer for GPU)')
    # Preprocessing
    parser.add_argument('-n', '--num_workers', type=int, default=32, help='Number of workers for data loading and processing')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size for data loading and processing')
    args = parser.parse_args()

    # Load config
    config = load_config(os.path.join(args.video2rms_ckpt_dir, 'opts.yml'))
    config.data.video_fps = config.data.video_samples / config.data.audio_samples
    config.data.video_width = 344
    config.data.video_height = 256
    config.data.training_files = []
    config.data.test_files = [os.path.join(args.output_dir, 'features', 'temp_file_list.txt')]
    config.data.rgb_feature_dirs = [os.path.join(args.output_dir, 'features', 'feature_rgb_bninception_dim1024_30fps')]
    config.data.flow_feature_dirs = [os.path.join(args.output_dir, 'features', 'feature_flow_bninception_dim1024_30fps')]
    config.freeze()
    
    # Rename videos
    for video_path in glob(os.path.join(args.video_dir, '*.mp4')) + glob(os.path.join(args.video_dir, '*.avi')):
        os.rename(video_path, os.path.join(os.path.dirname(video_path), os.path.basename(video_path).replace('_', '@')))

    processed_video_paths = preprocess_videos(args.video_dir, config, args.output_dir, 
                                              args.device_id, args.num_workers, args.batch_size)
    # processed_video_paths = []
    device = torch.device(f'cuda:{args.device_id}' if args.device_id >= 0 else 'cpu')
    generate_audio(processed_video_paths, args.prompt, args.prompt_type, args.epoch, 
                   args.video2rms_ckpt_dir, args.rms2sound_ckpt_dir, config, args.output_dir, device, args.batch_size)
    
    # Rename videos back to original names
    for video_path in glob(os.path.join(args.video_dir, '*.mp4')) + glob(os.path.join(args.video_dir, '*.avi')):
        os.rename(video_path, os.path.join(os.path.dirname(video_path), os.path.basename(video_path).replace('@', '_')))
    for video_path in glob(os.path.join(args.output_dir, 'video', '*.mp4')) + glob(os.path.join(args.output_dir, 'video', '*.avi')):
        os.rename(video_path, os.path.join(os.path.dirname(video_path), os.path.basename(video_path).replace('@', '_')))
    for audio_path in glob(os.path.join(args.output_dir, 'audio', '*.wav')):
        os.rename(audio_path, os.path.join(os.path.dirname(audio_path), os.path.basename(audio_path).replace('@', '_')))
