from typing import Union, List
from glob import glob
import os
import soundfile as sf
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
import librosa

from config import _C as config
from data_utils import VideoAnnotation, RMS
from audioldm_train.conditional_models import CLAPAudioEmbeddingClassifierFreev2

def _check_waveform_shape(waveform:torch.tensor):
    # waveform.shape: [batch, t_steps]
    if not len(waveform.shape) == 2:
        if len(waveform.shape) == 1:
            return waveform.unsqueeze(0)
        else:
            return None
    return waveform

def crop_or_pad(waveform:torch.tensor, target_len:int) -> torch.tensor:
    assert len(waveform.shape) == 2, f"waveform.shape: {waveform.shape}" # batch x samples
    if waveform.shape[1] > target_len:
        return waveform[:, :target_len]
    elif waveform.shape[1] < target_len:
        return F.pad(waveform, (0, target_len - waveform.shape[1]), mode='constant', value=0)
    assert waveform.shape[1] == target_len, f"waveform.shape: {waveform.shape}"
    return waveform
    
def cos_similarity_audio(clap_model, waveform1, waveform2) -> torch.tensor:
    # waveform: [bs, t_steps]
    waveform1 = _check_waveform_shape(waveform1)
    waveform2 = _check_waveform_shape(waveform2)
    if waveform1 is None:
        raise ValueError("waveform1 shape is not valid")
    if waveform2 is None:
        raise ValueError("waveform2 shape is not valid")
        
    original_embed_mode = clap_model.embed_mode
    with torch.no_grad():
        clap_model.embed_mode = "audio"
        audio_emb1 = clap_model(waveform1.cuda())
        audio_emb2 = clap_model(waveform2.cuda())
        similarity = F.cosine_similarity(audio_emb1, audio_emb2, dim=2)
    clap_model.embed_mode = original_embed_mode
    return similarity.squeeze()

def get_clap_score(prompt:Union[List[str], torch.Tensor, np.ndarray], generated:Union[torch.Tensor, np.ndarray],
                    clap_pretrained_path:str='', sr:int=16000, device:str='cuda:0') -> torch.tensor:
    clap = CLAPAudioEmbeddingClassifierFreev2(
        pretrained_path=clap_pretrained_path,
        sampling_rate=sr,
        embed_mode="audio",
        amodel="HTSAT-base",
    ).to(device)

    if isinstance(prompt, list):
        similarity = clap.cos_similarity(
            torch.FloatTensor(generated).to(device), prompt
        )
    elif isinstance(prompt, torch.Tensor) or isinstance(prompt, np.ndarray):
        if isinstance(prompt, np.ndarray):
            prompt = torch.FloatTensor(prompt)
        similarity = cos_similarity_audio(
            clap,
            torch.FloatTensor(generated).to(device), torch.FloatTensor(prompt).to(device)
        )
    else :
        raise ValueError(f"prompt type {type(prompt)} is not supported")
    
    return similarity

def get_EL1(ground_truth:Union[torch.Tensor, np.ndarray], generated:Union[torch.Tensor, np.ndarray],
            rms_len:int, nframes:int, hop:int) \
    -> torch.tensor:
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy()
    if isinstance(generated, torch.Tensor):
        generated = generated.cpu().numpy()
    assert ground_truth.shape == generated.shape, f"ground_truth.shape: {ground_truth.shape}, generated.shape: {generated.shape}"
        
    ground_truth_rms = np.array([RMS.get_rms(ground_truth[i], nframes, hop) for i in range(ground_truth.shape[0])])
    generated_rms = np.array([RMS.get_rms(generated[i], nframes, hop) for i in range(generated.shape[0])])
    
    assert ground_truth_rms.shape == generated_rms.shape == (ground_truth.shape[0], rms_len), \
        f"ground_truth_rms.shape: {ground_truth_rms.shape}, generated_rms.shape: {generated_rms.shape}"
    
    return F.l1_loss(torch.FloatTensor(ground_truth_rms), torch.FloatTensor(generated_rms), reduction='mean')

def get_audio_files(target_dir:str) -> List[str]:
    audio_file_list = list(os.listdir(target_dir))
    audio_file_list = [audio_file for audio_file in audio_file_list if audio_file.endswith(".wav")]
    assert len(set(audio_file_list)) == len(audio_file_list), "There are duplicated audio files"
    return audio_file_list

def read_audio(audio_file:str, sample_rate:int=16000) -> torch.tensor:
    audio, sr_file = sf.read(audio_file)
    # assert sr_file == sample_rate, f"sample rate is not matched: {sr_file} != {sample_rate}"
    if not sr_file == sample_rate:
        print(f"Resample audio {audio_file}: {sr_file} -> {sample_rate}")
        audio = librosa.resample(audio, orig_sr=sr_file, target_sr=sample_rate)
    return torch.from_numpy(audio).float()
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--el1', action='store_true', default=False, help='calculate EL1')
    parser.add_argument('--clap', action='store_true', default=False, help='calculate CLAP')
    parser.add_argument('--text', action='store_true', default=False, help='calculate text-audio similarity')
    parser.add_argument('--clap_pretrained_path', type=str, 
                        default='./ckpt/clap_music_speech_audioset_epoch_15_esc_89.98.pt', 
                        help='CLAP pretrained path')
    parser.add_argument('--generated_dir', type=str, required=True, help='generated audio directory')
    parser.add_argument('--ground_truth_dir', type=str, required=True, help='ground truth audio directory')
    parser.add_argument('--csv_path', type=str, default='./eval_v2s_audio.csv', help='csv file path')
    parser.add_argument('--annotation_dir', type=str, help='annotation directory',
                        default='/mnt/GreatestHits/vis-data/')
    
    args = parser.parse_args()
    
    config.freeze()

    print('Check audio files...')
    generated_audio_list = sorted(get_audio_files(args.generated_dir))
    ground_truth_audio_list = sorted(get_audio_files(args.ground_truth_dir))

    # check if two lists have same values
    assert len(generated_audio_list) == len(ground_truth_audio_list), f"Different number of files in two directories. {len(generated_audio_list)} != {len(ground_truth_audio_list)}"
    assert generated_audio_list == ground_truth_audio_list, f"Different files in two directories:\n{set(generated_audio_list) - set(ground_truth_audio_list)},\n{set(ground_truth_audio_list) - set(generated_audio_list)}"


    print('Calculate scores...')
    if args.el1:
        el1_list = []
    if args.clap:
        similarity_list = []
    # batch processing
    for i in tqdm(range(0, len(generated_audio_list), args.batch_size)):
        generated_audio_list_batch = generated_audio_list[i:min(i+args.batch_size, len(generated_audio_list))]
        ground_truth_audio_list_batch = ground_truth_audio_list[i:min(i+args.batch_size, len(generated_audio_list))]
        generated_audio_batch = torch.stack([read_audio(os.path.join(args.generated_dir, generated_audio_fname), 
                                                        sample_rate=config.data.audio_sample_rate) 
                                            for generated_audio_fname in generated_audio_list_batch], dim=0)
        ground_truth_audio_batch = torch.stack([read_audio(os.path.join(args.ground_truth_dir, ground_truth_audio_fname), 
                                                           sample_rate=config.data.audio_sample_rate) 
                                                for ground_truth_audio_fname in ground_truth_audio_list_batch], dim=0)
        if args.text:
            text_prompt_batch = []
            for generated_audio_fname in generated_audio_list_batch:
                videoname, index = generated_audio_fname.replace('.wav', '').split('_')
                text_prompt_batch.append(VideoAnnotation.get_text_prompt(
                    annot_dir=args.annotation_dir, 
                    videoname=videoname, 
                    index=int(index), 
                    length=10
                ))
        
        if args.clap:
            if config.data.audio_samples <= 10:
                waveform_sample_len = int(config.data.audio_sample_rate * 10.24)
            else:
                waveform_sample_len = config.data.audio_samples * config.data.audio_sample_rate
            ground_truth_audio_batch = crop_or_pad(ground_truth_audio_batch, waveform_sample_len)
            generated_audio_batch = crop_or_pad(generated_audio_batch, waveform_sample_len)
            
            if args.text:
                similarity_batch = get_clap_score(text_prompt_batch, generated_audio_batch, 
                                                clap_pretrained_path=args.clap_pretrained_path, 
                                                sr=config.data.audio_sample_rate, device=args.device)
            else:
                similarity_batch = get_clap_score(ground_truth_audio_batch, generated_audio_batch, 
                                                clap_pretrained_path=args.clap_pretrained_path, 
                                                sr=config.data.audio_sample_rate, device=args.device)
            similarity_list.extend(similarity_batch.tolist())
        
        if args.el1:
            waveform_sample_len = config.data.audio_samples * config.data.audio_sample_rate
            generated_audio_batch = crop_or_pad(generated_audio_batch, waveform_sample_len)
            ground_truth_audio_batch = crop_or_pad(ground_truth_audio_batch, waveform_sample_len)
            
            el1_batch = get_EL1(ground_truth_audio_batch, generated_audio_batch,
                                rms_len=config.data.rms_samples, nframes=config.data.rms_nframes, hop=config.data.rms_hop)
            el1_list.extend([el1_batch.item()] * len(generated_audio_list_batch))

        
    print('Result...')
    print(f'# of audio files: {len(generated_audio_list)}')
    if args.el1:
        print(f"Mean EL1: {np.mean(el1_list)}")
        print(f"Std EL1: {np.std(el1_list)}")
    if args.clap:
        print(f"Mean similarity: {np.mean(similarity_list)}")
        print(f"Std similarity: {np.std(similarity_list)}")
    # save it in csv (ground_truth_dir, generated_dir, el1_mean, el1_std, similarity_mean, similarity_std)
    df = pd.DataFrame({
        'ground_truth_dir': [args.ground_truth_dir],
        'generated_dir': [args.generated_dir],
        'el1_mean': [np.mean(el1_list)] if args.el1 else [None],
        'el1_std': [np.std(el1_list)] if args.el1 else [None],
        'similarity_mean': [np.mean(similarity_list)] if args.clap else [None],
        'similarity_std': [np.std(similarity_list)] if args.clap else [None]
    })
    # if file does not exist write header
    if not os.path.isfile(args.csv_path):
        df.to_csv(args.csv_path, header='column_names')
    else:
        df.to_csv(args.csv_path, mode='a', header=False)