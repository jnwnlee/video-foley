import os
from glob import glob
import numpy as np
from tqdm import tqdm
import os.path as P
import argparse
from multiprocessing import Pool
from functools import partial

def execCmd(cmd):
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text

def pipline_align(video_path, output_dir):
    video_name = '.'.join(os.path.basename(video_path).split('.')[:-1]).replace('_denoised', '') + ".mp4"
    audio_name = video_name.replace(".mp4", ".wav")

    # Extract Original Audio
    ori_audio_dir = P.join(output_dir, "audio_ori")
    os.makedirs(ori_audio_dir, exist_ok=True)
    os.system(f"ffmpeg -i {video_path} -loglevel error -f wav -vn -y {P.join(ori_audio_dir, audio_name)}")

    # Cut Video According to Audio
    align_video_dir = P.join(output_dir, "videos_align")
    os.makedirs(align_video_dir, exist_ok=True)
    duration = execCmd(f"ffmpeg -i {P.join(ori_audio_dir, audio_name)}  2>&1 | grep 'Duration' | cut -d ' ' -f 4 | sed s/,//")
    duration = duration.replace('\n', "")
    os.system("ffmpeg -ss 0 -t {} -i {} -loglevel error -c:v libx264 -c:a aac -strict experimental -b:a 98k -y {}".format(
            duration, video_path, P.join(align_video_dir, video_name)))


def pipline_cut(video_id, metadata_dir, preproc_dir, output_dir, fps, sr, duration_target):
    video_name, onset_id, material = video_id.split("_")
    # video_name = os.path.basename(video_path)
    # audio_name = video_name.replace(".mp4", ".wav")

    # Cut Video
    cut_video_dir = P.join(output_dir, material, f"videos_{duration_target}s")
    os.makedirs(cut_video_dir, exist_ok=True)
    os.system("ffmpeg -ss {} -to {} -i {} -loglevel error -c:v libx264 -c:a aac -strict experimental -b:a 98k -y {}".format(
            int(onset_id)*duration_target, (int(onset_id)+1)*duration_target, 
            P.join(preproc_dir, "videos_align", video_name+".mp4"), P.join(cut_video_dir, video_name+'_'+onset_id+".mp4")))
    video_name = video_name+'_'+onset_id+".mp4"
    audio_name = video_name.replace(".mp4", ".wav")
    # assert video length is equal to duration_target
    margin_of_error = 0.05
    video_length = float(execCmd(f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {P.join(cut_video_dir, video_name)}").strip()) 
    assert abs(video_length - duration_target) <= margin_of_error, f"{video_name} length: {video_length}"
    

    # Extract Audio
    cut_audio_dir = P.join(output_dir, material, f"audio_{duration_target}s")
    os.makedirs(cut_audio_dir, exist_ok=True)
    os.system("ffmpeg -i {} -loglevel error -f wav -vn -y {}".format(
            P.join(cut_video_dir, video_name), P.join(cut_audio_dir, audio_name)))

    # change audio sample rate
    sr_audio_dir = P.join(output_dir, material, f"audio_{duration_target}s_{sr}hz")
    os.makedirs(sr_audio_dir, exist_ok=True)
    os.system("ffmpeg -i {} -loglevel error -ac 1 -ab 16k -ar {} -y {}".format(
            P.join(cut_audio_dir, audio_name), sr, P.join(sr_audio_dir, audio_name)))
    
    # mute audio before the first onset
    annotation_path = P.join(metadata_dir, f"{video_name.split('_')[0]}_times.txt")
    with open(annotation_path, "r") as f:
        for line in f.readlines():
            if float(line.strip().split()[0]) >= int(onset_id)*duration_target:
                first_onset = float(line.strip().split()[0]) % duration_target
                if first_onset >= duration_target:
                    raise ValueError(f"{video_id} first_onset: {first_onset}, duration_target: {duration_target}")
                break
    muted_audio_dir = P.join(output_dir, material, f"audio_{duration_target}s_{sr}hz_muted")
    os.makedirs(muted_audio_dir, exist_ok=True)
    os.system("ffmpeg -i {} -loglevel error -af \"volume=enable='between(t,0,{})':volume=0\" -y {}".format(
        P.join(sr_audio_dir, audio_name), first_onset, P.join(muted_audio_dir, audio_name)))

    # change video fps
    fps_audio_dir = P.join(output_dir, material, f"videos_{duration_target}s_{fps}fps")
    os.makedirs(fps_audio_dir, exist_ok=True)
    os.system("ffmpeg -y -i {} -loglevel error -r {} -c:v libx264 -strict -2 {}".format(
            P.join(cut_video_dir, video_name), fps, P.join(fps_audio_dir, video_name)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", default="/media/daftpunk4/dataset/GreatestHits/vis-data")
    parser.add_argument("-p", "--preproc_dir", default="/media/daftpunk4/dataset/GreatestHits/features_preproc")
    parser.add_argument("-o", "--output_dir", default="/media/daftpunk4/dataset/GreatestHits/features")
    parser.add_argument("-d", "--duration", type=int, default=10)
    parser.add_argument("-a", '--audio_sample_rate', type=str, default='16000')
    parser.add_argument("-v", '--video_fps', type=str, default='30')
    parser.add_argument("-n", '--num_worker', type=int, default=32)
    args = parser.parse_args()
    input_dir = args.input_dir
    preproc_dir = args.preproc_dir
    output_dir = args.output_dir
    duration_target = args.duration
    sr = args.audio_sample_rate
    fps = args.video_fps
    
    # video_paths = glob(P.join(input_dir, "*_denoised.mp4"))
    # video_paths.sort()

    # # Align Video and Audio
    # with Pool(args.num_worker) as p:
    #     for _ in tqdm(p.imap_unordered(partial(pipline_align, output_dir=preproc_dir, 
    #                                            sr=sr, fps=fps, duration_target=duration_target), video_paths), 
    #                   total=len(video_paths)):
    #         pass

    filelists = glob(P.join(preproc_dir, "filelists", "*.txt"))
    video_ids = []
    for filelist in filelists:
        material = P.basename(filelist).split('_')[0]
        with open(filelist, "r") as f:
            video_ids.extend([line.strip()+'_'+material for line in f.readlines()])
        
    # Cut and Change Rate
    with Pool(args.num_worker) as p:
        for _ in tqdm(p.imap_unordered(partial(pipline_cut, metadata_dir=input_dir, preproc_dir=preproc_dir, 
                                               output_dir=output_dir, sr=sr, fps=fps, duration_target=duration_target), video_ids),
                      total=len(video_ids)):
            pass
