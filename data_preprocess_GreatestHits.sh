directory="path/to/dataset/GreatestHits"  # Path to the directory containing folders
soundlist=""  # Initialize soundlist as an empty string

# Split training/testing from given list
python preprocess/gen_list.py \
-i ${directory}/vis-data \
-o ${directory}/features_preproc/filelists \
-d 10

# Data preprocessing. We will first pad all videos to 10s and change video FPS and audio
# sampling rate.
python preprocess/extract_audio_and_video.py \
-i ${directory}/vis-data \
-p ${directory}/features_preproc \
-o ${directory}/features \
--duration 10 --audio_sample_rate 16000 --video_fps 30 --num_worker 20

# Loop through each folder in the specified directory and add it to soundlist
for folder in "$directory"/features/*/; do
    foldername=$(basename "$folder")  # Extract the name of the folder
    soundlist+="$foldername "  # Append the folder name to soundlist, enclosed in quotes
done

echo $soundlist  # Print the list of sound types

for soundtype in $soundlist 
do

    # Generating RGB frame and optical flow.
    CUDA_VISIBLE_DEVICES=0 python preprocess/extract_rgb_flow_raft.py \
    -i ${directory}/features/${soundtype}/videos_10s_30fps \
    -o ${directory}/features/${soundtype}/OF_10s_30fps \
    -f 30 -l 10 -n 16 -d 0 -b 32

    #Extract Mel-spectrogram from audio
    python preprocess/extract_mel_spectrogram.py \
    -i ${directory}/features/${soundtype}/audio_10s_16000hz \
    -o ${directory}/features/${soundtype}/melspec_10s_16000hz \
    -l 160000 -n 20

    # Extract RGB feature
    CUDA_VISIBLE_DEVICES=0 python -m preprocess.extract_feature \
    -t filelists/GreatestHits/${soundtype}_train.txt \
    -m RGB \
    -i ${directory}/features/${soundtype}/OF_10s_30fps \
    -o ${directory}/features/${soundtype}/feature_rgb_bninception_dim1024_30fps \
    -j 16

    CUDA_VISIBLE_DEVICES=0 python -m preprocess.extract_feature \
    -t filelists/GreatestHits/${soundtype}_test.txt \
    -m RGB \
    -i ${directory}/features/${soundtype}/OF_10s_30fps \
    -o ${directory}/features/${soundtype}/feature_rgb_bninception_dim1024_30fps \
    -j 16

    #Extract optical flow feature
    CUDA_VISIBLE_DEVICES=0 python -m preprocess.extract_feature \
    -t filelists/GreatestHits/${soundtype}_train.txt \
    -m Flow \
    -i ${directory}/features/${soundtype}/OF_10s_30fps \
    -o ${directory}/features/${soundtype}/feature_flow_bninception_dim1024_30fps \
    -j 16

    CUDA_VISIBLE_DEVICES=0 python -m preprocess.extract_feature \
    -t filelists/GreatestHits/${soundtype}_test.txt \
    -m Flow \
    -i ${directory}/features/${soundtype}/OF_10s_30fps \
    -o ${directory}/features/${soundtype}/feature_flow_bninception_dim1024_30fps \
    -j 16

done
