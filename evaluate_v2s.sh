
gt_audio_dir=path/to/gt
gen_dir=./
gen_folders=(
    "Video-Foley/audio_prompt"
    "Video-Foley/text_prompt"
    # Add more gen_folders here
)

source ~/.bashrc
conda deactivate
conda activate v2s
for gen_folder in ${gen_folders[@]}; do
    CUDA_VISIBLE_DEVICES=0 python evaluate_v2s.py \
        --el1 --clap \
        --clap_pretrained_path ./ckpt/clap_music_speech_audioset_epoch_15_esc_89.98.pt \
        --ground_truth_dir ${gt_audio_dir} \
        --generated_dir ${gen_dir}${gen_folder}/audio \
        --csv_path eval_v2s_audio.csv
done


# (if you made a separate conda environment for FADTK)
# conda deactivate
# conda activate fadtk
for gen_folder in ${gen_folders[@]}; do
    CUDA_VISIBLE_DEVICES=0 fadtk panns-wavegram-logmel \
    ${gt_audio_dir} ${gen_dir}${gen_folder}/audio eval_v2s_fad.csv --inf
done

for gen_folder in ${gen_folders[@]}; do
    CUDA_VISIBLE_DEVICES=0 fadtk clap-2023 \
    ${gt_audio_dir} ${gen_dir}${gen_folder}/audio eval_v2s_fad.csv --inf
done