# set epoch and micro variable
epoch=500
average=micro
ckpt_dirs=(
    "/home/junwon/video-foley-model"
    # Add more ckpt_dirs here
)

for ckpt_dir in ${ckpt_dirs[@]}; do
    CUDA_VISIBLE_DEVICES=0 python evaluate.py -c ${ckpt_dir} -e ${epoch} -a ${average} # --preserve_correct_0 # E-L1
done
