#!/bin/bash

export NAME="garden_flythrough"
# export NAME="dl3d140-165f5af8bfe32f70595a1c9393a6e442acf7af019998275144f605b89a306557"
echo "Running job for 5 view img2img"
python demo.py \
    --data_path /home/stone/dev/projects/qc_avatar/stable-virtual-camera/assets_demo_cli \
    --data_items $NAME \
    --task img2img \
    --num_inputs 5 \
    --video_save_fps 10 \
    --T 16
mv work_dirs ${NAME}_img2img_5_T16
echo "Completed job for img2img 5 views"

