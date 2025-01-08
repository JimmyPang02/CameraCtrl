#!/bin/bash

# your path
OUTPUT_PATH='/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/project/CameraCtrl/output' 
SD15_PATH='/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/model/stable-diffusion-v1-5'
SUBFOUDER_NAME='unet'
ADV3_MM_CKPT='/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/model/animatediff/v3_sd15_mm.ckpt'
CAMERACTRL_CKPT='/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/model/cameractrl/CameraCtrl.ckpt'


torchrun --nproc_per_node=1 --master_port=25000 inference.py \
      --out_root ${OUTPUT_PATH} \
      --ori_model_path ${SD15_PATH} \
      --unet_subfolder ${SUBFOUDER_NAME} \
      --motion_module_ckpt ${ADV3_MM_CKPT} \
      --pose_adaptor_ckpt ${CAMERACTRL_CKPT} \
      --model_config configs/train_cameractrl/adv3_256_384_cameractrl_relora.yaml \
      --visualization_captions assets/cameractrl_prompts.json \
      --use_specific_seeds \
      --trajectory_file assets/pose_files/0f47577ab3441480.txt \
      --n_procs 1 