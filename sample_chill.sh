accelerate launch --multi_gpu sample.py --pretrained_model_name_or_path "/ssd/zhaohanqing/msws/diffusers/examples/model_lab/chilloutmix" \
  --model_path "chillout_train_ffhq/checkpoint-20000"  \
  --image_path "images/asuka.png" \
  --train_batch_size 2 \
  --finetune_steps 15 --reg_weight 0.1 --resolution 512 \
  --prompt "a pencil sketch of face." "an oil paint of face." "a photo of face surrounded by sunflowers" "a photo of face in red shirt" "a photo of face" --placeholder_token face \
  --num_samples 2 --learning_rate 1.6e-5 --train_text_encoder --mixed_precision bf16 \
  --output_dir chillout_train_ffhq/checkpoint-20000/samples_lr16_reg_0.1_f15_asuka_bf16

