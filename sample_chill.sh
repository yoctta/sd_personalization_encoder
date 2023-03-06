accelerate launch --multi_gpu sample.py --pretrained_model_name_or_path "/ssd/zhaohanqing/msws/diffusers/examples/model_lab/chilloutmix" \
  --model_path "chillout_train_ffhq/checkpoint-70000" --final_checkpoint \
  --image_path "temp/images/asuka.png" \
  --train_batch_size 6 \
  --finetune_steps 15 --reg_weight 0.1 --resolution 512 \
  --prompt "a photo of face with sunglasses." "an photo of face on the beach." "a photo of face surrounded by sunflowers" "a photo of face in red shirt"  --placeholder_token face \
  --num_samples 6 --learning_rate 1.6e-5 --train_text_encoder --mixed_precision bf16 \
  --output_dir temp/samples_lr16_reg_0.1_f15_asuka_bf16

