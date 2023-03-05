accelerate launch --multi_gpu sample.py --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1" \
  --model_path "new_train_ffhq/checkpoint-140000" --final_checkpoint \
  --image_path "images/asuka.png" \
  --train_batch_size 2 \
  --finetune_steps 15 --reg_weight 0.1 --resolution 768 \
  --prompt "a pencil sketch of person." "an oil paint of person." "a photo of person surrounded by sunflowers" "a photo of person in red shirt" "a photo of person" \
  --num_samples 2 --learning_rate 1.6e-5  --train_text_encoder --mixed_precision bf16 \
  --output_dir new_train_ffhq/checkpoint-140000/samples_lr16_reg_0.1_f15_asuka_bf16

