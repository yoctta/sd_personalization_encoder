import argparse
import itertools
import math
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import DistributedDataParallelKwargs
import PIL
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from dataset import Simpledataset
from diffusers.loaders import AttnProcsLayers
from diffusers.models.cross_attention import LoRACrossAttnProcessor
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from copy import deepcopy
import itertools
from encoder_model import IDEncoder
logger = get_logger(__name__)

def train_step(accelerator,batch,vae,noise_scheduler,id_encoder,text_encoder,unet,optimizer,lr_scheduler=None,reg_weight=0.01):
    if 'latents' in batch:
        latents=batch['latents']
    else:
        latents = vae.encode(batch["pixel_values"]).latent_dist.sample().detach()
        latents = latents * 0.18215
    noise = torch.randn(latents.shape,device=latents.device,dtype=latents.dtype)
    bsz = latents.shape[0]
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    if 'text_encoder_states' in batch:
        text_encoder_states=batch['text_encoder_states']
    else:
        text_encoder_states=text_encoder(batch["input_ids"])[0]
    id_f=id_encoder(batch,noisy_latents,timesteps,text_encoder_states)
    loss_reg=torch.mean(torch.norm(id_f,dim=1)**2)
    input_placeholder_pos=batch["input_placeholder_pos"].unsqueeze(-1)
    input_ids=batch["input_ids"]
    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])
    hidden_states = text_encoder.text_model.embeddings(input_ids=input_ids, position_ids=None)
    id_f=id_f.unsqueeze(1)
    hidden_states=id_f*input_placeholder_pos*0.1+hidden_states #mul 0.1
    #print("id_f",id_f.shape,"input_placeholder_pos",input_placeholder_pos.shape,"hidden_states",hidden_states.shape)
    bsz, seq_len = input_shape
    causal_attention_mask = text_encoder.text_model._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(hidden_states.device)
    encoder_outputs = text_encoder.text_model.encoder(
        inputs_embeds=hidden_states,
        attention_mask=None,
        causal_attention_mask=causal_attention_mask
    )
    last_hidden_state = encoder_outputs[0]
    encoder_hidden_states = text_encoder.text_model.final_layer_norm(last_hidden_state)
    # Predict the noise residual
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    loss_simple = F.mse_loss(model_pred, target)
    loss=loss_simple+reg_weight*loss_reg
    accelerator.backward(loss)
    optimizer.step()
    if lr_scheduler is not None:
        lr_scheduler.step()
    optimizer.zero_grad()
    return dict(loss_simple=loss_simple.detach().item(),loss_reg=loss_reg.detach().item())


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--images_dir", type=str, default=None, required=True, help="A folder containing the training data."
    )
    parser.add_argument("--lora_rank",type=int,default=32)
    parser.add_argument("--placeholder_token", type=str, default='person')
    parser.add_argument("--resolution",type=int,default=768)
    parser.add_argument("--max_train_steps",type=int,default=None)
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--output_dir",type=str,default='runs')
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1)
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=True,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--reg_weight", type=float, default=0.01)
    parser.add_argument("--save_steps", type=int, default=50000)
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank


    return args


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir
        #kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )


    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer",use_auth_token=True)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder",use_auth_token=True)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae",use_auth_token=True)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet",use_auth_token=True)
    ######### id_encoder
    id_encoder=IDEncoder(unet_model=args.pretrained_model_name_or_path)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    for i in [unet,vae,text_encoder]:
        freeze_params(i.parameters())
        i.eval()
        i.to(accelerator.device)
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRACrossAttnProcessor(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,rank=args.lora_rank
        )

    unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(unet.attn_processors)
    accelerator.register_for_checkpointing(lora_layers)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        itertools.chain(id_encoder.parameters(),lora_layers.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

    train_dataset = Simpledataset(
        args.images_dir,
        tokenizer,
        id_encoder.feature_extractor.preprocess,
        size=args.resolution,
        placeholder_token=args.placeholder_token
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    id_encoder,lora_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        id_encoder,lora_layers, optimizer, train_dataloader, lr_scheduler
    )
    id_encoder.module.feature_extractor.set_device(accelerator.device, dtype=weight_dtype)
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
    
    torch.cuda.empty_cache()
    #accelerator.free_memory()
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("personalization_encoder", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    for epoch in range(first_epoch,args.num_train_epochs):
        unet.train()
        id_encoder.train()
        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            losses=train_step(accelerator,batch,vae,noise_scheduler,id_encoder,text_encoder,unet,optimizer,lr_scheduler,args.reg_weight)
            progress_bar.update(1)
            global_step += 1
            if global_step % args.save_steps == 0:
                if accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

            logs = {"loss_simple": losses['loss_simple'],"loss_reg": losses['loss_reg'], "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            if global_step >= args.max_train_steps:
                break
        accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        save_path=os.path.join(args.output_dir,f"checkpoint-{global_step}")
        os.makedirs(save_path,exist_ok=True)
        accelerator.unwrap_model(unet).save_attn_procs(os.path.join(save_path,"LORA_module"))
        state_dict = accelerator.unwrap_model(id_encoder).state_dict()
        torch.save(state_dict, os.path.join(save_path, "id_encoder.pth"))
    accelerator.end_training()

def load_model(path,model="stabilityai/stable-diffusion-2-1",final_ckpt=True):
    id_encoder=IDEncoder(unet_model=model)
    pipe = DiffusionPipeline.from_pretrained(model)
    if final_ckpt:
        id_encoder.load_state_dict(torch.load(os.path.join(path,"id_encoder.pth"),map_location='cpu'))
        pipe.unet.load_attn_procs(os.path.join(path,"LORA_module"))
    else:
        id_encoder.load_state_dict(torch.load(os.path.join(path,"pytorch_model.bin"),map_location='cpu'))
        pipe.unet.load_attn_procs(os.path.join(path,"pytorch_model_1.bin"))
    return id_encoder,pipe

def finetune(image,pipe,id_encoder,mixed_precision='no',learning_rate=1e-6,train_batch_size=1,train_steps=15,\
    text='a photo of person',placeholder_token='person',resize=768,prompts=None,output_dir='',num_samples=2,train_text_encoder=False,reg_weight=0.1):
    tokenizer=pipe.tokenizer
    text_encoder=pipe.text_encoder
    vae=pipe.vae.eval()
    unet=pipe.unet
    accelerator = Accelerator(mixed_precision=mixed_precision)
    raw_image=image
    with torch.no_grad():
        image = image.resize((resize, resize))
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        latents = vae.encode(image).latent_dist.sample().repeat(train_batch_size,1,1,1)
        latents = latents * 0.18215
        input_ids=tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.repeat(train_batch_size,1)
        place_holder_id=tokenizer.encode(placeholder_token)[1]
        text_encoder_states=text_encoder(input_ids)[0]
        input_placeholder_pos=input_ids==place_holder_id
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    for module in [unet, text_encoder,id_encoder]:
        module.train()
        for param in module.parameters():
            param.requires_grad = True
    unet.enable_gradient_checkpointing()
    #learning_rate = learning_rate *train_batch_size * accelerator.num_processes
    optimizer = torch.optim.AdamW(itertools.chain(id_encoder.parameters(),unet.parameters(),text_encoder.text_model.encoder.parameters(),text_encoder.text_model.final_layer_norm.parameters())\
        ,lr=learning_rate,betas=(0.9,0.999),weight_decay=1e-2,eps=1e-8)
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    image_features=id_encoder.feature_extractor.encode_image(id_encoder.feature_extractor.preprocess_images(raw_image)).repeat(train_batch_size,1)
    id_encoder.feature_extractor.set_device(accelerator.device, dtype=weight_dtype,only_unet=True)
    if train_text_encoder:
        text_encoder.gradient_checkpointing_enable()
        id_encoder,unet,encoder_layers,final_layer_norm, optimizer = accelerator.prepare(id_encoder,unet,text_encoder.text_model.encoder,text_encoder.text_model.final_layer_norm,optimizer)
        text_encoder.text_model.encoder=encoder_layers
        text_encoder.text_model.final_layer_norm=final_layer_norm
        pipe.text_encoder.text_model.embeddings.to(device=accelerator.device,dtype=weight_dtype)
    else:
        id_encoder,unet, optimizer = accelerator.prepare(id_encoder,unet,optimizer)
        text_encoder.eval().to(device=accelerator.device)
        for param in text_encoder.parameters():
            param.requires_grad = False
    batch=dict(latents=latents.to(accelerator.device),text_encoder_states=text_encoder_states.to(accelerator.device),input_ids=input_ids.to(accelerator.device),\
            input_placeholder_pos=input_placeholder_pos.to(accelerator.device),image_features=image_features.to(accelerator.device))
    progress_bar = tqdm(range(train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    for step in range(train_steps):
        loss=train_step(accelerator,batch,vae,noise_scheduler,id_encoder,text_encoder,unet,optimizer,None,reg_weight=reg_weight)
        progress_bar.update(1)
        progress_bar.set_postfix(**loss)
    accelerator.wait_for_everyone()
    if train_text_encoder:
        pipe.text_encoder.text_model.encoder=accelerator.unwrap_model(text_encoder.text_model.encoder).eval().to(accelerator.device,dtype=weight_dtype)
        pipe.text_encoder.text_model.final_layer_norm=accelerator.unwrap_model(text_encoder.text_model.final_layer_norm).eval().to(accelerator.device,dtype=weight_dtype)
    else:
        text_encoder.to(dtype=weight_dtype)
    pipe.unet=accelerator.unwrap_model(unet).eval().to(accelerator.device,dtype=weight_dtype)
    id_encoder=accelerator.unwrap_model(id_encoder).eval().to(accelerator.device,dtype=weight_dtype)
    vae.to(device=accelerator.device,dtype=weight_dtype)
    if type(prompts)==str:
        sample(id_encoder,pipe,(resize,resize),prompts,accelerator,num_samples,output_dir,place_holder_id,batch['image_features'][:1].to(dtype=weight_dtype),batch['latents'][:1].to(dtype=weight_dtype))
    elif type(prompts)==list:
        for prompt in prompts:
            sample(id_encoder,pipe,(resize,resize),prompt,accelerator,num_samples,output_dir,place_holder_id,batch['image_features'][:1].to(dtype=weight_dtype),batch['latents'][:1].to(dtype=weight_dtype))

@torch.no_grad()
def sample(id_encoder,pipe,image_size,prompt,accelerator,num_samples,output_dir,place_holder_id,image_features,source_latents,guidance_scale=7.5,num_inference_steps=100):
    #pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    height,width = image_size
    if prompt is not None and isinstance(prompt, str):
        prompt=[prompt]
    if prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    device = accelerator.device
    weight_dtype=pipe.text_encoder.dtype
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    text_inputs=torch.repeat_interleave(text_inputs,num_samples,dim=0)
    do_classifier_free_guidance = guidance_scale > 1.0
    if do_classifier_free_guidance:
        uncond_tokens = [""] * batch_size
        uncond_input = pipe.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        negative_prompt_embeds = torch.repeat_interleave(pipe.text_encoder(uncond_input.input_ids.to(device),None)[0],num_samples,dim=0)
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
    num_channels_latents = pipe.unet.in_channels
    latents = pipe.prepare_latents(batch_size * num_samples,num_channels_latents,height,width,weight_dtype,device,None)
    source_latents=source_latents.repeat(batch_size * num_samples,1,1,1)
    input_placeholder_pos=(text_inputs==place_holder_id).to(device) #BxL
    if accelerator.is_main_process:
        print(input_placeholder_pos)
        os.makedirs(output_dir,exist_ok=True)
    prompt_embeds = pipe.text_encoder(text_inputs.to(device),None)[0]
    batch_input=dict(image_features=image_features.repeat(num_samples,1))
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(None, 0.0)
    noise = torch.randn(latents.shape,device=latents.device,dtype=latents.dtype)
    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            #id_f=id_encoder(batch_input,pipe.scheduler.add_noise(source_latents, noise, t),t,prompt_embeds) #BxD
            id_f=id_encoder(batch_input,latents,t,prompt_embeds)*0.1 ##mul 0.1
            if do_classifier_free_guidance:
                noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds+id_f.unsqueeze(1)*input_placeholder_pos.unsqueeze(-1)])).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds+id_f.unsqueeze(1)*input_placeholder_pos.unsqueeze(-1)).sample
            latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()
    image = pipe.decode_latents(latents)
    image = pipe.numpy_to_pil(image)
    for i in range(batch_size):
        for j in range(num_samples):
            rank=num_samples*accelerator.process_index+j
            image[i*num_samples+j].save(os.path.join(output_dir,f'{prompt[i]}-{rank}.png'))
        

if __name__ == "__main__":
    main()
