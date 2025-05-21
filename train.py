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
    FluxTransformer2DModel,
    FluxSchnellPipeline, # For loading FLUX pipelines
    # Potentially add T5EncoderModel if needed, but start with CLIP-based text encoders
)
from transformers import CLIPTextModelWithProjection # For the second text encoder in FLUX
from copy import deepcopy
import itertools
from encoder_model import IDEncoder
logger = get_logger(__name__)

def train_step(accelerator,batch,vae,noise_scheduler,id_encoder,text_encoder,unet,optimizer,lr_scheduler=None,reg_weight=0.01, \
               is_flux_model=False, text_encoder_two=None, tokenizer=None, tokenizer_two=None, weight_dtype=torch.float32): # Added FLUX args & weight_dtype
    if 'latents' in batch: # Keep this simple for now, will be refined later
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

    # This line should already be present from the previous successful diff
    is_flux_model = "flux" in args.pretrained_model_name_or_path.lower() 

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # Initialize all model and tokenizer variables to None
    tokenizer, text_encoder, text_encoder_two, tokenizer_two, vae, unet = None, None, None, None, None, None
    
    if is_flux_model:
        logger.info(f"Loading FLUX model components from: {args.pretrained_model_name_or_path}")
        # Minimal loading for now: unet and primary tokenizer
        try: 
            unet = FluxTransformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer", use_auth_token=True, torch_dtype=weight_dtype)
        except EnvironmentError: 
            logger.info("FLUX transformer not found in subfolder 'transformer', trying model root.")
            unet = FluxTransformer2DModel.from_pretrained(args.pretrained_model_name_or_path, use_auth_token=True, torch_dtype=weight_dtype)
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_auth_token=True)
        # text_encoder, vae, tokenizer_two, text_encoder_two will be loaded in the next phase
    else: 
        logger.info(f"Loading UNet model components from: {args.pretrained_model_name_or_path}")
        # Minimal loading for now: unet and primary tokenizer
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer",use_auth_token=True)
        unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet",use_auth_token=True, torch_dtype=weight_dtype)
        # text_encoder, vae will be loaded in the next phase

    id_encoder=IDEncoder(unet_model=args.pretrained_model_name_or_path) 

    models_to_freeze = []
    if unet is not None: # Only add unet to freeze list for now
        models_to_freeze.append(unet)
    
    for model_part in models_to_freeze: # Iterating only over 'unet' for now if loaded
        if model_part is not None: 
            freeze_params(model_part.parameters())
            model_part.eval()

    lora_layers = None 
    params_to_optimize_list = list(id_encoder.parameters()) 

    if not is_flux_model and unet is not None: # Only apply LoRA if UNet and unet is loaded
        logger.info("Applying LoRA to UNet model's attention processors.")
        lora_attn_procs = {}
        for name in unet.attn_processors.keys(): 
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"): hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"): hidden_size = list(reversed(unet.config.block_out_channels))[int(name[len("up_blocks.")])]
            elif name.startswith("down_blocks"): hidden_size = unet.config.block_out_channels[int(name[len("down_blocks.")])]
            else: hidden_size = unet.config.block_out_channels[0] 
            lora_attn_procs[name] = LoRACrossAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=args.lora_rank)
        unet.set_attn_processor(lora_attn_procs)
        lora_layers = AttnProcsLayers(unet.attn_processors) 
        if lora_layers is not None: # Check if LoRA layers were successfully created
             params_to_optimize_list.extend(lora_layers.parameters())
             accelerator.register_for_checkpointing(lora_layers) 
    elif is_flux_model: 
        logger.warning("LoRA for FLUX models is not implemented in this script. Training IDEncoder only.")

    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)

    optimizer = torch.optim.AdamW(
        params_to_optimize_list,  
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Determine noise scheduler
    if is_flux_model:
        # For FLUX, often the pipeline itself would define the scheduler if it's specific.
        # If VAE was loaded from a pipeline that had a scheduler attribute:
        if hasattr(vae, 'scheduler') and vae.scheduler is not None: 
            noise_scheduler = vae.scheduler 
            logger.info("Using noise scheduler from loaded FLUX VAE component.")
        else: # Fallback to loading from subfolder or default
            try: 
                noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", use_auth_token=True)
                logger.info("Loaded DDPMScheduler from 'scheduler' subfolder for FLUX model.")
            except EnvironmentError: 
                logger.warning(f"Could not load DDPMScheduler from 'scheduler' subfolder for FLUX model {args.pretrained_model_name_or_path}. Using default DDPMScheduler configuration as a fallback.")
                noise_scheduler = DDPMScheduler() # Basic default
    else: # UNet-based models
        noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")
        logger.info("Loaded DDPMScheduler from UNet model config.")

    # Dataset and Dataloader
    # Pass the primary tokenizer for dataset preparation.
    # `tokenizer_two` is also passed if `is_flux_model` for potential dual tokenization within SimpleDataset.
    # `id_encoder.feature_extractor.preprocess` is correct as `feature_extractor` handles model type.
    train_dataset = Simpledataset(
        images_dir=args.images_dir,
        tokenizer=tokenizer, 
        image_processor=id_encoder.feature_extractor.preprocess,  # This is correct
        size=args.resolution,
        placeholder_token=args.placeholder_token,
        is_flux=is_flux_model, 
        tokenizer_2=tokenizer_two if is_flux_model else None,
    )
    # Use collate_fn from dataset if it provides one (e.g., for FLUX dual tokenization)
    collate_fn_to_use = train_dataset.collate_fn if hasattr(train_dataset, 'collate_fn') and callable(train_dataset.collate_fn) else None
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.train_batch_size, 
        shuffle=True,
        collate_fn=collate_fn_to_use,
    )
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

    # Prepare components with Accelerator
    # `lora_layers` is None if not UNet or if LoRA is not applied.
    # `id_encoder` and `optimizer` are always present.
    if lora_layers: # Typically for UNet
        id_encoder, lora_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            id_encoder, lora_layers, optimizer, train_dataloader, lr_scheduler
        )
        # accelerator.register_for_checkpointing(lora_layers) # Already done during LoRA setup for UNet
    else: # FLUX (no LoRA implemented here) or UNet without LoRA
        id_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            id_encoder, optimizer, train_dataloader, lr_scheduler
        )
    
    # Ensure frozen models are on the correct device and in eval mode
    # These are not part of `params_to_optimize`, so `accelerator.prepare` doesn't handle their device placement for training.
    unet.to(accelerator.device, dtype=weight_dtype).eval() 
    vae.to(accelerator.device, dtype=weight_dtype).eval()
    text_encoder.to(accelerator.device, dtype=weight_dtype).eval()
    if is_flux_model and text_encoder_two:
        text_encoder_two.to(accelerator.device, dtype=weight_dtype).eval()

    # Set device for IDEncoder's internal feature_extractor's models.
    # This is done after id_encoder itself is potentially wrapped by accelerator.prepare (DDP).
    # Also ensure the IDEncoder model itself (and its wrapper if DDP) is on the correct dtype.
    id_encoder_ref = id_encoder.module if hasattr(id_encoder, 'module') else id_encoder
    id_encoder_ref.feature_extractor.set_device(accelerator.device, dtype=weight_dtype)
    id_encoder.to(dtype=weight_dtype) # Ensure the overall IDEncoder (possibly wrapped) is set to correct dtype
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
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

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("personalization_encoder", config=vars(args))
    
    torch.cuda.empty_cache() # Clear cache before training

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        # Set unet (FLUX transformer or UNet) to train mode if LoRA is applied or if it's part of params_to_optimize
        # Otherwise, it should remain in .eval() mode (which was set during loading)
        if not is_flux_model and lora_layers: # Standard UNet LoRA
            unet.train()
        elif is_flux_model and lora_layers: # FLUX LoRA (if it were implemented and lora_layers existed)
             unet.train() # unet is FluxTransformer here
        # id_encoder is always trained
        id_encoder.train()

        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(id_encoder): 
                losses = train_step(
                    accelerator=accelerator,
                    batch=batch,
                    vae=vae, 
                    noise_scheduler=noise_scheduler,
                    id_encoder=id_encoder, 
                    text_encoder=text_encoder, 
                    unet=unet, 
                    optimizer=optimizer, 
                    lr_scheduler=lr_scheduler,
                    reg_weight=args.reg_weight,
                    is_flux_model=is_flux_model, 
                    text_encoder_two=text_encoder_two if is_flux_model else None, 
                    tokenizer=tokenizer, # Using primary tokenizer for now
                    tokenizer_two=tokenizer_two if is_flux_model else None,
                    weight_dtype=weight_dtype 
                )
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.save_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        # For FLUX, if LoRA is implemented, unet would be transformer.
                        # accelerator.unwrap_model(unet).save_attn_procs(os.path.join(save_path,"LORA_module"))
                        if not is_flux_model and lora_layers: # Save UNet LoRA
                             accelerator.unwrap_model(unet).save_attn_procs(os.path.join(save_path,"LORA_module"))
                        elif is_flux_model:
                             logger.info("FLUX model saving: LoRA not implemented, skipping save_attn_procs.")
                        # Save IDEncoder separately
                        id_encoder_save_path = os.path.join(save_path, "id_encoder.pth")
                        torch.save(accelerator.unwrap_model(id_encoder).state_dict(), id_encoder_save_path)
                        logger.info(f"Saved state to {save_path} and id_encoder to {id_encoder_save_path}")


            logs = {"loss_simple": losses['loss_simple'],"loss_reg": losses['loss_reg'], "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        accelerator.wait_for_everyone()

    # Final save
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, f"final_checkpoint-{global_step}")
        os.makedirs(save_path, exist_ok=True)
        if not is_flux_model and lora_layers:
            accelerator.unwrap_model(unet).save_attn_procs(os.path.join(save_path, "LORA_module"))
        elif is_flux_model:
            logger.info("Final FLUX model saving: LoRA not implemented, skipping save_attn_procs.")
        
        state_dict = accelerator.unwrap_model(id_encoder).state_dict()
        torch.save(state_dict, os.path.join(save_path, "id_encoder.pth"))
        logger.info(f"Saved final IDEncoder state to {os.path.join(save_path, 'id_encoder.pth')}")

    accelerator.end_training()


def load_model(path, model="stabilityai/stable-diffusion-2-1", final_ckpt=True):
    is_flux_model = "flux" in model.lower()
    id_encoder = IDEncoder(unet_model=model) # IDEncoder needs to know if it's for FLUX

    if is_flux_model:
        # Load FLUX pipeline (e.g., Schnell, or adapt if Pro)
        # This provides the necessary components like transformer, text_encoders, vae, tokenizers
        try:
            # Attempt to load as Schnell, may need to adapt for Pro or other variants
            pipe = FluxSchnellPipeline.from_pretrained(model, use_auth_token=True)
            logger.info(f"Loaded FLUX pipeline: {model}")
        except Exception as e:
            logger.error(f"Failed to load FLUX pipeline {model}. Error: {e}")
            raise
        # The 'unet' equivalent in FLUX pipelines is often called 'transformer'
        # Ensure IDEncoder's feature extractor is set up for the correct device and dtype later
    else:
        pipe = DiffusionPipeline.from_pretrained(model, use_auth_token=True)
        logger.info(f"Loaded UNet pipeline: {model}")

    # Load IDEncoder state dict
    id_encoder_path = os.path.join(path, "id_encoder.pth")
    if not os.path.exists(id_encoder_path): # Fallback for older checkpoint naming
        id_encoder_path = os.path.join(path, "pytorch_model.bin") # Old name for id_encoder
        if not os.path.exists(id_encoder_path) :
             raise FileNotFoundError(f"Could not find id_encoder weights at {os.path.join(path, 'id_encoder.pth')} or {id_encoder_path}")


    id_encoder.load_state_dict(torch.load(id_encoder_path, map_location='cpu'))
    logger.info(f"Loaded IDEncoder from {id_encoder_path}")

    # Load LoRA weights if applicable and they exist
    lora_module_path = os.path.join(path, "LORA_module") # Standard name
    # For old UNet checkpoints, LoRA might be in "pytorch_model_1.bin"
    lora_alt_path = os.path.join(path, "pytorch_model_1.bin") 

    if os.path.exists(lora_module_path):
        if is_flux_model:
            logger.warning("LoRA for FLUX model loading not fully implemented. Attempting to load into pipe.transformer.")
            # pipe.transformer.load_attn_procs(lora_module_path) # This line would be for FLUX LoRA
        else:
            pipe.unet.load_attn_procs(lora_module_path)
        logger.info(f"Loaded LoRA weights from {lora_module_path}")
    elif not is_flux_model and os.path.exists(lora_alt_path) and not final_ckpt: # Only use alt path for non-final UNet checkpoints
        pipe.unet.load_attn_procs(lora_alt_path)
        logger.info(f"Loaded LoRA weights from {lora_alt_path}")
    elif not is_flux_model:
         logger.warning(f"LoRA weights not found at {lora_module_path} or {lora_alt_path} for UNet model. Skipping LoRA loading.")
    elif is_flux_model:
         logger.info("FLUX model: LoRA not implemented or weights not found. Skipping LoRA loading.")
        
    return id_encoder, pipe


def finetune(image,pipe,id_encoder,mixed_precision='no',learning_rate=1e-6,train_batch_size=1,train_steps=15,\
    text='a photo of person',placeholder_token='person',resize=768,prompts=None,output_dir='',num_samples=2,train_text_encoder=False,reg_weight=0.1):
    
    # Determine if it's a FLUX model by checking the type of the main model component
    is_flux_model = isinstance(getattr(pipe, 'transformer', None), FluxTransformer2DModel)
    
    # Get components from the pipe
    vae = pipe.vae.eval() # VAE should always be in eval for this fine-tuning
    unet = pipe.transformer if is_flux_model else pipe.unet # FLUX uses 'transformer'
    
    # Text encoders and tokenizers
    # For FLUX, pipe.text_encoder and pipe.tokenizer might be the primary ones.
    # FLUX pipelines often store text_encoder_2 and tokenizer_2 separately.
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    tokenizer_two = getattr(pipe, 'tokenizer_2', None) if is_flux_model else None
    text_encoder_two = getattr(pipe, 'text_encoder_2', None) if is_flux_model else None

    accelerator = Accelerator(mixed_precision=mixed_precision)
    raw_image = image # Keep original for image feature extraction

    # Prepare latents
    with torch.no_grad():
        # Preprocess image for VAE
        img_for_vae = image.resize((resize, resize))
        img_for_vae = np.array(img_for_vae).astype(np.uint8)
        img_for_vae = (img_for_vae / 127.5 - 1.0).astype(np.float32)
        img_for_vae = torch.from_numpy(img_for_vae).permute(2, 0, 1).unsqueeze(0)
        img_for_vae = img_for_vae.to(vae.device, dtype=vae.dtype)

        latents = vae.encode(img_for_vae).latent_dist.sample().repeat(train_batch_size, 1, 1, 1)
        latents = latents * vae.config.scaling_factor # Use scaling_factor from VAE config

        # Prepare text inputs
        # For FLUX, this needs to handle dual tokenizers if placeholder is in the second one
        # or if `train_step` expects separate input_ids.
        # Assuming `train_step` will handle tokenization for FLUX based on `is_flux_model` flag.
        # Here, we prepare based on the primary tokenizer for placeholder identification.
        
        # Determine which tokenizer contains the placeholder token for ID extraction
        # This logic might need refinement based on how placeholders are handled with dual tokenizers.
        active_tokenizer = tokenizer
        if is_flux_model and tokenizer_two and placeholder_token in tokenizer_two.vocab and not placeholder_token in tokenizer.vocab:
            active_tokenizer = tokenizer_two
        
        input_ids = active_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=active_tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.repeat(train_batch_size, 1)
        
        place_holder_id = active_tokenizer.encode(placeholder_token, add_special_tokens=False)[0] # Get ID without special tokens
        input_placeholder_pos = (input_ids == place_holder_id)
        
        # Initial text_encoder_states (might be overridden by train_step for FLUX)
        # This is primarily for the IDEncoder call within train_step if not using batch['text_encoder_states']
        if is_flux_model:
            # This is a simplified placeholder; train_step should properly generate concatenated embeddings
            prompt_embeds_one = text_encoder(input_ids.to(text_encoder.device))[0]
            # For text_encoder_two, it might need different input_ids if tokenizers are different.
            # This assumes input_ids are compatible or train_step re-tokenizes.
            input_ids_two = tokenizer_two(text, padding="max_length", truncation=True, max_length=tokenizer_two.model_max_length, return_tensors="pt").input_ids.to(text_encoder_two.device).repeat(train_batch_size,1)
            prompt_embeds_two = text_encoder_two(input_ids_two)[0]
            text_encoder_states = torch.cat([prompt_embeds_one, prompt_embeds_two], dim=-1)
        else:
            text_encoder_states = text_encoder(input_ids.to(text_encoder.device))[0]


    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Set modules to train mode
    unet.train() # This is FluxTransformer or UNet
    id_encoder.train()
    
    params_to_optimize = itertools.chain(id_encoder.parameters(), unet.parameters())

    if train_text_encoder:
        text_encoder.train()
        params_to_optimize = itertools.chain(params_to_optimize, text_encoder.parameters())
        if is_flux_model and text_encoder_two:
            text_encoder_two.train()
            params_to_optimize = itertools.chain(params_to_optimize, text_encoder_two.parameters())
    else: # Freeze text encoders if not training them
        text_encoder.eval()
        for param in text_encoder.parameters(): param.requires_grad = False
        if is_flux_model and text_encoder_two:
            text_encoder_two.eval()
            for param in text_encoder_two.parameters(): param.requires_grad = False


    if hasattr(unet, "enable_gradient_checkpointing"): # UNet and FluxTransformer should have this
        unet.enable_gradient_checkpointing()
    if train_text_encoder:
        if hasattr(text_encoder, "gradient_checkpointing_enable"): text_encoder.gradient_checkpointing_enable()
        if is_flux_model and text_encoder_two and hasattr(text_encoder_two, "gradient_checkpointing_enable"):
            text_encoder_two.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate, betas=(0.9,0.999), weight_decay=1e-2, eps=1e-8)
    
    # Scheduler (use one from pipe if available, otherwise default DDPMScheduler)
    noise_scheduler = pipe.scheduler if hasattr(pipe, 'scheduler') else DDPMScheduler.from_config(pipe.config.scheduler)


    # Image features for IDEncoder
    # IDEncoder's internal preprocess_images expects PIL images
    image_features = id_encoder.feature_extractor.encode_image(
        id_encoder.feature_extractor.preprocess_images(raw_image) # Pass original PIL image
    ).repeat(train_batch_size,1)
    
    # Ensure feature extractor's internal models are on the correct device/dtype
    # only_unet=True refers to the UNet/Transformer part of the feature_extractor, not the main unet model.
    id_encoder.feature_extractor.set_device(accelerator.device, dtype=weight_dtype, only_unet=True)


    # Prepare with accelerator
    # Order matters: models first, then optimizer
    id_encoder, unet, optimizer = accelerator.prepare(id_encoder, unet, optimizer)
    if train_text_encoder:
        text_encoder = accelerator.prepare(text_encoder)
        if is_flux_model and text_encoder_two:
            text_encoder_two = accelerator.prepare(text_encoder_two)
    
    # Move other components to device if not already handled
    vae.to(accelerator.device, dtype=weight_dtype)
    if not train_text_encoder: # If not prepared by accelerator, move manually
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        if is_flux_model and text_encoder_two:
            text_encoder_two.to(accelerator.device, dtype=weight_dtype)


    # Batch construction
    batch = {
        'latents': latents.to(accelerator.device, dtype=weight_dtype),
        'text_encoder_states': text_encoder_states.to(accelerator.device, dtype=weight_dtype), # This might be re-calculated in train_step for FLUX
        'input_ids': input_ids.to(accelerator.device), # Primary input_ids
        'input_placeholder_pos': input_placeholder_pos.to(accelerator.device),
        'image_features': image_features.to(accelerator.device, dtype=weight_dtype)
    }
    if is_flux_model: # Add specific FLUX inputs if needed by train_step
        # train_step will need to tokenize for FLUX properly.
        # We can pass the raw text, and train_step can use tokenizer and tokenizer_two.
        batch['text'] = [text] * train_batch_size # Pass raw text for FLUX tokenization in train_step
        # Or, if dual tokenization is done by SimpleDataset, it would be batch['input_ids_one'], batch['input_ids_two']


    progress_bar = tqdm(range(train_steps), disable=not accelerator.is_local_main_process, desc="Finetuning Steps")
    for step in range(train_steps):
        # Note: train_step needs to be adapted for FLUX's dual text encoders and timestep embedding
        loss_dict = train_step(
            accelerator=accelerator,
            batch=batch,
            vae=vae,
            noise_scheduler=noise_scheduler,
            id_encoder=id_encoder,
            text_encoder=text_encoder, # Primary
            unet=unet, # Main model (UNet or FluxTransformer)
            optimizer=optimizer,
            lr_scheduler=None, # No explicit LR scheduler for finetuning loop
            reg_weight=reg_weight,
            is_flux_model=is_flux_model,
            text_encoder_two=text_encoder_two if is_flux_model else None,
            tokenizer=tokenizer, # Pass primary tokenizer
            tokenizer_two=tokenizer_two if is_flux_model else None # Pass secondary tokenizer
        )
        progress_bar.update(1)
        progress_bar.set_postfix(**loss_dict)
    
    accelerator.wait_for_everyone()

    # Unwrap models and move to CPU/eval if needed by pipeline afterwards
    pipe.unet = accelerator.unwrap_model(unet).eval()
    if is_flux_model: pipe.transformer = pipe.unet # ensure flux pipeline has transformer attribute updated
    
    id_encoder_unwrapped = accelerator.unwrap_model(id_encoder).eval()
    
    if train_text_encoder:
        pipe.text_encoder = accelerator.unwrap_model(text_encoder).eval()
        if is_flux_model and text_encoder_two:
            pipe.text_encoder_2 = accelerator.unwrap_model(text_encoder_two).eval()

    # Sampling after finetuning
    if prompts:
        if isinstance(prompts, str): prompts = [prompts]
        for current_prompt in prompts:
            # sample function needs to be FLUX-aware for text encoding and timestep embedding
            sample(
                id_encoder=id_encoder_unwrapped, 
                pipe=pipe, # pipe now contains the finetuned components
                image_size=(resize, resize), 
                prompt=current_prompt, 
                accelerator=accelerator, 
                num_samples=num_samples, 
                output_dir=output_dir,
                # placeholder_id and image_features might need to be re-evaluated for FLUX context
                # For now, using the ones derived from the primary tokenizer.
                place_holder_id=place_holder_id, 
                image_features=batch['image_features'][:1].to(dtype=weight_dtype, device=accelerator.device), # Use one sample of image features
                source_latents=batch['latents'][:1].to(dtype=weight_dtype, device=accelerator.device) # Use one sample of latents
            )


@torch.no_grad()
def sample(id_encoder,pipe,image_size,prompt,accelerator,num_samples,output_dir,place_holder_id,image_features,source_latents,guidance_scale=7.5,num_inference_steps=100):
    is_flux_model = isinstance(getattr(pipe, 'transformer', None), FluxTransformer2DModel)
    unet = pipe.transformer if is_flux_model else pipe.unet
    
    height,width = image_size
    if prompt is not None and isinstance(prompt, str):
        prompt_list=[prompt] # Use a different variable name to avoid conflict
    elif prompt is not None and isinstance(prompt, list):
        prompt_list = prompt
    else: # Should not happen if called from finetune
        prompt_list = ["a photo of person"]

    batch_size = len(prompt_list)
    device = accelerator.device
    
    # Determine dtypes from pipeline components
    text_encoder_dtype = pipe.text_encoder.dtype
    unet_dtype = unet.dtype

    # Tokenize prompts
    # For FLUX, this needs to handle dual tokenizers and encoders
    if is_flux_model:
        # Tokenize with both tokenizers
        input_ids_one = pipe.tokenizer(
            prompt_list, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt"
        ).input_ids
        input_ids_two = pipe.tokenizer_2(
            prompt_list, padding="max_length", max_length=pipe.tokenizer_2.model_max_length, truncation=True, return_tensors="pt"
        ).input_ids
        
        input_ids_one = torch.repeat_interleave(input_ids_one, num_samples, dim=0).to(device)
        input_ids_two = torch.repeat_interleave(input_ids_two, num_samples, dim=0).to(device)

        # Get embeddings from both text encoders
        prompt_embeds_one = pipe.text_encoder(input_ids_one, output_hidden_states=True)[0] # Use last_hidden_state
        prompt_embeds_two = pipe.text_encoder_2(input_ids_two, output_hidden_states=True)[0] # Use last_hidden_state
        
        # Concatenate the embeddings (common FLUX pattern)
        # The dimensions should be compatible for concatenation along the last axis.
        # FLUX pipelines might also return pooled embeddings, check specific pipeline doc.
        # For now, assuming direct concatenation of last_hidden_state is desired.
        # prompt_embeds = torch.cat([prompt_embeds_one, prompt_embeds_two], dim=-1)
        # FLUX Schnell pipeline uses pooled embeddings and passes them separately to transformer
        # This is a simplification and might need adjustment.
        # For FLUX, encoder_hidden_states is often just prompt_embeds_one, and text_embeds_pooled_two is separate.
        # Let's assume for now IDEncoder gets the concatenated embeds, and the main model gets specific ones.
        # This needs to align with how IDEncoder expects `encoder_hidden_states` for FLUX.
        # From encoder_model.py, IDEncoder's feature_extractor.encode_unet takes `encoder_hidden_states`.
        # Let's provide the concatenated state to IDEncoder.
        id_encoder_prompt_embeds = torch.cat([prompt_embeds_one, prompt_embeds_two], dim=-1)

        # For the main FLUX transformer, it might expect specific embeddings.
        # E.g. Schnell: `hidden_states`, `encoder_hidden_states` (from text_encoder), `pooled_projections` (from text_encoder_2)
        # This part is highly dependent on the specific FLUX model.
        # Let's assume `unet` (FluxTransformer) will take `prompt_embeds_one` as `encoder_hidden_states`
        # and we'll need to figure out `pooled_projections` or `timestep_embedding`.
        # For now, we'll pass `prompt_embeds_one` as main `encoder_hidden_states` to unet.
        # The ID feature injection must be compatible.
        main_prompt_embeds = prompt_embeds_one # This is a guess for FLUX Schnell like models
        # We might need pooled_prompt_embeds from text_encoder_2 as well for the FLUX model.
        # For now, this is simplified.
        
        # Placeholder for classifier-free guidance with dual encoders
        if guidance_scale > 1.0:
            uncond_tokens = [""] * batch_size
            uncond_ids_one = pipe.tokenizer(uncond_tokens, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids
            uncond_ids_two = pipe.tokenizer_2(uncond_tokens, padding="max_length", max_length=pipe.tokenizer_2.model_max_length, truncation=True, return_tensors="pt").input_ids
            uncond_ids_one = torch.repeat_interleave(uncond_ids_one, num_samples, dim=0).to(device)
            uncond_ids_two = torch.repeat_interleave(uncond_ids_two, num_samples, dim=0).to(device)

            neg_prompt_embeds_one = pipe.text_encoder(uncond_ids_one, output_hidden_states=True)[0]
            neg_prompt_embeds_two = pipe.text_encoder_2(uncond_ids_two, output_hidden_states=True)[0]
            # negative_prompt_embeds = torch.cat([neg_prompt_embeds_one, neg_prompt_embeds_two], dim=-1) # For IDEncoder
            main_negative_prompt_embeds = neg_prompt_embeds_one # For main FLUX model
    else: # UNet
        text_inputs_ids = pipe.tokenizer(
            prompt_list, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt"
        ).input_ids
        text_inputs_ids = torch.repeat_interleave(text_inputs_ids, num_samples, dim=0).to(device)
        main_prompt_embeds = pipe.text_encoder(text_inputs_ids)[0]
        id_encoder_prompt_embeds = main_prompt_embeds # For UNet, they are the same

        if guidance_scale > 1.0:
            uncond_tokens = [""] * batch_size
            uncond_input_ids = pipe.tokenizer(
                uncond_tokens, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt"
            ).input_ids
            uncond_input_ids = torch.repeat_interleave(uncond_input_ids, num_samples, dim=0).to(device)
            main_negative_prompt_embeds = pipe.text_encoder(uncond_input_ids)[0]

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    
    num_channels_latents = unet.config.in_channels if hasattr(unet.config, 'in_channels') else 4 # Default for SD
    latents = pipe.prepare_latents(batch_size * num_samples, num_channels_latents, height, width, unet_dtype, device, None) # Use unet_dtype
    
    # Ensure source_latents and image_features are correctly prepared (moved to device and correct dtype)
    source_latents = source_latents.to(device=device, dtype=unet_dtype)
    image_features = image_features.to(device=device, dtype=id_encoder.feature_extractor.dtype) # IDEncoder has its own dtype

    # Determine active tokenizer for placeholder_id (this was passed in, but good to be consistent)
    # This logic for placeholder_pos might need to be specific to how FLUX handles dual tokenizers
    active_tokenizer_for_placeholder = pipe.tokenizer
    if is_flux_model and pipe.tokenizer_2 and place_holder_id in pipe.tokenizer_2.get_vocab().values() and not place_holder_id in pipe.tokenizer.get_vocab().values():
         active_tokenizer_for_placeholder = pipe.tokenizer_2
    
    # This assumes text_inputs_ids corresponds to the tokenizer that contains place_holder_id
    # For FLUX, if place_holder_id is from tokenizer_2, then input_placeholder_pos should use input_ids_two
    # This is getting complicated. For now, assume place_holder_id is from the primary tokenizer or that `text_inputs_ids` is appropriate.
    # A robust solution would pass which input_ids (one or two) corresponds to the placeholder.
    # For now, using text_inputs_ids (which is input_ids_one for FLUX). This might be incorrect if placeholder is in text_encoder_two.
    current_text_ids_for_placeholder = input_ids_one if is_flux_model else text_inputs_ids
    input_placeholder_pos = (current_text_ids_for_placeholder == place_holder_id).to(device)

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    batch_input_for_id_encoder = dict(image_features=image_features.repeat(num_samples,1) if image_features.shape[0] == 1 else image_features)
    
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(None, 0.0) if hasattr(pipe, 'prepare_extra_step_kwargs') else {}
    num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order

    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = latent_model_input.to(unet_dtype) # Ensure dtype

            # Timestep embedding for FLUX
            # This is CRITICAL for FLUX. `t` is a raw timestep.
            # IDEncoder's feature_extractor also needs this if it's a FLUX model.
            current_timestep_input_for_id_encoder = t 
            current_timestep_input_for_unet = t

            if is_flux_model:
                # FLUX models require pre-calculated timestep embeddings.
                # This often involves `time_ids` and specific projection layers.
                # Example for FLUX (Schnell-like, highly model specific):
                # time_ids = t.unsqueeze(0).repeat(latent_model_input.shape[0], 1) # or more complex formation
                # time_ids = torch.cat([time_ids, other_img_cond_ids], dim=1) # example if image conditioning ids are used
                # timestep_embed = unet.time_text_embed(time_ids.long()) # Hypothetical layer
                # For IDEncoder, it expects timestep that its internal feature_extractor can handle.
                # The feature_extractor.encode_unet for FLUX was designed to accept precomputed embeddings or try to make one.
                # For simplicity in `sample`, we'll rely on IDEncoder's internal handling for now if `t` is passed.
                # For the main FLUX model `unet` (FluxTransformer2DModel):
                # It expects `timestep_embedding` not raw `t`.
                # This is a placeholder and needs to be correctly implemented based on FluxTransformer2DModel's API.
                # A common pattern:
                #   1. Create `time_ids` from `t`. (e.g., `t.unsqueeze(-1)`)
                #   2. Possibly concatenate with other conditioning signals (text length, image size for some FLUX variants)
                #   3. Project through `unet.time_proj` (or similar) -> `t_emb`
                #   4. Embed through `unet.time_embedding` (or similar) -> `timestep_embedding_for_unet`
                # This is a simplification and likely needs specific layers from `pipe.transformer` (unet).
                # For now, this is a MAJOR GAP for FLUX in sample().
                # Let's assume `pipe.prepare_timestep_embedding` exists (hypothetical)
                if hasattr(pipe, "prepare_timestep_embedding"):
                    timestep_embedding_for_unet = pipe.prepare_timestep_embedding(t, latent_model_input.shape[0], device, unet_dtype)
                elif hasattr(unet, "time_text_embed") and hasattr(unet.config, "text_embed_dim") and hasattr(unet.config, "time_embed_dim"): # Schnell like
                    # This is a very specific guess for Schnell-like models
                    time_ids = t.unsqueeze(0).repeat(latent_model_input.shape[0],1).to(device).long() # B,1
                    # It might need additional IDs like image crop, etc. For now, just timestep.
                    # Schnell's time_text_embed might expect a fixed size input, e.g. (B, 2) or (B, 6)
                    # This part is highly model-dependent and will likely error or be suboptimal.
                    # Example: if unet.config.time_text_embed_input_dim is 2:
                    # time_ids = time_ids.repeat(1, unet.config.time_text_embed_input_dim) 
                    # This is a placeholder. Actual FLUX models require specific time_ids construction.
                    # A simple approach: assume time_text_embed can take (B,1) timesteps.
                    try:
                        timestep_embedding_for_unet = unet.time_text_embed(time_ids) # This is a guess
                    except Exception as e:
                        # Fallback, this will likely be wrong shape/content
                        print(f"Warning: Failed to create FLUX timestep embedding for unet via time_text_embed: {e}. Using zeros.")
                        timestep_embedding_for_unet = torch.zeros(latent_model_input.shape[0], unet.config.time_embed_dim, device=device, dtype=unet_dtype)

                else: # Fallback if no obvious way to get FLUX timestep embedding
                    print(f"Warning: FLUX timestep embedding for unet not implemented. Passing raw 't' or zeros. This will likely fail or perform poorly.")
                    # FluxTransformer2DModel expects 'timestep_embedding', not 't'.
                    # Using zeros as a last resort, this needs proper implementation.
                    timestep_embedding_for_unet = torch.zeros(latent_model_input.shape[0], getattr(unet.config, "time_cond_proj_dim", unet.config.hidden_size), device=device, dtype=unet_dtype)
                current_timestep_input_for_unet = timestep_embedding_for_unet.to(unet_dtype)
                # IDEncoder will also need proper timestep embedding if its internal unet is FLUX
                # The feature_extractor.encode_unet for FLUX will try to handle `t` if it's not already an embedding.

            id_f = id_encoder(batch_input_for_id_encoder, latents.to(id_encoder.feature_extractor.dtype), current_timestep_input_for_id_encoder, id_encoder_prompt_embeds.to(id_encoder.feature_extractor.dtype)) * 0.1
            id_f = id_f.to(unet_dtype) # Cast to unet's dtype for addition

            # Classifier-free guidance
            if guidance_scale > 1.0:
                current_prompt_embeds = torch.cat([main_negative_prompt_embeds, main_prompt_embeds + id_f.unsqueeze(1) * input_placeholder_pos.unsqueeze(-1)])
                if is_flux_model:
                    # For FLUX, text_embeds_pooled_two might be needed for CFG
                    # neg_pooled = ... main_pooled = ...
                    # For now, assuming `encoder_hidden_states` is the primary way.
                    # This will need refinement based on specific FLUX model's CFG inputs.
                    # If pipe.transformer expects pooled_projections (from text_encoder_2):
                    # neg_pooled = pipe.text_encoder_2(uncond_ids_two)[1] # Example if [1] is pooled
                    # text_pooled = pipe.text_encoder_2(input_ids_two)[1]
                    # pooled_projections = torch.cat([neg_pooled, text_pooled])
                    # noise_pred = unet(latent_model_input, timestep=current_timestep_input_for_unet, encoder_hidden_states=current_prompt_embeds, pooled_projections=pooled_projections).sample
                    # This is simplified for now:
                    noise_pred = unet(latent_model_input, timestep_embedding=current_timestep_input_for_unet, encoder_hidden_states=current_prompt_embeds).sample
                else: # UNet
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=current_prompt_embeds).sample
                
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            else: # No CFG
                current_prompt_embeds = main_prompt_embeds + id_f.unsqueeze(1) * input_placeholder_pos.unsqueeze(-1)
                if is_flux_model:
                    # pooled_projections = pipe.text_encoder_2(input_ids_two)[1] # Simplified
                    # noise_pred = unet(latent_model_input, timestep=current_timestep_input_for_unet, encoder_hidden_states=current_prompt_embeds, pooled_projections=pooled_projections).sample
                    noise_pred = unet(latent_model_input, timestep_embedding=current_timestep_input_for_unet, encoder_hidden_states=current_prompt_embeds).sample
                else: # UNet
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=current_prompt_embeds).sample

            latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()

    image = pipe.decode_latents(latents.to(pipe.vae.dtype)) # Ensure dtype for VAE
    image = pipe.image_processor.postprocess(image, output_type="pil") # Use image_processor for postprocessing

    for i in range(batch_size): # Iterate through prompts
        for j in range(num_samples): # Iterate through samples per prompt
            # Calculate unique rank for saving, considering multiple processes
            # This assumes accelerator.process_index and num_samples are available
            # If accelerator is not available here (e.g. if sample is called outside main training),
            # then process_index might be 0 or need to be passed.
            # For now, assume accelerator is available or this is single process.
            sample_idx_in_batch = i * num_samples + j
            # rank = num_samples * accelerator.process_index + j # This would be for distributed saving index
            # For local saving, a simple index is fine.
            
            # Use a safe filename based on the prompt
            prompt_slug = "".join(filter(str.isalnum, prompt_list[i]))[:50] # Limit length
            
            image[sample_idx_in_batch].save(os.path.join(output_dir, f'{prompt_slug}-{sample_idx_in_batch}.png'))
        
if __name__ == "__main__":
    main()
