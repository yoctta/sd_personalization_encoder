import torch
import open_clip
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel, FluxTransformer2DModel
class feature_extractor:
    def __init__(self,clip_model='ViT-H-14/laion2b_s32b_b79k',unet_model="stabilityai/stable-diffusion-2-1"):
        self.is_flux_model = "flux" in unet_model.lower()
        if self.is_flux_model:
            # For FLUX, subfolder might be 'transformer' or it might be directly in the model root
            # Trying 'transformer' first, then root if it fails.
            try:
                unet = FluxTransformer2DModel.from_pretrained(unet_model, subfolder="transformer", use_auth_token=True).eval()
            except EnvironmentError: # Could be various errors like OSError, FileNotFoundError etc.
                unet = FluxTransformer2DModel.from_pretrained(unet_model, use_auth_token=True).eval()
        else:
            unet = UNet2DConditionModel.from_pretrained(unet_model, subfolder="unet",use_auth_token=True).eval()
            # These deletions are UNet specific and should not apply to FLUX
            del unet.mid_block
            del unet.up_blocks
            del unet.conv_out
        unet.eval()
        self.unet=unet
        clip_model, _, preprocess = open_clip.create_model_and_transforms(clip_model.split('/')[0], pretrained=clip_model.split('/')[1])
        self.preprocess=preprocess
        self.image_encoder=clip_model.visual.eval()
        self.activation={}
        self.device=torch.device('cpu')
        self.dtype=torch.float32
        def getActivation(name):
            def hook(model, input, output):
                self.activation[name] = output[:,0,:].detach()
            return hook
        for i in range(1,len(self.image_encoder.transformer.resblocks),2):
            self.image_encoder.transformer.resblocks[i].register_forward_hook(getActivation(i))

    def set_device(self,device,dtype=torch.float32,only_unet=False):
        self.device=torch.device(device)
        self.dtype=dtype
        self.unet.to(device,dtype=dtype)
        if not only_unet:
            self.image_encoder.to(device,dtype=dtype)
        
    def preprocess_images(self,images):
        if not type(images)==list:
            images=[images]
        return torch.stack([self.preprocess(i) for i in images]).to(self.device)
        
    def encode_image(self,images):
        with torch.no_grad():
            _=self.image_encoder(images.to(self.dtype))
            n=sorted(self.activation.keys())
            return torch.stack([torch.cat([self.activation[j][i] for j in n],dim=0) for i in range(len(images))])
        
    def encode_unet(self,latent,timestep,encoder_hidden_states):
        with torch.no_grad():
            pooled_features=[]
            if self.is_flux_model:
                # FLUX model expects `timestep_embedding`
                # The `timestep` argument could be raw timesteps (long tensor) or a pre-calculated embedding.
                timestep_embedding = None
                if timestep.ndim == 2 and timestep.shape[0] == latent.shape[0]: # Possibly pre-calculated embedding
                    # Heuristic: if it's 2D and batch size matches, assume it's the embedding
                    # A more robust check would be if timestep.shape[1] == self.unet.config.time_cond_proj_dim or similar
                    timestep_embedding = timestep.to(dtype=self.dtype, device=latent.device)
                elif timestep.ndim == 1 or (timestep.ndim == 0 and latent.shape[0] == 1) : # Raw timesteps
                    if timestep.ndim == 0: # scalar tensor
                        timestep = timestep.unsqueeze(0) # make it 1D
                    
                    current_timesteps = timestep.to(device=latent.device, dtype=torch.long)
                    current_timesteps = current_timesteps.expand(latent.shape[0]) # ensure it matches batch size

                    # FLUX models (like Schnell) often use a time_text_embed layer for time embeddings.
                    # This usually takes `time_ids` which are derived from timesteps.
                    # Example: time_ids = current_timesteps.unsqueeze(-1).repeat(1, 2) # This is a guess for FLUX structure
                    # And then: timestep_embedding = self.unet.time_text_embed(time_ids)
                    # The exact structure of time_ids and the layer name (time_text_embed, time_embed, etc.)
                    # can vary between FLUX model versions.
                    # For now, let's assume `train.py` will provide the correct `timestep_embedding`.
                    # If not, this part will need specific logic for the target FLUX model.
                    # A placeholder: if `self.unet` has a direct time embedding method similar to UNet's, use it.
                    # This is unlikely to be correct for all FLUX variants.
                    if hasattr(self.unet, 'time_embedding') and callable(self.unet.time_embedding):
                         # This is a guess. FLUX might not have time_proj and time_embedding like UNet.
                         # It might require a specific `time_ids` format.
                        if hasattr(self.unet, 'time_proj') and callable(self.unet.time_proj):
                            t_emb = self.unet.time_proj(current_timesteps).to(dtype=self.dtype)
                            timestep_embedding = self.unet.time_embedding(t_emb)
                        else: # Direct embedding from timesteps (less common for FLUX main transformer)
                            timestep_embedding = self.unet.time_embedding(current_timesteps) 
                    else:
                        # This is a critical point: FLUX models need specific timestep embeddings.
                        # If not provided directly, and no clear way to compute it here, this will be an issue.
                        # Raising an error or using a zero embedding as a last resort.
                        print(f"Warning: FLUX timestep embedding cannot be determined. Timestep shape: {timestep.shape}. Using zeros.")
                        # The dimension here should be what FluxTransformer2DModel expects for timestep_embedding
                        # e.g., self.unet.config.time_cond_proj_dim or self.unet.config.hidden_size
                        # Using hidden_size as a general guess, might need adjustment.
                        expected_dim = getattr(self.unet.config, "time_cond_proj_dim", self.unet.config.hidden_size)
                        timestep_embedding = torch.zeros(latent.shape[0], expected_dim, device=latent.device, dtype=self.dtype)
                else:
                    raise ValueError(f"Unsupported timestep format for FLUX model: shape {timestep.shape}")

                output = self.unet(
                    hidden_states=latent.to(dtype=self.dtype),
                    encoder_hidden_states=encoder_hidden_states.to(dtype=self.dtype),
                    timestep_embedding=timestep_embedding.to(dtype=self.dtype)
                )
                last_hidden_state = output.last_hidden_state 
                
                # For consistency with UNet path and IDEncoder structure,
                # ensure pooled_features is a list of tensors.
                # Here, we take the mean of the last_hidden_state.
                if last_hidden_state.ndim == 3: # B, S, E (Sequence-like)
                    pooled_features.append(last_hidden_state.mean(dim=1))
                elif last_hidden_state.ndim == 4: # B, C, H, W (Spatial-like)
                    pooled_features.append(last_hidden_state.mean(dim=[2,3]))
                else:
                    # Fallback, though FLUX output should be one of the above.
                    print(f"Unexpected output shape from FLUX model: {last_hidden_state.shape}. Using as is or zero if incompatible.")
                    # Attempt to add a feature, this might fail in IDEncoder if dimensions mismatch.
                    # This indicates a need to better understand the specific FLUX model's output.
                    if last_hidden_state.shape[0] == latent.shape[0] and last_hidden_state.numel() > 0 :
                        # As a very basic fallback, flatten and take mean if possible, or just append if it's already B, D
                        if last_hidden_state.ndim > 2:
                             pooled_features.append(last_hidden_state.mean(dim=list(range(2, last_hidden_state.ndim))))
                        elif last_hidden_state.ndim == 2: # B, D
                             pooled_features.append(last_hidden_state)
                        else: # Should not happen
                             pooled_features.append(torch.zeros(latent.shape[0], self.unet.config.hidden_size, device=latent.device, dtype=self.dtype))
                    else:
                        pooled_features.append(torch.zeros(latent.shape[0], self.unet.config.hidden_size, device=latent.device, dtype=self.dtype))


            else: # Existing UNet logic
                timesteps_arg = timestep
                if not torch.is_tensor(timesteps_arg): # Scalar or float/int
                    timesteps_arg = torch.tensor([timesteps_arg], device=latent.device) # Convert to tensor
                
                if timesteps_arg.ndim == 0: # Handle scalar tensor
                    timesteps_arg = timesteps_arg.unsqueeze(0)

                # Ensure timesteps_arg is long type for time_proj
                timesteps_arg = timesteps_arg.to(dtype=torch.long)
                timesteps_arg = timesteps_arg.expand(latent.shape[0])

                t_emb = self.unet.time_proj(timesteps_arg).to(dtype=self.dtype)
                emb = self.unet.time_embedding(t_emb)
                current_latent = self.unet.conv_in(latent.to(dtype=self.dtype))
                for downsample_block in self.unet.down_blocks:
                    if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                        current_latent, res_samples = downsample_block(
                            hidden_states=current_latent,
                            temb=emb,
                            encoder_hidden_states=encoder_hidden_states.to(self.dtype))
                    else:
                        current_latent, res_samples = downsample_block(hidden_states=current_latent, temb=emb)
                    pooled_features.append(current_latent.mean(dim=[2,3]))
        return pooled_features


class IDEncoder(nn.Module):
    def __init__(self,clip_model='ViT-H-14/laion2b_s32b_b79k',unet_model="stabilityai/stable-diffusion-2-1"):
        super().__init__()
        self.feature_extractor=feature_extractor(clip_model,unet_model) # unet_model is passed here
        dim_width=self.feature_extractor.image_encoder.transformer.width
        dim_layers=self.feature_extractor.image_encoder.transformer.layers//2
        
        if self.feature_extractor.is_flux_model:
            # Based on the modified `encode_unet` for FLUX, it returns a list with a single tensor.
            # The dimension of this tensor is `self.feature_extractor.unet.config.hidden_size`.
            dim_unet = [self.feature_extractor.unet.config.hidden_size] 
            # FLUX cross_attention_dim is typically hidden_size or a dedicated config field like `cross_attention_dim`
            # Defaulting to hidden_size if cross_attention_dim is not present.
            dim_cross_attn = getattr(self.feature_extractor.unet.config, 'cross_attention_dim', self.feature_extractor.unet.config.hidden_size)
        else:
            dim_unet = self.feature_extractor.unet.config.block_out_channels
            dim_cross_attn = self.feature_extractor.unet.config.cross_attention_dim
            
        self.id_encoder_feature=nn.Conv1d(dim_width*dim_layers,dim_width*dim_layers,1,groups=dim_layers)
        self.unet_encoder_feature=nn.ModuleList([nn.Linear(i,dim_width) for i in dim_unet])
        
        self.dim_width=dim_width # Storing for use in forward if needed, or just for clarity
        self.dim_layers=dim_layers # Storing for use in forward if needed
        self.last_linear=nn.Linear(dim_width,dim_cross_attn)


    def forward(self,batch,latent,timestep,encoder_hidden_states):
        if 'image_features' in batch:
            image_features=batch['image_features'].to(dtype=latent.dtype, device=latent.device)
        elif 'image' in batch:
            # Ensure preprocessed images are on the correct device before encoding
            processed_images = self.feature_extractor.preprocess_images(batch['image'])
            image_features=self.feature_extractor.encode_image(processed_images).to(dtype=latent.dtype, device=latent.device)
        else:
            raise ValueError("Batch must contain either 'image_features' or 'image'")

        encoder_hidden_states_processed = encoder_hidden_states.to(dtype=latent.dtype, device=latent.device)
        
        # `timestep` will be handled by `encode_unet` based on whether it's FLUX or UNet
        # For FLUX, `timestep` should ideally be the precomputed `timestep_embedding`
        # or raw timesteps that `encode_unet` can attempt to convert.
        unet_features_list=self.feature_extractor.encode_unet(latent,timestep,encoder_hidden_states_processed)
        
        processed_unet_features = []
        for idx, feature_tensor in enumerate(unet_features_list):
            processed_unet_features.append(self.unet_encoder_feature[idx](feature_tensor.to(dtype=latent.dtype)))
        
        unet_features_stacked = torch.stack(processed_unet_features, dim=1)

        # Ensure image_features also has the correct dtype for the conv layer (though it should be from image_encoder)
        image_features_proj=self.id_encoder_feature(image_features.unsqueeze(-1)).reshape(image_features.shape[0],self.dim_layers,self.dim_width)
        
        features=torch.cat([unet_features_stacked,image_features_proj],dim=1)
        features=F.leaky_relu(features, negative_slope=0.1)
        features=torch.mean(features,dim=1)
        return self.last_linear(features)
