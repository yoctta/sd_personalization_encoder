import torch
import open_clip
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
class feature_extractor:
    def __init__(self,clip_model='ViT-H-14/laion2b_s32b_b79k',unet_model="stabilityai/stable-diffusion-2-1"):
        unet = UNet2DConditionModel.from_pretrained(unet_model, subfolder="unet",use_auth_token=True).eval()
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
        timesteps=timestep
        with torch.no_grad():
            pooled_features=[]
            if not torch.is_tensor(timesteps):
                if isinstance(timestep, float):
                    dtype = torch.float64
                else:
                    dtype = torch.int64
                timesteps = torch.tensor([timesteps], dtype=dtype, device=latent.device)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(latent.device)
            timesteps = timesteps.expand(latent.shape[0])
            t_emb = self.unet.time_proj(timesteps).to(dtype=self.dtype)
            emb = self.unet.time_embedding(t_emb)
            latent = self.unet.conv_in(latent.to(dtype=self.dtype))
            for downsample_block in self.unet.down_blocks:
                if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                    latent, res_samples = downsample_block(
                        hidden_states=latent,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states.to(self.dtype))
                else:
                    latent, res_samples = downsample_block(hidden_states=latent, temb=emb)
                pooled_features.append(latent.mean(dim=[2,3]))
        return pooled_features
        
        
        
        

class IDEncoder(nn.Module):
    def __init__(self,clip_model='ViT-H-14/laion2b_s32b_b79k',unet_model="stabilityai/stable-diffusion-2-1"):
        super().__init__()
        self.feature_extractor=feature_extractor(clip_model,unet_model)
        dim_width=self.feature_extractor.image_encoder.transformer.width
        dim_layers=self.feature_extractor.image_encoder.transformer.layers//2
        dim_unet = self.feature_extractor.unet.config.block_out_channels
        dim_cross_attn=self.feature_extractor.unet.config.cross_attention_dim
        self.id_encoder_feature=nn.Conv1d(dim_width*dim_layers,dim_width*dim_layers,1,groups=dim_layers)
        self.unet_encoder_feature=nn.ModuleList([nn.Linear(i,dim_width) for i in dim_unet])
        self.last_linear=nn.Linear(dim_width,dim_cross_attn)
        self.dim_width=dim_width
        self.dim_layers=dim_layers
    def forward(self,batch,latent,timestep,encoder_hidden_states):
        if 'image_features' in batch:
            image_features=batch['image_features']
        elif 'image' in batch:
            image_features=self.feature_extractor.encode_image(batch['image']).to(dtype=latent.dtype)
        unet_features=self.feature_extractor.encode_unet(latent,timestep,encoder_hidden_states)
        unet_features=torch.stack([f(i.to(dtype=latent.dtype)) for f,i in zip(self.unet_encoder_feature,unet_features)],dim=1)
        image_features_proj=self.id_encoder_feature(image_features.unsqueeze(-1)).reshape(image_features.shape[0],self.dim_layers,self.dim_width)
        features=torch.cat([unet_features,image_features_proj],dim=1)
        features=F.leaky_relu(features, negative_slope=0.1)
        features=torch.mean(features,dim=1)
        return self.last_linear(features)
