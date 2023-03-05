from train import finetune, load_model
from PIL import Image
import os
import argparse
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_name_or_path',type=str,default="stabilityai/stable-diffusion-2-1")
    parser.add_argument('--model_path',type=str,default=None)
    parser.add_argument('--final_checkpoint', action='store_true')
    parser.add_argument('--train_text_encoder', action='store_true')
    parser.add_argument('--image_path',type=str,default=None)
    parser.add_argument("--mixed_precision",type=str,default="no",choices=["no", "fp16", "bf16"])
    parser.add_argument('--learning_rate',type=float,default=1e-6)
    parser.add_argument('--reg_weight',type=float,default=0.1)
    parser.add_argument('--train_batch_size',type=int,default=1)
    parser.add_argument('--finetune_steps',type=int,default=15)
    parser.add_argument('--prompt',type=str,nargs='+')
    parser.add_argument('--placeholder_token',type=str,default='person')
    parser.add_argument('--resolution',type=int,default=768)
    parser.add_argument('--output_dir',type=str,default=None)
    parser.add_argument('--num_samples',type=int,default=1)
    
    args=parser.parse_args()
    if not args.output_dir:
        args.output_dir=os.path.join(args.model_path,'sampled_images')
        
    id_encoder,pipe=load_model(args.model_path,args.pretrained_model_name_or_path,args.final_checkpoint)
    image=Image.open(args.image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    finetune(image,pipe,id_encoder,mixed_precision=args.mixed_precision,learning_rate=args.learning_rate,train_batch_size=args.train_batch_size,\
        train_steps=args.finetune_steps,text='a photo of '+args.placeholder_token,placeholder_token=args.placeholder_token,resize=args.resolution,\
        prompts=args.prompt,output_dir=args.output_dir,num_samples=args.num_samples,train_text_encoder=args.train_text_encoder,reg_weight=args.reg_weight)
    