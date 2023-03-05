from torch.utils.data import Dataset
import torch
import numpy as np
import PIL
from PIL import Image
import random
from torchvision import transforms
import os


my_templates=["a photo of {}"]
    

class Simpledataset(Dataset):
    def __init__(
        self,
        image_dir,
        tokenizer,
        preprocess,
        size=512,
        placeholder_token='person'
    ):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.placeholder_token = placeholder_token
        self.size=size
        self.preprocess=preprocess
        self.place_holder_id=self.tokenizer.encode(self.placeholder_token)[1]
        self.ids=os.listdir(image_dir)
        self.templates = [i.format(self.placeholder_token) for i in my_templates]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        example = {}
        img_path=os.path.join(self.image_dir,self.ids[i])
        image = Image.open(img_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        example["image"]=self.preprocess(image)
        text = random.choice(self.templates)
        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        example["input_placeholder_pos"]=example["input_ids"]==self.place_holder_id
        image = image.resize((self.size, self.size))
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example
