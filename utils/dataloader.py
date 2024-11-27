import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np


class ESD_DZSeg(Dataset):
    def __init__(self, num_classes, image_dir, mask_dir, transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.num_classes = num_classes
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
    

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
    
        if self.transform:
            image, mask = self.transform[0](image), self.transform[1](mask)
            if self.num_classes == 2 or self.num_classes == 1:
                mask[mask==2] = 1
        
        return image, mask

class Resize:
    def __init__(self, resize_size=532):
        self.resize_size = resize_size
    def __call__(self, image):
        resize_transform = transforms.Resize((self.resize_size, self.resize_size))
        image = resize_transform(image)
        return image

class ToTensor:
    def __call__(self, image):
        image = transforms.ToTensor()(image)
        return image
    
class ToTensor_mask():
    def __init__(self, type='int'):
        self.type = type
    def __call__(self, mask):
        if self.type == 'int':
            mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        elif self.type == 'float':
            mask = torch.from_numpy(np.array(mask, dtype=np.float32))
        else:
            raise ValueError('unknown type')
        return mask

class Normalize:
    def __call__(self, image):
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        return image
    

def get_dataloader(data_root='./dataset', prompt_type='None', image_size=532, batch_size=8, num_classes=2):
    
    if prompt_type == 'None':
        train_image_path = 'images/training'
        val_image_path = 'images/validation'
    else:
        train_image_path = f'images_{prompt_type}/training'
        val_image_path = f'images_{prompt_type}/validation'
        
    train_image_dir = os.path.join(data_root, train_image_path)
    train_mask_dir = os.path.join(data_root, 'annotations/training')

    val_image_dir = os.path.join(data_root, val_image_path)
    val_mask_dir = os.path.join(data_root, 'annotations/validation')
    
    image_transform = transforms.Compose([Resize(image_size), ToTensor(), Normalize()])
    mask_transform = transforms.Compose([Resize(image_size), ToTensor_mask(type='int')])

    train_dataset = ESD_DZSeg(num_classes=num_classes, image_dir=train_image_dir, mask_dir=train_mask_dir, transform=(image_transform, mask_transform))
    
    val_dataset = ESD_DZSeg(num_classes=num_classes, image_dir=val_image_dir, mask_dir=val_mask_dir, transform=(image_transform, mask_transform))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)

    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = get_dataloader(batch_size=1)
    print(len(train_loader), len(val_loader)) 

    for batch_idx, (image, mask) in enumerate(val_loader):
        print(f'image shape: {image.shape}, target shape: {mask.shape}')
        break

    
        