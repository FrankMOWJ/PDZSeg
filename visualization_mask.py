import torch
import torch.nn.functional as F
from arch.dino_seg import PDZSeg
import torch
import numpy as np
import cv2
import os
import torch.nn.functional as F
from PIL import Image
from utils.dataloader import Resize, ToTensor, Normalize
from torchvision import transforms
import argparse

origin_image_size = (1010, 1310)

def get_parser(**kwargs):
    parser = argparse.ArgumentParser(description='RCMNet')

    # Device
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help='Device to use for training, either "cuda:0" or "cpu".')

    # Dataset directories
    parser.add_argument('--crop_size', type=int, default=532, help='Crop size of images.')
    parser.add_argument('--prompt_image_dir', type=str, required=True, help='directory of the image.')
    parser.add_argument('--clean_image_dir', type=str, default=None, help='directory of raw image.')

    # Model parameters
    parser.add_argument('--backbone_size', type=str, default='base', required=True, choices=['base', 'large', 'gaint'], help='Backbone size.')
    # parser.add_argument('--num_classes', type=int, default=2, required=True, help='Number of classes.')
    # choices=['segformer', 'base', 'segmenter', 'FCNHead', 'ASPPHead'],
    parser.add_argument('--decoder_type', type=str, default='segformer', help='Type of decoder to use.')

    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint', help='Checkpoint file direction.')
    parser.add_argument('--dst_dir', type=str, default='./visualization', help='visualization result save direction.')
    
    parser.add_argument('--opacity', type=float, default=0.3, help='opacity of the mask.')
    
    return parser.parse_args()


def visualization(image_folder, model, device, clean_image_folder=None, dst_folder='./visualization/', opacity=0.3):
    '''
    Function:
        Visualization of images with predictive segmentation mask
    Inputs:
        image_folder: contain images with visual prompt
        clean_image_folder: raw image for visualization, if not provided,
                            only visualize the mask without overlaying on the raw image
        dst_folder: destination folder
    '''
    
    image_transform = transforms.Compose([Resize(532), ToTensor(), Normalize()])
    cnt = 0
    
    os.makedirs(dst_folder, exist_ok=True)
    
    for filename in os.listdir(image_folder):
        cnt += 1
        print(cnt)
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path)

        transformed_image = image_transform(image).to(device)

        pred_seg = model.interface(transformed_image.unsqueeze(0))
        pred_seg = torch.argmax(F.softmax(pred_seg, dim=1), dim=1)

        pred_seg = F.interpolate(pred_seg.unsqueeze(0).float(), size=origin_image_size, mode='bilinear').squeeze(0).cpu().permute(1, 2, 0).numpy()

        pred_seg = pred_seg.astype('uint8')

        # Create blue-colored mask: (0, 0, 255) for blue color
        blue_mask = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
        blue_mask = np.where(pred_seg == 1, [0,0,255], blue_mask).astype('uint8')
        
        if clean_image_folder is not None:
            clean_image_path = os.path.join(clean_image_folder, filename)
            ori_image =  cv2.cvtColor(cv2.imread(clean_image_path), cv2.COLOR_BGR2RGB)

            image_fin=cv2.addWeighted(blue_mask, opacity, ori_image, 1, 0)
            cv2.imwrite(os.path.join(dst_folder, filename), cv2.cvtColor(image_fin, cv2.COLOR_BGR2RGB)) 
        else:
            cv2.imwrite(os.path.join(dst_folder, filename), cv2.cvtColor(blue_mask, cv2.COLOR_BGR2RGB)) 


if __name__ == "__main__":
    args = get_parser()
    DEVICE = args.device
    num_classes = 2
    prompt_image_folder = args.prompt_image_dir
    clean_image_folder = args.clean_image_dir
    dst_folder = args.dst_dir
    opacity = args.opacity
    model = PDZSeg(backbone_size=args.backbone_size, r=4, lora_layer=None, image_shape=args.crop_size, num_classes=num_classes, \
                        decode_type = 'linear4', decoder_type=args.decoder_type).to(DEVICE)
    
    model.load_parameters(args.checkpoint_path, DEVICE)
    
    visualization(prompt_image_folder, model, DEVICE, clean_image_folder, dst_folder, opacity)