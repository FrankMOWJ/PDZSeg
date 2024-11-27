import torch
import torch.nn.functional as F
from arch.dino_seg import PDZSeg
from utils.constant import *
from utils.dataloader import get_dataloader
from utils.metrics import ComputeIoU, dice_coefficient
import argparse

def get_parser(**kwargs):
    parser = argparse.ArgumentParser(description='PDZSeg')

    # Device
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help='Device to use for training, either "cuda:0" or "cpu".')

    # Dataset directories
    parser.add_argument('--crop_size', type=int, default=532, help='Crop size of images.')
    parser.add_argument('--data_root', type=str, default='./dataset', help='directiory of the dataset.')
    parser.add_argument('--prompt_type', type=str, choices=['None', 'longScribble', 'bbox', 'shortScribble', 'point'], 
                        default='None', help='type of visual prompt to test')
    
    # Model parameters
    parser.add_argument('--backbone_size', type=str, default='base', required=True, choices=['base', 'large', 'gaint'], help='Backbone size.')
    # parser.add_argument('--num_classes', type=int, default=2, required=True, help='Number of classes.')
    # choices=['segformer', 'base', 'segmenter', 'FCNHead', 'ASPPHead'],
    parser.add_argument('--decoder_type', type=str, default='segformer', help='Type of decoder to use.')

    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint', help='Checkpoint file direction.')
    
    return parser.parse_args()


def evaluate_metrics(model, dataloader, num_classes, device):
    model.eval()
    compute_iou = ComputeIoU(num_class=num_classes)
    dice = {0: 0.0, 1: 0.0}
    with torch.no_grad():
        for (images, masks) in  dataloader:

            images, masks = images.to(device), masks.to(device)
                                                        
            pred_seg = model(images, masks).logits
            pred_seg = torch.argmax(F.softmax(pred_seg, dim=1), dim=1)

            compute_iou(pred_seg, masks) # the values of the confusion matrix are continuously added

            for i in range(num_classes):
                dice[i] += dice_coefficient(masks, pred_seg, i)
    
    mean_dice = 0.0
    for i in range(num_classes):
        dice[i] = dice[i] / len(dataloader)
        mean_dice += dice[i]
    ious = compute_iou.get_ious()
    miou = compute_iou.get_miou(None)
  
    print(f'ious per class:{ious}')
    print(f'miou: {miou}')
    print(f'dice per class: {dice}')
    print(f'mDice: {mean_dice / num_classes}')


if __name__ == "__main__":
    args = get_parser()
    backbone_size = 'base'
    decoder_type = 'segformer' # 'baseHead'
    DEVICE = torch.device(args.device)
    image_size = args.crop_size
    num_classes = 2
    prompt_type = args.prompt_type
    
    data_root = args.data_root
    checkpoint_path = args.checkpoint_path
    
    model = PDZSeg(backbone_size=backbone_size, r=4, lora_layer=None, image_shape=image_size, num_classes=num_classes, \
                        decode_type = 'linear4', decoder_type=decoder_type).to(DEVICE)
    
    model.load_parameters(checkpoint_path, DEVICE)

    train_loader, val_loader = get_dataloader(data_root, prompt_type, image_size=image_size)
    
    evaluate_metrics(model, val_loader, num_classes, DEVICE)


    '''robustness'''
    # for corrupt in corrupt_type:
    #     for corrupt_strength in [3]:
    #         print(f'type: {corrupt}, degree: {corrupt_strength}')
    #         corrupt_image_dir = os.path.join(val_corrupted_dir, f'{corrupt}/{corrupt_strength}_point')
    #         train_loader, val_loader = get_dataloader(2, image_size, train_image_dir, train_mask_dir, train_depth_dir, train_confidence_dir, \
    #                                             corrupt_image_dir, val_mask_dir, val_depth_dir, val_confidence_dir, 1, \
    #                                             confidence_normalization)
            
    #         miou, mDice = evaluate_corrupt(model, val_loader, DEVICE, corrupt, corrupt_strength)
    #         print()
            


    