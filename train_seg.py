import torch
from utils.dataloader import get_dataloader
from utils.metrics import ComputeIoU
from arch.dino_seg import PDZSeg
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os
import logging
from datetime import datetime

# best miou
best_miou = 0.0

def get_parser(**kwargs):
    parser = argparse.ArgumentParser(description='PDZSeg')
    
    # Seed
    parser.add_argument('--seed', type=int, default=3407, help='Random seed.')

    # Device
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help='Device to use for training, either "cuda:0" or "cpu".')

    # Dataset directories
    parser.add_argument('--crop_size', type=int, default=532, help='Crop size of images.')
    parser.add_argument('--data_root', type=str, default='./dataset', help='directiory of the dataset.')
    parser.add_argument('--prompt_type', type=str, choices=['None', 'longScribble', 'bbox', 'shortScribble', 'point'], 
                        default='None', help='type of visual prompt to use')

    # Model parameters
    parser.add_argument('--backbone_size', type=str, default='base', required=True, choices=['base', 'large', 'gaint'], help='Backbone size.')
    parser.add_argument('--num_classes', type=int, default=2, required=True, help='Number of classes.')
    parser.add_argument('--ignore_index', type=list, default=[], help='Indices to ignore during training.')
    # choices=['segformer', 'base', 'segmenter', 'FCNHead', 'ASPPHead'],
    parser.add_argument('--decoder_type', type=str, default='segformer', help='Type of decoder to use.')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs for training.')
    parser.add_argument('--init_lr', type=float, default=0.001, help='Initial learning rate.')

    # Log and checkpoint
    parser.add_argument('--no_log', action='store_true', help='whethere to do logging.')
    parser.add_argument('--log_dir', type=str, default='./log', help='Log file direction.')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='Checkpoint file direction.')
    
    # resume 
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from.')
    # suffix
    parser.add_argument('--suffix', type=str, default=None, help='suffix added to log and checkpoint name.')
    
    return parser.parse_args()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model, val_result: float, checkpoint_path, epoch, logger):
    global best_miou
    if val_result >= best_miou:
        logger.info(f'save checkpoint at Epoch {epoch}')
        model.save_parameters(checkpoint_path)
        best_miou = val_result

def evaluate(model, dataloader, num_classes, device, logger, ignore_index=[]):
    model.eval()
    global best_miou
    compute_iou = ComputeIoU(num_class=num_classes)
    with torch.no_grad():
        for (images, masks) in  dataloader:

            images, masks = images.to(device), masks.to(device)                                     
            pred_seg = model(images, masks).logits
            pred_seg = torch.argmax(F.softmax(pred_seg, dim=1), dim=1)

            compute_iou(pred_seg, masks)

    ious = compute_iou.get_ious()
    miou = compute_iou.get_miou(ignore=ignore_index)
    best_miou = max(best_miou, miou)
    cfsmatrix = compute_iou.get_cfsmatrix()

    logger.info(f'ious per class:{ious}')
    logger.info(f'miou: {miou}')
    # if you want to see the confuse matrix, uncomment it
    # logger.info(f'confuse matrix: {cfsmatrix}')
    
    return miou


def train(model, train_loader, val_loader, optimizer, num_epochs, device, save_dir, logger):
    model.to(device)
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        for (images, masks) in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images, masks)
            pred_seg, loss = outputs.logits, outputs.loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)

        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        miou = evaluate(model, val_loader, num_classes, device, logger)
        scheduler.step()

        if epoch % 1 == 0 and epoch != 0:
            checkpoint_path = f'{save_dir}/best.pt'
            save_checkpoint(model, miou, checkpoint_path, epoch, logger)
        
    
    logger.info('Finished Training')

if __name__ == "__main__":
    args = get_parser()
    
    seed_everything(args.seed)
    num_classes = args.num_classes
    decoder_type = args.decoder_type
    epoch = args.epoch
    batch_size = args.batch_size
    init_lr = args.init_lr
    DEVICE = args.device
    log_dir = args.log_dir
    image_size = args.crop_size
    backbone_size = args.backbone_size
    prompt_type = args.prompt_type
    time_stamp = datetime.now().strftime('%Y%m%d-%H%M')
    name = f'{backbone_size}_{decoder_type}_{prompt_type}_{image_size}_lr{init_lr}_epoch{epoch}' if args.suffix is None else \
            f'{backbone_size}_{decoder_type}_{prompt_type}_{image_size}_lr{init_lr}_epoch{epoch}_{args.suffix}'

    log_save_dir =  rf'{log_dir}/{name}/{time_stamp}'
    if not args.no_log:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(rf'{log_dir}/{name}/{time_stamp}', exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f'{log_save_dir}/log.txt'),
                        logging.StreamHandler()
                    ] if not args.no_log else [
                        logging.StreamHandler()
                    ])
    logger = logging.getLogger(__name__)
    logger.info('************* Setting *************')
    logger.info(f'   Seed = {args.seed}')
    logger.info(f'   Visual prompt type = {prompt_type}')
    logger.info(f'   Image size = {args.crop_size}')
    logger.info(f'   Decoder = {decoder_type}')
    logger.info(f'   Num_classes = {num_classes}')
    logger.info(f'   Epoch = {epoch}')
    logger.info(f'   Batch size = {batch_size}')
    logger.info(f'   lr = {init_lr}')
    logger.info(f'************************************')
    # Instantiate Surgical-DINO
    model = PDZSeg(backbone_size=backbone_size, r=4, lora_layer=None, image_shape=(image_size,image_size), num_classes=num_classes, \
                        decode_type='linear4', decoder_type=args.decoder_type).to(DEVICE)
    
    # data
    train_loader, val_loader = get_dataloader(args.data_root, prompt_type, image_size, batch_size, num_classes)
    
    if args.resume is not None:
        model.load_parameters(f'{args.resume}', DEVICE)
        print('evaluating...')
        evaluate(model, val_loader, num_classes, DEVICE, logger)
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    # scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch, verbose=True)
    checkpoint_save_dir =  log_save_dir
    train(model, train_loader, val_loader, optimizer, epoch, DEVICE, checkpoint_save_dir, logger)

    model.save_parameters( f'{checkpoint_save_dir}/last.pt')
    logger.info(f'best val miou: {best_miou}')
    logger.info(f'log and checkpoint are saved at {log_save_dir}')

