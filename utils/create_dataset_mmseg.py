import numpy as np
import cv2
import os
from PIL import Image

src_folder = '/PATH/TO/DATASET' # ./dataset
dst_root = os.path.join(src_folder, 'annotations_mmseg')

os.makedirs(dst_root, exist_ok=True)

def change_mask():
    for dataset_type in ['training', 'validation']:
        dst_folder = os.path.join(dst_folder, dataset_type)
        for i, mask_file in enumerate(os.listdir(src_folder)):
            print(f'[{i+1} / {len(os.listdir(src_folder))}]')
            mask_path = os.path.join(src_folder, mask_file)
            mask = Image.open(mask_path)
            mask = np.array(mask)
            
            mask[mask==2] = 1
            
            assert (np.unique(mask) == [0, 1]).all() == True

            # save new mask
            save_path = os.path.join(dst_folder, mask_file)
            mask.save(save_path)
        
change_mask()
