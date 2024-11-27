import cv2
import numpy as np
import os

dataset_split = ['training', 'validation']

for dataset in dataset_split:
    src_image_folder = rf'./dataset/images/{dataset}'
    mask_folder = rf'./dataset/annotations/{dataset}'
    dst_folder = rf'./dataset/images_longScribble/{dataset}'

    os.makedirs(dst_folder, exist_ok=True)

    cnt = 0

    for filename in os.listdir(src_image_folder):
        cnt += 1
        image_path = os.path.join(src_image_folder, filename)
        mask_path = os.path.join(mask_folder, filename)
        
        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        img, mask = np.array(img), np.array(mask)

        output = np.where(mask[:, :, None] == 2, [0, 0, 255], img).astype('uint8')

        save_path = os.path.join(dst_folder, filename)
        cv2.imwrite(save_path, output)
        
        print(cnt)
        # break
