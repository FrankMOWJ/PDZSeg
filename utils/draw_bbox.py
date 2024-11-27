import cv2
import numpy as np
import os


dataset_split = ['training', 'validation']

for dataset in dataset_split:
    src_image_folder = rf'./dataset/images/{dataset}'
    mask_folder = rf'./dataset/annotations/{dataset}'
    dst_folder = rf'./dataset/images_bbox/{dataset}'
    os.makedirs(dst_folder, exist_ok=True)

    cnt = 0


    for filename in os.listdir(src_image_folder):
        cnt += 1
        image_path = os.path.join(src_image_folder, filename)
        mask_path = os.path.join(mask_folder, filename)
        
        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask>=1] = 1
        
        # find the contours 
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)  # get the external rectangle of the contour
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)


        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        save_path = os.path.join(dst_folder, filename)
        cv2.imwrite(save_path, output)
        
        print(cnt)
        # break
