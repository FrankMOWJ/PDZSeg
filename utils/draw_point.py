import cv2
import numpy as np
import os
from PIL import Image

def random_click(mask, point_labels = 1):
    # check if all masks are black
    max_label = max(set(mask.flatten()))
    if max_label == 0:
        point_labels = max_label
    # max agreement position
    indices = np.argwhere(mask == max_label) 
    # ramdom choose a point 
    return point_labels, indices[np.random.randint(len(indices))]

def draw_red_point(image, point, radius=15):
    """
    image: BGR format of image read by OpenCV
    point: (x,y)
    radius: r
    """
    img_copy = image.copy()
    # draw a red point
    cv2.circle(img_copy, (int(point[0]), int(point[1])), radius, (255,0,0), -1)
    return img_copy

dataset_split = ['training', 'validation']

for dataset in dataset_split:
    src_image_folder = rf'./dataset/images/{dataset}'
    mask_folder = rf'./dataset/annotations/{dataset}'
    dst_folder = rf'./dataset/images_point/{dataset}'
    
    os.makedirs(dst_folder, exist_ok=True)

    cnt = 0
    for filename in os.listdir(src_image_folder):
        cnt += 1
        image_path = os.path.join(src_image_folder, filename)
        mask_path = os.path.join(mask_folder, filename)
        
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        
        # get cordinate 
        _, cordinate = random_click(np.array(mask)) 
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        redLine_image = draw_red_point(image, cordinate[::-1])
        
        cv2.imwrite(os.path.join(dst_folder, filename), cv2.cvtColor(redLine_image, cv2.COLOR_BGR2RGB))
        print(cnt)
        # break   
