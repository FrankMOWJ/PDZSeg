import cv2 
import os

# Put the corresponding path here
image_folder = r''
mask_folder = r''
dst_dolder = r''

os.makedirs(dst_dolder, exist_ok=True)

def binarization(*images, threshold=0):
    """
    Binarize images.
    
    Args:
    *images: Any number of input images.
    threshold: Threshold value for binarization, default is 0.
    
    Returns:
    If only one image is provided, return a single binary image.
    If multiple images are provided, return a tuple containing all the binary images.
    """
    # List to store all binary images
    binary_images = []
    
    # Process each image
    for image in images:
        if image is None:
            print("Warning: Empty image detected, skipping this image.")
            continue
            
        # Convert to grayscale if the image is in color
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image  # Already a grayscale image
            
        # Apply binarization
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        binary_images.append(binary)
    
    # Return result
    return binary_images[0] if len(binary_images) == 1 else tuple(binary_images)


def find_contour(*binary_mask):
    all_contours = []
    
    for mask in binary_mask:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.append(contours)
        
    return all_contours[0] if len(all_contours) == 1 else tuple(all_contours)


def overlay_contours_on_image(image, contours_list, colors=None):
    """
    Overlay multiple sets of contours on the original image, each set with a different color.
    
    Args:
    image: The original image.
    contours_list: A list/tuple containing multiple sets of contours.
    colors: A list of colors, where each set of contours corresponds to a color. Default is None, which uses preset colors.
    
    Returns:
    The image with contours overlayed.
    """
    # Default colors: Red, Green, Blue, Yellow
    default_colors = [
        (0, 0, 255),    # red
    ]
    
    # Use provided colors or default colors
    colors = colors if colors else default_colors
    assert len(contours_list) == len(colors)
    
    # Copy the original image to avoid modifying it
    result = image.copy()
    
    # Draw each set of contours on the image
    for contours, color in zip(contours_list, colors):
        cv2.drawContours(result, contours, -1, color, 2)
    
    return result


def visualizate_with_mask_contour():
    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)
        mask_path = os.path.join(mask_folder, filename)
        
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)
        
        bin_mask = binarization(mask)
        
        contour = find_contour(*bin_mask)
        
        colors = [
           (0, 0, 255) 
        ]
       
        # overly the contour on the raw image
        result = overlay_contours_on_image(
            image, 
            contour, 
            colors
        )
       
        output_path = os.path.join(dst_dolder, filename)
        cv2.imwrite(output_path, result)


if __name__ == "__main__":
    visualizate_with_mask_contour()
        

       