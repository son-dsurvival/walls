import numpy as np
import cv2
from ultralytics import YOLO

# Function to generate masks and save the mask image
def generate_mask(image_path, model, mask_path):
    results = model(image_path)
    result = results[0]  # Perform object detection
    image= cv2.imread(image_path)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

# Loop through each detected object and apply segmentation mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

# Loop over detected masks
    for seg_mask, cls in zip(result.masks.data, result.boxes.cls):
        if int(cls) == 0:  # Class 0 = 'wall'
            bin_mask = (seg_mask.cpu().numpy() > 0.5).astype(np.uint8) * 255
            bin_mask = cv2.resize(bin_mask, (image.shape[1], image.shape[0]))
            mask = cv2.bitwise_or(mask, bin_mask)
    # Optional: visualize mask over original image
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite(mask_path, masked_image)
    return mask_path

image_path = 'image.png'
model = YOLO("best.pt")
mask_path = "mask.png"

# Generate the mask and read it back as grayscale
generate_mask(image_path, model, mask_path)
mask = cv2.imread(mask_path, 0)  # Load as grayscale
image = cv2.imread(image_path)

# Desired color (B, G, R) - let's say red
color = (0, 0, 255)



# Broadcast mask to 3D
overlay_color = (0, 0, 255)  # Red in BGR
color_layer = np.full_like(image, overlay_color)

# Set transparency (alpha): 0 = no effect, 1 = full red
alpha = 0.5  # You can tune this

# Expand mask to 3 channels
mask_3ch = (mask == 255)[:, :, None].astype(np.uint8)

# Blend: only where mask == 255
blended = image.copy()
blended = blended.astype(np.float32)
blended = blended * (1 - alpha * mask_3ch) + color_layer * (alpha * mask_3ch)
blended = blended.astype(np.uint8)
cv2.imwrite("result.png", blended)