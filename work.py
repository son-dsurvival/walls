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
    cv2.imwrite(mask_path, mask)
    return mask_path

image_path = 'image.png'
model = YOLO("best.pt")
mask_path = "mask.png"

# Generate the mask and read it back as grayscale
generate_mask(image_path, model, mask_path)
mask = cv2.imread(mask_path, 0)  # Load as grayscale
image = cv2.imread(image_path)


# Define overlay color (BGR)
overlay_color = (0, 0, 255)  # Red
alpha = 0.5

# Create color overlay
color_layer = np.full_like(image, overlay_color, dtype=np.float32)

# Prepare mask
mask_bool = (mask == 255).astype(np.float32)  # (H, W)
mask_3ch = np.repeat(mask_bool[:, :, None], 3, axis=2)  # (H, W, 3)

# Convert image to float
image = image.astype(np.float32)

# Blend where mask is active
blended = image * (1 - alpha * mask_3ch) + color_layer * (alpha * mask_3ch)
blended = np.clip(blended, 0, 255).astype(np.uint8)
cv2.imwrite("result.png", blended)