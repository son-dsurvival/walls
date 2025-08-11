import numpy as np
import cv2
from ultralytics import YOLO

# Function to generate masks and save the mask image
def generate_mask(image_path, model, mask_path, y=282, x=190):
    # 1) Run the SEGMENTATION model (must be a *-seg model, not detect-only)
    results = model(image_path)
    r = results[0]
    image = cv2.imread(image_path)
    H, W = image.shape[:2]

    if r.masks is None:
        raise ValueError("Model returned no masks. Use a segmentation model (e.g., yolov8n-seg).")

    # 2) Build an integer-labeled mask: 0=background, 1..N = instances
    label_mask = np.zeros((H, W), dtype=np.uint16)
    for i, m in enumerate(r.masks.data):                   # m: (h,w) tensor
        m_np = m.cpu().numpy().astype(np.uint8)            # 0/1
        m_np = cv2.resize(m_np, (W, H), interpolation=cv2.INTER_NEAREST)
        label_mask[m_np.astype(bool)] = i + 1              # overwrite is fine

    # 3) Pick the clicked instance and black-out everything else
    label_val = int(label_mask[y, x])                      # NOTE: (row=y, col=x)
    if label_val == 0:
        raise ValueError("Clicked background (label 0).")
    seg_mask = (label_mask == label_val)

    if image.ndim == 2:                                    # grayscale
        seg_image = np.where(seg_mask, image, 0)
    else:                                                  # BGR (OpenCV)
        seg_image = np.where(seg_mask[..., None], image, 0)

    cv2.imwrite(mask_path, seg_mask)
    return seg_mask

image_path = 'WhatsApp Image 2025-08-09 at 13.58.55_38625e19.jpg'
model = YOLO("best.pt")
mask_path = "mask.png"

# Generate the mask and read it back as grayscale
mask=generate_mask(image_path, model, mask_path)

image = cv2.imread(image_path)  # BGR
overlay = image.copy()
colour=(0, 0, 255)
alpha=0.5
# Apply pure colour to the segment region
overlay[mask] = colour

# Blend overlay with original image
blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

# Save the result with colour baked in
cv2.imwrite("result.png", blended)