"""
Helmet Inference Script

- Load trained Faster R-CNN model
- Run inference on a single image
- Apply custom NMS logic to filter overlapping boxes
- Visualize the detection results
"""

import os
import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import box_iou

from Config.config import model_params
from Model.faster_rcnn import FasterRCNN

# ==== SETTINGS ====

MODEL_PATH = "./checkpoints/faster_rcnn_helmet.pth"
IMAGE_PATH = "./test/JPEGImages/test24_mov-0035_jpg.rf.ec62b8339dcc6d0221af1bde7b590380.jpg"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3  # background, helmet, no-helmet

# ==== LOAD MODEL ====

model = FasterRCNN(model_params, num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("✅ Model loaded successfully.")

# ==== LOAD IMAGE ====

image = Image.open(IMAGE_PATH).convert("RGB")
transform = T.ToTensor()
image_tensor = transform(image).unsqueeze(0).to(device)

# ==== PREDICT ====

with torch.no_grad():
    _, outputs = model(image_tensor)

boxes = outputs['boxes'].cpu()
scores = outputs['scores'].cpu()
labels = outputs['labels'].cpu()

# Filter low score boxes
keep = scores > 0.5
boxes = boxes[keep]
scores = scores[keep]
labels = labels[keep]

# ==== CUSTOM NMS ====

if len(boxes) > 1:
    ious = box_iou(boxes, boxes)
    ious.fill_diagonal_(0)

    keep_mask = torch.ones(len(boxes), dtype=torch.bool)

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if ious[i, j] > 0.7:
                if labels[i] != labels[j]:
                    if (labels[i] == 1 and scores[i] >= scores[j] - 0.05):
                        keep_mask[j] = False
                    elif (labels[j] == 1 and scores[j] >= scores[i] - 0.05):
                        keep_mask[i] = False
                    else:
                        keep_mask[j if scores[i] >= scores[j] else i] = False
                else:
                    keep_mask[j if scores[i] >= scores[j] else i] = False

    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    labels = labels[keep_mask]

# ==== VISUALIZE ====

fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(image)

for box, score, label in zip(boxes, scores, labels):
    if score < 0.6:
        continue

    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    color = 'lime' if label == 1 else 'red'
    label_str = 'helmet' if label == 1 else 'no-helmet'

    ax.add_patch(patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=color, facecolor='none'))
    ax.text(x1, y1, f'{label_str} {score:.2f}', color=color, fontsize=12, weight='bold')

plt.axis('off')
plt.show()
