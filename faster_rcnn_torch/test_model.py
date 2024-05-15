import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
import re
from shapely.geometry import Polygon, LineString, Point
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from torch.optim import SGD, Adam, Adadelta
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from torch.utils.data._utils.collate import default_collate
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
from math import radians, cos, sin
import cv2
import ast

# Define the backbone
backbone = resnet_fpn_backbone('resnet50', pretrained=True)
# Create the model
model = FasterRCNN(backbone, num_classes=num_classes)  # num_classes includes the background
model.load_state_dict(torch.load('./exported_model.pt'))
model.eval()  # Set the model to evaluation mode

# Convert DataFrame to a dictionary
label_dict = dict(zip(unique_labels['label'], unique_labels['name']))

# Load the image
image_path = './data/ds2_dense/images/lg-101766503886095953-aug-beethoven--page-4.png'
image = Image.open(image_path).convert("L")
image_tensor = F.to_tensor(image).unsqueeze(0)  # Convert image to tensor

# Perform prediction
with torch.no_grad():
    predictions = model(image_tensor)

image = Image.open(image_path).convert("RGB")
# Draw predictions on the image
draw = ImageDraw.Draw(image)
# Specify a larger font size for the annotations
font = ImageFont.load_default()  # Currently, load_default does not support size adjustment in PIL

for element in range(len(predictions[0]['boxes'])):
    boxes = predictions[0]['boxes'][element].cpu().numpy().astype(int)
    label_id = predictions[0]['labels'][element].item()
    score = predictions[0]['scores'][element].item()

    # Look up the label name using the label dictionary
    label_name = label_dict.get(label_id, 'Unknown')  # Default to 'Unknown' if not found

    if score > 0.5:  # filter out low-confidence predictions
        draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])], outline='red', width=3)
        draw.text((boxes[0], boxes[1]-40), f'{label_name}:{score:.2f}', fill='blue', font=font)  
# Save or show image
image.show()