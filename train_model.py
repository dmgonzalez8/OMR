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
from datetime import datetime

from preprocess_data import *
from image_distortion import *

"""start distortion transform wrappers"""
def random_distortion(image):
    # Your distortion logic here
    return image  # Return a PIL Image

class MyCustomTransform:
    def __call__(self, img):
        return random_distortion(img)
"""end distortion transform wrappers"""

"""start helper functions for model"""
def get_model(num_classes):
    # Load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(num_features, num_classes)
    
    return model

def collate_fn(batch):
    images = [item[0] for item in batch]  # Extract images
    batch_targets = [item[1] for item in batch]  # Extract targets

    # Stack images using torch.stack to create a batch
    images = torch.stack(images, 0)

    # Initialize the structure for collated targets
    # We need a list of dictionaries
    collated_targets = []
    keys = batch_targets[0].keys()

    # We iterate through each batch item to separate their components properly
    for index in range(len(batch)):  # Loop over items in the batch
        single_target = {}
        for key in keys:
            # We extract the component for each target key from each batch individually
            # This avoids any mixing of data across the batch items
            single_target[key] = batch_targets[index][key]
        collated_targets.append(single_target)

    return images, collated_targets
"""end helper functions for model"""

class MusicScoreDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        """
        Args:
            dataframe (DataFrame): Pandas DataFrame containing annotations.
            root_dir (string): Directory with all the images.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations = dataframe
        self.root_dir = root_dir
        # Set default transforms if none are provided
        if transform is None:
            self.transform = self.transform = transforms.Compose([
                MyCustomTransform(),  # Apply your custom distortion
                transforms.ToTensor(),  # Convert images to tensors
                transforms.Normalize(mean=[0], std=[1])  # set for grayscale
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations['filename'].iloc[idx])
        image = Image.open(img_name).convert("L") # use grayscale
        if self.transform:
            image = self.transform(image)
        a_boxes = torch.as_tensor(self.annotations['padded_a_bbox'].iloc[idx], dtype=torch.float32)
        o_boxes = torch.as_tensor(self.annotations['padded_o_bbox'].iloc[idx], dtype=torch.float32)
        durations = torch.as_tensor(self.annotations['duration'].iloc[idx], dtype=torch.float32)
        rel_positions = torch.as_tensor(self.annotations['rel_position'].iloc[idx], dtype=torch.float32)
        duration_masks = torch.as_tensor(self.annotations['duration_mask'].iloc[idx], dtype=torch.int32)
        rel_position_masks = torch.as_tensor(self.annotations['rel_position_mask'].iloc[idx], dtype=torch.int32)
        labels = torch.as_tensor(self.annotations['label'].iloc[idx], dtype=torch.int64)
        image_id = torch.tensor([self.annotations['img_id'].iloc[idx]], dtype=torch.int64)
        a_area = torch.as_tensor([self.annotations['a_area'].iloc[idx]], dtype=torch.float32)
        o_area = torch.as_tensor([self.annotations['o_area'].iloc[idx]], dtype=torch.float32)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)  # Assuming no crowd

        target = {}
        target["aboxes"] = a_boxes
        target["oboxes"] = o_boxes
        target["labels"] = labels
        target["durations"] = durations
        target["rel_positions"] = rel_positions
        target["duration_masks"] = duration_masks
        target["rel_position_masks"] = rel_position_masks
        target["image_id"] = image_id
        target["a_area"] = a_area
        target["o_area"] = o_area
        target["iscrowd"] = iscrowd

        return image, target
    
def train(model, data_loader, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            # Ensure targets are dictionaries and move them to the appropriate device
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            # Forward and backward passes
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} of {num_epochs}, Loss: {losses.item()}")
        
def train_faster_rcnn(train_df, test_df, image_directory):
    # make the DataSet
    # dataset = MusicScoreDataset(train_df, image_directory)

    # Setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = len(unique_labels) + 1  # Define the number of classes including background
    model = get_model(num_classes).to(device)
    data_loader = DataLoader(MusicScoreDataset(train_df, image_directory), 
                             batch_size=2, shuffle=True, collate_fn=collate_fn)
    optimizer = SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # Start training
    train(model, data_loader, optimizer)

    # export the model
    torch.save(model.state_dict(), f'./faster_rcnn_{str(datetime.now())}.pt')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process data into pickle binaries for DataLoader')
    parser.add_argument('json_directory', type=str, help='Root directory with annotation JSONs.')
    parser.add_argument('image_directory', type=str, help='Directory containing images.')
    parser.add_argument('labels_path', type=str, help='Path to csv file with labels')

    args = parser.parse_args()
    
    # preprocess the data and get aggregated dataframes
    train_df, test_df = preprocess_data(args.json_directory, args.labels_path)
    # train the model
    train_faster_rcnn(train_df, test_df, args.image_directory)