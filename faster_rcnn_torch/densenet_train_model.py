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
from torch.optim import SGD, Adam, Adadelta, AdamW, RMSprop
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from torch.utils.data._utils.collate import default_collate
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
from math import radians, cos, sin
import cv2
import ast
from datetime import datetime
import argparse
from torch.nn import DataParallel
from torchvision.models.detection.backbone_utils import BackboneWithFPN
import warnings
warnings.filterwarnings("ignore")

from preprocess_data import *
from image_preprocessing import *

"""start helper functions for model"""
def get_model(num_classes):
    # Load a DenseNet backbone pre-trained on ImageNet
    densenet_backbone = torchvision.models.densenet121(pretrained=True)
    # Remove the classifier part (fully-connected layer) at the end of DenseNet
    backbone = densenet_backbone.features

    # DenseNet outputs feature maps with 1024 channels, so we need to specify that here
    backbone.out_channels = 1024

    # Create an FPN on top of the DenseNet backbone
    # This will create a pyramid of feature maps with different resolutions
    backbone_with_fpn = BackboneWithFPN(backbone,
                                         return_layers={ '0': '0', '1': '1', '2': '2', '3': '3' },
                                         in_channels_list=[256, 512, 1024, 1024],
                                         out_channels=256,
                                         extra_blocks=None)

    # Use the FPN backbone to construct Faster R-CNN
    model = FasterRCNN(backbone_with_fpn, num_classes=num_classes)

    # Replace the head predictor with a new one
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

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
        masks = torch.as_tensor(self.annotations['padded_mask'].iloc[idx], dtype=torch.float32)
        durations = torch.as_tensor(self.annotations['duration'].iloc[idx], dtype=torch.float32)
        rel_positions = torch.as_tensor(self.annotations['rel_position'].iloc[idx], dtype=torch.float32)
        duration_masks = torch.as_tensor(self.annotations['duration_mask'].iloc[idx], dtype=torch.int32)
        rel_position_masks = torch.as_tensor(self.annotations['rel_position_mask'].iloc[idx], dtype=torch.int32)
        labels = torch.as_tensor(self.annotations['label'].iloc[idx], dtype=torch.int64)
        # image_id = torch.tensor([self.annotations['img_id'].iloc[idx]], dtype=torch.int64)
        a_area = torch.as_tensor([self.annotations['padded_a_area'].iloc[idx]], dtype=torch.float32)
        o_area = torch.as_tensor([self.annotations['padded_o_area'].iloc[idx]], dtype=torch.float32)
        mask_area = torch.as_tensor([self.annotations['padded_mask_area'].iloc[idx]], dtype=torch.float32)
        iscrowd = torch.zeros((len(a_boxes)), dtype=torch.int64)  # Assuming no crowd

        target = {}
        # there must be boxes for the model
        target["boxes"] = a_boxes
        # target["oboxes"] = o_boxes
        # target["masks"] = masks
        # there must be labels
        target["labels"] = labels
        target["durations"] = durations
        target["rel_positions"] = rel_positions
        target["duration_masks"] = duration_masks
        target["rel_position_masks"] = rel_position_masks
        # target["image_id"] = image_id
        target["a_area"] = a_area
        # target["o_area"] = o_area
        # target["mask_area"] = mask_area
        target["iscrowd"] = iscrowd

        return image, target


def train(device, model, model_file, train_loader, test_loader, optimizer, num_epochs=10, checkpoint_interval=5):
    """
    Train the model for a specified number of epochs and save periodic checkpoints.

    Parameters:
        device (torch.device): The device to use for training (e.g., CPU or GPU).
        model (torch.nn.Module): The object detection model to train.
        model_file (str): The base file name for model checkpoints.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        test_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        num_epochs (int): Number of epochs to train the model.
        checkpoint_interval (int): Interval for saving model checkpoints.
    """
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f"Starting epoch {epoch + 1}.")

        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            # Forward and backward passes
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # Accumulate running loss
            running_loss += losses.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        # Save checkpoint every `checkpoint_interval` epochs
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_filename = f'{model_file}_epoch_{epoch + 1}.pt'
            torch.save(model.state_dict(), checkpoint_filename)
            print(f"Saved model checkpoint at epoch {epoch + 1} to: {checkpoint_filename}")



def main(json_directory, optim, batch=2, num_epochs=5, checkpoint=None):
    image_directory = os.path.join(json_directory, 'processed_images/')
    train_df = pd.read_pickle(os.path.join(json_directory, 'deepscores_train.pkl'))
    test_df = pd.read_pickle(os.path.join(json_directory, 'deepscores_test.pkl'))
    unique_labels = pd.read_pickle(os.path.join(json_directory, 'unique_labels.pkl'))

    # Force the device to CPU
    device = torch.device('cpu')

    num_classes = len(unique_labels) + 1
    model = get_model(num_classes).to(device)

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        print("Loaded checkpoint model from:", checkpoint)

    # Adjust the DataLoader's `num_workers` to 0 or 1 for CPU training
    train_loader = DataLoader(MusicScoreDataset(train_df, image_directory), num_workers=1,
                              batch_size=batch, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(MusicScoreDataset(test_df, image_directory), num_workers=1,
                             batch_size=batch, shuffle=False, collate_fn=collate_fn)

    # Configure the optimizer
    if optim == "SGD":
        optimizer = SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        model_filename = f'{datetime.now().strftime("%Y%m%d%H%M")}_SGD'
    elif optim == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
        model_filename = f'{datetime.now().strftime("%Y%m%d%H%M")}_adamW'
    elif optim == "RMSprop":    
        optimizer = RMSprop(model.parameters(), lr=0.001, alpha=0.99, eps=1e-08, weight_decay=0.0001, momentum=0.9)
        model_filename = f'{datetime.now().strftime("%Y%m%d%H%M")}_RMSprop'
    elif optim == "Adadelta":    
        optimizer = Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0.0001)
        model_filename = f'{datetime.now().strftime("%Y%m%d%H%M")}_adadelta'
    else:
        optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
        model_filename = f'{datetime.now().strftime("%Y%m%d%H%M")}_adam'

    # Train the model on the CPU
    train(device, model, model_filename, train_loader, test_loader, optimizer, num_epochs)

    # Save the final model
    filename = f'./faster_rcnn_{datetime.now().strftime("%Y%m%d%H%M%S")}.pt'
    torch.save(model.state_dict(), filename)
    print(f"Saved model to: " + filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process data into pickle binaries for DataLoader')
    parser.add_argument('json_directory', type=str, help='Root directory with annotation JSONs.')
    parser.add_argument('optimizer', type=str, help='SGD, Adam, Adadelta, AdamW, RMSprop')
    parser.add_argument('batch_size', type=int, help='Images per batch.')
    parser.add_argument('num_epochs', type=int, help='How long to train.')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to the model checkpoint to continue training.')

    args = parser.parse_args()
    main(args.json_directory, args.optimizer, args.batch_size, args.num_epochs, args.checkpoint)
