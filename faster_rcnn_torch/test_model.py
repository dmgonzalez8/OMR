import argparse
import os
import torch
import pandas as pd
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from dataset import MusicScoreDataset  # Make sure to define this in your imports

def get_model(num_classes, checkpoint_path=None):
    model = fasterrcnn_resnet50_fpn(pretrained=True if checkpoint_path is None else False)
    num_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(num_features, num_classes)

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    
    return model

def main(args):
    unique_labels = pd.read_pickle(args.unique_labels)
    num_classes = len(unique_labels) + 1

    model = get_model(num_classes, args.checkpoint)
    model.eval()

    dataset = MusicScoreDataset(pd.read_pickle(args.dataframe), args.images_dir, transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))

    for idx in range(len(dataset)):
        image, target = dataset[idx]
        prediction = model([image])

        draw = ImageDraw.Draw(image)
        for element in prediction[0]['boxes']:
            draw.rectangle([(element[0], element[1]), (element[2], element[3])], outline='red', width=3)
        
        image.save(os.path.join(args.output_dir, f"output_{idx}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a pre-trained Faster R-CNN model on annotated musical scores.")
    parser.add_argument("--checkpoint", help="Path to the model checkpoint file.", required=True)
    parser.add_argument("--dataframe", help="Path to the pandas dataframe pickle file.", required=True)
    parser.add_argument("--images_dir", help="Directory containing images.", required=True)
    parser.add_argument("--unique_labels", help="Pickle file containing unique labels.", required=True)
    parser.add_argument("--output_dir", help="Directory to save output images.", required=True)
    
    args = parser.parse_args()
    main(args)
