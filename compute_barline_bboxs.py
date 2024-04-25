import os
import json
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

def analyze_image_for_barlines(image_path):
    """ 
    Analyze an image to find barlines, return a DataFrame with bounding box details. 
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or path is incorrect")
    
    height, width = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lower_bound = np.array([0, 172, 198])  # Lower bound of the color
    upper_bound = np.array([0, 172, 198])  # Upper bound of the color

    mask = cv2.inRange(image_rgb, lower_bound, upper_bound)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    data = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 2 or h < 2:
            x = x - 1 if w < 2 else x
            y = y - 1 if h < 2 else y
            w = max(2, w)
            h = max(2, h)
        orthogonal_bbox = [float(x), float(y), float(x + w), float(y + h)]

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # Add each box to the data list
        data.append({
            'filename': image_path.split('/')[-1].replace('_seg',''),
            'a_bbox': orthogonal_bbox,
            'o_bbox': [float(coord) for xs in box.tolist() for coord in xs],
            'area': w*h,
            'width': width,
            'height': height
        })

    # Create a DataFrame from the collected data
    return pd.DataFrame(data)

def process_dataset(json_directory, segmentation_directory, output_directory):
    json_files = [f for f in os.listdir(json_directory) if f.endswith('.json')]
    
    for json_file in tqdm(json_files):
        with open(os.path.join(json_directory, json_file), 'r') as file:
            data = json.load(file)
            images_df = pd.DataFrame(data['images'])
            filenames = images_df['filename'].tolist()

        output_csv = json_file.replace('.json', '.csv')
        output_path = os.path.join(output_directory, output_csv)

        # Prepare an empty DataFrame to collect all barline data
        barlines_data = pd.DataFrame(columns=['filename', 'a_bbox', 'o_bbox', 'area',
                                              'width', 'height', 'ann_id',
                                              'label', 'padded_bbox', 'duration', 
                                              'rel_position', 'duration_mask', 
                                              'rel_position_mask'])
        barlines_data.to_csv(output_path)
        
        ann_id = -1
        # Process each image
        for filename in tqdm(filenames):
            file_path = os.path.join(segmentation_directory,
                                     filename.replace('.png','_seg.png'))
            annotations = analyze_image_for_barlines(file_path)
            if not annotations.empty:
                annotations['ann_id'] = [ann_id - i for i in range(len(annotations))]
                ann_id -= len(annotations)  # Decrement ann_id for next file
                annotations['label'] = 156
                annotations['padded_bbox'] = annotations['a_bbox']
                annotations['duration'] = -1
                annotations['rel_position'] = 0
                annotations['duration_mask'] = 0
                annotations['rel_position_mask'] = 0
                barlines_data = pd.concat([barlines_data, annotations],
                                          ignore_index=True)
        
        # Save all collected barline data to CSV after processing all files
        if not barlines_data.empty:
            barlines_data.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process datasets to find barlines.')
    parser.add_argument('json_directory', type=str, help='Directory containing JSON files.')
    parser.add_argument('segmentation_directory', type=str, help='Directory containing segmented images.')
    parser.add_argument('output_directory', type=str, help='Output directory for CSV files.')
    
    args = parser.parse_args()
    
    process_dataset(args.json_directory, 
                    args.segmentation_directory, 
                    args.output_directory)

