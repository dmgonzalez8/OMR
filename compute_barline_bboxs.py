import os
import json
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing

def calculate_distance(df, merge_radius):
    centers = df['a_bbox'].apply(lambda x: ((x[0]+x[2])/2, (x[1]+x[3])/2))
    center_x, center_y = zip(*centers)
    center_x = np.array(center_x)
    center_y = np.array(center_y)
    distances = np.sqrt((center_x[:, None] - center_x[None, :]) ** 2 + (center_y[:, None] - center_y[None, :]) ** 2)
    return pd.DataFrame(distances < merge_radius)

def merge_close_boxes(data, merge_radius=20):
    # Calculate which boxes are close to each other
    is_close = calculate_distance(data, merge_radius)

    # Group close boxes
    groups = []
    for idx, row in enumerate(is_close.itertuples(index=False, name=None)):
        if not any([idx in group for group in groups]):
            close_indices = list(np.where(row)[0])
            groups.append(close_indices)

    # Merge groups
    merged_data = []
    for group in groups:
        subset = data.iloc[group]
        x_min = subset['a_bbox'].apply(lambda x: x[0]).min()
        y_min = subset['a_bbox'].apply(lambda x: x[1]).min()
        x_max = subset['a_bbox'].apply(lambda x: x[2]).max()
        y_max = subset['a_bbox'].apply(lambda x: x[3]).max()
        merged_data.append({
            'filename': subset.iloc[0]['filename'],
            'a_bbox': [x_min, y_min, x_max, y_max],
            'o_bbox': [x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min],
            'padded_a_bbox': [x_min, y_min, x_max, y_max],
            'padded_o_bbox': [x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min],
            'area': (x_max - x_min) * (y_max - y_min),
            'width': -1, # this is updated later (it's the image dimension, NOT the bbox)
            'height': -1 # this is updated later (it's the image dimension, NOT the bbox)
        })

    # Create a new DataFrame with the merged data
    return pd.DataFrame(merged_data)

def analyze_image_for_measures(image_path):

    ## Eren's code here
    
    return pd.DataFrame(data)

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
            # 'o_bbox': [float(coord) for xs in box.tolist() for coord in xs],
            # 'padded_a_bbox': orthogonal_bbox,
            # 'padded_o_bbox': [float(coord) for xs in box.tolist() for coord in xs],
            # 'area': w*h,
            # 'width': width,
            # 'height': height
        })

    # Create a DataFrame from the collected data
    return pd.DataFrame(data)


def process_file(json_file, json_directory, segmentation_directory, merge_radius):
    print(f'Processing {json_file}')
    output_directory = json_directory

    with open(os.path.join(json_directory, json_file), 'r') as file:
        data = json.load(file)
        images_df = pd.DataFrame(data['images'])
        filenames = images_df['filename'].tolist()

    output_csv = json_file.replace('.json', '_barlines.csv')
    output_path = os.path.join(output_directory, output_csv)

    # Prepare an empty DataFrame to collect all barline data
    barlines_data = pd.DataFrame(columns=['filename', 'a_bbox', 'o_bbox', 
                                            'padded_a_bbox', 'padded_o_bbox', 
                                            'area', 'width', 'height', 'ann_id',
                                            'label', 'duration', 'rel_position', 
                                            'duration_mask', 'rel_position_mask'])
    # barlines_data.to_csv(output_path)
    
    ann_id = -1
    # Process each image
    for filename in tqdm(filenames):
        file_path = os.path.join(segmentation_directory,
                                    filename.replace('.png','_seg.png'))
        # Get barlines in a df
        barline_annotations = analyze_image_for_barlines(file_path)
        # Eren - Get measures in a df
        measure_annotation = analyze_image_for_measures(file_path)
        # concat measures and barlines
        annotations = pd.concat([barline_annotations, measure_annotation],
                                        ignore_index=True)
        if not annotations.empty:
            # merge duplicates
            annotations = merge_close_boxes(annotations, merge_radius)
            # update other cells
            annotations['ann_id'] = [ann_id - i for i in range(len(annotations))]
            ann_id -= len(annotations)  # Decrement ann_id for next file
            annotations['label'] = 156
            annotations['duration'] = -1
            annotations['rel_position'] = 0
            annotations['duration_mask'] = 0
            annotations['rel_position_mask'] = 0
            barlines_data = pd.concat([barlines_data, annotations],
                                        ignore_index=True)
    
    # Save all collected barline data to CSV after processing all files
    if not barlines_data.empty:
        barlines_data.to_csv(output_path, index=False)


def process_dataset(json_directory, segmentation_directory, n_jobs=4, merge_radius=None):
    json_files = [f for f in os.listdir(json_directory) if f.endswith('.json')]

    # Number of processes to run in parallel
    pool = multiprocessing.Pool(n_jobs)
    
    for json_file in json_files:
        pool.apply_async(process_file, args=(json_file, json_directory, 
                                             segmentation_directory, merge_radius))
    
    pool.close()
    pool.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process datasets to find barlines.')
    parser.add_argument('json_directory', type=str, help='Directory containing JSON files.')
    parser.add_argument('segmentation_directory', type=str, help='Directory containing segmented images.')
    parser.add_argument('n_jobs', type=int, help='How many processes to spawn.')
    parser.add_argument('merge_radius', type=int, help='Maximum distance in pixels to merge.')

    args = parser.parse_args()
    
    process_dataset(args.json_directory, 
                    args.segmentation_directory,
                    args.n_jobs, args.merge_radius)
    print('Done processing.')
