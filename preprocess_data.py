import os
import re
import ast
import json
import math
from math import radians, cos, sin
import random
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon, LineString, Point
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
from image_preprocessing import *

"""start of helper functions for remapping old cat_id to new label"""
def map_cat_ids_to_classes(class_mapping, cat_ids):
    # replace each cat_id list with corresponding new label list
    return [class_mapping.get(str(cat_id)) for cat_id in cat_ids]

def clean_labels(label_list):
    # set comprehension to remove duplicates and filter out None values
    return list({label for label in label_list if label is not None})
    
def select_highest_precedence(label_list):
    # return the label with highest precedence
    return max(label_list)
"""end of helper functions for remapping old cat_id to new label"""

"""start of helper functions for relative positions and durations"""
def extract_info(comment):
    # extract duration and relative position from comments
    duration = re.search(r'duration:(\d+);', comment)
    rel_position = re.search(r'rel_position:(-?\d+);', comment)
    return [int(duration.group(1)) if duration else None, 
            int(rel_position.group(1)) if rel_position else None]
##
## Eric- add code here to impute the relative positions
# Function to estimate missing 'rel_position' based on known positions
def estimate_missing_positions(series):
    # Use some method to estimate missing values
    return series.interpolate(method='linear', limit_direction='both')


def process_dataframe(df):
    # First, extract info from comments
    df[['duration', 'rel_position']] = df['comments'].apply(extract_info).tolist()

    # Apply the imputation function to the 'rel_position' column grouped by 'image_id'
    df['rel_position'] = df.groupby('img_id')['rel_position'].transform(estimate_missing_positions)

    return df
##
"""end of helper functions for relative positions and durations"""

"""start of bounding box helper functions"""
def add_padding_oriented(obb):
    xs = obb[::2]  # Extract all x coordinates
    ys = obb[1::2]  # Extract all y coordinates
    
    # Check for zero width or height
    if max(xs) == min(xs):
        xs = [x - 1 if x == min(xs) else x + 1 for x in xs]
    if max(ys) == min(ys):
        ys = [y - 1 if y == min(ys) else y + 1 for y in ys]
    
    # Combine xs and ys back into a single list
    adjusted_obb = [val for pair in zip(xs, ys) for val in pair]
    return adjusted_obb

def add_padding(bbox):
    if len(bbox)==4:
        x_min, y_min, x_max, y_max = bbox
        if x_min == x_max:
            x_min -= 1
            x_max += 1
        if y_min == y_max:
            y_min -= 1
            y_max += 1
        return [x_min, y_min, x_max, y_max]
    else:
        return add_padding_oriented(bbox)
    
def corners_to_yolo(bbox, img_width, img_height):
    polygon = Polygon([(bbox[i], bbox[i + 1]) for i in range(0, len(bbox), 2)])
    min_rect = polygon.minimum_rotated_rectangle

    # Check if the minimum rotated rectangle is a point
    if isinstance(min_rect, Point):
        # Handle the case where the shape is a point by creating a small box around it
        x, y = min_rect.x, min_rect.y
        min_rect = Polygon([(x-1, y-1), (x+1, y-1), (x+1, y+1), (x-1, y+1)])
        return 'invalid'

    # check if symbol is a line and add 1 px padding if so (almost always stems)
    elif isinstance(min_rect, LineString):
        # Handle the case where the shape is a line by padding
        x_coords, y_coords = zip(*min_rect.coords)
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        min_rect = Polygon([(min_x-1, min_y-1), (max_x+1, min_y-1), (max_x+1, max_y+1), (min_x-1, max_y+1)])

    corners = np.array(min_rect.exterior.coords)
    edge1 = np.linalg.norm(corners[0] - corners[1])
    edge2 = np.linalg.norm(corners[1] - corners[2])
    width = max(edge1, edge2)
    height = min(edge1, edge2)
    center = min_rect.centroid.coords[0]
    center_x = center[0]
    center_y = center[1]
    angle = np.rad2deg(np.arctan2(corners[1][1] - corners[0][1], corners[1][0] - corners[0][0]))

    center_x /= img_width
    center_y /= img_height
    width /= img_width
    height /= img_height

    return [center_x, center_y, width, height, angle]
    
def to_yolo(row):
    # convert corners to YOLO format for each row in the DataFrame
    return corners_to_yolo(row['o_bbox'], row['width'], row['height'])

def calculate_bbox_area(bbox):
    if len(bbox) == 4:  # Orthogonal bbox [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = bbox
        return int((x_max - x_min) * (y_max - y_min))
    elif len(bbox) == 8:  # Oriented bbox [x1, y1, x2, y2, x3, y3, x4, y4]
        # Rearrange into (x, y) pairs
        points = [(bbox[i], bbox[i + 1]) for i in range(0, len(bbox), 2)]
        # Create a polygon and calculate the area
        polygon = Polygon(points)
        return int(polygon.area)
    else:
        raise ValueError("Invalid bounding box format")
"""end of bounding box helper functions"""

"""start barline processing helpers"""
def convert_str_to_list(coord_str):
    return ast.literal_eval(coord_str)

#
# more code here to compute full barlines/measures
# input - df from below
# output - df to be concatenated below
#
"""end barline processing helpers"""

def preprocess_data(json_directory, labels_path):
    """
    Add docstring
    """
    # Load the annotation data and labels from their respective files
    with open(json_directory+'deepscores_train.json') as file:
        train_json = json.load(file)
    with open(json_directory+'deepscores_test.json') as file:
        test_json = json.load(file)
    raw_labels = pd.read_csv(labels_path)
    raw_labels.head()
    print("Read JSON complete.")

    # make the final list of labels for Torch
    unique_labels = raw_labels[['label', 'name']]
    unique_labels = unique_labels.drop_duplicates(subset=['label'])
    unique_labels = unique_labels.sort_values(by=['label']).reset_index(drop=True)
    print("Make unique label list complete.")

    # extract the images and annotations (obboxs) tables
    train_images = pd.DataFrame(train_json['images'])
    train_obboxs = pd.DataFrame(train_json['annotations']).T
    test_images = pd.DataFrame(test_json['images'])
    test_obboxs = pd.DataFrame(test_json['annotations']).T
    train_images.rename(columns={'id': 'img_id'}, inplace=True)
    test_images.rename(columns={'id': 'img_id'}, inplace=True)
    print("Extracted tables from JSON.")

    # update the annotation/obboxs table with the new labels (existing labels have issues)
    class_mapping = dict(zip(raw_labels['old_id'].astype(str), raw_labels['label']))
    train_obboxs['label'] = train_obboxs['cat_id'].apply(lambda x: map_cat_ids_to_classes(class_mapping, x))
    test_obboxs['label'] = test_obboxs['cat_id'].apply(lambda x: map_cat_ids_to_classes(class_mapping, x))
    train_obboxs['label'] = train_obboxs['label'].apply(clean_labels)
    test_obboxs['label'] = test_obboxs['label'].apply(clean_labels)
    train_obboxs['label'] = train_obboxs['label'].apply(select_highest_precedence)
    test_obboxs['label'] = test_obboxs['label'].apply(select_highest_precedence)
    print("Updated labels to new mapping.")

    # update the annotations/obboxs table with the rel_position and duration extracted from comments
    train_obboxs[['duration', 'rel_position']] = train_obboxs['comments'].apply(extract_info).tolist()
    test_obboxs[['duration', 'rel_position']] = test_obboxs['comments'].apply(extract_info).tolist()
    # create a mask for the duration to mark where the duration is relevent
    train_obboxs['duration_mask'] = train_obboxs['duration'].notna().astype(int)
    test_obboxs['duration_mask'] = test_obboxs['duration'].notna().astype(int)
    # set items with no duration to -1
    train_obboxs['duration'] = train_obboxs['duration'].replace(np.nan,-1)
    test_obboxs['duration'] = test_obboxs['duration'].replace(np.nan,-1)
    # create a mask for the rel_position to mark where the rel_position is relevent
    train_obboxs['rel_position_mask'] = train_obboxs['rel_position'].notna().astype(int)
    test_obboxs['rel_position_mask'] = test_obboxs['rel_position'].notna().astype(int)
    ##
    ## Eric- the two lines of code can be adjusted to use your vectorize function
    # set items with no rel_position to 50 (nothing has a position this high)
    train_obboxs['rel_position'] = train_obboxs['rel_position'].replace(np.nan,50)
    test_obboxs['rel_position'] = test_obboxs['rel_position'].replace(np.nan,50)
    print("Imputed durations and positions.")

    # padding used to be here

    # clean up the dataframes and cast columns to correct type
    train_obboxs.reset_index(inplace=True)
    test_obboxs.reset_index(inplace=True)
    train_obboxs.drop(['cat_id','comments'], axis=1, inplace=True)
    test_obboxs.drop(['cat_id','comments'], axis=1, inplace=True)
    train_obboxs.rename(columns={'index': 'ann_id'}, inplace=True)
    test_obboxs.rename(columns={'index': 'ann_id'}, inplace=True)
    train_obboxs['ann_id'] = train_obboxs['ann_id'].astype(int)
    test_obboxs['ann_id'] = test_obboxs['ann_id'].astype(int)
    train_obboxs['area'] = train_obboxs['area'].astype(int)
    test_obboxs['area'] = test_obboxs['area'].astype(int)
    train_obboxs['img_id'] = train_obboxs['img_id'].astype(int)
    test_obboxs['img_id'] = test_obboxs['img_id'].astype(int)
    print("Cleaned up tables.")

    # join the bounding box annotations and image information
    train_data = pd.merge(train_obboxs, train_images, on='img_id', how='inner')
    test_data = pd.merge(test_obboxs, test_images, on='img_id', how='inner')
    train_data.drop('ann_ids', axis=1, inplace=True)
    test_data.drop('ann_ids', axis=1, inplace=True)
    print("Merged bboxs and images together.")

    # concat the barlines annotations
    train_barlines = pd.read_csv(json_directory+'deepscores_train_barlines.csv')
    test_barlines = pd.read_csv(json_directory+'deepscores_test_barlines.csv')
    train_barlines['a_bbox'] = train_barlines['a_bbox'].apply(convert_str_to_list)
    train_barlines['o_bbox'] = train_barlines['o_bbox'].apply(convert_str_to_list)
    test_barlines['a_bbox'] = test_barlines['a_bbox'].apply(convert_str_to_list)
    test_barlines['o_bbox'] = test_barlines['o_bbox'].apply(convert_str_to_list)

    #
    ## Eren add code here to process the barlines into full measure lines or bboxes
    ## train_barlines = process_barlines(train_barlines)
    #
    
    train_data = pd.concat([train_data, train_barlines], ignore_index=True)
    test_data = pd.concat([test_data, test_barlines], ignore_index=True)
    print("Processed barlines into main df.")
    # areas/yolo used to be here

    # AGGREGATE everything so that each row contains all the data for a single image
    train_data_agg = train_data.groupby('filename').agg({
        'ann_id': lambda x: list(x),
        'a_bbox': lambda x: list(x),
        'o_bbox': lambda x: list(x),
        'area': lambda x: list(x),
        'duration': lambda x: list(x),
        'duration_mask': lambda x: list(x),
        'rel_position': lambda x: list(x), 
        'rel_position_mask': lambda x: list(x),
        'label': lambda x: list(x),
        'img_id': 'first',  # assuming all entries per image have the same img_id
        'width': 'first',   # assuming all entries per image have the same width
        'height': 'first'  # assuming all entries per image have the same height
    }).reset_index()
    test_data_agg = test_data.groupby('filename').agg({
        'ann_id': lambda x: list(x),
        'a_bbox': lambda x: list(x),
        'o_bbox': lambda x: list(x),
        'area': lambda x: list(x),
        'duration': lambda x: list(x),
        'duration_mask': lambda x: list(x),
        'rel_position': lambda x: list(x), 
        'rel_position_mask': lambda x: list(x),
        'label': lambda x: list(x),
        'img_id': 'first',  # assuming all entries per image have the same img_id
        'width': 'first',   # assuming all entries per image have the same width
        'height': 'first'  # assuming all entries per image have the same height
    }).reset_index()
    print("Aggregated dfs down to single row per image.")

    # add a flag for resizing (will be set by future functions)
    train_data_agg['resized'] = 0
    test_data_agg['resized'] = 0
    # get the width and height for the model- all bigger/smaller images will be resized to this 
    median_width = int(train_data_agg['width'].median())
    median_height = int(train_data_agg['height'].median())
 
    # image distortion - make a new folder to hold the processed images
    image_path = os.path.join(json_directory, 'images')
    output_path = os.path.join(json_directory, 'processed_images')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for index, row in tqdm(train_data_agg.iterrows()):
        source_image = os.path.join(image_path, row['filename'])
        a_bboxes = row['a_bbox']
        o_bboxes = row['o_bbox']
        resized_img = None
        new_abbs, new_obbs = None, None
        # resize if needed
        if row['width'] != median_width or row['height'] != median_height:
            resized_img, new_abbs, new_obbs = resize_image(source_image, 
                                                          a_bboxes, 
                                                          o_bboxes, 
                                                          median_width, 
                                                          median_height)
            # update df
            train_data_agg.at[index, 'a_bbox'] = new_abbs
            train_data_agg.at[index, 'o_bbox'] = new_obbs
            train_data_agg.at[index, 'width'] = median_width
            train_data_agg.at[index, 'height'] = median_height
            train_data_agg.at[index, 'resized'] = 1

        if resized_img is not None:
            output_img, new_abbs, new_obbs = apply_random_distortion(resized_img, 
                                                                     new_abbs, 
                                                                     new_obbs)
        else:
            output_img, new_abbs, new_obbs = apply_random_distortion(source_image, 
                                                                     a_bboxes, 
                                                                     o_bboxes)
        # write image to disk
        output_img_path = os.path.join(output_path, row['filename'])
        output_img.save(output_img_path)  # Save the processed image
        # update df
        train_data_agg.at[index, 'a_bbox'] = new_abbs
        train_data_agg.at[index, 'o_bbox'] = new_obbs

    for index, row in tqdm(test_data_agg.iterrows()):
        source_image = os.path.join(image_path, row['filename'])
        a_bboxes = row['a_bbox']
        o_bboxes = row['o_bbox']
        resized_img = None
        new_abbs, new_obbs = None, None
        # resize if needed
        if row['width'] != median_width or row['height'] != median_height:
            resized_img, new_abbs, new_obbs = resize_image(source_image, 
                                                          a_bboxes, 
                                                          o_bboxes, 
                                                          median_width, 
                                                          median_height)
            # update df
            test_data_agg.at[index, 'a_bbox'] = new_abbs
            test_data_agg.at[index, 'o_bbox'] = new_obbs
            test_data_agg.at[index, 'width'] = median_width
            test_data_agg.at[index, 'height'] = median_height
            test_data_agg.at[index, 'resized'] = 1

        if resized_img is not None:
            output_img, new_abbs, new_obbs = apply_random_distortion(resized_img, 
                                                                     new_abbs, 
                                                                     new_obbs)
        else:
            output_img, new_abbs, new_obbs = apply_random_distortion(source_image, 
                                                                     a_bboxes, 
                                                                     o_bboxes)
        # write image to disk
        output_img_path = os.path.join(output_path, row['filename'])
        output_img.save(output_img_path)  # Save the processed image
        # update df
        test_data_agg.at[index, 'a_bbox'] = new_abbs
        test_data_agg.at[index, 'o_bbox'] = new_obbs
    print("Image resize and distortion complete.")

    ## compute the features that depend on the final bounding boxes here
    # add 2px padding to any bounding box with a dimension < 2px
    train_data_agg['padded_a_bbox'] = train_data_agg['a_bbox'].apply(lambda bboxes: [add_padding(bbox) for bbox in bboxes])
    train_data_agg['padded_o_bbox'] = train_data_agg['o_bbox'].apply(lambda bboxes: [add_padding(bbox) for bbox in bboxes])
    test_data_agg['padded_a_bbox'] = test_data_agg['a_bbox'].apply(lambda bboxes: [add_padding(bbox) for bbox in bboxes])
    test_data_agg['padded_o_bbox'] = test_data_agg['o_bbox'].apply(lambda bboxes: [add_padding(bbox) for bbox in bboxes])
    # compute the features that depend on the final bounding boxes here
    # compute the padding here
    # compute the areas here
    #     # add 2px padding to any bounding box with a dimension < 2px
    # train_obboxs['padded_a_bbox'] = train_obboxs['a_bbox'].apply(adjust_bbox)
    # test_obboxs['padded_a_bbox'] = test_obboxs['a_bbox'].apply(adjust_bbox)
    # train_obboxs['padded_o_bbox'] = train_obboxs['o_bbox'].apply(adjust_bbox)
    # test_obboxs['padded_o_bbox'] = test_obboxs['o_bbox'].apply(adjust_bbox)
    print("Added padding where needed.")

    ## add a column with bounding boxes in (center x, center y, W, H, R)*normalized format
    # tqdm.pandas() 
    # train_data_agg['yolo_box'] = train_data_agg.apply(lambda row: [to_yolo(bbox, row['width'], row['height']) for bbox in row['o_bbox']], axis=1)
    # test_data_agg['yolo_box'] = test_data_agg.apply(lambda row: [to_yolo(bbox, row['width'], row['height']) for bbox in row['o_bbox']], axis=1)
    # print("Computed yolo coords.")

    # add area columns 
    train_data_agg['a_area'] = train_data_agg['a_bbox'].apply(lambda bboxes: [calculate_bbox_area(bbox) for bbox in bboxes])
    test_data_agg['a_area'] = test_data_agg['a_bbox'].apply(lambda bboxes: [calculate_bbox_area(bbox) for bbox in bboxes])
    train_data_agg['o_area'] = train_data_agg['o_bbox'].apply(lambda bboxes: [calculate_bbox_area(bbox) for bbox in bboxes])
    test_data_agg['o_area'] = test_data_agg['o_bbox'].apply(lambda bboxes: [calculate_bbox_area(bbox) for bbox in bboxes])
    print("Computed areas.")

    #  save processed data
    train_data_agg.to_csv(json_directory+'train_df_for_model.csv')
    test_data_agg.to_csv(json_directory+'test_df_for_model.csv')
    print("Processing complete, saved csv files.")

    return train_data_agg, test_data_agg
