import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
import re
from shapely.geometry import Polygon, LineString, Point
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
from math import radians, cos, sin
import cv2
import ast

def get_image_with_bboxes(image_path_or_bin, bounding_boxes, show=False):
    """
    Displays an image with bounding boxes drawn over it. Can handle both orthogonal and oriented bounding boxes.

    :param image_path: The path to the image file.
    :param bounding_boxes: A list of tuples representing the bounding boxes.
                           For orthogonal boxes: (x1, y1, x2, y2)
                           For oriented boxes: (x1, y1, x2, y2, x3, y3, x4, y4)
    :param image_binary: An alternative to image_path, a PIL image object to be used directly.
    :param oriented: Boolean flag to indicate whether the bounding boxes are oriented.
    """
    # Load the image
    if type(image_path_or_bin) is str:
        image = Image.open(image_path_or_bin)
    else:
        image = image_path_or_bin

    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    
    if len(bounding_boxes[0])==8:
        # Draw each bounding box
        for bbox in bounding_boxes:
            # Assumes oriented bounding box is given as (x1, y1, x2, y2, x3, y3, x4, y4)
            # We need to reorganize this into [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            points = [(bbox[i], bbox[i+1]) for i in range(0, len(bbox), 2)]
            draw.polygon(points, outline='red', width=2)
    else: 
        # Draw each bounding box
        for bbox in bounding_boxes:
            # Orthogonal bounding box (x1, y1, x2, y2)
            draw.rectangle(bbox, outline='red', width=2)

    # Show the image
    if show: image.show()
    return image

def apply_blur(image_path_or_bin, blur_radius=None):
    
    # Load the image
    if type(image_path_or_bin) is str:
        image = Image.open(image_path_or_bin)
    else:
        image = image_path_or_bin
    
    # set the intensity of the blur
    if blur_radius is None:
        blur_radius = random.randint(1, 3)
        
    # Define the area to blur (x1, y1, x2, y2)
    x1 = random.randint(0, image.width*0.5)
    y1 = random.randint(0, image.width*0.5)
    x2 = x1 + int(image.width*0.5) 
    y2 = y1 + int(image.width*0.5) 

    # Crop the area, apply blur, and paste back
    cropped_area = image.crop((x1, y1, x2, y2))
    blurred_area = cropped_area.filter(ImageFilter.BoxBlur(radius=blur_radius))
    image.paste(blurred_area, (x1, y1, x2, y2))

    return image

def apply_zoom(image_path_or_bin, bounding_boxes, min_zoom=0.8, max_zoom=1.2):
    
    # Load the image
    if type(image_path_or_bin) is str:
        image = Image.open(image_path_or_bin)
    else:
        image = image_path_or_bin
        
    original_width, original_height = image.size
    scale_factor = np.random.uniform(min_zoom, max_zoom)

    # New dimensions
    new_width, new_height = int(original_width * scale_factor), int(original_height * scale_factor)

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    # Calculate the new image's padding offsets
    pad_width = (original_width - new_width) // 2
    pad_height = (original_height - new_height) // 2

    # Create a new image with a white background
    result_image = Image.new('L', (original_width, original_height), 'white')
    result_image.paste(resized_image, (pad_width, pad_height))

    # Adjust bounding boxes
    adjusted_bboxes = []
    if len(bounding_boxes[0])==8:
        for coords in bounding_boxes:
            adjusted_coords = []
            for i in range(0, len(coords), 2):
                new_x = coords[i] * scale_factor + pad_width
                new_y = coords[i+1] * scale_factor + pad_height
                adjusted_coords.extend([new_x, new_y])
            adjusted_bboxes.append(tuple(adjusted_coords))
    else:
        for x1, y1, x2, y2 in bounding_boxes:
            new_x1 = x1 * scale_factor + pad_width
            new_y1 = y1 * scale_factor + pad_height
            new_x2 = x2 * scale_factor + pad_width
            new_y2 = y2 * scale_factor + pad_height
            adjusted_bboxes.append((new_x1, new_y1, new_x2, new_y2))

    return result_image, adjusted_bboxes

def apply_rotation(image_path_or_bin, bounding_boxes, max_rotation_deg=5):
    # Load the image
    if type(image_path_or_bin) is str:
        image = Image.open(image_path_or_bin)
    else:
        image = image_path_or_bin
    
    width, height = image.size

    # Random rotation angle between -5 and 5 degrees
    angle = np.random.uniform(-max_rotation_deg, max_rotation_deg)

    # Rotate image with a white background
    rotated_image = image.rotate(angle, expand=True, fillcolor='white')  # Ensure background is white

    # Crop the image to the original dimensions
    new_width, new_height = rotated_image.size
    left = (new_width - width) // 2
    top = (new_height - height) // 2
    right = left + width
    bottom = top + height
    rotated_image = rotated_image.crop((left, top, right, bottom))

    # Calculate new bounding boxes
    new_bounding_boxes = []
    rad_angle = math.radians(-angle)  # Negative to rotate the points back
    for box in bounding_boxes:
        new_box = []
        if len(bounding_boxes[0])==8:
            points = [(box[i], box[i + 1]) for i in range(0, len(box), 2)]
        else:
            points = [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]

        for x, y in points:
            # Translate point to origin
            tx = x - width / 2
            ty = y - height / 2
            # Rotate point
            new_x = (math.cos(rad_angle) * tx - math.sin(rad_angle) * ty) + width / 2
            new_y = (math.sin(rad_angle) * tx + math.cos(rad_angle) * ty) + height / 2
            new_box.extend([new_x, new_y])

        if len(bounding_boxes[0])==8:
            new_bounding_boxes.append(new_box)  # Store the four corner points for OBBs
        else:
            # Convert back to bounding box format
            min_x, min_y = min(new_box[::2]), min(new_box[1::2])
            max_x, max_y = max(new_box[::2]), max(new_box[1::2])
            new_bounding_boxes.append((min_x, min_y, max_x, max_y))

    return rotated_image, new_bounding_boxes

def apply_skew(image_path_or_bin, bounding_boxes, 
               horizontal_warp_range=0.05, vertical_warp_range=0.05):
    # Load the image
    if type(image_path_or_bin) is str:
        image = cv2.imread(image_path_or_bin)
    else: # convert PIL to CV2
        image_array = np.array(image_path_or_bin.convert('RGB'))
        image = image_array[:, :, ::-1]
    
    height, width = image.shape[:2]

    # Randomly select warp ratios within the specified ranges
    horizontal_warp = np.random.uniform(-horizontal_warp_range, horizontal_warp_range)
    vertical_warp = np.random.uniform(-vertical_warp_range, vertical_warp_range)

    # Define source points (original corners of the image)
    src_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # Define destination points for warping
    dst_points = np.float32([
        [0, 0], 
        [width + horizontal_warp * width, vertical_warp * height], 
        [0, height - vertical_warp * height], 
        [width, height]
    ])

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Warp the image using the transformation matrix
    warped_image = cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    warped_image = Image.fromarray(warped_image).convert('L')
    
    # Adjust bounding boxes
    new_bounding_boxes = []
    for box in bounding_boxes:
        new_box = []
        # Transform each corner of the bounding box
        for i in range(0, len(box), 2):
            point = np.array([[[box[i], box[i+1]]]], dtype='float32')
            # Apply the transformation matrix
            transformed_point = cv2.perspectiveTransform(point, matrix)
            new_box.extend(transformed_point[0][0])
        if not len(bounding_boxes[0])==8:
            # For non-oriented boxes, recalculate to orthogonal bounds
            min_x, min_y = min(new_box[::2]), min(new_box[1::2])
            max_x, max_y = max(new_box[::2]), max(new_box[1::2])
            new_box = [min_x, min_y, max_x, max_y]
        new_bounding_boxes.append(new_box)

    return warped_image, new_bounding_boxes # this is a PIL image

def apply_warp(image_path_or_bin, bounding_boxes, max_skew_factor=0.05):
    # Load the image
    if type(image_path_or_bin) is str:
        image = cv2.imread(image_path_or_bin)
    else: # convert PIL to CV2
        image_array = np.array(image_path_or_bin.convert('RGB'))
        image = image_array[:, :, ::-1]
    
    height, width = image.shape[:2]

    # Define source points (original corners of the image)
    src_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # Randomly apply skew factors to each corner
    dst_points = np.float32([
        [np.random.uniform(-max_skew_factor, max_skew_factor) * width, 
         np.random.uniform(-max_skew_factor, max_skew_factor) * height],
        [width + np.random.uniform(-max_skew_factor, max_skew_factor) * width, 
         np.random.uniform(-max_skew_factor, max_skew_factor) * height],
        [np.random.uniform(-max_skew_factor, max_skew_factor) * width, 
         height + np.random.uniform(-max_skew_factor, max_skew_factor) * height],
        [width + np.random.uniform(-max_skew_factor, max_skew_factor) * width, 
         height + np.random.uniform(-max_skew_factor, max_skew_factor) * height]
    ])

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Warp the image using the transformation matrix
    warped_image = cv2.warpPerspective(image, matrix, (width, height), 
                                       borderMode=cv2.BORDER_CONSTANT, 
                                       borderValue=(255, 255, 255))

    # Convert warped image back to PIL format and convert to grayscale
    warped_image_pil = Image.fromarray(warped_image[:, :, ::-1]).convert('L')

    # Adjust bounding boxes
    new_bounding_boxes = []
    for box in bounding_boxes:
        transformed_points = []
        # Transform each corner of the bounding box
        for i in range(0, len(box), 2):
            point = np.array([[[box[i], box[i + 1]]]], dtype='float32')
            transformed_point = cv2.perspectiveTransform(point, matrix)
            transformed_points.extend(transformed_point[0][0])

        # Recalculate the bounding box to contain all points
        xs = transformed_points[0::2]
        ys = transformed_points[1::2]
        new_box = [min(xs), min(ys), max(xs), max(ys)]
        new_bounding_boxes.append(new_box)

    return warped_image_pil, new_bounding_boxes

def resize_image(image_path_or_bin, orthogonal_bboxes, oriented_bboxes, 
                 target_width, target_height):
    # Load the image (PIL)
    if type(image_path_or_bin) is str:
        image = Image.open(image_path_or_bin)
    else:
        image = image_path_or_bin

    ## you might need to update the above code if you are not using PIL
    ## add code here to resize and update bboxes
    ## needs to process both 
    ## return image binary PIL
    ## return bounding boxed as a list

    return resized_image_pil, new_orthogonal_bboxes, new_oriented_bboxes

"""add a main function which applies random distortion to images"""