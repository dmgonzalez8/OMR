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
    distances = np.sqrt((center_x[:, None] - center_x[None, :])
                        ** 2 + (center_y[:, None] - center_y[None, :]) ** 2)
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
            # this is updated later (it's the image dimension, NOT the bbox)
            'width': -1,
            # this is updated later (it's the image dimension, NOT the bbox)
            'height': -1
        })

    # Create a new DataFrame with the merged data
    return pd.DataFrame(merged_data)


def analyze_image_for_measures(image_path):

    # Eren's code here

    # format for output
    data.append({
        'filename': image_path.split('/')[-1].replace('_seg', ''),
        'a_bbox': orthogonal_bbox,
        'o_bbox': oriented_bbox
    })

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
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
            'filename': image_path.split('/')[-1].replace('_seg', ''),
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


def convert_str_to_list(coord_str):
    return ast.literal_eval(coord_str)


# Function to detect if bboxes are on the same line
def on_same_line(bbox1, bbox2):
    # Extract the y-coordinates of the top left corner of the bounding boxes
    y1_top_left = bbox1[1]
    y2_top_left = bbox2[1]

    # Check if the y-coordinates are within a 1-2 pixel range
    return abs(y1_top_left - y2_top_left) <= 2


def check_and_assign_measure(group, current_index, measure_num):
    # Check for adjacent line measures and assign the same measure number
    for j in range(current_index + 1, len(group)):
        if group.iat[j, group.columns.get_loc('measure_number')] == -1:
            if abs(group.iat[current_index, group.columns.get_loc('x_coord')] - group.iat[j, group.columns.get_loc('x_coord')]) <= 2 and \
               abs(group.iat[current_index, group.columns.get_loc('line_number')] - group.iat[j, group.columns.get_loc('line_number')]) == 1:
                group.iat[j, group.columns.get_loc(
                    'measure_number')] = measure_num
                # Recursively check the next line
                check_and_assign_measure(group, j, measure_num)


def group_measures_by_line(group):
    # Sort the group by line number and measure number
    sorted_group = group.sort_values(by=['line_number', 'measure_number'])
    measure_memo = set()
    # Dictionary to hold the bounding boxes grouped by line and measure
    bbox_groups = {}
    special_measure_count = 1  # Counter for special measures

    # Iterate over each line
    for line_number in sorted_group['line_number'].unique():
        line_data = sorted_group[sorted_group['line_number']
                                 == line_number].copy()

        # Check if there is only one row in this line
        if len(line_data) == 1:

            current_measure = line_data['measure_number'].values[0]

            if current_measure in measure_memo:
                continue
            else:
                measure_memo.add(current_measure)

            measure_key_initial = f"initial_special_measure_{special_measure_count}"

            measure_key_final = f"final_special_measure_{special_measure_count}"

            # Initialize measure key if not present
            if measure_key_initial not in bbox_groups:
                bbox_groups[measure_key_initial] = []

            # Initialize measure key if not present
            if measure_key_final not in bbox_groups:
                bbox_groups[measure_key_final] = []

            # Original bbox
            bbox_groups[measure_key_initial].append(
                line_data['o_bbox'].values[0].copy())

            # Original bbox
            bbox_groups[measure_key_final].append(
                line_data['o_bbox'].values[0].copy())

            # Append all bounding boxes for the same measure
            for coords in group[group['measure_number'] == current_measure]['o_bbox'].values:
                bbox_groups[measure_key_initial].append(coords)
                bbox_groups[measure_key_final].append(coords)

            # Modified bbox with certain indices set to 0
            modified_bbox_initial = line_data['o_bbox'].values[0].copy()
            for index in [0]:
                modified_bbox_initial[index] = 0  # Set specified indices to 0
            bbox_groups[measure_key_initial].append(modified_bbox_initial)

            # Modified bbox with certain indices set to 1960
            modified_bbox_final = line_data['o_bbox'].values[0].copy()
            for index in [0]:
                # Set specified indices to 1960
                modified_bbox_final[index] = 1960
            bbox_groups[measure_key_final].append(modified_bbox_final)

            special_measure_count += 1

        else:
            # Iterate through the measures in the line normally
            for i in range(len(line_data) - 1):  # -1 because we look ahead one measure
                current_measure = line_data.iloc[i]
                next_measure = line_data.iloc[i + 1]

                # Check if the next measure number is consecutive
                if current_measure['measure_number'] + 1 == next_measure['measure_number']:
                    # Create the key as 'measure_current_current+1'
                    key = f"measure_{current_measure['measure_number']}_{next_measure['measure_number']}"

                    # If the key doesn't exist, create it and assign an empty list
                    if key not in bbox_groups:
                        bbox_groups[key] = []

                    # Append the current and next bounding boxes to the list under the key
                    bbox_groups[key].append(current_measure['o_bbox'])
                    bbox_groups[key].append(next_measure['o_bbox'])

    return bbox_groups

# Function to numerically encode the bar lines based on their y-coordinates


def process_group(group, filename):

    # Sort by y-coordinate first to group by line, then by x-coordinate to maintain left-to-right order
    # Assuming 'o_bbox' is a list with the structure [x1, y1, x2, y2, x3, y3, x4, y4]
    # where (x1, y1) are the coordinates for the top-left corner of the bbox
    group = group.sort_values(by=['o_bbox']).reset_index(drop=True)

    # Split the 'o_bbox' column into separate x and y columns for sorting
    group['x_coord'] = group['o_bbox'].apply(lambda x: x[0])  # x1 coordinate
    group['y_coord'] = group['o_bbox'].apply(lambda x: x[1])  # y1 coordinate
    group = group.sort_values(by=['y_coord', 'x_coord']).reset_index(drop=True)

    # Initialize line number
    line_number = 0
    group['line_number'] = line_number  # Initialize line number column

    # Iterate through each bbox
    for i in range(1, len(group)):
        # If the y-coordinates change significantly, it's a new line
        if not on_same_line(group.iloc[i-1]['o_bbox'], group.iloc[i]['o_bbox']):
            line_number += 1
            # probably error
            if line_number == 100:
                print(f"Line number exceeded 100 for {filename}")
        # Assign the current line number
        group.iat[i, group.columns.get_loc('line_number')] = line_number

    # Normalize the y-coordinates within each line
    for line in group['line_number'].unique():
        mean_y = group[group['line_number'] == line]['y_coord'].mean()
        group.loc[group['line_number'] == line, 'y_coord'] = mean_y

    # Identify and remove redundant barlines
    max_x_per_line = group.groupby('line_number')['x_coord'].max()
    group = group[group.apply(
        lambda row: row['x_coord'] < max_x_per_line[row['line_number']], axis=1)]

    # Initialize measure numbers
    # Initialize with -1 to denote unassigned measures
    group['measure_number'] = -1
    measure_number = 0

    group = group.sort_values(by=['y_coord', 'x_coord'], ascending=[
                              True, False]).reset_index(drop=True)

    # Assign measure numbers by comparing each bbox to every other bbox
    for i in range(len(group)):
        if group.iat[i, group.columns.get_loc('measure_number')] == -1:
            group.iat[i, group.columns.get_loc(
                'measure_number')] = measure_number
            check_and_assign_measure(group, i, measure_number)
            measure_number += 1  # Increment measure number after fully exploring all consecutive lines

        if measure_number == 200:
            # Error handling if too many measures
            print(f"Measure number exceeded 200 for {filename}")

    # Drop the helper columns after processing
    group.drop(['x_coord', 'y_coord'], axis=1, inplace=True)

    return group


def process_bboxes_initial(data, tolerance=2):
    # Group by min_y and max_y and find the minimum min_x for each group, within a tolerance
    grouped_data = {}
    new_bboxes = {}  # Use this to store new entries

    for key, box in data.items():
        if key.startswith('measure'):
            min_x, min_y, max_x, max_y = box[0], box[1], box[2], box[5]

            # Check if min_x is within the tolerance to continue processing
            if min_x < tolerance:
                continue  # Skip this entry if it doesn't meet criteria

            similar_y_key_found = False
            # Check each y_key in grouped_data to see if it's within the tolerance level
            for y_key in grouped_data.keys():
                if (abs(y_key[0] - min_y) <= tolerance) and (abs(y_key[1] - max_y) <= tolerance):
                    # If found, update the similar y_key with the min of min_x values
                    grouped_data[y_key] = min(grouped_data[y_key], min_x)
                    similar_y_key_found = True
                    break

            # If no similar y_key was found, add this box as a new entry
            if not similar_y_key_found:
                grouped_data[(min_y, max_y)] = min_x

    # Create new bounding boxes using the real min_x and the y_key values
    count = 0
    for y_key, real_min_x in grouped_data.items():
        if real_min_x == 0:
            continue  # Skip entries with real_min_x set to zero
        new_key = f"initial_measure_{count}"
        new_bboxes[new_key] = [0, y_key[0], real_min_x,
                               y_key[0], real_min_x, y_key[1], 0, y_key[1]]
        count += 1

    # Update the original data dictionary with new entries
    data.update(new_bboxes)

    return data


def process_bboxes_final_new(row, tolerance=2):

    # Extract data from the row
    data = row['new_measure_bbox']
    width = row['width']
    height = row['height']

    # Group by min_y and max_y and find the maximum min_x for each group, within a tolerance
    grouped_data = {}
    new_bboxes = {}

    for key, box in data.items():
        if key.startswith('measure'):
            min_x, min_y, max_x, max_y = box[0], box[1], box[2], box[5]

            # This will be used to check if we have a similar y_key already in grouped_data
            similar_y_key_found = False

            # Check each y_key in grouped_data to see if it's within the tolerance level
            for y_key in grouped_data.keys():
                if (abs(y_key[0] - min_y) <= tolerance) and (abs(y_key[1] - max_y) <= tolerance):
                    # If found, update the similar y_key with the max of max_x values
                    grouped_data[y_key] = max(grouped_data[y_key], max_x)
                    similar_y_key_found = True
                    break

            # If no similar y_key was found, add this box as a new entry
            if not similar_y_key_found:
                grouped_data[(min_y, max_y)] = max_x

    # Form the new bounding boxes using the real max_x and the y_key values
    count = 0
    for y_key, real_max_x in grouped_data.items():
        new_key = f"final_measure_{count}"
        new_bboxes[new_key] = [width, y_key[0], real_max_x,
                               y_key[0], real_max_x, y_key[1], width, y_key[1]]
        count += 1

    # Update the original data dictionary with new entries
    data.update(new_bboxes)

    return data

# Function to process each dictionary and extract min/max coordinates


def calc_coordinates(bbox_dict):
    result = {}
    for key, bboxes in bbox_dict.items():
        min_x = min(min(bbox[0], bbox[2], bbox[4], bbox[6]) for bbox in bboxes)
        min_y = min(min(bbox[1], bbox[3], bbox[5], bbox[7]) for bbox in bboxes)
        max_x = max(max(bbox[0], bbox[2], bbox[4], bbox[6]) for bbox in bboxes)
        max_y = max(max(bbox[1], bbox[3], bbox[5], bbox[7]) for bbox in bboxes)
        result[key] = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]
    return result


def parse_list_string(s):
    s = s.strip("[]")
    return [int(item) for item in s.split(',') if item.strip()]


def analyze_image_for_measures(df, json_path):

    # df['a_bbox'] = df['a_bbox'].apply(convert_str_to_list)
    # df['o_bbox'] = df['o_bbox'].apply(convert_str_to_list)

    # df['a_bbox'] = df['a_bbox'].apply(json.loads)
    # df['o_bbox'] = df['o_bbox'].apply(json.loads)

    # df['a_bbox'] = df['a_bbox'].apply(parse_list_string)
    # df['o_bbox'] = df['o_bbox'].apply(parse_list_string)

    # Sort by 'filename' first, then by y-coordinate, then by x-coordinate
    df = df.iloc[np.lexsort((df['o_bbox'].apply(lambda x: x[0]),
                             df['o_bbox'].apply(lambda x: x[1]),
                             df['filename']))]
    # get the measure number and line numbers for the bars
    df = df.groupby('filename', group_keys=False).apply(
        lambda x: process_group(x, x.name), include_groups=True).reset_index(drop=True)
    df = df.sort_values(by=['filename', 'line_number', 'measure_number'])  # -

    # Create a new df which includes filename, and coordinates of the measures
    grouped_bbox_data = df.groupby("filename").apply(
        group_measures_by_line, include_groups=False)
    grouped_bbox_df = pd.DataFrame(grouped_bbox_data).reset_index()

    # drop the line number, measure number in the expanded df for concatenation
    df.drop(['line_number', 'measure_number'], axis=1, inplace=True)  # -

    # Load JSON data into a dictionary
    with open(f'''{json_path}''') as file:
        data1 = json.load(file)

    train_images = pd.DataFrame(data1['images'])

    # Update the width and height of the images
    filename_to_dimensions = dict(zip(train_images['filename'], zip(
        train_images['width'], train_images['height'])))

    # Use map to update 'width' and 'height' columns in measures_df based on filename
    grouped_bbox_df['width'] = grouped_bbox_df['filename'].map(
        lambda x: filename_to_dimensions.get(x, (np.nan, np.nan))[0])
    grouped_bbox_df['height'] = grouped_bbox_df['filename'].map(
        lambda x: filename_to_dimensions.get(x, (np.nan, np.nan))[1])

    # Apply the function, 0 is the column which includes the coordinates of the measures as a dictionary
    grouped_bbox_df['measure_bbox'] = grouped_bbox_df[0].apply(
        calc_coordinates)

    # Apply the function to add the initial measure at each line
    grouped_bbox_df['new_measure_bbox'] = grouped_bbox_df['measure_bbox'].apply(
        process_bboxes_initial)

    # Apply the function to add the final measure at each line
    grouped_bbox_df['new_measure_bbox_updated'] = grouped_bbox_df.apply(
        process_bboxes_final_new, axis=1)

    # Drop the previously created columns
    grouped_bbox_df.drop(
        [0, 'measure_bbox', 'new_measure_bbox'], axis=1, inplace=True)  # -

    # Create a new df using the existing one
    # Create a new DataFrame to hold the expanded data
    expanded_rows = []

    # Iterate over each row in the DataFrame
    for index, row in grouped_bbox_df.iterrows():
        filename = row['filename']
        bbox_dict = row['new_measure_bbox_updated']

        # Iterate over each item in the dictionary to create new rows
        for key, value in bbox_dict.items():
            expanded_rows.append({'filename': filename, 'o_bbox': value})

    # Create a new DataFrame from the expanded rows
    expanded_df = pd.DataFrame(expanded_rows)

    # Create 'a_bbox' column by selecting specific indices directly
    expanded_df['a_bbox'] = expanded_df['o_bbox'].apply(
        lambda x: [x[0], x[1], x[4], x[5]])

    expanded_df['padded_o_bbox'] = expanded_df['o_bbox']
    expanded_df['padded_a_bbox'] = expanded_df['a_bbox']

    expanded_df['area'] = expanded_df['a_bbox'].apply(
        lambda x: (x[2] - x[0]) * (x[3] - x[1]))
    expanded_df['width'] = expanded_df['a_bbox'].apply(lambda x: (x[2] - x[0]))
    expanded_df['height'] = expanded_df['a_bbox'].apply(
        lambda x: (x[3] - x[1]))

    expanded_df['label'] = 157
    expanded_df['duration'] = -1
    expanded_df['rel_position'] = 0
    expanded_df['duration_mask'] = 0
    expanded_df['rel_position_mask'] = 0

    # Find the smallest value in the column of the other DataFrame
    min_value = df['ann_id'].min()

    # Calculate the start point for ann_id
    start_point = min_value - 1

    # Create the ann_id column by generating a sequence starting from start_point
    expanded_df['ann_id'] = start_point - expanded_df.index

    # Concatenate the new dataframe with the original after adding the above columns
    concatenated_df = pd.concat(
        [df, expanded_df[df.columns]], ignore_index=True)

    return concatenated_df


def process_file(json_file, json_directory, segmentation_directory, merge_radius):
    print(f'Processing {json_file}')
    output_directory = json_directory

    with open(os.path.join(json_directory, json_file), 'r') as file:
        data = json.load(file)
        images_df = pd.DataFrame(data['images'])
        filenames = images_df['filename'].tolist()

    output_csv = json_file.replace('.json', '_barlines.csv')
    output_path = os.path.join(output_directory, output_csv)
    output_pkl = json_file.replace('.json', '_barlines.pkl')
    output_bin = os.path.join(output_directory, output_pkl)

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
                                 filename.replace('.png', '_seg.png'))
        # Get barlines in a df
        barline_annotations = analyze_image_for_barlines(file_path)

        if not barline_annotations.empty:
            # merge duplicates
            barline_annotations = merge_close_boxes(
                barline_annotations, merge_radius)
            # update other cells
            barline_annotations['ann_id'] = [
                ann_id - i for i in range(len(barline_annotations))]
            # Decrement ann_id for next file
            ann_id -= len(barline_annotations)
            barline_annotations['label'] = 156
            barline_annotations['duration'] = -1
            barline_annotations['rel_position'] = 0
            barline_annotations['duration_mask'] = 0
            barline_annotations['rel_position_mask'] = 0
            barlines_data = pd.concat([barlines_data, barline_annotations],
                                      ignore_index=True)

    # Eren - Get measures in a df
    barlines_data = analyze_image_for_measures(
        barlines_data, json_path=json_file)

    # Save all collected barline data to CSV after processing all files
    if not barlines_data.empty:
        barlines_data.to_csv(output_path, index=False)
        barlines_data.to_pickle(output_bin)


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
    parser = argparse.ArgumentParser(
        description='Process datasets to find barlines.')
    parser.add_argument('json_directory', type=str,
                        help='Directory containing JSON files.')
    parser.add_argument('segmentation_directory', type=str,
                        help='Directory containing segmented images.')
    parser.add_argument('n_jobs', type=int,
                        help='How many processes to spawn.')
    parser.add_argument('merge_radius', type=int,
                        help='Maximum distance in pixels to merge.')

    args = parser.parse_args()

    process_dataset(args.json_directory,
                    args.segmentation_directory,
                    args.n_jobs, args.merge_radius)

    print('Done processing.')
