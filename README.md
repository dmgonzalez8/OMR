![Optical Music Recognition](images/banner2.png)

<p align="center" style="font-size: 24px;">
  <strong>Optical Music Recognition</strong>
</p>

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Models and Techniques](#models-and-techniques)
  - [Model Training](#model-training)
  - [Faster R-CNN Training](#faster-r-cnn-training)
- [Image Processing](#image-processing)
  - [Image Scaling and Bounding Boxes](#image-scaling-and-bounding-boxes)
  - [Advanced Image Manipulations](#advanced-image-manipulations)
  - [Annotation Adjustment](#annotation-adjustment)
- [Barline Extraction and Measure Detection](#barline-extraction-and-measure-detection)
- [Results: YOLO Models](#yolo-models-results)
- [Results: Faster R-CNN Models](#faster-r-cnn-models-results)
- [Future Work](#future-work)
- [Contributors](#contributors)
- [Acknowledgements](#acknowledgements)

## Overview
This project focuses on training models to detect and classify musical symbols using the <a href="https://zenodo.org/records/4012193">The DeepScoresV2</a> dataset.

## Dataset
<a href="https://zenodo.org/records/4012193">The DeepScoresV2</a>  dataset, comprising high-resolution images of musical scores annotated with precise symbol locations and types, serves as the training and validation ground for our models. This dataset is pivotal for training due to its diversity in musical notation and complexity.

## Models and Techniques

### Model Training
We have employed You Only Look Once (YOLO) model for two tasks:

- **YOLO for All Symbols**
- **YOLO for Measures**

#### YOLO Training Details
We trained our models using the following configurations:
- **Models**: `YOLOv8x.pt`, `YOLOv8m.pt`, `YOLOv8x.pt`
- **Data Configuration**: `deep_scores.yaml`
- **Batch Size**: 5
- **Image Size**: [704, 992]
- **Optimizer**: Automatically selected based on training conditions
- **Additional Settings**:
  - **Pretrained**: Yes, leveraging <a href="https://github.com/ultralytics/ultralytics">Ultralytics'</a>  pretrained models to accelerate convergence.
  - **Patience**: 0, to avoid early stopping.
  - **Rectangle Training**: Enabled, to optimize loading of rectangular images.

**Folder Structure and Configuration**
- YOLO training requires a specific folder format and a `.yaml` file detailing the labels and their corresponding numbers, as well as paths for folders containing images, separated into train and test sets.

### Faster R-CNN Training

#### Current pipeline: 

[preprocess_data.py](preprocess_data.py) >> [train_model.py](train_model.py) (>> [test_model.py](test_model.py))

For our Faster R-CNN model, we trained with various optimizers under the following conditions:
- **Epochs**: 100
- **Optimizers**:
  - **SGD**: Learning rate of 0.005, momentum of 0.9, weight decay of 0.0005.
  - **AdamW**: Learning rate of 0.001, betas of (0.9, 0.999), eps of 1e-08, weight decay of 0.01.
  - **RMSprop**: Learning rate of 0.001, alpha of 0.99, eps of 1e-08, weight decay of 0.0001, momentum of 0.9.
  - **Adadelta**: Learning rate of 1.0, rho of 0.9, eps of 1e-06, weight decay of 0.0001.

## Image Processing

### Image Scaling and Bounding Boxes
To standardize our dataset, we:
- Scaled all images to a median size of **1960x2772** and added padding as necessary to maintain aspect ratio integrity.
- Computed absolute coordinates for bounding boxes in both YOLO format and oriented 4-corner-coordinate format.
- Adjusted bounding boxes for new image sizes, considering different scaling factors for x and y axes due to int rounding, despite their negligible differences.

### Advanced Image Manipulations
In the [`image_preprocessing.py`](image_preprocessing.py) file, we have implemented functions to apply distortions, warping, and various other transformations. These processes create diverse training images, enhancing the robustness of our models against real-world variations in musical scores.

### Annotation Adjustment
For symbols without explicit relative positions, we:
- Imputed the relative positions by finding the closest staff and calculating the distance from the center line.
- Utilized the staff lines as relative positions (-4, -2, 0, 2, 4) to determine the scaling size in pixels.

## Barline Extraction and Measure Detection

### Barline Processing
We extracted barlines from segmented images using the color properties defined in [`compute_barline_bboxs.py`](compute_barline_bboxs.py). This script automatically scans all JSON files in the directory and processes them using multiprocessing.


### Measure Detection
The algorithm in [`compute_barline_bboxs.py`](compute_barline_bboxs.py) detects measures from barlines using recursion. Currently, measures are detected only for the default 5 staff lines. We need to improve the algorithm to capture notes that are positioned above or below these lines.


## YOLO Models Results

The results section for YOLO includes two parts: one for symbol detection & classification, and the other for measure detection. We have trained four models, three for symbol detection and classification, and one for measure detection. For symbol detection, we trained YOLOv8n (500 + 500 epochs), YOLOv8m (500 epochs), and YOLOv8x (500). We achieved highest mean Average Precision (mAP) at an Intersection Over Union (IOU) of 0.5 of ... across all classes using YOLOv8x. After observing improvements in the training efficiency as the models' number of parameters increased, we decided to use YOLOv8x for measure detection. For measure detection we trained one model using YOLOv8x(500 epochs). 

### Symbol Detection & Classification Using YOLO:

First let's see the histogram chart showing the class imbalance in the data:


<p align="center">
  <kbd style="border: 1px solid #ccc; box-shadow: 4px 4px 5px #888;">
    <img src="images/yolo_for_symbols_v8n/histogram.jpg" width="400" alt="Histogram of Symbol Class Frequency"/>
  </kbd>
</p>
<p align="center"><em>Figure 1: Histogram of Symbol Class Frequency</em></p>

The confusion matrix below represents the performance of the YOLOv8n model after 1000 epochs, highlighting the prediction accuracy across various symbol classes. Each cell shows the proportion of predictions made for a predicted class versus the true class, with a perfect prediction represented by a 1.0 in the diagonal cells. Please refer to the provided class index [file](deep_scores.yaml) to determine which symbol each class index corresponds to.

<p align="center">
  <kbd style="border: 1px solid #ccc; box-shadow: 4px 4px 5px #888;">
    <img src="images/yolo_for_symbols_v8n/confusion_matrix_normalized_1000epochs.png" width="500" alt="Normalized Confusion Matrix for Symbol Detection using YOLOv8n"/>
  </kbd>
</p>
<p align="center"><em>Figure 2: Normalized Confusion Matrix for Symbol Detection using YOLOv8x (This is from YOLOv8n, I will change this by tomorrow evening.)</em></p>

After training the YOLOv8n model for 500 epochs for approximately 9 hours, here are the losses:

<p align="center">
  <kbd style="border: 1px solid #ccc; box-shadow: 4px 4px 5px #888;">
    <img src="images/yolo_for_symbols_v8n/losses_500epochs.png" width="400" alt="Symbol Classification Using yolov8n After 500 Epochs"/>
  </kbd>
</p>
<p align="center"><em>Figure 3: Symbol Classification Using YOLOv8n After 500 Epochs</em></p>
  
Both training and validation box loss, as well as classification loss, decrease sharply initially and then level off, showing that the model is learning to predict the bounding boxes and classify the music symbols. The distribution-focused loss follows a similar trend, improving the model's distribution predictions. Additionally, precision and recall metrics consistently increased, demonstrating that the model's accuracy and completeness in symbol detection are improving. Although the rate of decrease in the losses diminished towards the end of the 500 epochs, we wanted to see whether the decrease would continue or if it would stabilize around similar levels. Therefore, we decided to train for an additional 500 epochs and observed the following Precision-Confidence Curve and Precision-Recall Curve.

<p align="center">
  <kbd style="border: 1px solid #ccc; box-shadow: 4px 4px 5px #888; margin-right: 10px;">
    <img src="images/yolo_for_symbols_v8n/P_curve_1000epochs.png" width="400" alt="Precision-Confidence Curve of YOLOv8n Model for Symbols Detection After 1000 Epochs"/>
  </kbd>
  
  <kbd style="border: 1px solid #ccc; box-shadow: 4px 4px 5px #888;">
    <img src="images/yolo_for_symbols_v8n/PR_curve_1000epochs.png" width="400" alt="Precision-Recall Curves of YOLOv8n Model for Symbols Detection After 1000 Epochs"/> 
  </kbd>
</p>
<p align="center">
  <em>Figure 4 and 5: Precision and Precision-Recall Curves of YOLOv8n Model for Symbols Detection After 1000 Epochs</em>
</p>

The Precision-Confidence Curve now achieves a precision of 0.95 at a confidence level of 0.990 for all classes, while the Precision-Recall Curve has a mean Average Precision (mAP) at an Intersection Over Union (IOU) of 0.5 of 0.560 across all classes.These values indicate an improvement from the previous results observed at the end of the initial 500 epochs, demonstrating that the model remains open to training even though the decrease in loss is quite slow compared to the first epochs. Please see the [documentation](/images/yolo_for_symbols_v8n/) for the plots after 500 epochs. After realizing that it becomes harder to improve the models with each epoch, we decided to use the YOLOv8m model, which has approximately 25.9 million parameters—about 8 times more than the YOLOv8n model with 3.2 million parameters. Using the same settings, we trained it for 500 epochs and observed the following results:

<p align="center">
  <kbd style="border: 1px solid #ccc; box-shadow: 4px 4px 5px #888; margin-right: 10px;">
    <img src="images/yolo_for_symbols_v8m/P_curve.png" width="400" alt="Precision-Confidence Curve of YOLOv8m Model for Symbols Detection After 500 Epochs"/>
  </kbd>
  
  <kbd style="border: 1px solid #ccc; box-shadow: 4px 4px 5px #888;">
    <img src="images/yolo_for_symbols_v8m/PR_curve.png" width="400" alt="Precision-Recall Curves of YOLOv8m Model for Symbols Detection After 500 Epochs"/> 
  </kbd>
</p>
<p align="center">
  <em>Figure 6 and 7: Precision and Precision-Recall Curves of YOLOv8m Model for Symbols Detection After 500 Epochs</em>
</p>

This took approximately 9 hours, similar to the YOLOv8n model, which was expected to have a longer duration based on the documentation from the Ultralytics page. We realized afterward that the YOLOv8n model did not fully utilize the GPU. Please see this [plot](/images/yolo_for_symbols_v8m/losses.png) for their losses. Training for only 500 epochs and observing better results than from 1000 epochs of training with the YOLOv8n model hinted that as the number of parameters increases, training becomes more efficient in this case. Consequently, we decided to try the YOLOv8x model, and here are the results we observed:

<p align="center">
  <kbd style="border: 1px solid #ccc; box-shadow: 4px 4px 5px #888; margin-right: 10px;">
    <img src="images/yolo_for_symbols_v8x/P_curve.png" width="400" alt="Precision-Confidence Curve of YOLOv8x Model for Symbols Detection After 500 Epochs"/>
  </kbd>
  <kbd style="border: 1px solid #ccc; box-shadow: 4px 4px 5px #888;">
    <img src="images/yolo_for_symbols_v8x/PR_curve.png" width="400" alt="Precision-Recall Curves of YOLOv8x Model for Symbols Detection After 500 Epochs"/> 
  </kbd>
</p>
<p align="center">
  <em>Figure 8 and 9: Precision and Precision-Recall Curves of YOLOv8x Model for Symbols Detection After 500 Epochs (by tomorrow evening)</em>
</p>

Above plots shows that training with YoloV8x model seems to yield the most efficent losses per each epoch, although its training time is considereably higher than the YOLOv8n and YOLOv8m models which was approximately took ... hours. And here is a sample predicted from test set using this model:

<p align="center">
  <kbd style="border: 1px solid #ccc; box-shadow: 4px 4px 5px #888;">
    <img src="images/yolo_for_symbols_v8n/lg-110143839-aug-gonville-.png" width="600" alt="Predicted Sample From Test Set "lg-110143839-aug-gonville-.png""/>
  </kbd>
</p>
<p align="center"><em>Figure 10: Predicted Sample From Test Set "lg-110143839-aug-gonville-.png" Using the YOLOv8x (Sample is predicted using YOLOv8n, YOLOv8x is still on training, I will change this by tomorrow evening.)</em></p>

### Measure Detection Using YOLO
Lastly, we trained a YOLOv8x model for 500 epochs to detect only the measures after seeing it is outperforming the others, and here is the results

<p align="center">
  <kbd style="border: 1px solid #ccc; box-shadow: 4px 4px 5px #888;">
    <img src="images/yolo_for_measures_v8x/results.png" width="600" alt="Symbol Classification Using yolov8n After 500 Epochs"/>
  </kbd>
</p>
<p align="center"><em>Figure 11: Measure Detection Using YOLOv8x After 500 Epochs (by tomorrow evening) </em></p>

And here is a sample predicted from test set using this model:

<p align="center">
  <kbd style="border: 1px solid #ccc; box-shadow: 4px 4px 5px #888;">
    <img src="images/yolo_for_measures_v8x/lg-110143839-aug-gonville-.png" width="600" alt="Predicted Sample From Test Set "lg-110143839-aug-gonville-.png" Using the YOLOv8x"/>
  </kbd>
</p>
<p align="center"><em>Figure 12: Predicted Sample From Test Set "lg-110143839-aug-gonville-.png" Using the YOLOv8x (by tomorrow evening) </em></p>

## Faster R-CNN Models Results

<p align="center">
  <kbd style="border: 1px solid #ccc; box-shadow: 4px 4px 5px #888;">
    <img src="images/image_1.png" width="400" alt="Symbol Classification Using R-CNN"/>
  </kbd>
</p>
<p align="center"><em>Figure 13: Symbol Classification Using R-CNN</em></p>

## Future Work
To further enhance our optical music recognition system, we plan to focus on several key areas:
- **Training Set Refinement**: The variability in training image quality, particularly issues with cramped spacing in some scores, indicates a need to refine our training set. By selecting or creating images with cleaner, more consistent spacing, we can improve model training and performance.
- **End-to-End OMR System**: The next major goal is to develop an end-to-end optical music recognition system that converts music scores into XML files. This includes processing not only traditionally scanned images but also photographs that are not perfectly aligned or warped. Enhancements in image segmentation will be crucial to achieve this goal.

## Contributors


## Acknowledgements
We extend our gratitude to the creators of the DeepScores dataset for their comprehensive collection of musical symbols and annotations, which significantly contributed to the depth of our analysis and training processes.
