from ultralytics import YOLO
import squarify
import matplotlib.pyplot as plt
import argparse
import logging
import cv2
import os
import random
import pandas as pd
import matplotlib.image as mpimg
import seaborn as sns
import torch

sns.set_style('darkgrid')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", help="Log threshold (default=DEBUG)", type=str, default='DEBUG')
    parser.add_argument("--config", help="Path to the config file (default=config.ini)", type=str, default='config.ini')
    parser.add_argument("--latency_test", help="Enable latency measurement", action="store_true")
    args = parser.parse_args()

    numericLogLevel = getattr(logging, args.log.upper(), None)
    if not isinstance(numericLogLevel, int):
        raise ValueError('Invalid log level: %s' % numericLogLevel)

    logging.basicConfig(
        level=numericLogLevel, 
        format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# Check if PyTorch is using the GPU
print("Is CUDA available? ", torch.cuda.is_available())
print("Device name: ", torch.cuda.get_device_name(0))
# Set the device to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device is on: {device}')

train_images = "/kaggle/input/pothole-dataset/train/images"
train_labels = "/kaggle/input/pothole-dataset/train/labels"

test_images = "/kaggle/input/pothole-dataset/test/images"
test_labels = "/kaggle/input/pothole-dataset/test/images"

val_images = "/kaggle/input/pothole-dataset/valid/images"
val_labels = "/kaggle/input/pothole-dataset/valid/labels"

# Load an image using OpenCV
image = cv2.imread("/kaggle/input/pothole-dataset/test/images/0004_jpg.rf.f92ab952cd8544f887caf35fcccbcd10.jpg")

# Get the size of the image
height, width, channels = image.shape
print(f"The image has dimensions {width}x{height} and {channels} channels.")


# Loading a pretrained model
model = YOLO('yolo11n.pt')

# free up GPU memory
torch.cuda.empty_cache()

# Training the model
model.train(data = '/kaggle/input/pothole-dataset/data.yaml',
            epochs = 20,
            imgsz = (height, width, channels),
            seed = 42,
            batch = 8,
            workers = 4)