from ultralytics import YOLO
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

def check_data_paths(train_images: str, train_labels: str, test_images: str, test_labels: str, val_images:str, val_labels: str) -> bool:
    """
    Check if the data paths exist
    Attributes:
        train_images: str: Path to the training images
        train_labels: str: Path to the training labels
        test_images: str: Path to the test images
        test_labels: str: Path to the test labels
        val_images: str: Path to the validation images
        val_labels: str: Path to the validation labels
    Returns:
        bool: True if all paths exist
    Raises:
        FileNotFoundError: If any of the paths do not exist
    """
    if not os.path.exists(train_images):
        raise FileNotFoundError(f"Training images path does not exist: {train_images}")
    if not os.path.exists(train_labels):
        raise FileNotFoundError(f"Training labels path does not exist: {train_labels}")
    if not os.path.exists(test_images):
        raise FileNotFoundError(f"Test images path does not exist: {test_images}")
    if not os.path.exists(test_labels):
        raise FileNotFoundError(f"Test labels path does not exist: {test_labels}")
    if not os.path.exists(val_images):
        raise FileNotFoundError(f"Validation images path does not exist: {val_images}")
    if not os.path.exists(val_labels):
        raise FileNotFoundError(f"Validation labels path does not exist: {val_labels}")
    return True

def get_images_size(train_images_path: str) -> list[int, int, int]:
    """
    Get the size of the images
    Attributes:
        train_images_path: str: Path to the training images
    Returns:
        list[int, int, int]: Height, Width and Channels of the image
    """
    # get first image name of train_images_path
    image_name = os.listdir(train_images_path)[0]
    image = cv2.imread(train_images + "/" + image_name)

    # Get the size of the image
    height, width, channels = image.shape
    return [height, width, channels]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", help="Log threshold (default=DEBUG)", type=str, default='DEBUG')
    parser.add_argument("--datafile", help="Path to the data.yml file (yolo config file)", type=str, default='./dataset/data.yaml')
    parser.add_argument("--train", help="Path to the training data", type=str, default='./dataset/train')
    parser.add_argument("--val", help="Path to the validation data", type=str, default='./dataset/valid')
    parser.add_argument("--test", help="Path to the test data", type=str, default='./dataset/test')
    parser.add_argument("--model", help="Path to the config file (default=yolov8n)", type=str, default='yolov8n')
    parser.add_argument("--epochs", help="Number of epochs (default=100)", type=int, default=250)
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
    logging.info("Is CUDA available? ", torch.cuda.is_available()[0])
    logging.info("Device name: ", torch.cuda.get_device_name(0))
    # Set the device to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Device is on: {device}')

    # Configure the data paths
    train_images: str = args.train + "/images"
    train_labels: str = args.train + "/labels"
    test_images: str = args.test + "/images"
    test_labels: str = args.test + "/labels"
    val_images: str = args.val + "/images"
    val_labels: str = args.val + "/labels"
    check_data_paths(train_images, train_labels, test_images, test_labels, val_images, val_labels)

    # Load a sample image to get the size -> image_info[0] = height, image_info[1] = width, image_info[2] = channels
    image_info: list[int, int, int] = get_images_size()

    # Check the name of the model
    modelName: str = args.model
    if not modelName.endswith('.pt'):
        modelName += '.pt'

    # Loading a pretrained model
    model = YOLO(modelName)

    # free up GPU memory
    torch.cuda.empty_cache()

    # Training the model
    model.train(data=args.datafile,
                epochs=args.epochs,
                imgsz=(image_info[0], image_info[1], image_info[2]),
                seed=42,
                batch=8,
                workers=4,
                patience=10)  # Early stopping if no improvement after 10 epochs
    
    # Evaluate the model
    results = model.evaluate(data=args.datafile)
    logging.info(results)

    # Export model 
    model.export()

    # Export model as dla format, optimized for jetson
    model.export('model.dla')
    