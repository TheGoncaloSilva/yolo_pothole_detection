# YOLO Pothole Detection

Potholes are a major problem in urban areas, causing damage to vehicles and posing safety risks to drivers and pedestrians. This project aims to leverage the YOLO (You Only Look Once) object detection algorithm to accurately detect and classify potholes in real-time, providing a valuable tool for road maintenance and safety improvements.

# Getting Started

To get started, clone this repository using ssh, with the command:

```bash
git clone git@github.com:TheGoncaloSilva/yolo_pothole_detection.git
```

## Downloading dataset

Due to the size of the dataset used, it needs to be hosted in an external platform. Since we are using a dataset provided by available in [Kaggle](https://www.kaggle.com/datasets/ryukijanoramunae/pothole-dataset), follow the next commands to download and set it in the correct folder structure:

### For Linux

```bash
curl -L -o ./dataset.zip https://www.kaggle.com/api/v1/datasets/download/ryukijanoramunae/pothole-dataset~
unzip ./dataset.zip -d dataset
rm ./dataset.zip
```

### For Windows
```powershell
Invoke-WebRequest -Uri "https://www.kaggle.com/api/v1/datasets/download/ryukijanoramunae/pothole-dataset" -OutFile "./dataset.zip" -UseBasicParsing
Expand-Archive -Path ".\dataset.zip" -DestinationPath ".\dataset" -Force
rm ./dataset.zip
```

## Dependencies

Python is the chosen language to train and evaluate the model, so first, make sure you have it installed. Preferably, a version between `3.8 - 3.11`. Version `3.12+` aren't advisable, at the moment.

To install the dependencies for this project, you just need to use the provided `requirements` file:
```bash
pip install -r requirements.txt
```

# Training

## Pre-trained

### Epochs

### 

## Sources

[YOLO training tips](https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/#training-settings)