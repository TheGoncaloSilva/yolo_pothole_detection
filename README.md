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
curl -L -o ./dataset.zip https://www.kaggle.com/api/v1/datasets/download/ryukijanoramunae/pothole-dataset
unzip ./dataset.zip -d dataset
rm ./dataset.zip
```

Next, navigate to the `dataset` folder and print the directory absolute path with the command:
```bash
pwd
```

Note of the printed path and edit the `data.yml`, by placing that path in the place of the `{PATH}` variable:

```yml
# Better to use absolute links than relative
train: {PATH}/train/images
val: {PATH}/valid/images
test: {PATH}/test/images
```

### For Windows

```powershell
Invoke-WebRequest -Uri "https://www.kaggle.com/api/v1/datasets/download/ryukijanoramunae/pothole-dataset" -OutFile "./dataset.zip" -UseBasicParsing
Expand-Archive -Path ".\dataset.zip" -DestinationPath ".\dataset" -Force
rm ./dataset.zip
```

Next, navigate to the `dataset` folder and print the directory absolute path with the command:
```powershell
cd
```

Note of the printed path and edit the `data.yml`, by placing that path in the place of the `{PATH}` variable:

```yml
# Better to use absolute links than relative
train: {PATH}/train/images
val: {PATH}/valid/images
test: {PATH}/test/images
```

## Dependencies

Python is the chosen language to train and evaluate the model, so first, make sure you have it installed. Preferably, a version between `3.8 - 3.11`. Version `3.12+` aren't advisable, at the moment.

It's highly advisable, to first start a virtual environment using the command:
```bash
python3 -m venv venv
source venv/bin/activate # used to activate the virtual environment, every time a new shell is created
```

To install the dependencies for this project, you just need to use the provided `requirements` file:

```bash
pip install -r requirements.txt
```

# Dataset

Data distribution... (how much for train, val and test?)

# Training

## Pre-trained

### Epochs

* epochs at 250 -> best

### Patience

Early stopping is a regularization technique that halts training when the model's performance on the validation set stops improving. The patience parameter controls how long the training should continue without improvement before stopping.
Patience specifies the number of epochs to wait after the last improvement in the validation metric (e.g., validation loss or mAP) before stopping. -> positive integer

#### Impact of Patience

Higher patience:

The model will have more time to improve, even if performance stagnates temporarily.
Useful if the training process involves noisy validation metrics or requires more epochs to converge.
However, it can lead to longer training times and increase the risk of overfitting if the model continues training unnecessarily.
Lower patience:

The model stops sooner after performance stops improving.
Useful to save time and prevent overfitting, especially if the model is prone to diminishing returns quickly.
However, it may stop prematurely before the model fully converges.

For example:

* If patience = 5, and validation loss hasn't improved for 5 consecutive epochs, training stops.

##### Practical Recommendations

Small datasets or noisy validation metrics: Use a higher patience value (e.g., 10–20 epochs) to ensure the model doesn't stop prematurely due to random fluctuations.
Large datasets or when overfitting is a concern: Use a lower patience value (e.g., 3–5 epochs) to prevent overfitting and save training time.

## Sources

[YOLO training tips](https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/#training-settings)

tips:

* early dropping to prevent overfiting
