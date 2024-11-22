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

| Use | nº images |
| Test | 891 |
| Train | 11068 |
| Valid | 1808 |

# Training

## Pre-trained

### Epochs

Each epoch represents a full pass over the entire dataset. Adjusting this value can affect training duration and model performance [source](https://www.ultralytics.com/glossary/epoch)
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

### Background images
Background images are images with no objects that are added to a dataset to reduce False Positives (FP). We recommend about 0-10% background images to help reduce FPs (COCO has 1000 background images for reference, 1% of the total). No labels are required for background images.

### Batch size

Refers to the number of training examples utilized in one iteration of model training. It significantly influences the efficiency and speed of training, as well as model performance. By breaking the training dataset into smaller batches, computational resources are used more efficiently, and gradient updates occur more frequently, leading to faster convergence. [source](https://www.ultralytics.com/glossary/batch-size)

Smaller batch sizes can lead to faster learning and less opportunity for overfitting, whereas larger batch sizes can leverage parallel computation power for more efficient training. The right balance depends on the specific application and available hardware.

Batch size affects various aspects of model training:

Training Speed: Larger batch sizes utilize the computational resources effectively, often accelerating training. However, they require more memory, potentially limiting their use in resource-constrained environments.
Generalization: Smaller batch sizes introduce more noise in training, which can help models generalize better by avoiding overfitting. This randomness can be beneficial for models in real-world scenarios like AI in Self-Driving.
Convergence Stability: Smaller batches may result in more unstable convergence due to the high variance in gradient estimation, while larger batches offer smoother convergence.

#### Usage

Set as an integer :
* e.g.: `batch=16`
* auto mode for 60% GPU memory utilization: `batch=-1`
* auto mode with specified utilization fraction `batch=0.70`.

#### Compare performance

see launching performance for jetson, kaggle and other device

### Workers

Number of worker threads for data loading (per RANK if Multi-GPU training). Influences the speed of data preprocessing and feeding into the model, especially useful in multi-GPU setups.

### Seed

Sets the random seed for training, ensuring reproducibility of results across runs with the same configurations.

### Optimizer

Choice of optimizer for training. Options include SGD, Adam, AdamW, NAdam, RAdam, RMSProp etc., or auto for automatic selection based on model configuration. Affects convergence speed and stability.

### Initial Learning Rate

lr0 -> Initial learning rate (i.e. SGD=1E-2, Adam=1E-3) . Adjusting this value is crucial for the optimization process, influencing how rapidly model weights are updated.

### Final Learning Rate

lrf -> Final learning rate as a fraction of the initial rate = (lr0 * lrf), used in conjunction with schedulers to adjust the learning rate over time.

### Momentum

Momentum factor for SGD or beta1 for Adam optimizers, influencing the incorporation of past gradients in the current update.

### Dropout 

Dropout rate for regularization in classification tasks, preventing overfitting by randomly omitting units during training.

### Maybe try augmentation?

### Hyperparameters (Optional)

### Improve training speed 

* [Reduce dataset size](https://github.com/ultralytics/ultralytics/issues/4695)
* [Changing the optimizer to Adam and adjusting the learning rate](https://github.com/ultralytics/ultralytics/issues/5717) -> no difference

# Validate the model

# Results

## Check distances of detection

## Prediction time


## Sources

* [YOLO training tips](https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/#training-settings)
* [Install](https://medium.com/@manyi.yim/pytorch-on-mac-m1-gpu-installation-and-performance-698442a4af1e) pytorch in macos Apple Silicon and [here](https://stackoverflow.com/questions/68820453/how-to-run-pytorch-on-macbook-pro-m1-gpu)

### Papers

Keywords:
* YOLO
* Object Detection
* Computer Vision
* Deep Neural Networks
* Deep Learning
* Neural Networks

Papers:
* [Object Detection in 20 Years: A Survey](https://ieeexplore.ieee.org/abstract/document/10028728/keywords#keywords)
* [Object Detection With Deep Learning: A Review](https://ieeexplore.ieee.org/abstract/document/8627998?casa_token=EEZOjMDhn5MAAAAA:ijCq364P3WpbJ_luSdnz5Xkt_7BFUGCBbBCx9ZIhzPF9DEuO9anO-JnkGyscmletsBC9amG0s98)
* [A review of Yolo Algorithm development](https://www.sciencedirect.com/science/article/pii/S1877050922001363)
* [Object detection using YOLO: challenges, architectural successors, datasets and applications](https://link.springer.com/article/10.1007/s11042-022-13644-y)
* [A Comprehensive Review of YOLO Architectures in Computer Vision: From YOLOv1 to YOLOv8 and YOLO-NAS](https://www.mdpi.com/2504-4990/5/4/83)
* [YOLO-v1 to YOLO-v8, the Rise of YOLO and Its Complementary Nature toward Digital Manufacturing and Industrial Defect Detection](https://www.mdpi.com/2075-1702/11/7/677)
* [Comparison of CNN and YOLO for Object Detection](https://koreascience.kr/article/JAKO202011161035249.page)


tips:

* early dropping to prevent overfiting

Use screen:
```bash
screen -s {window_name}
screen -ad # Run detached
screen -ls # List windows
screen -r {window_id}
```