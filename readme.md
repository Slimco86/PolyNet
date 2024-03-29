# PolyNet
![Example](/media/example1.gif "Output example")
## Introduction
Polynet is a multitask, human-centric network aiming to provide a detailed analysis and tracking for multi-human view scenes. The main objective and the challenge is to achieve real-time on edge-device performance to make it feasible for certain applications and to fulfill GDPR requirements. In particular, the tasks can be split into three categories: detection, classification, and landmark estimation.
### Detection:
1. Person bounding box (2d)
2. Face bounding box (2d)
### Classification:
1. Gender
2. Ethnicity
3. Emotion
4. Age
5. Skin condition
### Landmarks:
1. Face landmarks (70 key-points)
2. Pose landmarks (18 key-points)

## List of content:
1. Architecture design
2. Data format
3. Requirements
3. Training
4. Detection
5. TODO list


## Architecture design
Due to the performance requirements and testing purposes, it is chosen to utilize [EfficientDet](https://arxiv.org/abs/1911.09070) network as the core architecture. Specifically, the accent is put on the smallest version of the network d0. The base code is taken from [Zylo117](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) repository. Initial EfficientDet architecture looks as follows: 
![EfficientDet architecture](/media/EfficientDet.jpeg "EfficientDet architecture")
The network is fully-convolutional with a performant feature extraction stem and bidirectional feature pyramid network, introduced in the paper. As well, the network benefits of extensive application of Swish activation function. Classification and bounding box prediction, in this project, is done exactly as it is performed in the classical EfficientDet network. For each pair of regression/classification tasks additional head is attached to the network. Each classification is combined either with person bbox (full body) or with face bbox (face) prediction task according to relevancy. For example, emotion classification is combined with face bounding box prediction. As a loss function [Focal Loss](https://arxiv.org/abs/1708.02002) is utilized.

For landmark prediction, an additional module/head has been developed which is performed on the same input features as all the other heads. The module takes features on multiple scales extracted from bifpn (p1,p2,p3,p4,p5) and performs a [HourGlass](https://arxiv.org/abs/1603.06937)-like upscaling to predict a probability heatmap for landmarks positions on the original input scale. The architecture is shown below:
![PoseMap architecture](/media/PoseMap.png "PoseMap architecture")

The major difference with the method proposed in the paper above, is that the network tries to predict all the landmarks at once, on a single channel heatmap, rather than a single landmark per channel. This is governed by the fact that the total amount of the landmarks on the output is a variable and initially is unknown. As a result, this induces an additional postprocessing step for landmark identification and filtering.

For optimizing this head, [Adaptive Wing Loss](https://arxiv.org/abs/1904.07399) is used, which allows to penalize missing the correct prediction and slightly relaxes the penalty for false positives.  

## Data Format

The data is supposed to be provided in the $\*$.JSON format, where each person on the image is labled with the id, starting from 1, for example "id1".
Each person has the following fields:
1. <span style="color:red">"person_bbox"</span> (int) (1,4)
2. <span style="color:red">"face_bbox"</span> (int) (1,4)
3. <span style="color:red">"age"</span> (int) 1
4. <span style="color:red">"gender"</span> (str) ["Male", "Female", "unknown"]
5. <span style="color:red">"race"</span> (str) ["white", "black", "latino hispanic", "middle eastern", "asian", "indian", "unknown"]
6. <span style="color:red">*"skin"*</span> (str) ["acne", "blackhead", "bodybuilders", "body_fat", "eczema", "good_skin", "hives", "ichtyosis", "men_skin", "normal_skin", "psoriasis, "rosacea", "seborrheic_dermatitis", "tired_eyes", "virtiligo", "unknown"]
7. <span style="color:red">"emotions"</span> (str) ["happy", "sad", "neutral", "angry", "fear", "surprise", "disgust", "unknown"]
8. <span style="color:red">"pose"</span> (int) [18,3]
9. <span style="color:red">"face_landmarks"</span> (str) [70,2]

For classification tasks, the list of available classes is specified in the config.json file.
The name of the label document must correspond to the name of the image + "_meta" tag. For example, for img123.jpg --> img123_meta.json.
An example dataset is available [here](). Similar data can be collected from the internet using [this]() web scraper from my other repository.
In order to make sure that the data is correct I suggest running the dataset through dataset checks available in the misc folder and manually check it by the visual data tool from [this](https://github.com/Slimco86/DataViewer) repository.

## Reqirements

## Training
The training was performed under a very basic setup. Single GPU(V100), batch_size of 32. No data augmentation was used, yet. [OneCycleLR](https://pytorch.org/docs/stable/optim.html) scheduler was used for learning rate update over the total of 4000 epochs. It is observed that the Focal Loss induces higher gradients compared to Adaptive Wing Loss and, as a result, converges faster, as demonstrated in the figure below.
![Training loss](/media/loss.png "training loss")
Thus, the tasks are not optimized simultaneously, as their corresponding optimums happened at different epochs.
In the future, an adaptive loss combination has to be applied in order to equalize the convergence speed from multiple tasks.
To train the network on your dataset you can use the [train.py](/Model/train.py) script. The training procedure includes the parameter "heads", which has to be used to specify which heads you want to attach and train. The available options are given in the list:
1. "person"
2. "face"
3. "pose"
4. "age"
5. "face_landmarks"
6. "gender"
7. "emotions"
8. "skin"

The network, loss and postprocessing would adapt according to specified heads, so no further work has to be done.
## Detection

## TODO list

1. Add data Augmentation
2. Extend PoseMap to larger scales to improve multiscale detection
3. Implement adaptive weighting for the total combined loss 
4. Extend a postprocessing
