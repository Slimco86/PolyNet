# PolyNet
## Introduction
Polynet is a multitask, human-centric network aiming to provide a detailed analysis and tracking for multi-human view scenes. The main objective and the challange is to achieve real-time on edge-device performance to make it feasible for certain applications and to fulfill GDPR requirements. In particular, the tasks can be split into three categories: dtection, classification and landmark estimation.
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
![EfficientDet architecture](/description_images/EfficientDet.jpg "EfficientDet architecture")
The network is fully-convolutional with performant feature extraction stem and bidirectional feature pyramid network, introduced in the paper. As well, the network benefits of extensive application of Swish activation function. Classification and bounding box prediction, in this project, is done exactly as it is performed  in the classical EfficientDet network. For each pair of regression/classification task additional head is attached to the network. Each classification is combined either with person bbox (full body) or with face bbox (face) prediction task according to relevancy. For example, emotion calssification are combined with face bounding box prediction.

For landmark prediction an additional module/head has been developed which is performed on the same input features as other heads. The module takes features on multiple scales extracted from bifpn (f3,f4,f5) and performes a [HourGlass](https://arxiv.org/abs/1603.06937)-like upscaling to predict a probability mask for landmarks positions on the original input scale. The major difference with the method proposed in the paper above, is that the network tries to predict all the landmarks at once, on a single channel, as the total amount of the landmarks on the output is a variable and initially is unknown.
