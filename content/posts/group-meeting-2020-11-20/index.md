---
title: "group-share-meeting - 2020/11/20"
date: 2020-11-20T15:40:51+09:00
description: Group share meeting
draft: false
hideToc: false
enableToc: true
enableTocContent: true
tocPosition: inner
tags:
- Machine learning
- Deep learning
categories:
- cv
image: images/feature1/markdown.png
---



## Title:

## Pose2Seg：Detection Free Human Instance Segmentation

### Abstract:

The standard approach to image instance segmentation is to perform the object detection first, and then segment the object from the detection bounding-box. More recently, deep learning methods like Mask R-CNN [14] perform them jointly. However, little research takes into account the uniqueness of the “human” category, which can be well defined by the pose skeleton. Moreover, the human pose skeleton can be used to better distinguish instances with heavy occlusion than using bounding-boxes. In this pa- per, we present a brand new pose-based instance segmen- tation framework1 for humans which separates instances based on human pose, rather than proposal region de- tection. We demonstrate that our pose-based framework can achieve better accuracy than the state-of-art detection- based approach on the human instance segmentation prob- lem, and can moreover better handle occlusion. Further- more, there are few public datasets containing many heavily occluded humans along with comprehensive annotations, which makes this a challenging problem seldom noticed by researchers. Therefore, in this paper we introduce a new benchmark “Occluded Human (OCHuman)”2, which fo- cuses on occluded humans with comprehensive annotations including bounding-box, human pose and instance masks. This dataset contains 8110 detailed annotated human in- stances within 4731 images. With an average 0.67 Max- IoU for each person, OCHuman is the most complex and challenging dataset related to human instance segmenta- tion. Through this dataset, we want to emphasize occlusion as a challenging problem for researchers to study.

### Link:

[https://arxiv.org/pdf/1803.10683.pdf](https://arxiv.org/pdf/1803.10683.pdf)

网络框架主要由Affine-Align, Skeleton features和SegModule三部分组成。首先，将有人体姿态标注的图像作为输入，用基础网络（resnet50FPN）提取特征；接着通用Affine-Align operation基于人体动作将ROIs对齐为统一大小（本文中为64*64），同时为图中每个人体生成骨架特征；将上述两者concate之后传给SegModule对图中每个人体进行分割；最后，将Affine-Align operation中的所得仿射变换矩阵H对图中人物反转对齐，得到最终分割结果。

### 关键技术:

1.Affine-Align Operation
Affine-Align的作用与Faster RCNN的ROI Pooling和Mask RCNN的ROI Align类似，都是将ROI对齐成特定大小。但是与它们不同的是，Affine-Align是基于人物的动作对齐，而不是边界框。通过人类动作蕴涵的信息，AffineAlign操作可以把奇怪的人类动作拉直，然后将重叠的人分开。

2.Skeleton Features
骨架特征的提取采用的是Realtime multi-person 2d pose estimation using part affinity fields中的方法，通过part confidence maps进行身体关节点检测，然后用PAFs进行关节点进行连接，最后将它们结合起来，得到图像中每个实例的骨架特征。

3.SegModule
SegModule始于一个7*7，步长为2的卷积层，接着是几个标准残差unit，以便为RoI实现足够大的感受野。之后，通过双线性上采样层来扩大分辨率，并且使用另一个残余单元以及1个1*1的卷积层来预测最终结果。其中10个残差单元的这种结构可以实现大约50个像素的感受野。



