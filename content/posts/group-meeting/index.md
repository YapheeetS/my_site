---
title: "group-share-meeting"
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

# 2020/11/13

## Title:

## Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting

### Abstract:

Modeling complex spatial and temporal correlations in the correlated time series data is indispensable for understanding the traffic dynamics and predicting the future status of an evolving traffic system. Recent works focus on designing complicated graph neural network architectures to capture shared patterns with the help of pre-defined graphs. In this paper, we argue that learning node-specific patterns is essential for traffic forecasting while the pre-defined graph is avoidable. To this end, we propose two adaptive modules for enhancing Graph Convolutional Network (GCN) with new capabilities: 1) a Node Adaptive Parameter Learning (NAPL) module to capture node-specific patterns; 2) a Data Adaptive Graph Generation (DAGG) module to infer the inter-dependencies among different traffic series automatically. We further propose an Adaptive Graph Convolutional Recurrent Network (AGCRN) to capture fine-grained spatial and temporal correlations in traffic series automatically based on the two modules and recurrent networks. Our experiments on two real-world traffic datasets show AGCRN outperforms state-of-the-art by a significant margin without pre-defined graphs about spatial connections.

### Link:

[https://arxiv.org/pdf/2007.02842.pdf](https://arxiv.org/pdf/2007.02842.pdf)

GCN+GRU, 分解卷积核(N,C,F -> N,d * d,C,F ), 数据自适应确定邻接矩阵(DAGG)


## Title:

## Pose Guided Person Image Generation

### Abstract:

This paper proposes the novel Pose Guided Person Generation Network (PG2) that allows to synthesize person images in arbitrary poses, based on an image of that person and a novel pose. Our generation framework PG2 utilizes the pose information explicitly and consists of two key stages: pose integration and image refinement. In the first stage the condition image and the target pose are fed into a U-Net-like network to generate an initial but coarse image of the person with the target pose. The second stage then refines the initial and blurry result by training a U-Net-like generator in an adversarial way. Extensive experimental results on both 128×64 re-identification images and 256×256 fashion photos show that our model generates high-quality person images with convincing details.

### Link:

[https://arxiv.org/pdf/1705.09368.pdf]([https://arxiv.org/pdf/1705.09368.pdf])

### 生成网络

网络包含两个生成网络,分别为G1,G2.

生成网络G1输入为condition image和target pos的串联,生成粗略的姿势图像,即coarse result.

生成网络G2,将condition image,与生成网络G1的输入串联,输入G2,生成一个difference map.

将G1,G2生成图像相加得到最后的生成图像,即refined result.

### 判别网络

判别网络用于判别输入图像是真实图像(target image)还是假的图像(生成图像).

即将target image 与condition image串联,输入判别网络,希望判别网络判别为real(预测值为1).

将refined result 与condition image串联,输入判别网络,希望判别网络判别为fake(预测值为0).



## Title:

## How to Distribute Computation in Networks

### Abstract:

In network function computation is as a means to reduce the required communication flow in terms of number of bits transmitted per source symbol. However, the rate region for the function computation problem in general topologies is an open problem, and has only been considered under certain restrictive assumptions (e.g. tree networks, linear functions, etc.). In this paper, we propose a new perspective for distributing computation, and formulate a flow-based delay cost minimization problem that jointly captures the costs of communications and computation. We introduce the notion of entropic surjectivity as a measure to determine how sparse the function is and to understand the limits of computation. Exploiting Little’s law for stationary systems, we provide a connection between this new notion and the computation processing factor that reflects the proportion of flow that requires communications. This connection gives us an understanding of how much a node (in isolation) should compute to communicate the desired function within the network without putting any assumptions on the topology. Our analysis characterizes the functions only via their entropic surjectivity, and provides insight into how to distribute computation. We numerically test our technique for search, MapReduce, and classification tasks, and infer for each task how sensitive the processing factor to the entropic surjectivity is.

### Link:

[https://arxiv.org/pdf/1912.03531.pdf]([https://arxiv.org/pdf/1912.03531.pdf])



## Title:

## Explainable Reinforcement Learning Through a Causal Lens

### Abstract:

Prominent theories in cognitive science propose that humans understand and represent the knowledge of the world through causal relationships. In making sense of the world, we build causal models in our mind to en- code cause-effect relations of events and use these to explain why new events happen by referring to coun- terfactuals — things that did not happen. In this pa- per, we use causal models to derive causal explanations of the behaviour of model-free reinforcement learning agents. We present an approach that learns a struc- tural causal model during reinforcement learning and encodes causal relationships between variables of inter- est. This model is then used to generate explanations of behaviour based on counterfactual analysis of the causal model. We computationally evaluate the model in 6 domains and measure performance and task pre- diction accuracy. We report on a study with 120 partic- ipants who observe agents playing a real-time strategy game (Starcraft II) and then receive explanations of the agents’ behaviour. We investigate: 1) participants’ un- derstanding gained by explanations through task pre- diction; 2) explanation satisfaction and 3) trust. Our results show that causal model explanations perform better on these measures compared to two other base- line explanation models.

### Link:

[https://arxiv.org/pdf/1905.10958.pdf]([https://arxiv.org/pdf/1905.10958.pdf])





# 2020/11/20

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



