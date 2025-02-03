# TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation

Ehsan Adeli3, Yan Wang4, Le Lu1, Qihang Yu1, Xiangde Luo2, Jieneng Chen1, Yongyi Lu5, Alan L. Yuille1, and Yuyin Zhou3

1Johns Hopkins University

2University of Electronic Science and Technology of China

3Stanford University

4East China Normal University

5PAII Inc.

arXiv:2102.04306v1 · [cs.CV] · 8 Feb 2021

# Abstract

Medical image segmentation is an essential prerequisite for developing healthcare systems, especially for disease diagnosis and treatment planning. On various medical image segmentation tasks, the u-shaped architecture, also known as U-Net, has become the de-facto standard and achieved tremendous success. However, due to the intrinsic locality of convolution operations, U-Net generally demonstrates limitations in explicitly modeling long-range dependency. Transformers, designed for sequence-to-sequence prediction, have emerged as alternative architectures with innate global self-attention mechanisms, but can result in limited localization abilities due to insufficient low-level details. In this paper, we propose TransUNet, which merits both Transformers and U-Net, as a strong alternative for medical image segmentation. On one hand, the Transformer encodes tokenized image patches from a convolution neural network (CNN) feature map as the input sequence for extracting global contexts. On the other hand, the decoder upsamples the encoded features which are then combined with the high-resolution CNN feature maps to enable precise localization. We argue that Transformers can serve as strong encoders for medical image segmentation tasks, with the combination of U-Net to enhance finer details by recovering localized spatial information. TransUNet achieves superior performances to various competing methods on different medical applications including multi-organ segmentation and cardiac segmentation. Code and models are available at https://github.com/Beckschen/TransUNet.

# 1 Introduction

Convolutional neural networks (CNNs), especially fully convolutional networks (FCNs) [8], have become dominant in medical image segmentation. Among different variants, U-Net [12], which consists of a symmetric encoder-decoder network with skip-connections to enhance detail retention, has become the de-facto choice. Based on this line of approach, tremendous success has been achieved in a wide range of medical applications such as cardiac segmentation from