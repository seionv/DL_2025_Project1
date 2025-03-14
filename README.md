# DL_2025_Project1
This is the NYU deep learning course mini-project for using ResNet Architecture to detect the CIFAR-10 set.

## Team 
- Feiyu Jia (netID: fj2182)
- Tianzan Min (netID: tm4485)
- Xiaoyu Liu (netID: xl5808)

## Overview
This project aims to train a ResNet model that achieves over 90% accuracy with fewer than 5 million trainable parameters. We explored various fine-tuning techniques from both the model architecture and training process to achieve this goal. Specifically, we sought an optimal combination of parameter-related hyperparameters and a suitable model architecture to enhance performance. During training, we applied techniques such as the AdamW optimizer, data augmentation, cosine annealing, and warm-up training to further improve the model’s performance. For evaluation, we used our model to perform the classification task on the CIFAR-10 dataset.

Model Structure:
-------------------------|---------------------------|------------
        Layer (type)     |          Output Shape     |    Param #
---|---|---
            Conv2d-1     |      [-1, 64, 32, 32]       |    1,728
       BatchNorm2d-2     |      [-1, 64, 32, 32]      |       128
            Conv2d-3     |      [-1, 64, 32, 32]      |    36,864
       BatchNorm2d-4      |     [-1, 64, 32, 32]      |       128
            Conv2d-5     |      [-1, 64, 32, 32]      |    36,864
       BatchNorm2d-6     |      [-1, 64, 32, 32]        |     128
     BasicResBlock-7     |      [-1, 64, 32, 32]       |        0
            Conv2d-8    |       [-1, 64, 32, 32]     |     36,864
       BatchNorm2d-9     |      [-1, 64, 32, 32]     |        128
           Conv2d-10    |       [-1, 64, 32, 32]      |    36,864
      BatchNorm2d-11      |     [-1, 64, 32, 32]     |        128
    BasicResBlock-12      |     [-1, 64, 32, 32]     |          0
           Conv2d-13      |     [-1, 64, 32, 32]     |     36,864
      BatchNorm2d-14      |     [-1, 64, 32, 32]    |         128
           Conv2d-15   |        [-1, 64, 32, 32]      |    36,864
      BatchNorm2d-16     |      [-1, 64, 32, 32]    |         128
    BasicResBlock-17     |      [-1, 64, 32, 32]   |            0
           Conv2d-18    |      [-1, 128, 16, 16]     |     73,728
      BatchNorm2d-19    |      [-1, 128, 16, 16]     |        256
           Conv2d-20     |     [-1, 128, 16, 16]     |    147,456
      BatchNorm2d-21      |    [-1, 128, 16, 16]     |        256
           Conv2d-22      |    [-1, 128, 16, 16]     |      8,192
      BatchNorm2d-23    |      [-1, 128, 16, 16]      |       256
    BasicResBlock-24    |      [-1, 128, 16, 16]     |          0
           Conv2d-25    |      [-1, 128, 16, 16]     |    147,456
      BatchNorm2d-26    |      [-1, 128, 16, 16]     |        256
           Conv2d-27    |      [-1, 128, 16, 16]     |    147,456
      BatchNorm2d-28    |      [-1, 128, 16, 16]     |        256
    BasicResBlock-29    |      [-1, 128, 16, 16]     |          0
           Conv2d-30    |      [-1, 128, 16, 16]     |    147,456
      BatchNorm2d-31    |      [-1, 128, 16, 16]     |        256
           Conv2d-32    |      [-1, 128, 16, 16]     |    147,456
      BatchNorm2d-33    |      [-1, 128, 16, 16]     |        256
    BasicResBlock-34    |      [-1, 128, 16, 16]     |          0
           Conv2d-35    |      [-1, 128, 16, 16]     |    147,456
      BatchNorm2d-36    |      [-1, 128, 16, 16]     |        256
           Conv2d-37    |      [-1, 128, 16, 16]     |    147,456
      BatchNorm2d-38    |      [-1, 128, 16, 16]     |        256
    BasicResBlock-39    |      [-1, 128, 16, 16]     |          0
           Conv2d-40    |      [-1, 128, 16, 16]     |    147,456
      BatchNorm2d-41    |      [-1, 128, 16, 16]     |        256
           Conv2d-42    |      [-1, 128, 16, 16]     |    147,456
      BatchNorm2d-43    |      [-1, 128, 16, 16]     |        256
    BasicResBlock-44    |      [-1, 128, 16, 16]      |         0
           Conv2d-45    |        [-1, 256, 8, 8]     |    294,912
      BatchNorm2d-46    |        [-1, 256, 8, 8]      |       512
           Conv2d-47    |        [-1, 256, 8, 8]      |   589,824
      BatchNorm2d-48    |        [-1, 256, 8, 8]     |        512
           Conv2d-49    |        [-1, 256, 8, 8]    |      32,768
      BatchNorm2d-50    |        [-1, 256, 8, 8]    |         512
    BasicResBlock-51    |        [-1, 256, 8, 8]    |           0
           Conv2d-52    |        [-1, 256, 8, 8]    |     589,824
      BatchNorm2d-53    |        [-1, 256, 8, 8]    |         512
           Conv2d-54    |        [-1, 256, 8, 8]    |     589,824
      BatchNorm2d-55    |        [-1, 256, 8, 8]    |         512
    BasicResBlock-56    |        [-1, 256, 8, 8]    |           0
           Conv2d-57    |        [-1, 256, 8, 8]    |     589,824
      BatchNorm2d-58    |        [-1, 256, 8, 8]    |         512
           Conv2d-59    |        [-1, 256, 8, 8]    |     589,824
      BatchNorm2d-60    |        [-1, 256, 8, 8]    |         512
    BasicResBlock-61    |        [-1, 256, 8, 8]    |           0
         Identity-62    |        [-1, 256, 8, 8]    |           0
           Linear-63    |               [-1, 10]    |       2,570
---|---|---

Total params: 4,918,602
Trainable params: 4,918,602
Non-trainable params: 0

## Results
Our final model achieved a test accuracy of 92% after 50 training epochs and had 4,918,602 trainable parameters.

After training:

Train Loss|Test Loss|Train Acc.(%)|Test Acc.(%) |Learning Rate
--------|---------|---------|---------|-----
0.18543|0.20701|93.45|93.92|0.00025447

Best result:

Best Model Saved|Test loss|Test acc
---|---|---
Best model|0.20615|0.93928

Kaggle Score:

Private Score|Public Score
-------------|------------
0.80894|0.80067