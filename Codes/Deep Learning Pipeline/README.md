# Deep Learning Pipeline for ABP Estimation using NABNet
This repository contains the end-to-end 1D Convolutional Neural Network (1D-CNN) based Segmentation Pipeline to estimate ABP from PPG, VPG, APG and ECG waveforms. The estimated BP values (SBP and DBP) from an UNet based autoencoder can be found from these links:  

Fold 1 (Part 4):  
Fold 2 (Part 3):  
Fold 3 (Part 2):  
Fold 4 (Part 1):  

These estimated BP values are for the 4 channel (PPG, VPG, APG, ECG) approach. Details of the approach in this paper [1]. Similar approach was followed to estimate BP values for this study. Mentionable that these BP values can be predicted by any other classical or deep learning approach, only the same set of signals need to be used for both BP prediction and ABP segmentation pipelines as discussed in this paper [2].


## Reference:
[1] S. Mahmud et al., "A Shallow U-Net Architecture for Reliably Predicting Blood Pressure (BP) from Photoplethysmogram (PPG) and Electrocardiogram (ECG) Signals", Sensors, vol. 22, no. 3, p. 919, 2022. Available: https://www.mdpi.com/1424-8220/22/3/919. [Accessed 18 August 2022].  

[2]  
