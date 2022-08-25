# NABNet
## Introduction
ABP Waveform Estimation from PPG, PPG derivatives and ECG using Deep Learning based 1D-Segmentation models. This repository contains files related to the paper called "NABNet: A Nested Attention-guided BiConvLSTM Network for a robust prediction of Blood Pressure components from reconstructed Arterial Blood Pressure waveforms using PPG and ECG Signals"  
## ABP Estimation Pipeline  
Our proposed ABP estimation pipeline divides the task into two parts viz. BP prediction and ABP pattern estimation. Combining outcomes from both sub-pipelines provides with the final estimation ABP waveforms. Mentionable that this paper covers only the ABP segmentation task and uses predicted BP values from this articles published in MDPI Sensors [1]. Nevertheless, any other robust BP prediction method can be used as well.  
![ABP Estimation Pipeline](https://github.com/Sakib1263/NABNet/blob/main/Documents/Pipeline.png "ABP Estimation Pipeline")  
**Proposed ABP Estimation End2End Pipeline Block Diagram**  
## NABNet Architecture  
The proposed NABNet has been built on the UNet++ segmentation model [2]. Instead of direct skip connections, NABNet implements attetion-guided BiConvLSTM blocks as shown in the Figure below. NABNet also has Multi-attention BiConvLSTM blocks for the inner convolutional blocks.  
![NABNet Architecture](https://github.com/Sakib1263/NABNet/blob/main/Documents/NABNet.png "NABNet Architecture")  
**NABNet Architecture Breakdown**  
## Model Performance  
## Citation Request  
## References  
**[1]** S. Mahmud et al., "A Shallow U-Net Architecture for Reliably Predicting Blood Pressure (BP) from Photoplethysmogram (PPG) and Electrocardiogram (ECG) Signals", Sensors, vol. 22, no. 3, p. 919, 2022. Available: https://www.mdpi.com/1424-8220/22/3/919.  
**[2]** Zhou, Z., Siddiquee, M., Tajbakhsh, N., & Liang, J. (2021). UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation. Arxiv-vanity.com. Retrieved 30 August 2021, from https://www.arxiv-vanity.com/papers/1912.05074/. 
**[3]**  
