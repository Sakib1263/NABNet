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
## Model Qualitative Performance  
While quantitative performances can be explored from the article itself, here we share some visualizations from the paper showing the robust qualitative performance of our approach.  
![NABNet Performance Figure 1](https://github.com/Sakib1263/NABNet/blob/main/Documents/1-s2.0-S1746809422007017-gr7.png "NABNet Performance Figure 1")  
**NABNet Performance in Estimating ABP from various PPG and ECG Morphology**  
![NABNet Performance Figure 2](https://github.com/Sakib1263/NABNet/blob/main/Documents/1-s2.0-S1746809422007017-gr8_lrg.png "NABNet Performance Figure 2")  
**NABNet Performance in Retaining Cardiovascular Anomalies (CVDs) from corresponding PPG and ECG signals**  
## Citation Request  
If you use out preprocessed data, code or any other materials in your work, please cite the following articles:
```
@article{MAHMUD2023104247,
title = {NABNet: A Nested Attention-guided BiConvLSTM network for a robust prediction of Blood Pressure components from reconstructed Arterial Blood Pressure waveforms using PPG and ECG signals},
journal = {Biomedical Signal Processing and Control},
volume = {79},
pages = {104247},
year = {2023},
issn = {1746-8094},
doi = {https://doi.org/10.1016/j.bspc.2022.104247},
url = {https://www.sciencedirect.com/science/article/pii/S1746809422007017},
author = {Sakib Mahmud and Nabil Ibtehaz and Amith Khandakar and M. {Sohel Rahman} and Antonio {JR. Gonzales} and Tawsifur Rahman and Md {Shafayet Hossain} and Md. {Sakib Abrar Hossain} and Md. {Ahasan Atick Faisal} and Farhan {Fuad Abir} and Farayi Musharavati and Muhammad {E. H. Chowdhury}},
keywords = {NABNet, Arterial Blood Pressure (ABP), Photoplethysmogram (PPG), Electrocardiogram (ECG), BP Prediction, ABP Estimation, Signal to Signal Synthesis, Signal Reconstruction, Guided Attention, Bidirectional Convolutional LSTM, 1D-Segmentation}}

@Article{s22030919,
AUTHOR = {Mahmud, Sakib and Ibtehaz, Nabil and Khandakar, Amith and Tahir, Anas M. and Rahman, Tawsifur and Islam, Khandaker Reajul and Hossain, Md Shafayet and Rahman, M. Sohel and Musharavati, Farayi and Ayari, Mohamed Arselene and Islam, Mohammad Tariqul and Chowdhury, Muhammad E. H.},
TITLE = {A Shallow U-Net Architecture for Reliably Predicting Blood Pressure (BP) from Photoplethysmogram (PPG) and Electrocardiogram (ECG) Signals},
JOURNAL = {Sensors},
VOLUME = {22},
YEAR = {2022},
NUMBER = {3},
ARTICLE-NUMBER = {919},
URL = {https://www.mdpi.com/1424-8220/22/3/919},
PubMedID = {35161664},
ISSN = {1424-8220},
DOI = {10.3390/s22030919}}
```
## References  
**[1]** S. Mahmud et al., "A Shallow U-Net Architecture for Reliably Predicting Blood Pressure (BP) from Photoplethysmogram (PPG) and Electrocardiogram (ECG) Signals", Sensors, vol. 22, no. 3, p. 919, 2022. Available: https://www.mdpi.com/1424-8220/22/3/919.  
**[2]** Zhou, Z., Siddiquee, M., Tajbakhsh, N., & Liang, J. (2021). UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation. Arxiv-vanity.com. Retrieved 30 August 2021, from https://www.arxiv-vanity.com/papers/1912.05074/.  
**[3]** S. Mahmud et al., "NABNet: A Nested Attention-guided BiConvLSTM network for a robust prediction of Blood Pressure components from reconstructed Arterial Blood Pressure waveforms using PPG and ECG signals", Biomedical Signal Processing and Control, vol. 79, no. 2, p. 104247, 2022. Available: https://www.sciencedirect.com/science/article/abs/pii/S1746809422007017?via%3Dihub.  
