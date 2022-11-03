# Deep Learning Pipeline for ABP Estimation using NABNet
This repository contains the end-to-end 1D Convolutional Neural Network (1D-CNN) based Segmentation Pipeline to estimate ABP from PPG, VPG, APG and ECG waveforms. The estimated BP values (SBP and DBP) from the shallow UNet based autoencoder from [1] and trained segmentation model trained on the same dataset can be found from these links:  

* Trained Segmentation Model Fold 1: https://drive.google.com/file/d/1hgD3APmcylf54RoGpjSGbPLMrRQA3Jkd/view?usp=sharing  
* SBP Fold 1: https://drive.google.com/file/d/1-iaHpspm3XVVVosn6M6QDJWqlLyVjH7y/view?usp=sharing
* DBP Fold 1: https://drive.google.com/file/d/1-eTFPKWgDysc8Ah8mers_pKvGX327C8c/view?usp=sharing
* SBP Fold 2: https://drive.google.com/file/d/1Ei7PxaCEIIqhu_0A7Lc3JQnqJ84hw1JC/view?usp=sharing
* DBP Fold 2: https://drive.google.com/file/d/1D7bjldaG1ZO0UlHI3-lo_mYXJVG6PW-X/view?usp=sharing
* SBP Fold 3: https://drive.google.com/file/d/1-BiMWmbfKnVK3rN15tLHS5UxFlj5M3ax/view?usp=sharing
* DBP Fold 3: https://drive.google.com/file/d/1-AnRNW09qL5Kh4zHb4PBgcodYc48ed94/view?usp=sharing
* SBP Fold 4: https://drive.google.com/file/d/1-_Wz7W3Qrdu5iRWXjtULUeUmb2CL74xf/view?usp=sharing
* DBP Fold 4: https://drive.google.com/file/d/1-ZObf25po8F_PxOpAZQIiTRZgple6iOR/view?usp=sharing

These estimated BP values are for the 4 channel (PPG, VPG, APG, ECG) approach. Details of the approach in this paper [1]. Similar approach was followed to estimate BP values for this study.  

Mentionable that these BP values can be predicted by any other classical or deep learning approach, only the same set of signals need to be used for both BP prediction and ABP segmentation pipelines as discussed in this paper [2]. Estimated BP values and the trained segmentation model can loaded directly into the pipeline, tested and evaluated on the respective test folds. This model takes in normalized PPG, VPG, APG and ECG as the inputs and normalized, segmented ABP as the output [2].  


## Reference:
**[1]** S. Mahmud et al., "A Shallow U-Net Architecture for Reliably Predicting Blood Pressure (BP) from Photoplethysmogram (PPG) and Electrocardiogram (ECG) Signals", Sensors, vol. 22, no. 3, p. 919, 2022. Available: https://www.mdpi.com/1424-8220/22/3/919. [Accessed 18 August 2022].  
**[2]** S. Mahmud et al., "NABNet: A Nested Attention-guided BiConvLSTM network for a robust prediction of Blood Pressure components from reconstructed Arterial Blood Pressure waveforms using PPG and ECG signals", Biomedical Signal Processing and Control, vol. 79, no. 2, p. 104247, 2022. Available: https://www.sciencedirect.com/science/article/abs/pii/S1746809422007017?via%3Dihub.   
