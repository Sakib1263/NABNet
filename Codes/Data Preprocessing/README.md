# Data-Preprocessing
This folder contains the MATLAB (.m) files used to prepare the dataset for this experiment. The "The Cuff-Less Blood Pressure Estimation dataset" present in the UCI repository is a processed subset of the MIMIC-III Waveform Dataset. For this study, we download the dataset, preprocess it, segment it and create four sets from the existing four parts in the dataset. Three of the four parts was used for training and the remaining one for testing. These codes can be used to prepare the dataset from the raw dataset or use the readymade datasets provided in the "Datasets" folder. The files are described below in brief:
## Create_UCI_Dataset
This piece of code automatically takes in the raw dataset and prepares a train and test set after all pre-processing. But the raw dataset should be present in the same directory (at least should have been downloaded in the local disk prior running the code). If the follower wants to modify/add/remove any data-preprocessing the feature or wants to understand the process instead of blindly using the preapre the dataset, they can play use this code along with the sub-functions mentioned below.
## Fix_Baseline_Drift
This function fixes the baseline drift of the PPG, ABP and ECG signals just after cropping the segment from the long sequence, prior to any further processing. There are two versions of it.
## PPG_Diff
This function takes in PPG signal and computes it first and second derivatives, namely VPG and APG, respectively. This code also filters the derivated signals which creates a delay which needs to be adjusted in all the signals (e.g., ECG or ABP) in order to maintain the concurrency. So, this function also returns the amount of delay incurred due to the consecutive digital filtering. There is a version of this code which performs Phase Shift (PS) correction during alignment as an extra last step.
## crop_signal_delay
This function takes in the delay returned by the "PPG_Diff" function and applies it to other signals to maintan the alingment in the time domain.
## Remove_Bad_Signals
This function takes in PPG and ABP signals and through some novel methods removes the segment if either PPG or ABP breaks certain thresholds.
## NotchFilterIIR
Implements an IIR based Notch Filter, can be used to remove powerline components from the signals (e.g., 50/60 Hz).
## filtbutter
This 'p' function can be used to implement a robust and flexible ButterWorth BandPass Filter. This file can be only be used, cannot be modified.
## Create_Folds
This piece of code can be used to create 'n' folds (non-stratified) from a dataset.
## hist
Hist can be used produce Histogram of the dataset.
## violin
This custom function can be used produce Violin Plot of the dataset.
