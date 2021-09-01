# Data-Preprocessing
This folder contains the MATLAB (.m) files used to prepare the dataset for this experiment. The follower has the option for using these codes to prepare the dataset from the raw dataset (Link given in the 'Create_Dataset.m' file) or use the readymade datasets provided in the "Datasets" folder. The files are discussed below in brief:
# Create_Dataset
This piece of code automatically takes in the raw dataset and prepares a train and test set after all pre-processing. But the raw dataset should be present in the same directory (at least should have been downloaded in the local disk prior running the code). If the follower wants to modify/add/remove any data-preprocessing the feature or wants to understand the process instead of blindly using the preapre the dataset, they can play use this code along with the sub-functions mentioned below.
# Fix_Baseline_Drift
This function fixes the baseline drift of the PPG, ABP and ECG signals just after cropping the segment from the long sequence, prior to any further processing.
# PPG_Diff
This function takes in PPG signal and computes it first and second derivatives, namely VPG and APG, respectively. This code also filters the derivated signals which creates a delay which needs to be adjusted in all the signals (e.g., ECG or ABP) in order to maintain the concurrency. So, this function also returns the amount of delay incurred due to the consecutive digital filtering.
# crop_signal_delay
This function takes in the delay returned by the "PPG_Diff" function and applies it to other signals to maintan the alingment in the time domain.
# Remove_Bad_Signals
This function takes in PPG and ABP signals and through some novel method of signal processing removes the segment if either PPG or ABP breaks certain conditions or thresholds.
