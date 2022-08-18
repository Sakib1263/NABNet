%% Create X_Fold Cross Validation Set, here X = 5
load('Train_Dataset_Large.mat');
Data_Length = length(PPG);
signal_length = 1024;
ABP_GRND = ABP;
Fold_num = 5;
VL = floor(Data_Length / Fold_num);
for i=1:Fold_num
    j = i-1;
    if (i == 1)
        PPG_1 = [];
        ABP_1 = [];
        ABP_GRND_1 = [];
        ECG_1 = [];
    else
        PPG_1 = PPG(:,1:VL*j);
        ABP_1 = ABP(:,1:VL*j);
        ABP_GRND_1 = ABP_GRND(:,1:VL*j);
        ECG_1 = ECG(:,1:VL*j);
    end
    if (i == Fold_num)
        PPG_2 = [];
        ABP_2 = [];
        ABP_GRND_2 = [];
        ECG_2 = [];
    else
        PPG_2 = PPG(:,VL*i+1:VL*Fold_num);
        ABP_2 = ABP(:,VL*i+1:VL*Fold_num);
        ABP_GRND_2 = PPG(:,VL*i+1:VL*Fold_num);
        ECG_2 = PPG(:,VL*i+1:VL*Fold_num);
    end

    Train_PPG = horzcat(PPG_1,PPG_2);
    PPG_MAX = max(max(Train_PPG));
    PPG_MIN = min(min(Train_PPG));
    Train_PPG = (Train_PPG - PPG_MIN)/PPG_MAX;
    Train_ABP = horzcat(ABP_1,ABP_2);
    ABP_MAX = max(max(Train_ABP));
    ABP_MIN = min(min(Train_ABP));
    Train_ABP = (Train_ABP - ABP_MIN)/ABP_MAX;
    Train_ABP_GRND = horzcat(ABP_GRND_1,ABP_GRND_2);
    Train_ECG = horzcat(ECG_1,ECG_2);
    %
    Val_PPG = PPG(:,VL*j+1:VL*i);
    Val_PPG = (Val_PPG - PPG_MIN)/PPG_MAX;
    Val_ABP = ABP(:,VL*j+1:VL*i);
    Val_ABP = (Val_ABP - ABP_MIN)/ABP_MAX;
    Val_ABP_GRND = ABP_GRND(:,VL*j+1:VL*i);
    Val_ECG = ECG(:,VL*j+1:VL*i);
    %
    train_path_mat = sprintf('Folds1/Fold %d/Train_Data_Fold_%d.mat',i,i);
    val_path_mat = sprintf('Folds1/Fold %d/Val_Data_Fold_%d.mat',i,i);
    %
    save(train_path_mat,'Train_PPG','Train_ABP','Train_ABP_GRND','Train_ECG','ABP_MAX','ABP_MIN','PPG_MAX','PPG_MIN','-v7.3');
    save(val_path_mat,'Val_PPG','Val_ABP','Val_ABP_GRND','Val_ECG','-v7.3');
    %
    train_path_hdf5 = sprintf('Folds1/Fold %d/Train_Data_Fold_%d.h5',i,i);
    val_path_hdf5 = sprintf('Folds1/Fold %d/Val_Data_Fold_%d.h5',i,i);
    %
    h5create(train_path_hdf5,'/PPG',[signal_length length(Train_PPG)]);
    h5create(train_path_hdf5,'/ABP',[signal_length length(Train_PPG)]);
    h5create(train_path_hdf5,'/ECG',[signal_length length(Train_PPG)]);
    h5create(train_path_hdf5,'/ABP_GRND',[signal_length length(Train_PPG)]);
    h5create(train_path_hdf5,'/ABP_MAX',[1 1]);
    h5create(train_path_hdf5,'/ABP_MIN',[1 1]);
    h5create(train_path_hdf5,'/PPG_MAX',[1 1]);
    h5create(train_path_hdf5,'/PPG_MIN',[1 1]);
    h5write(train_path_hdf5,'/PPG',Train_PPG);
    h5write(train_path_hdf5,'/ABP',Train_ABP);
    h5write(train_path_hdf5,'/ECG',Train_ECG);
    h5write(train_path_hdf5,'/ABP_GRND',Train_ABP_GRND);
    h5write(train_path_hdf5,'/ABP_MAX',ABP_MAX);
    h5write(train_path_hdf5,'/ABP_MIN',ABP_MIN);
    h5write(train_path_hdf5,'/PPG_MAX',PPG_MAX);
    h5write(train_path_hdf5,'/PPG_MIN',PPG_MIN);
    %
    h5create(val_path_hdf5,'/PPG',[signal_length length(Val_PPG)]);
    h5create(val_path_hdf5,'/ABP',[signal_length length(Val_PPG)]);
    h5create(val_path_hdf5,'/ECG',[signal_length length(Val_PPG)]);
    h5create(val_path_hdf5,'/ABP_GRND',[signal_length length(Val_PPG)]);
    h5write(val_path_hdf5,'/PPG',Val_PPG);
    h5write(val_path_hdf5,'/ABP',Val_ABP);
    h5write(val_path_hdf5,'/ECG',Val_ECG);
    h5write(val_path_hdf5,'/ABP_GRND',Val_ABP_GRND);
end
%% Plotting Sample Data
start_point = 1;
end_point = 1024;
Signal_Number = 250;

A = Train_PPG_Fold5(:,Signal_Number);
B = Train_ABP_Fold5(:,Signal_Number);
C = Train_ABP_GRND_Fold5(:,Signal_Number);
D = Train_ECG_Fold5(:,Signal_Number);

figure;
subplot(2,2,1);
plot(A);
title('Photoplethysmogram (PPG)');
subplot(2,2,2);
plot(B);
title('Arterial Blood Pressure (ABP) Normalized');
subplot(2,2,3);
plot(C);
title('Arterial Blood Pressure (ABP)');
subplot(2,2,4);
plot(D);
title('ECG');
