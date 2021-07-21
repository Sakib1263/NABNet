%% Split UCI Train-Test Data
clear;
load('UCI Dataset/Part_1.mat');
load('UCI Dataset/Part_2.mat');
load('UCI Dataset/Part_3.mat');
load('UCI Dataset/Part_4.mat');
UCI_Train_Dataset = horzcat(Part_1, Part_2, Part_3);
UCI_Test_Dataset = Part_4;
%% Prepare UCI Train and Test Datasets
warning('off');
% Input Signal Length should be at least 60 samples more than the Output, or more
input_signal_length = 1100;
output_signal_length = 1024;
% Make or Set Destination Directory
train_path_mat = sprintf('UCI_Train_Dataset_v2.mat');
train_path_hdf5 = sprintf('UCI_Train_Dataset_v2.h5');
test_path_mat = sprintf('UCI_Test_Dataset_v2.mat');
test_path_hdf5 = sprintf('UCI_Test_Dataset_v2.h5');
%
for c = 1:2
    if c == 1
        dataset_size = length(UCI_Train_Dataset);
    elseif c == 2
        dataset_size = length(UCI_Test_Dataset);
    end
    % No resampling required.
    % The Data for this dataset has been collected at 125Hz Sampling Frequency by Default
    counter = 0;
    bad_signal_count = 0;
    %
    PPG_TOT_UCI = zeros(output_signal_length,300000);
    VPG_TOT_UCI = zeros(output_signal_length,300000);
    APG_TOT_UCI = zeros(output_signal_length,300000);
    ECG_TOT_UCI = zeros(output_signal_length,300000);
    ECG_RPR_UCI = zeros(output_signal_length,300000);
    ABP_TOT_UCI = zeros(output_signal_length,300000);
    ABP_RNorm_UCI = zeros(output_signal_length,300000);
    SBP_TOT_UCI = zeros(1,300000);
    DBP_TOT_UCI = zeros(1,300000);
    
    for i = 1:dataset_size
        if c == 1
            cell = UCI_Train_Dataset{1,i};
        elseif c == 2
            cell = UCI_Test_Dataset{1,i};
        end
        cell_length = length(cell);
        if cell_length < input_signal_length
            continue;
        end
        PPG = cell(1,:);
        ABP = cell(2,:);
        ECG = cell(3,:);
        sample_num = floor(cell_length/input_signal_length);
        L_temp = sample_num*input_signal_length;
        PPG = PPG(1:L_temp);
        ABP = ABP(1:L_temp);
        ECG = ECG(1:L_temp);
        for ii = 0:(sample_num-1)
            counter = counter+1;
            Iter_Left = dataset_size - i;
            %
            PPG_PP = normalize(Fix_Baseline_Drift(PPG(input_signal_length*ii+1:input_signal_length*ii+input_signal_length)),'range');
            %
            ABP_PP1 = ABP(input_signal_length*ii+1:input_signal_length*ii+input_signal_length);
            ABP_PP2 = Fix_Baseline_Drift(ABP(input_signal_length*ii+1:input_signal_length*ii+input_signal_length));
            ABP_PP1_AMP = max(ABP_PP1) - min(ABP_PP1);
            ABP_PP2_AMP = max(ABP_PP2) - min(ABP_PP2);
            ABP_PP = (ABP_PP2*(ABP_PP1_AMP/ABP_PP2_AMP))+min(ABP_PP1);
            ABP_PP_RNorm = normalize(ABP_PP,'range');
            %
            ECG_PP = normalize(Fix_Baseline_Drift(ECG(input_signal_length*ii+1:input_signal_length*ii+input_signal_length)),'range');
            ECG_RPR_PP = normalize(sgolayfilt(ECG_Peak_Removal(ECG_PP),7,21),'range');
            %
            [PPG_PP,VPG_PP,APG_PP,ECG_PP,ABP_PP,ABP_PP_RNorm,ECG_RPR_PP] = PPG_diff(PPG_PP,ECG_PP,ABP_PP,ABP_PP_RNorm,ECG_RPR_PP,input_signal_length,output_signal_length);
            %
            Decision = Remove_Bad_Signals(PPG_PP,ABP_PP,output_signal_length);
            TF = mean(isnan(ABP_PP));
            SBP = max(ABP_PP);
            DBP = min(ABP_PP);
            BP_DIFF = SBP - DBP;
            if (TF > 0)||((SBP > 190)||(SBP < 80)||(DBP > 120)||(DBP < 50)||(BP_DIFF < 20)||(BP_DIFF > 120))||(Decision == 0)
                bad_signal_count = bad_signal_count + 1;
                counter = counter - 1;
                continue
            else
                PPG_TOT_UCI(:,counter) = PPG_PP';
                VPG_TOT_UCI(:,counter) = VPG_PP';
                APG_TOT_UCI(:,counter) = APG_PP';
                ECG_TOT_UCI(:,counter) = ECG_PP';
                ECG_RPR_UCI(:,counter) = ECG_RPR_PP';
                ABP_TOT_UCI(:,counter) = ABP_PP';
                ABP_RNorm_UCI(:,counter) = ABP_PP_RNorm';
                SBP_TOT_UCI(1,counter) = SBP;
                DBP_TOT_UCI(1,counter) = DBP;
            end
            fprintf('Samples Collected = %d, Iteration Left = %d, Bad Signal Count = %d\n',counter,Iter_Left,bad_signal_count);
        end
    end

    PPG = PPG_TOT_UCI(:,1:counter);
    VPG = VPG_TOT_UCI(:,1:counter);
    APG = APG_TOT_UCI(:,1:counter);
    ECG = ECG_TOT_UCI(:,1:counter);
    ECG_RPR = ECG_RPR_UCI(:,1:counter);
    ABP = ABP_TOT_UCI(:,1:counter);
    ABP_RNorm = ABP_RNorm_UCI(:,1:counter);
    SBP = SBP_TOT_UCI(1,1:counter);
    DBP = DBP_TOT_UCI(1,1:counter);
    %
    ABP_GRND = ABP;
    ABP_MAX = max(max(ABP));
    ABP_MIN = min(min(ABP));
    ABP_AMP = ABP_MAX - ABP_MIN;
    ABP = (ABP - ABP_MIN)/ABP_AMP;
    
    if c == 1
        save(train_path_mat,'PPG','VPG','APG','ECG','ECG_RPR','ABP','SBP','DBP','ABP_GRND','ABP_RNorm','ABP_AMP','ABP_MIN','-v7.3');
        %
        h5create(train_path_hdf5,'/PPG',[output_signal_length length(PPG)]);
        h5create(train_path_hdf5,'/ABP',[output_signal_length length(PPG)]);
        h5create(train_path_hdf5,'/VPG',[output_signal_length length(PPG)]);
        h5create(train_path_hdf5,'/APG',[output_signal_length length(PPG)]);
        h5create(train_path_hdf5,'/ECG',[output_signal_length length(PPG)]);
        h5create(train_path_hdf5,'/ECG_RPR',[output_signal_length length(PPG)]);
        h5create(train_path_hdf5,'/SBP',[1 length(PPG)]);
        h5create(train_path_hdf5,'/DBP',[1 length(PPG)]);
        h5create(train_path_hdf5,'/ABP_AMP',[1 1]);
        h5create(train_path_hdf5,'/ABP_MIN',[1 1]);
        h5create(train_path_hdf5,'/ABP_GRND',[output_signal_length length(PPG)]);
        h5create(train_path_hdf5,'/ABP_RNorm',[output_signal_length length(PPG)]);
        h5write(train_path_hdf5,'/PPG',PPG);
        h5write(train_path_hdf5,'/ABP',ABP);
        h5write(train_path_hdf5,'/VPG',VPG);
        h5write(train_path_hdf5,'/APG',APG);
        h5write(train_path_hdf5,'/ECG',ECG);
        h5write(train_path_hdf5,'/ECG_RPR',ECG_RPR);
        h5write(train_path_hdf5,'/SBP',SBP);
        h5write(train_path_hdf5,'/DBP',DBP);
        h5write(train_path_hdf5,'/ABP',ABP);
        h5write(train_path_hdf5,'/ABP_AMP',ABP_AMP);
        h5write(train_path_hdf5,'/ABP_MIN',ABP_MIN);
        h5write(train_path_hdf5,'/ABP_GRND',ABP_GRND);
        h5write(train_path_hdf5,'/ABP_RNorm',ABP_RNorm);
    elseif c == 2
        save(test_path_mat,'PPG','VPG','APG','ECG','ECG_RPR','ABP','SBP','DBP','ABP_GRND','ABP_AMP','ABP_MIN','-v7.3');
        %
        h5create(test_path_hdf5,'/PPG',[output_signal_length length(PPG)]);
        h5create(test_path_hdf5,'/ABP',[output_signal_length length(PPG)]);
        h5create(test_path_hdf5,'/VPG',[output_signal_length length(PPG)]);
        h5create(test_path_hdf5,'/APG',[output_signal_length length(PPG)]);
        h5create(test_path_hdf5,'/ECG',[output_signal_length length(PPG)]);
        h5create(test_path_hdf5,'/ECG_RPR',[output_signal_length length(PPG)]);
        h5create(test_path_hdf5,'/SBP',[1 length(PPG)]);
        h5create(test_path_hdf5,'/DBP',[1 length(PPG)]);
        h5create(test_path_hdf5,'/ABP_AMP',[1 1]);
        h5create(test_path_hdf5,'/ABP_MIN',[1 1]);
        h5create(test_path_hdf5,'/ABP_GRND',[output_signal_length length(PPG)]);
        h5create(test_path_hdf5,'/ABP_RNorm',[output_signal_length length(PPG)]);
        h5write(test_path_hdf5,'/PPG',PPG);
        h5write(test_path_hdf5,'/ABP',ABP);
        h5write(test_path_hdf5,'/VPG',VPG);
        h5write(test_path_hdf5,'/APG',APG);
        h5write(test_path_hdf5,'/ECG',ECG);
        h5write(test_path_hdf5,'/ECG_RPR',ECG_RPR);
        h5write(test_path_hdf5,'/SBP',SBP);
        h5write(test_path_hdf5,'/DBP',DBP);
        h5write(test_path_hdf5,'/ABP_AMP',ABP_AMP);
        h5write(test_path_hdf5,'/ABP_MIN',ABP_MIN);
        h5write(test_path_hdf5,'/ABP_GRND',ABP_GRND);
        h5write(test_path_hdf5,'/ABP_RNorm',ABP_RNorm);
    end
end
%% Create X_Fold Cross Validation Train-Test-Validation Set, here X = 5 by Default
% clear
% load('Train_Dataset_3CH.mat');
% Data_Length = length(PPG);
% signal_length = 1024;
% Fold_num = 5;
% VL = floor(Data_Length / Fold_num);
% % Main Loop
% for i=1:Fold_num
%     fprintf('Current Fold Number: %d\n',i);
%     j = i-1;
%     if (i == 1)
%         PPG_1 = [];
%         VPG_1 = [];
%         APG_1 = [];
%         ABP_1 = [];
%         ABP_GRND_1 = [];
%     else
%         PPG_1 = PPG(:,1:VL*j);
%         VPG_1 = VPG(:,1:VL*j);
%         APG_1 = APG(:,1:VL*j);
%         ABP_1 = ABP(:,1:VL*j);
%         ABP_GRND_1 = ABP_GRND(:,1:VL*j);
%     end
%     if (i == Fold_num)
%         PPG_2 = [];
%         VPG_2 = [];
%         APG_2 = [];
%         ABP_2 = [];
%         ABP_GRND_2 = [];
%     else
%         PPG_2 = PPG(:,VL*i+1:VL*Fold_num);
%         VPG_2 = VPG(:,VL*i+1:VL*Fold_num);
%         APG_2 = APG(:,VL*i+1:VL*Fold_num);
%         ABP_2 = ABP(:,VL*i+1:VL*Fold_num);
%         ABP_GRND_2 = ABP_GRND(:,VL*i+1:VL*Fold_num);
%     end
% 
%     Train_PPG = horzcat(PPG_1,PPG_2);
%     Train_VPG = horzcat(VPG_1,VPG_2);
%     Train_APG = horzcat(APG_1,APG_2);
%     Train_ABP = horzcat(ABP_1,ABP_2);
%     Train_ABP_GRND = horzcat(ABP_GRND_1,ABP_GRND_2);
%     ABP_MAX = max(max(Train_ABP_GRND));
%     ABP_MIN = min(min(Train_ABP_GRND));
%     ABP_AMP = ABP_MAX - ABP_MIN;
%     %
%     Val_PPG = PPG(:,VL*j+1:VL*i);
%     Val_VPG = VPG(:,VL*j+1:VL*i);
%     Val_APG = APG(:,VL*j+1:VL*i);
%     Val_ABP = ABP(:,VL*j+1:VL*i);
%     Val_ABP_GRND = ABP_GRND(:,VL*j+1:VL*i);
%     % Make or Set Destination Directory
%     train_path_mat = sprintf('Folds_3CH/Fold %d/Train_Data_Fold_%d.mat',i,i);
%     val_path_mat = sprintf('Folds_3CH/Fold %d/Val_Data_Fold_%d.mat',i,i);
%     train_path_hdf5 = sprintf('Folds_3CH/Fold %d/Train_Data_Fold_%d.h5',i,i);
%     val_path_hdf5 = sprintf('Folds_3CH/Fold %d/Val_Data_Fold_%d.h5',i,i);
%     save(train_path_mat,'Train_PPG','Train_VPG','Train_APG','Train_ABP','Train_ABP_GRND','ABP_AMP','ABP_MIN','-v7.3');
%     save(val_path_mat,'Val_PPG','Val_VPG','Val_APG','Val_ABP','Val_ABP_GRND','-v7.3');
%     %
%     h5create(train_path_hdf5,'/PPG',[signal_length length(Train_PPG)]);
%     h5create(train_path_hdf5,'/VPG',[signal_length length(Train_PPG)]);
%     h5create(train_path_hdf5,'/APG',[signal_length length(Train_PPG)]);
%     h5create(train_path_hdf5,'/ABP',[signal_length length(Train_PPG)]);
%     h5create(train_path_hdf5,'/ABP_GRND',[signal_length length(Train_PPG)]);
%     h5create(train_path_hdf5,'/ABP_AMP',[1 1]);
%     h5create(train_path_hdf5,'/ABP_MIN',[1 1]);
%     h5write(train_path_hdf5,'/PPG',Train_PPG);
%     h5write(train_path_hdf5,'/VPG',Train_VPG);
%     h5write(train_path_hdf5,'/APG',Train_APG);
%     h5write(train_path_hdf5,'/ABP',Train_ABP);
%     h5write(train_path_hdf5,'/ABP_GRND',Train_ABP_GRND);
%     h5write(train_path_hdf5,'/ABP_AMP',ABP_AMP);
%     h5write(train_path_hdf5,'/ABP_MIN',ABP_MIN);
%     %
%     h5create(val_path_hdf5,'/PPG',[signal_length length(Val_PPG)]);
%     h5create(val_path_hdf5,'/VPG',[signal_length length(Val_PPG)]);
%     h5create(val_path_hdf5,'/APG',[signal_length length(Val_PPG)]);
%     h5create(val_path_hdf5,'/ABP',[signal_length length(Val_PPG)]);
%     h5create(val_path_hdf5,'/ABP_GRND',[signal_length length(Val_PPG)]);
%     h5write(val_path_hdf5,'/PPG',Val_PPG);
%     h5write(val_path_hdf5,'/VPG',Val_VPG);
%     h5write(val_path_hdf5,'/APG',Val_APG);
%     h5write(val_path_hdf5,'/ABP',Val_ABP);
%     h5write(val_path_hdf5,'/ABP_GRND',Val_ABP_GRND);
% end