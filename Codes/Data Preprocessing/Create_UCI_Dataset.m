%% Split UCI Train-Test Data
clear;
clc;
load('UCI Dataset/Part_1.mat');
load('UCI Dataset/Part_2.mat');
load('UCI Dataset/Part_3.mat');
load('UCI Dataset/Part_4.mat');
UCI_Train_Dataset = horzcat(Part_1, Part_2, Part_3);
UCI_Test_Dataset = Part_4;
%% Prepare UCI Train and Test Datasets
warning('off');
% Input Signal Length should be at least 60 samples more than the Output, or more
input_signal_length = 1120;
output_signal_length = 1024;
sampling_frequency = 125;
% Make or Set Destination Directory
fold_num = 1;
train_path_mat = sprintf('UCI_Train_Dataset_fold_%d.mat',fold_num);
train_path_hdf5 = sprintf('UCI_Train_Dataset_fold_%d.h5',fold_num);
test_path_mat = sprintf('UCI_Test_Dataset_fold_%d.mat',fold_num);
test_path_hdf5 = sprintf('UCI_Test_Dataset_fold_%d.h5',fold_num);
%
for c = 1:2
    if c == 1
        dataset_size = size(UCI_Train_Dataset);
        dataset_size = dataset_size(2);
    elseif c == 2
        dataset_size = size(UCI_Test_Dataset);
        dataset_size = dataset_size(2);
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
    ABP_TOT_UCI = zeros(output_signal_length,300000);
    ABP_RNorm_UCI = zeros(output_signal_length,300000);
    SBP_TOT_UCI = zeros(1,300000);
    DBP_TOT_UCI = zeros(1,300000);
    %
    for i = 1:dataset_size
        if c == 1
            cell = UCI_Train_Dataset{1,i};
        elseif c == 2
            cell = UCI_Test_Dataset{1,i};
        end
        cell_size = size(cell);
        cell_length = cell_size(2);
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
            if c == 1
                if counter > 147000
                    break
                end
            elseif c == 2
                if counter > 53000
                    break
                end
            end
            counter = counter+1;
            Iter_Left = dataset_size - i;
            %
            try
                PPG_PP = PPG(input_signal_length*ii+1:input_signal_length*ii+input_signal_length);
                PPG_PP = NotchFilterIIR(PPG_PP,50,sampling_frequency*2,25);
                PPG_PP = filtButter(PPG_PP,sampling_frequency*2,6,[0.01 30],'bandpass');
                PPG_PP = normalize(normalize(PPG_PP,'zscore'),'range');
                PPG_PP = Fix_Baseline_Drift(PPG_PP);
                PPG_PP = normalize(normalize(PPG_PP,'zscore'),'range');
            catch
                bad_signal_count = bad_signal_count + 1;
                counter = counter - 1;
                continue
            end
            %
            try
                ABP_PP1 = ABP(input_signal_length*ii+1:input_signal_length*ii+input_signal_length);
                ABP_PP2 = ABP_PP1;
                ABP_PP2 = NotchFilterIIR(ABP_PP2,50,sampling_frequency*2,25);
                ABP_PP2 = filtButter(ABP_PP2,sampling_frequency*2,6,[0.01 30],'bandpass');
                ABP_PP2 = Fix_Baseline_Drift(ABP_PP2);
                ABP_PP1_AMP = max(ABP_PP1) - min(ABP_PP1);
                ABP_PP2_AMP = max(ABP_PP2) - min(ABP_PP2);
                ABP_PP = (ABP_PP2*(ABP_PP1_AMP/ABP_PP2_AMP))+min(ABP_PP1);
            catch
                bad_signal_count = bad_signal_count + 1;
                counter = counter - 1;
                continue
            end
            %
            try
                ECG_PP = ECG(input_signal_length*ii+1:input_signal_length*ii+input_signal_length);
                ECG_PP = NotchFilterIIR(ECG_PP,50,sampling_frequency*2,25);
                ECG_PP = filtButter(ECG_PP,sampling_frequency*2,6,[0.05 40],'bandpass');
                ECG_PP = normalize(normalize(ECG_PP,'zscore'),'range');
                ECG_PP = Fix_Baseline_Drift(ECG_PP);
                ECG_PP = normalize(normalize(ECG_PP,'zscore'),'range');
            catch
                bad_signal_count = bad_signal_count + 1;
                counter = counter - 1;
                continue
            end
            %
            Decision = Remove_Bad_Signals(PPG_PP,ABP_PP,input_signal_length);
            TF1 = mean(isnan(PPG_PP));
            TF2 = mean(isnan(ABP_PP));
            TF = TF1+TF2;
            SBP = max(ABP_PP);
            DBP = min(ABP_PP);
            BP_DIFF = SBP - DBP;
            if (TF > 0)||((SBP > 190)||(SBP < 80)||(DBP > 120)||(DBP < 50)||(BP_DIFF < 20)||(BP_DIFF > 120))||(Decision == 0)
                bad_signal_count = bad_signal_count + 1;
                counter = counter - 1;
                continue
            else
                [PPG_PP,VPG_PP,APG_PP,ABP_PP,ABP_PP_RNorm,ECG_PP,delay] = PPG_diff(PPG_PP,ABP_PP,ECG_PP,input_signal_length,output_signal_length);
                TF3 = mean(isnan(ECG_PP));
                TF4 = mean(isnan(VPG_PP));
                TF5 = mean(isnan(APG_PP));
                TF = TF3+TF4+TF5;
                var_PPG = var(PPG_PP);
                var_VPG = var(VPG_PP);
                var_APG = var(APG_PP);
                var_ABP = var(ABP_PP);
                var_ECG = var(ECG_PP);
                if (TF > 0) || (var_PPG == 0) || (var_VPG == 0) || (var_APG == 0) || (var_ABP == 0) || (var_ECG == 0)
                    bad_signal_count = bad_signal_count + 1;
                    counter = counter - 1;
                    continue
                end
                PPG_TOT_UCI(:,counter) = PPG_PP';
                VPG_TOT_UCI(:,counter) = VPG_PP';
                APG_TOT_UCI(:,counter) = APG_PP';
                ECG_TOT_UCI(:,counter) = ECG_PP';
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
        save(train_path_mat,'PPG','VPG','APG','ECG','ABP','SBP','DBP','ABP_GRND','ABP_RNorm','ABP_AMP','ABP_MIN','-v7.3');
        %
        delete(train_path_hdf5);
        size_dataset = size(PPG);
        length = size_dataset(2);
        h5create(train_path_hdf5,'/PPG',[output_signal_length length]);
        h5create(train_path_hdf5,'/ABP',[output_signal_length length]);
        h5create(train_path_hdf5,'/VPG',[output_signal_length length]);
        h5create(train_path_hdf5,'/APG',[output_signal_length length]);
        h5create(train_path_hdf5,'/ECG',[output_signal_length length]);
        h5create(train_path_hdf5,'/SBP',[1 length]);
        h5create(train_path_hdf5,'/DBP',[1 length]);
        h5create(train_path_hdf5,'/ABP_AMP',[1 1]);
        h5create(train_path_hdf5,'/ABP_MIN',[1 1]);
        h5create(train_path_hdf5,'/ABP_GRND',[output_signal_length length]);
        h5create(train_path_hdf5,'/ABP_RNorm',[output_signal_length length]);
        h5write(train_path_hdf5,'/PPG',PPG);
        h5write(train_path_hdf5,'/ABP',ABP);
        h5write(train_path_hdf5,'/VPG',VPG);
        h5write(train_path_hdf5,'/APG',APG);
        h5write(train_path_hdf5,'/ECG',ECG);
        h5write(train_path_hdf5,'/SBP',SBP);
        h5write(train_path_hdf5,'/DBP',DBP);
        h5write(train_path_hdf5,'/ABP',ABP);
        h5write(train_path_hdf5,'/ABP_AMP',ABP_AMP);
        h5write(train_path_hdf5,'/ABP_MIN',ABP_MIN);
        h5write(train_path_hdf5,'/ABP_GRND',ABP_GRND);
        h5write(train_path_hdf5,'/ABP_RNorm',ABP_RNorm);
    elseif c == 2
        save(test_path_mat,'PPG','VPG','APG','ECG','ABP','SBP','DBP','ABP_GRND','ABP_AMP','ABP_MIN','-v7.3');
        %
        delete(test_path_hdf5);
        size_dataset = size(PPG);
        length = size_dataset(2);
        h5create(test_path_hdf5,'/PPG',[output_signal_length length]);
        h5create(test_path_hdf5,'/ABP',[output_signal_length length]);
        h5create(test_path_hdf5,'/VPG',[output_signal_length length]);
        h5create(test_path_hdf5,'/APG',[output_signal_length length]);
        h5create(test_path_hdf5,'/ECG',[output_signal_length length]);
        h5create(test_path_hdf5,'/SBP',[1 length]);
        h5create(test_path_hdf5,'/DBP',[1 length]);
        h5create(test_path_hdf5,'/ABP_AMP',[1 1]);
        h5create(test_path_hdf5,'/ABP_MIN',[1 1]);
        h5create(test_path_hdf5,'/ABP_GRND',[output_signal_length length]);
        h5create(test_path_hdf5,'/ABP_RNorm',[output_signal_length length]);
        h5write(test_path_hdf5,'/PPG',PPG);
        h5write(test_path_hdf5,'/ABP',ABP);
        h5write(test_path_hdf5,'/VPG',VPG);
        h5write(test_path_hdf5,'/APG',APG);
        h5write(test_path_hdf5,'/ECG',ECG);
        h5write(test_path_hdf5,'/SBP',SBP);
        h5write(test_path_hdf5,'/DBP',DBP);
        h5write(test_path_hdf5,'/ABP_AMP',ABP_AMP);
        h5write(test_path_hdf5,'/ABP_MIN',ABP_MIN);
        h5write(test_path_hdf5,'/ABP_GRND',ABP_GRND);
        h5write(test_path_hdf5,'/ABP_RNorm',ABP_RNorm);
    end
end
